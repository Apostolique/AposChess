// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
import { Chessground } from '@lichess-org/chessground';
import '@lichess-org/chessground/assets/chessground.base.css';
// brown.css is kept for its interactive-square styles + coord colors; the board
// texture and piece set it bundles are overridden by board-theme.css (blue2 + merida).
import '@lichess-org/chessground/assets/chessground.brown.css';
import './board-theme.css';
import './styles.css';

import { newGameState, parseFen, toFen, parseSquare, squareName, opponent } from './board.js';
import { applyMove, gameStatus, destsMap, legalMoves } from './engine.js';
import { hostGame, joinGame, normalizeCode, CODE_LENGTH } from './online.js';
import { findMatch } from './queue.js';
import { exportPgn, importPgn } from './pgn.js';

const boardEl = document.getElementById('board');
const $ = (id) => document.getElementById(id);

const ui = {
  mode: 'human-human', // 'human-human' | 'human-ai' | 'ai-ai' | 'online' | 'puzzle' | 'analysis' | 'editor'
  humanColor: 'white', // which side the human controls in 'human-ai'
  // AI strength per slot: a preset depth string '1'..'7', or 'custom'.
  strengthAi: '6',     // opponent strength in 'human-ai'
  strengthWhite: '6',  // per-colour strength in 'ai-ai'
  strengthBlack: '6',
  // Which evaluation drives each AI slot: 'handcrafted' or 'nn' (neural net).
  // Orthogonal to strength — the chosen engine still searches to the depth/time above.
  engineAi: 'handcrafted',
  engineWhite: 'handcrafted',
  engineBlack: 'handcrafted',
  // Selected neural-net name per slot (from the public/nn catalog); filled once the
  // manifest loads. Only consulted when that slot's engine is 'nn'.
  netAi: null,
  netWhite: null,
  netBlack: null,
  // Per-slot custom depth/timeout, used when that slot's strength is 'custom'.
  custom: {
    ai: { depth: 8, ms: 6000 },
    white: { depth: 8, ms: 6000 },
    black: { depth: 8, ms: 6000 },
  },
  maxMs: 6000,         // think-time cap (ms) for preset strengths
  delay: 450,          // ms pause before an AI move, so play is watchable
  running: false,      // AI-vs-AI loop active
  started: false,      // AI-vs-AI game has been started at least once (Pause/Resume vs Start)
  // AI-vs-AI opening variety: when on, the engine's first move (White) is forbidden
  // from matching its last few games, so consecutive games open differently.
  varyOpenings: false,
  recentOpenings: [],  // move keys (from*64+to) of recent games' opening moves, oldest first
  // AI-vs-AI: show each engine's own evaluation as a bar beside the board
  // (bottom engine left, top engine right). See the eval-bars section.
  showEvalBars: false,
  // Flip board: invert the point of view (board orientation, trays, eval-bar
  // sides). A pure view toggle, reset when the mode changes so mode-specific
  // auto-orientation (puzzles, online colour) starts out right.
  flipped: false,
};

// How many recent opening moves to remember/forbid, so the engines cycle through a
// few different first moves before one can recur.
const OPENING_HISTORY = 4;
// The standard initial position's FEN: the opening-variety filter only applies to a
// true fresh game from here, never to a loaded/edited starting position.
const STANDARD_START_FEN = toFen(newGameState());

let state = newGameState();
let status = gameStatus(state);
let aiTimer = null;

// Move/capture/check sounds: one reusable Audio element each, rewound before play
// so rapid consecutive moves (e.g. AI vs AI) still click. Public assets, so they
// resolve against Vite's base URL.
const moveSound = new Audio(import.meta.env.BASE_URL + 'sound/standard/Move.mp3');
const captureSound = new Audio(import.meta.env.BASE_URL + 'sound/standard/Capture.mp3');
const checkSound = new Audio(import.meta.env.BASE_URL + 'sound/standard/Check.mp3');
// Played once when an online opponent links up (host and joiner both hear it).
const connectSound = new Audio(import.meta.env.BASE_URL + 'sound/standard/SocialNotify.mp3');
// Check takes priority over capture (and covers checkmate, which is also a check).
function playMoveSound(capture, check) {
  const a = check ? checkSound : capture ? captureSound : moveSound;
  a.currentTime = 0;
  a.play().catch(() => {}); // ignore autoplay blocks before first interaction
}
function playConnectSound() {
  connectSound.currentTime = 0;
  connectSound.play().catch(() => {}); // gesture-unlocked by the Host/Join click
}
// Move history for review. Each entry is a snapshot after a ply; index 0 is the
// start position. `viewIndex` is the position shown on the board, which may lag
// the live game while the user steps back through earlier moves. Each entry also
// carries `evals` — what each engine's eval bar showed while that position was
// live (White's view; null = no opinion yet) — so review replays the bars too.
let history = [startEntry(state)];
let viewIndex = 0;

// FENs of every position in the live line, kept in lock-step with `history` and
// sent to the AI worker so its search knows the real game's repetitions — without
// this the engine has no game history and can shuffle a won position into a draw.
let repFens = [toFen(state)];

// The repetition positions worth sending to the worker for position `s`: only those
// since the last irreversible move (the last `halfmove` plies) can recur, so the
// engine's lookup set stays tiny. Older positions can never match (material/pawns
// differ), so dropping them changes nothing but speed.
const repWindow = (s) => repFens.slice(-(s.halfmove + 1));

// Board-editor mode: which side is to move once you leave the editor. The edited
// position only becomes a real game when you switch to a play mode (see the mode
// change handler), so during editing we don't maintain `state` from the board.
let editorTurn = 'white';

// Threefold-repetition tracking: count occurrences of each position over the live
// game. A position is identified by the first three FEN fields (piece placement,
// side to move, castling rights) — the same identity used for repetition in chess.
// The game is append-only (review changes viewIndex, not the live line), so a
// forward-only count is sufficient.
let posCounts = new Map();
const positionKey = (s) => toFen(s).split(' ', 3).join(' ');
function countPosition(s) {
  const k = positionKey(s);
  const n = (posCounts.get(k) || 0) + 1;
  posCounts.set(k, n);
  return n;
}
countPosition(state); // seed the start position

// Online (peer-to-peer) play. `online` is the active session (or null); `onlineColor`
// is the colour this client controls (host = White, joiner = Black); `onlineConnected`
// is true only once both peers are linked and the game is live.
let online = null;
let onlineColor = null;
let onlineConnected = false;
let isHost = false; // only the host assigns colours (and may swap sides)
let matchSession = null; // active matchmaking queue search (or null)
// True when the current online game came from matchmaking. Its private code is
// throwaway, so on a disconnect we go idle instead of keeping the lobby alive (that
// host-rehost behaviour is only useful for a code you deliberately shared).
let matchmade = false;

function startEntry(s) {
  return { state: s, lastMove: null, san: null, check: false, evals: { white: null, black: null } };
}

// Each AI colour gets its OWN worker — its own thread and its own persistent
// transposition table. In AI-vs-AI that lets both sides think at once: the side to
// move runs its real search while the other side ponders the reply it expects, the
// way two real engines trade ponder time. A worker *per colour* (not one shared
// worker serving whoever is on move) is what makes the warmth land where it's
// needed — White's pondering fills White's own table for White's next real search.
// you-vs-AI uses only the AI colour's worker; the other sits idle.
let aiThinking = false;
let lastCommitAt = performance.now(); // for pacing AI moves (the watch delay)
const PONDER_STEP_MS = 700; // ponder in short bursts so a worker stays responsive
// Per-colour AI state. `predicted` is this colour's guess of the opponent's next
// move (the `ponder` from its last real search); pondering searches the position
// after that guessed move. `searchSeq`/`ponderSeq` discard replies that a position
// change (new move, new game, stop) has superseded.
const ai = { white: makeAiSlot('white'), black: makeAiSlot('black') };
function makeAiSlot(color) {
  return { color, worker: null, searchSeq: 0, ponderSeq: 0, predicted: null, ponderState: null, ponderBest: 0 };
}
spawnAiWorkers();

function createAiWorker(slot) {
  const w = new Worker(new URL('./aiWorker.js', import.meta.url), { type: 'module' });
  w.onmessage = ({ data }) => {
    if (data.type === 'progress') onSearchProgress(slot, data);
    else if (data.type === 'ponder') onPonderResult(slot, data);
    else onSearchResult(slot, data);
  };
  w.onerror = (e) => {
    console.error(`AI worker (${slot.color}) error:`, e.message);
    aiThinking = false;
    updateStatusText();
  };
  return w;
}

// (Re)create both workers and reset their per-colour state, invalidating any
// in-flight replies (the recreated worker reuses the same slot object, so a stale
// message from the terminated one still fails the bumped seq check).
function spawnAiWorkers() {
  for (const c of ['white', 'black']) {
    if (ai[c].worker) ai[c].worker.terminate();
    ai[c].worker = createAiWorker(ai[c]);
    ai[c].searchSeq++;
    ai[c].ponderSeq++;
    ai[c].predicted = null;
    ai[c].ponderState = null;
    ai[c].ponderBest = 0;
  }
}

// Bump both colours' sequence counters so any pending search/ponder reply is
// discarded — a new position has superseded it (but the workers keep their warm
// tables, unlike cancelAi which throws them away).
function supersedeAi() {
  for (const c of ['white', 'black']) { ai[c].searchSeq++; ai[c].ponderSeq++; }
}

// Cancel any pending/in-flight AI work. Terminating is the only way to stop a deep
// search mid-think, so we replace both workers outright; that also discards their
// persistent transposition tables, which is correct for a hard reset.
function cancelAi() {
  clearTimeout(aiTimer);
  aiThinking = false;
  spawnAiWorkers();
  updateWakeLock();
}

// --- screen wake lock ---
// During automated AI play (esp. AI vs AI) there's no user input, so phones dim
// and then sleep. Hold a wake lock whenever it's the AI's turn to play; release
// it the rest of the time. Wake locks are auto-released when the tab is hidden,
// so we re-acquire on visibilitychange.
let wakeLock = null;
async function acquireWakeLock() {
  if (wakeLock || !('wakeLock' in navigator)) return;
  try {
    wakeLock = await navigator.wakeLock.request('screen');
    wakeLock.addEventListener('release', () => { wakeLock = null; });
  } catch { /* may be rejected (e.g. low battery, not visible); ignore */ }
}
function releaseWakeLock() {
  if (!wakeLock) return;
  wakeLock.release().catch(() => {});
  wakeLock = null;
}
function updateWakeLock() {
  if (aiToMove()) acquireWakeLock(); else releaseWakeLock();
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') updateWakeLock();
});

const cg = Chessground(boardEl, {
  fen: toFen(state),
  movable: { free: false, showDests: true, events: { after: onUserMove } },
  highlight: { lastMove: true, check: true },
  animation: { enabled: true, duration: 200 },
  events: { change: onBoardChange },
});

// Which colour the human may move right now (undefined = board locked).
function controllableColor() {
  if (status.over) return undefined;
  if (ui.mode === 'human-human') return state.turn;
  if (ui.mode === 'human-ai' && state.turn === ui.humanColor) return state.turn;
  if (ui.mode === 'online' && onlineConnected && state.turn === onlineColor) return state.turn;
  // Puzzle: the solver's side while the puzzle is live (not solved/revealed and not
  // waiting on the scripted reply); once it's done, the board opens up for free play,
  // so the side to move is hand-movable like analysis.
  if (ui.mode === 'puzzle' && puzzle) {
    if (puzzleDone()) return state.turn;
    if ((puzzlePhase === 'playing' || puzzlePhase === 'wrong') && state.turn === puzzleColor) return state.turn;
  }
  // Analysis: free play — the side to move is hand-movable (the other side becomes
  // movable once you navigate/fork to a ply where it's on move; see render()). The
  // engine never moves, it only suggests the best move via the eval-bar arrow.
  if (ui.mode === 'analysis') return state.turn;
  return undefined;
}

function aiToMove() {
  if (status.over) return false;
  if (ui.mode === 'human-ai') return state.turn !== ui.humanColor;
  if (ui.mode === 'ai-ai') return ui.running;
  // Puzzle mode never moves the engine automatically (analysis only suggests).
  return false;
}

// The colour shown at the bottom of the board (board orientation + bottom tray):
// the human's side in you-vs-AI, this client's side online, else White — then
// inverted while the Flip board toggle is on.
function viewColor() {
  let c = 'white';
  if (ui.mode === 'human-ai') c = ui.humanColor;
  else if (ui.mode === 'online') c = onlineColor || 'white';
  else if (ui.mode === 'puzzle') c = puzzleColor;
  else if (ui.mode === 'analysis') c = analysisOrient; // carried from the mode we entered from
  return ui.flipped ? opponent(c) : c;
}

function render() {
  // The editor owns the board directly (free placement); don't overwrite its pieces
  // from game state — just refresh the surrounding chrome. enterEditor() sets it up.
  if (ui.mode === 'editor') { updateStatusText(); applyAiLock(); return; }
  const entry = history[viewIndex];
  const atLive = viewIndex === history.length - 1;
  let color = atLive ? controllableColor() : undefined;
  let dests = (atLive && color) ? destsMap(status.legal) : new Map();
  // Analysis (and a finished puzzle): reviewing isn't read-only — EITHER side can move
  // from any viewed position, which FORKS the game there (the continuation after it is
  // discarded by onUserMove), so "what if" is one rewind away for both colours.
  // Everywhere else an off-live view stays locked.
  if (!atLive && (ui.mode === 'analysis' || (ui.mode === 'puzzle' && puzzleDone()))) {
    color = entry.state.turn;
    dests = destsMap(legalMoves(entry.state));
  }
  cg.set({
    fen: toFen(entry.state),
    turnColor: entry.state.turn,
    orientation: viewColor(),
    lastMove: entry.lastMove ? [squareName(entry.lastMove.from), squareName(entry.lastMove.to)] : undefined,
    check: entry.check ? entry.state.turn : false,
    // NEVER viewOnly:true — chessground only binds its mouse/touch handlers during
    // a redrawAll, and SKIPS binding them entirely while viewOnly is set (set()
    // alone never rebinds). So any redrawAll that lands while a review lock was on
    // (the eval bar appearing/hiding, an orientation change) would leave the board
    // permanently dead to input, in every mode, until a lucky future redrawAll.
    // The movable config below already locks the board fully on its own: with no
    // colour and no dests, nothing can be selected, dragged, or moved.
    viewOnly: false,
    // Explicitly restore non-editor settings (free placement, delete-on-drop-off,
    // and the move-destination dots) — cg.set merges, so the editor's overrides
    // would otherwise persist after leaving it.
    movable: { free: false, showDests: true, color, dests },
    draggable: { deleteOnDropOff: false },
  });
  renderTrays(entry);
  renderMoveList();
  updateStatusText();
  applyAiLock();
  syncEvalBar();
  // Puzzle mode owns its auto-shapes (syncEvalBar left them alone above): keep the
  // solved ✓ in sync with the viewed ply. Other modes manage shapes themselves.
  if (ui.mode === 'puzzle') paintPuzzleGlyph();
}

// --- material difference / advantage trays (Lichess-style) ---
const POINTS = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

function pointAdvantage(board, color) {
  let w = 0, b = 0;
  for (const p of board) {
    if (!p) continue;
    if (p.color === 'white') w += POINTS[p.role]; else b += POINTS[p.role];
  }
  const diff = w - b;
  return color === 'white' ? diff : -diff;
}

function renderTray(el, color, board) {
  // Lichess-style material diff, derived from the BOARD rather than capture
  // history: per role, show this side's surplus of pieces still standing. Equal
  // trades cancel per role for free, and promotion reads correctly — a pawn
  // promoting to a knight shows as being up a knight (capture history would keep
  // crediting the long-gone pawn, then "a knight" to whoever captures the
  // promoted piece, while the point score said something else entirely).
  const counts = { white: {}, black: {} };
  for (const p of board) {
    if (p) counts[p.color][p.role] = (counts[p.color][p.role] || 0) + 1;
  }
  const mine = counts[color], theirs = counts[opponent(color)];
  const surplus = [];
  for (const role of ['q', 'r', 'b', 'n', 'p']) { // descending value
    for (let i = (mine[role] || 0) - (theirs[role] || 0); i > 0; i--) surplus.push(role);
  }
  // Mono (single-colour) piece glyphs, like Lichess's material diff — whose pieces
  // they were is clear from which tray they're in, so colouring them is noise.
  el.querySelector('.caps').innerHTML = surplus
    .map((role) => `<span class="cap cap-${role}"></span>`)
    .join('');
  const adv = pointAdvantage(board, color);
  el.querySelector('.adv').textContent = adv > 0 ? `+${adv}` : '';
}

function renderTrays(entry) {
  // Bottom player matches board orientation; top player is the opponent.
  const bottom = viewColor();
  renderTray($('tray-bottom'), bottom, entry.state.board);
  renderTray($('tray-top'), opponent(bottom), entry.state.board);
}

// chessground fires this on any board mutation. In the editor (palette drops,
// drag-off deletions, Clear/Start) the board changes outside game state, and
// render() skips the editor entirely — so refresh the material trays from the
// edited placement here. Other modes drive the trays through render().
function onBoardChange() {
  if (ui.mode !== 'editor') return;
  const board = parseFen(`${cg.getFen()} w - - 0 1`).board;
  renderTrays({ state: { board } });
}

// Relocate the single #status element to where its message belongs: a concise live
// turn/thinking indicator sits in the to-move player's own tray (right of their
// material diff, on their side of the board), while everything else — puzzle prompts,
// game-over/draw banners, the editor and connection notices — goes below the board.
// `where` is 'below' or a colour; a colour maps to the matching tray via orientation.
// The 'below' banner resolves to whichever slot is showing: above the move list on
// desktop (the list has its own column there), or below the board on mobile (where
// that column drops underneath). Re-run placement when crossing the breakpoint.
const desktopMql = window.matchMedia('(min-width: 981px)');
desktopMql.addEventListener('change', updateStatusText);
function placeStatus(el, where) {
  const belowSlot = desktopMql.matches ? $('status-above') : $('status-below');
  const parent = where === 'below' ? belowSlot
    : where === viewColor() ? $('tray-bottom') : $('tray-top');
  if (el.parentElement !== parent) parent.appendChild(el);
  el.classList.toggle('status-side', where !== 'below');
}

function updateStatusText() {
  const el = $('status');
  if (ui.mode === 'editor') {
    el.classList.remove('over');
    el.textContent = `Editing — ${editorTurn === 'white' ? 'White' : 'Black'} to move. Switch Mode to play.`;
    placeStatus(el, 'below');
    return;
  }
  if (ui.mode === 'puzzle') {
    renderPuzzleMeta();
    el.classList.toggle('over', puzzlePhase === 'solved' || puzzlePhase === 'revealed');
    placeStatus(el, 'below');
    if (!puzzle) {
      el.textContent = puzzleCatalog === null ? 'Loading puzzles…'
        : puzzleCatalog.length === 0 ? 'No puzzles available.'
        : 'No puzzles match these filters.';
      return;
    }
    const side = puzzleColor === 'white' ? 'White' : 'Black';
    if (puzzlePhase === 'solved') el.textContent = 'Solved! Well done.';
    else if (puzzlePhase === 'revealed') el.textContent = 'Solution shown.';
    else if (puzzlePhase === 'wrong') el.textContent = 'Not the move — try again.';
    else if (state.turn !== puzzleColor) el.textContent = '…';
    else if (puzzle.kind === 'defense') el.textContent = `${side} to move — only one move doesn't lose. Find it${status.check ? ' (check!)' : ''}.`;
    else el.textContent = `${side} to move — find the best move${status.check ? ' (check!)' : ''}.`;
    return;
  }
  el.classList.toggle('over', status.over);
  // Live turn/thinking → the to-move player's tray; banners/results → below the board.
  let placement = 'below';
  if (status.result === 'checkmate') {
    el.textContent = `Checkmate — ${status.winner === 'white' ? 'White' : 'Black'} wins.`;
  } else if (status.result === 'stalemate') {
    el.textContent = 'Stalemate — draw.';
  } else if (status.result === 'insufficient-material') {
    el.textContent = 'Draw — insufficient material.';
  } else if (status.result === 'fifty-move') {
    el.textContent = 'Draw — fifty-move rule.';
  } else if (status.result === 'repetition') {
    el.textContent = 'Draw — threefold repetition.';
  } else if (ui.mode === 'online' && !onlineConnected) {
    el.textContent = 'Not connected — host or join a game.';
  } else if (aiThinking) {
    const side = state.turn === 'white' ? 'White' : 'Black';
    el.textContent = `${side} is thinking…`;
    placement = state.turn;
  } else if (ui.mode === 'analysis' && evalBar.pending) {
    // Analysis: the eval-bar/best-move-arrow search is running. Show the same
    // "thinking…" line as the play modes, for the side being analysed (the viewed
    // position's turn, which the eval-bar worker is searching).
    const side = evalBar.turn === 'white' ? 'White' : 'Black';
    el.textContent = `${side} is thinking…`;
    placement = evalBar.turn;
  } else {
    const side = state.turn === 'white' ? 'White' : 'Black';
    el.textContent = `${side} to move${status.check ? ' — check!' : ''}`;
    placement = state.turn;
  }
  placeStatus(el, placement);
}

// Apply one move to the live game and record it (state, status, history snapshot,
// repetition FEN). Shared by interactive play (`commit`) and bulk replay
// (`loadGame`); does no rendering, sound, or AI driving of its own.
function recordMove(move) {
  const pre = state;
  state = applyMove(pre, move);
  status = gameStatus(state);
  // Threefold repetition is a draw. Override only if the game isn't already over
  // (checkmate/stalemate/insufficient-material/fifty-move take precedence and may share the position).
  if (!status.over && countPosition(state) >= 3) {
    status = { over: true, check: status.check, legal: status.legal, result: 'repetition', winner: null };
  }
  history.push({
    state,
    lastMove: move,
    san: toSan(pre, move, status),
    check: status.check,
    // Carry the engines' opinions forward: the mover's score for this move is
    // already stamped on the pre-move entry (onSearchResult), so the new live
    // entry starts from both engines' latest views.
    evals: { ...history[history.length - 1].evals },
  });
  repFens.push(toFen(state));
}

function commit(move) {
  // A new position supersedes any pending search or ponder reply.
  clearTimeout(aiTimer);
  supersedeAi();
  aiThinking = false;
  const wasLive = viewIndex === history.length - 1;
  // If this is the opening move the variety filter constrained, remember it so the
  // next game's filter forbids it (checked before recordMove advances the ply).
  const isVariedOpening = ui.mode === 'ai-ai' && ui.varyOpenings
    && history.length === 1 && toFen(state) === STANDARD_START_FEN;
  recordMove(move);
  if (isVariedOpening) rememberOpening(move);
  if (wasLive) viewIndex = history.length - 1; // follow the game unless reviewing
  lastCommitAt = performance.now();
  // Only sound the new move if we're following the live game; while reviewing an
  // earlier position (e.g. rewound during an AI-vs-AI game) the move isn't shown,
  // so it shouldn't click.
  if (wasLive) playMoveSound(move.capture, status.check);
  render();
  driveAi();
}

// Long algebraic notation: always shows the from-square, so it needs no
// disambiguation and reads clearly even for the variant's jumps and knight
// slides. Adds '+'/'#' for check/checkmate and '=Q' for promotion.
function toSan(pre, move, st) {
  let s;
  if (move.castle === 'K') s = 'O-O';
  else if (move.castle === 'Q') s = 'O-O-O';
  else {
    const piece = pre.board[move.from];
    const letter = piece.role === 'p' ? '' : piece.role.toUpperCase();
    const sep = move.capture ? '×' : '–';
    const promo = move.promotion ? '=' + move.promotion.toUpperCase() : '';
    s = letter + squareName(move.from) + sep + squareName(move.to) + promo;
  }
  if (st.result === 'checkmate') s += '#';
  else if (st.check) s += '+';
  return s;
}

// --- move list & review navigation ---
function moveSpan(k) {
  return `<span class="move${k === viewIndex ? ' current' : ''}" data-i="${k}">${history[k].san}</span>`;
}

function renderMoveList() {
  const el = $('moves');
  if (history.length <= 1) { el.innerHTML = '<div class="empty">No moves yet.</div>'; return; }
  let html = '';
  for (let k = 1; k < history.length; k += 2) {
    const black = (k + 1 < history.length) ? moveSpan(k + 1) : '<span></span>';
    html += `<div class="moverow"><span class="moveno">${(k + 1) / 2}.</span>${moveSpan(k)}${black}</div>`;
  }
  el.innerHTML = html;
  const cur = el.querySelector('.current');
  // Scroll within the move list only — `scrollIntoView` would also scroll
  // every ancestor (the page itself on mobile) and yank the viewport around.
  if (cur) {
    const top = cur.offsetTop - el.clientTop;
    const bottom = top + cur.offsetHeight;
    if (top < el.scrollTop) el.scrollTop = top;
    else if (bottom > el.scrollTop + el.clientHeight) el.scrollTop = bottom - el.clientHeight;
  }
}

function goTo(index) {
  const clamped = Math.max(0, Math.min(history.length - 1, index));
  if (clamped === viewIndex) return;
  // Stepping/jumping forward replays moves: sound the one we land on (its capture
  // and check flags). Stepping back is silent.
  if (clamped > viewIndex) {
    const entry = history[clamped];
    playMoveSound(entry.lastMove && entry.lastMove.capture, entry.check);
  }
  viewIndex = clamped;
  render();
}

// Make the viewed ply the live end of the game: drop everything after it and
// rebuild the live bookkeeping (state, status, repetition counts) from the
// snapshot. Backs the rewind-and-fork in analysis mode.
function truncateToView() {
  history.length = viewIndex + 1;
  repFens.length = viewIndex + 1;
  const entry = history[viewIndex];
  state = entry.state;
  status = gameStatus(state);
  // Threefold tracking is forward-only, so recount over the surviving line.
  posCounts = new Map();
  for (const f of repFens) {
    const k = f.split(' ', 3).join(' '); // same identity as positionKey
    posCounts.set(k, (posCounts.get(k) || 0) + 1);
  }
}

// chessground reports a legal (from, to); resolve it to an engine move,
// asking for a promotion piece when several moves share that destination.
function onUserMove(orig, dest) {
  if (ui.mode === 'editor') return; // free placement: chessground keeps the move, no game logic
  // A move played while reviewing (enabled in analysis and in a finished puzzle —
  // everywhere else an off-live board is view-only) forks the game at the viewed ply:
  // the moves after it are discarded and this move becomes the new continuation.
  if (viewIndex !== history.length - 1) {
    if (ui.mode !== 'analysis' && !(ui.mode === 'puzzle' && puzzleDone())) { render(); return; }
    truncateToView();
  }
  const from = parseSquare(orig), to = parseSquare(dest);
  const matches = status.legal.filter((m) => m.from === from && m.to === to);
  if (matches.length === 0) { render(); return; }
  if (matches.length > 1 && matches[0].promotion) {
    askPromotion(state.turn).then((role) => {
      playLocalMove(matches.find((m) => m.promotion === role) || matches[0]);
    });
  } else {
    playLocalMove(matches[0]);
  }
}

// A move the local human made: commit it, and in online play relay it to the peer
// as a minimal {from,to,promotion} so the other client reconstructs the identical
// engine move from its own legalMoves (no chance of the two games diverging).
function playLocalMove(move) {
  // A live puzzle routes through the solution-checker; a finished one is free play,
  // so its moves just commit like any other mode.
  if (ui.mode === 'puzzle' && !puzzleDone()) { playPuzzleMove(move); return; }
  if (ui.mode === 'puzzle') { commit(move); return; }
  commit(move);
  if (ui.mode === 'online' && online && onlineConnected) {
    online.send({ t: 'move', from: move.from, to: move.to, promotion: move.promotion || null });
  }
}

// After every position change, point each engine-controlled colour at the right
// job: the side to move runs a real search; any other AI colour (only in AI-vs-AI)
// ponders the reply it expects, so its next real search starts warm. Both run
// concurrently on their own workers.
function driveAi() {
  clearTimeout(aiTimer);
  updateWakeLock();
  aiThinking = false;
  if (!status.over) {
    const movers = aiColors();
    for (const c of ['white', 'black']) {
      if (!movers.includes(c)) continue;
      if (c === state.turn) startSearch(ai[c]);   // its move: real search
      else startPonder(ai[c]);                     // idle: ponder the expected reply
    }
  }
  updateStatusText();
}

// Which colours are engine-controlled right now. In AI-vs-AI both sides are (while
// running) so one searches while the other ponders; in you-vs-AI only the AI side.
function aiColors() {
  if (status.over) return [];
  if (ui.mode === 'human-ai') return [opponent(ui.humanColor)];
  if (ui.mode === 'ai-ai') return ui.running ? ['white', 'black'] : [];
  // Puzzle mode never drives the AI automatically — analysis only shows the
  // engine's suggestion (the eval-bar best-move arrow), it never plays a move.
  return [];
}

// Think immediately (no pre-delay): the search overlaps the watch pause, so the AI
// is never idle. The result is held until at least `ui.delay` has elapsed since the
// last move, keeping AI-vs-AI watchable.
function startSearch(slot) {
  const seq = ++slot.searchSeq;
  slot.ponderSeq++; // a real search supersedes this colour's ponder chain
  aiThinking = true;
  const { depth, maxMs, engine, net } = aiParams(slot.color);
  slot.worker.postMessage({ type: 'search', seq, state, depth, maxMs, engine, net, posHistory: repWindow(state), exclude: openingExclude(), wasmUrl: WASM_URL });
}

// Move keys to forbid for the game's very first move, so an AI-vs-AI game doesn't
// replay a recent opening. Active only in AI-vs-AI with the option on, only on the
// opening ply of a standard fresh game (a loaded/edited start is left alone). The
// search applies this to White's first move; the game diverges from there.
function openingExclude() {
  if (ui.mode !== 'ai-ai' || !ui.varyOpenings) return undefined;
  if (history.length !== 1) return undefined;             // not the opening ply
  if (toFen(state) !== STANDARD_START_FEN) return undefined; // loaded/edited start — don't constrain
  return ui.recentOpenings.length ? ui.recentOpenings.slice() : undefined;
}

// Remember an opening move so later games avoid it (rolling window, most-recent last;
// re-playing a remembered move just moves it to the front of the window).
function rememberOpening(move) {
  const key = move.from * 64 + move.to;
  ui.recentOpenings = ui.recentOpenings.filter((k) => k !== key);
  ui.recentOpenings.push(key);
  while (ui.recentOpenings.length > OPENING_HISTORY) ui.recentOpenings.shift();
}

function onSearchResult(slot, data) {
  if (data.seq !== slot.searchSeq) return; // stale result for a superseded position
  slot.predicted = data.ponder || null;    // remember the predicted opponent reply
  // This engine's own opinion of the position it just searched (side-to-move
  // relative, i.e. this colour's view), stamped in White's view on the live
  // history entry — the position it searched (the seq check above rules out a
  // superseded position) — so review can replay the eval bars ply by ply.
  if (typeof data.score === 'number') {
    history[history.length - 1].evals[slot.color] = slot.color === 'white' ? data.score : -data.score;
    if (duelBarsVisible()) paintDuelBars();
  }
  // Only the side to move's real search yields a move to play (a ponder-side worker
  // never reaches here as a real search). Re-check the turn: the position may have
  // moved on while this reply was in flight.
  if (!data.move || slot.color !== state.turn || !aiToMove()) { aiThinking = false; updateStatusText(); return; }
  const move = data.move;
  const seq = data.seq;
  const wait = Math.max(0, lastCommitAt + ui.delay - performance.now());
  aiTimer = setTimeout(() => {
    if (slot.searchSeq !== seq || slot.color !== state.turn) return;
    commit(move);
  }, wait);
}

// A mid-search progress update from the side-to-move's worker: the engine's best
// score at the latest finished iterative-deepening depth, posted while the deeper
// iterations are still running, so the eval bar climbs in real time instead of only
// settling when the move is played. Same staleness guard and White-view convention
// as the final result — it stamps the live ply, which onSearchResult then refines
// with the deepest score.
function onSearchProgress(slot, data) {
  if (data.seq !== slot.searchSeq) return; // stale: the position moved on
  if (typeof data.score !== 'number') return;
  history[history.length - 1].evals[slot.color] = slot.color === 'white' ? data.score : -data.score;
  if (duelBarsVisible()) paintDuelBars();
}

// Ponder = think about the position we'd reach if the side to move plays the move
// this colour predicted for it. On a hit the next real search reuses the work; on a
// miss the table still holds useful overlap. Done in short bursts so a real move
// request (once the position changes) is picked up within one burst.
function startPonder(slot) {
  if (!slot.predicted) return; // nothing predicted yet (e.g. the first ply)
  const pm = status.legal.find((m) => m.from === slot.predicted.from && m.to === slot.predicted.to);
  if (!pm) return; // stale/illegal guess — skip pondering this turn
  slot.ponderState = applyMove(state, pm);
  slot.ponderBest = 0;
  const seq = ++slot.ponderSeq;
  const { depth, engine, net } = aiParams(slot.color);
  slot.worker.postMessage({ type: 'ponder', seq, state: slot.ponderState, depth, maxMs: PONDER_STEP_MS, engine, net, posHistory: repWindow(slot.ponderState), wasmUrl: WASM_URL });
}

function onPonderResult(slot, data) {
  if (data.seq !== slot.ponderSeq) return;
  if (slot.color === state.turn || !aiColors().includes(slot.color)) return; // no longer the idle side
  // Keep the idle engine's bar live during the opponent's turn: the ponder score is
  // its assessment of the position after the move it predicts the opponent will play
  // (side-to-move-relative there = this colour), shown in White's view on the live
  // ply just like a real search. If the prediction holds, the next real search picks
  // up seamlessly; if not, the bar corrects when that search starts.
  if (typeof data.score === 'number') {
    history[history.length - 1].evals[slot.color] = slot.color === 'white' ? data.score : -data.score;
    if (duelBarsVisible()) paintDuelBars();
  }
  const { depth, engine, net } = aiParams(slot.color);
  // Stop once we've searched to full strength or stopped making progress (e.g. a
  // forced line resolved), so we don't spin firing instant bursts.
  if (data.reached >= depth || data.reached <= slot.ponderBest) return;
  slot.ponderBest = data.reached;
  slot.worker.postMessage({ type: 'ponder', seq: data.seq, state: slot.ponderState, depth, maxMs: PONDER_STEP_MS, engine, net, posHistory: repWindow(slot.ponderState), wasmUrl: WASM_URL });
}

// --- puzzle mode ---
// Puzzles come from public/puzzles.json (mined by scripts/mine-puzzles.mjs —
// fetched at runtime like the nn catalog, so adding puzzles needs no rebuild).
// Each entry is { fen, moves, kind, themes, difficulty, mateIn? } where `moves` is
// the full solution line in from-to(-promotion) square names, solver's moves at the
// even indices. The solver plays the side to move in `fen`; correct moves commit
// into the normal game machinery (history, sounds, review all work), the scripted
// reply follows after the watch delay, and a wrong move snaps back — unless it
// delivers checkmate on the spot, which is accepted: the mined line is the unique
// *best* line, but mate is mate.
let puzzleCatalog = null;  // null until fetched; [] if the fetch failed or was empty
let puzzleList = [];       // current filtered, shuffled play order
let puzzleIdx = -1;        // position in puzzleList
let puzzleBeaten = new Set(); // puzzleList indices solved/revealed this session — they
                              // stay navigable (re-dealing one doesn't re-lock "Next")
let puzzle = null;         // the active catalog entry
let puzzleColor = 'white'; // the solver's side (the side to move in the puzzle FEN)
let puzzlePhase = 'idle';  // 'playing' | 'wrong' | 'solved' | 'revealed' | 'idle'
let puzzleStep = 0;        // index into puzzle.moves of the next expected move
let puzzleTimer = null;    // pending scripted reply / solution playback step
let puzzleSolvedPly = -1;  // history index of the solving move (-1 until solved); the ✓ pins to it

// A finished puzzle (solved or solution shown). The board then opens up for free
// exploration — either side hand-movable, rewind-and-fork like analysis — instead of
// staying locked, so you can try other continuations right where you are.
const puzzleDone = () => puzzlePhase === 'solved' || puzzlePhase === 'revealed';

// --- analysis mode ---
// Board orientation while in analysis (carried from whatever mode we entered from,
// then togglable via Flip board). See viewColor().
let analysisOrient = 'white';
// A snapshot of the puzzle session we entered analysis from (null = analysis was not
// reached from a puzzle). It drives the "Back to puzzle" button and is the data the
// button restores, so leaving analysis resumes the puzzle exactly where it was rather
// than re-dealing it. See snapshotPuzzleSession/resumePuzzleSession.
let puzzleSession = null;
// One-shot hand-off: set just before switching to puzzle mode so enterPuzzleMode
// resumes this saved session instead of dealing a fresh puzzle.
let puzzleResume = null;

// The analysis engine: a fixed strong default (the Opponent row isn't shown in
// puzzle mode — analysis wants one good answer, not a sparring partner's level).
// The loop-champion net at depth 7, our strongest eval; net resolved at request
// time (the catalog has loaded by the time you reach analysis), falling back to
// material if the catalog is unavailable. This drives the analysis eval bar AND
// the best-move arrow that shares its reply (see EVAL_BAR), so the two agree.
const PUZZLE_AI = { depth: 7, engine: 'nn', net: 'loop-champion' };

const puzzleUci = (m) => squareName(m.from) + squareName(m.to) + (m.promotion || '');

// Filter predicates for the two selects. Difficulty is the mining depth at which
// the engine first finds the key move (1 = sees it immediately).
const PUZZLE_DIFFICULTY = {
  easy: (p) => p.difficulty <= 2,
  medium: (p) => p.difficulty === 3 || p.difficulty === 4,
  hard: (p) => p.difficulty >= 5,
};
const PUZZLE_THEME = {
  mate: (p) => p.kind === 'mate',
  'razor-edge': (p) => p.themes.includes('razor-edge'), // the right move wins, everything else loses
  defense: (p) => p.kind === 'defense',                 // only-move saves
  jump: (p) => p.themes.includes('jump') || p.themes.includes('jump-capture'),
  'jump-block': (p) => p.themes.includes('jump-block'),
  knight: (p) => p.themes.includes('knight') || p.themes.includes('knight-block'),
  sacrifice: (p) => p.themes.includes('sacrifice'),
  promotion: (p) => p.themes.includes('promotion'),
};

async function enterPuzzleMode() {
  clearTimeout(puzzleTimer);
  // Returning from analysis (the "Back to puzzle" button): resume the saved session
  // instead of dealing a new puzzle. The catalog is already loaded (we were puzzling
  // before), so no fetch is needed.
  if (puzzleResume) { const s = puzzleResume; puzzleResume = null; resumePuzzleSession(s); return; }
  render(); // lock the board while the catalog loads
  if (!puzzleCatalog) {
    try {
      const url = new URL(import.meta.env.BASE_URL + 'puzzles.json', location.href).href;
      const data = await fetch(url).then((r) => r.json());
      puzzleCatalog = Array.isArray(data.puzzles) ? data.puzzles : [];
    } catch { puzzleCatalog = []; }
    if (ui.mode !== 'puzzle') return; // switched away while the fetch was in flight
  }
  refilterPuzzles();
  nextPuzzle();
}

// Capture everything needed to resume the current puzzle session after a detour
// through analysis: the live game (state/status/history/repFens/viewIndex/counts) and
// the puzzle tracking (which puzzle, whose move, phase, progress). The arrays are
// copied so analysis's forking/truncation can't mutate the saved session; the catalog
// and filtered play order (puzzleList) are left shared — analysis can't change them.
function snapshotPuzzleSession() {
  return {
    state, status,
    history: history.slice(),
    repFens: repFens.slice(),
    viewIndex,
    posCounts: new Map(posCounts),
    puzzle, puzzleColor, puzzlePhase, puzzleStep, puzzleIdx, puzzleSolvedPly,
  };
}

// Restore a session captured by snapshotPuzzleSession — the inverse: drop any analysis
// work in progress and re-establish the puzzle exactly where it was left.
function resumePuzzleSession(s) {
  clearTimeout(puzzleTimer);
  cancelAi();
  state = s.state;
  status = s.status;
  history = s.history;
  repFens = s.repFens;
  viewIndex = s.viewIndex;
  posCounts = s.posCounts;
  puzzle = s.puzzle;
  puzzleColor = s.puzzleColor;
  puzzlePhase = s.puzzlePhase;
  puzzleStep = s.puzzleStep;
  puzzleIdx = s.puzzleIdx;
  puzzleSolvedPly = s.puzzleSolvedPly ?? -1;
  lastCommitAt = performance.now();
  cg.setAutoShapes([]);
  applyModeVisibility();
  render();
}

// Rebuild the play order from the current filter selects: filter, then shuffle so
// "Next puzzle" walks a fresh random order (repeats only after the list wraps).
function refilterPuzzles() {
  const diff = PUZZLE_DIFFICULTY[$('puzzle-difficulty').value];
  const theme = PUZZLE_THEME[$('puzzle-theme').value];
  puzzleList = (puzzleCatalog || []).filter((p) => (!diff || diff(p)) && (!theme || theme(p)));
  for (let i = puzzleList.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [puzzleList[i], puzzleList[j]] = [puzzleList[j], puzzleList[i]];
  }
  puzzleIdx = -1;
  puzzleBeaten.clear();
}

function nextPuzzle() {
  if (puzzleList.length === 0) {
    clearTimeout(puzzleTimer);
    puzzle = null;
    puzzlePhase = 'idle';
    cg.setAutoShapes([]);
    applyModeVisibility();
    render();
    return;
  }
  puzzleIdx = (puzzleIdx + 1) % puzzleList.length;
  loadPuzzle(puzzleList[puzzleIdx]);
}

// Step back to the puzzle before this one in the session's walk. No wraparound: at
// the start of the list there's nothing earlier to revisit (the button is disabled
// there), so going back only ever re-deals puzzles already seen this session.
function prevPuzzle() {
  if (puzzleList.length === 0 || puzzleIdx <= 0) return;
  puzzleIdx--;
  loadPuzzle(puzzleList[puzzleIdx]);
}

// Reset the live game to the puzzle's position (the same bookkeeping as loadGame,
// minus the replay) and arm the solution tracking. When the catalog carries the
// lead-in (`prefen` + `opening`, the blunder that created the puzzle), start one
// ply earlier and animate the opponent playing it — seeing the move that just
// happened is half the orientation (and how lichess puzzles open).
function loadPuzzle(p) {
  clearTimeout(puzzleTimer);
  cancelAi();
  ui.running = false;
  ui.started = false;
  syncToggleLabel();
  state = parseFen(p.fen);
  status = gameStatus(state);
  let lead = null;
  if (p.prefen && p.opening) {
    const pre = parseFen(p.prefen);
    const preStatus = gameStatus(pre);
    lead = preStatus.legal.find((m) => puzzleUci(m) === p.opening) || null;
    if (lead) { state = pre; status = preStatus; } // fall back to the puzzle position if not
  }
  history = [startEntry(state)];
  repFens = [toFen(state)];
  viewIndex = 0;
  posCounts = new Map();
  countPosition(state);
  lastCommitAt = performance.now();
  puzzle = p;
  puzzleColor = lead ? opponent(state.turn) : state.turn; // the solver = the side to move in p.fen
  puzzlePhase = 'playing';
  puzzleStep = 0;
  puzzleSolvedPly = -1;
  cg.setAutoShapes([]);
  applyModeVisibility(); // keep the puzzle panel/buttons in sync with the new puzzle
  render();
  // Let the starting position paint, then play the blunder (commit animates it and
  // sounds it like any move); the board unlocks for the solver when it lands.
  if (lead) puzzleTimer = setTimeout(() => commit(lead), 650);
}

// Lichess-style feedback glyphs, drawn into a square's top-right corner (the
// customSvg cell is the 100×100 viewBox of the whole square). The green ✓ marks
// the move that solved the puzzle; the red ✗ flashes on a wrong try.
const GLYPH_SOLVED = '<g transform="translate(54,2)"><circle cx="22" cy="22" r="22" fill="#629924" stroke="#fff" stroke-width="2"/><path d="M11 23l7 7 13-15" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></g>';
const GLYPH_WRONG = '<g transform="translate(54,2)"><circle cx="22" cy="22" r="22" fill="#c0392b" stroke="#fff" stroke-width="2"/><path d="M13 13l18 18M31 13l-18 18" stroke="#fff" stroke-width="5" stroke-linecap="round"/></g>';

// The puzzle-mode auto-shapes, recomputed from the viewed position on every render
// (in puzzle mode syncEvalBar leaves shapes untouched, so this owns them). The green
// ✓ pins to the solving ply (puzzleSolvedPly), so it shows only while that exact move
// is in view: it vanishes when you play on or rewind, and returns when you step back
// to it. A transient wrong-move ✗ is drawn directly by playPuzzleMove, outside this
// path.
function paintPuzzleGlyph() {
  const entry = history[viewIndex];
  cg.setAutoShapes(
    puzzlePhase === 'solved' && viewIndex === puzzleSolvedPly && entry && entry.lastMove
      ? [{ orig: squareName(entry.lastMove.to), customSvg: { html: GLYPH_SOLVED } }]
      : [],
  );
}

// A solver move from the board. The expected move is accepted and committed; any
// other move that mates on the spot also solves the puzzle; everything else snaps
// back for another try.
function playPuzzleMove(move) {
  if (!puzzle || (puzzlePhase !== 'playing' && puzzlePhase !== 'wrong')) { render(); return; }
  clearTimeout(puzzleTimer); // cancel a pending wrong-move snap-back if the solver retries fast
  cg.setAutoShapes([]);
  if (puzzleUci(move) === puzzle.moves[puzzleStep]) {
    puzzleStep++;
    puzzlePhase = 'playing';
    const done = puzzleStep >= puzzle.moves.length;
    if (done) { puzzlePhase = 'solved'; puzzleSolvedPly = history.length; } // the move commit is about to append
    commit(move);
    if (done) playConnectSound();
    else schedulePuzzleReply();
    return;
  }
  const after = gameStatus(applyMove(state, move));
  if (after.result === 'checkmate') {
    puzzleStep = puzzle.moves.length;
    puzzlePhase = 'solved';
    puzzleSolvedPly = history.length; // the mating move commit is about to append
    commit(move);
    playConnectSound();
    return;
  }
  // Wrong: chessground has optimistically placed the piece on its target. Sound the
  // move (it visibly moved), flash a red ✗ there and leave the piece for a beat, then
  // render() snaps it back to its origin and clears the glyph (puzzlePhase 'wrong' →
  // paintPuzzleGlyph draws nothing).
  puzzlePhase = 'wrong';
  playMoveSound(move.capture, after.check);
  cg.setAutoShapes([{ orig: squareName(move.to), customSvg: { html: GLYPH_WRONG } }]);
  updateStatusText(); // "Not the move — try again." while the ✗ shows
  puzzleTimer = setTimeout(render, 600);
}

// Play the scripted opponent reply after the watch delay, then hand back to the
// solver. A reply missing from legalMoves means the catalog and engine disagree
// (e.g. a rule change since mining) — end the puzzle as solved rather than strand it.
function schedulePuzzleReply() {
  puzzleTimer = setTimeout(() => {
    const m = status.legal.find((x) => puzzleUci(x) === puzzle.moves[puzzleStep]);
    if (!m) { puzzlePhase = 'solved'; render(); return; }
    puzzleStep++;
    commit(m);
  }, ui.delay);
}

// Reveal: play out the rest of the line (from wherever the solver got to) with a
// readable pause between moves. The board locks (phase 'revealed').
function revealPuzzleSolution() {
  if (!puzzle || puzzlePhase !== 'playing' && puzzlePhase !== 'wrong') return;
  clearTimeout(puzzleTimer);
  goTo(history.length - 1); // make sure we're at the live position before extending it
  puzzlePhase = 'revealed';
  cg.setAutoShapes([]);
  const step = () => {
    if (puzzleStep >= puzzle.moves.length) return;
    const m = status.legal.find((x) => puzzleUci(x) === puzzle.moves[puzzleStep]);
    if (!m) return;
    puzzleStep++;
    commit(m);
    puzzleTimer = setTimeout(step, 600);
  };
  step();
}

function puzzleHint() {
  if (!puzzle || (puzzlePhase !== 'playing' && puzzlePhase !== 'wrong')) return;
  if (state.turn !== puzzleColor || viewIndex !== history.length - 1) return;
  cg.setAutoShapes([{ orig: puzzle.moves[puzzleStep].slice(0, 2), brush: 'green' }]);
}

// The line under the puzzle buttons: progress while solving; kind, difficulty and
// themes only once the puzzle is over (they'd spoil the answer).
function renderPuzzleMeta() {
  const el = $('puzzle-meta');
  // "Next puzzle" stays locked until the current one is solved/revealed — no skipping
  // ahead without an answer.
  const solvedish = puzzlePhase === 'solved' || puzzlePhase === 'revealed';
  if (solvedish && puzzleIdx >= 0) puzzleBeaten.add(puzzleIdx);
  // "Next" unlocks once the current puzzle is solved — or was already beaten earlier
  // this session, so stepping back to a finished puzzle keeps navigation free.
  $('puzzle-next').disabled = !solvedish && !puzzleBeaten.has(puzzleIdx);
  // Previous is free (those puzzles were already seen) — only the start of the walk
  // disables it.
  $('puzzle-prev').disabled = puzzleIdx <= 0;
  if (ui.mode !== 'puzzle' || !puzzle) { el.textContent = ''; return; }
  let s = `Puzzle ${puzzleIdx + 1} / ${puzzleList.length}`;
  if (solvedish) {
    s += puzzle.kind === 'mate' ? ` · mate in ${puzzle.mateIn}`
      : puzzle.kind === 'defense' ? ' · defensive save' : ' · winning tactic';
    s += ` · difficulty ${puzzle.difficulty}`;
    if (puzzle.themes.length) s += ` · ${puzzle.themes.join(', ')}`;
  }
  el.textContent = s;
}

// --- eval bars ---
// Lichess-style vertical gauges beside the board; the white fill's height is
// White's winning chances and the CSS height transition animates each update.
// Two users, mutually exclusive by mode:
//   - Analysis mode (left bar only): a dedicated THIRD worker evaluates the VIEWED
//     position (so review navigation and forking refresh the bar) and never competes
//     with the per-colour play workers; it's terminated whenever the bar hides, since
//     its idle transposition table holds ~20 MB. Requests are throttled to one in
//     flight: a position change while one is out marks it dirty and re-requests
//     on the reply, so mashing the review arrows queues at most one extra search.
//   - AI-vs-AI "Eval bars" option (both bars): each engine's OWN root score —
//     already carried by every search reply, so this costs no extra compute
//     and shows genuinely different opinions when the engines use different
//     evals. Scores are stored per history entry (what each bar showed while
//     that ply was live), so review navigation replays the bars ply by ply.
//     The bottom engine's bar sits left of the board, the top engine's right.
//     The live bars update in real time: the side to move streams its best score
//     after each search depth (onSearchProgress), and the idle engine refreshes its
//     bar on each ponder burst (onPonderResult), so neither bar sits frozen between
//     moves. The final per-ply value (onSearchResult) is what review replays.
// The analysis eval bar (and the best-move arrow that shares its reply) runs the same
// engine as "AI move" — loop-champion at depth 7 — so the bar, the arrow, and the move
// AI move would play all agree. It lives in its own worker, so the heavier nn search
// never blocks the UI; it just updates a touch more deliberately as you navigate.
const EVAL_BAR = { depth: PUZZLE_AI.depth, maxMs: ui.maxMs, engine: PUZZLE_AI.engine, net: PUZZLE_AI.net };
// Scores beyond this magnitude encode a forced mate (ai.js MATE = 1,000,000; no
// real centipawn eval comes anywhere near) — pin the bar to the winner's end.
const EVAL_BAR_MATE = 500_000;
const evalBar = { worker: null, seq: 0, pending: false, dirty: false, fen: null, turn: 'white' };
// Analysis "best-move arrow": reuse the eval-bar worker's reply (it already searches
// the viewed position) to draw an arrow to its best move, refreshed as you navigate
// or fork — no extra worker, no need to actually play the move.
let showBestArrow = false;

const analysisBarVisible = () => ui.mode === 'analysis';
const duelBarsVisible = () => ui.mode === 'ai-ai' && ui.showEvalBars;

function createEvalBarWorker() {
  const w = new Worker(new URL('./aiWorker.js', import.meta.url), { type: 'module' });
  w.onmessage = ({ data }) => {
    if (data.type !== 'search' || data.seq !== evalBar.seq) return;
    evalBar.pending = false;
    // A queued re-search picks up where this left off; it sets pending again and
    // refreshes the status, so don't clear the "thinking…" line here.
    if (evalBar.dirty) { evalBar.dirty = false; evalBar.fen = null; requestEvalBar(); return; }
    if (!analysisBarVisible()) return;
    // The search score is side-to-move-relative; the bar wants White's view.
    if (typeof data.score === 'number') paintEvalBar(evalBar.turn === 'white' ? data.score : -data.score);
    drawBestArrow(data.move);
    updateStatusText(); // the search settled — drop the "thinking…" line
  };
  w.onerror = (e) => console.error('Eval-bar worker error:', e.message);
  return w;
}

// Centipawns (White's view) -> percentage of the bar that is white. Lichess's
// cp -> winning-chance sigmoid, clamped so a non-mate eval always leaves a
// sliver of the losing side visible; only a forced mate pins the bar.
function evalBarPct(whiteCp) {
  if (whiteCp >= EVAL_BAR_MATE) return 100;
  if (whiteCp <= -EVAL_BAR_MATE) return 0;
  const winChance = 2 / (1 + Math.exp(-0.00368208 * whiteCp)) - 1;
  return Math.max(2, Math.min(98, 50 + 50 * winChance));
}

// Paint one bar element. `whiteCp` is centipawns from White's view (null = no
// evaluation yet, neutral); `who` labels the hover tooltip.
function paintEvalBarEl(el, whiteCp, who) {
  el.querySelector('.eval-fill').style.height = (whiteCp == null ? 50 : evalBarPct(whiteCp)) + '%';
  el.title = whiteCp == null ? `${who}: no evaluation yet`
    : whiteCp >= EVAL_BAR_MATE ? `${who}: White has forced mate`
    : whiteCp <= -EVAL_BAR_MATE ? `${who}: Black has forced mate`
    : `${who}: ${(whiteCp >= 0 ? '+' : '') + (whiteCp / 100).toFixed(1)}`;
}

// The puzzle bar (left, dedicated worker).
function paintEvalBar(whiteCp) { paintEvalBarEl($('eval-bar-left'), whiteCp, 'Eval'); }

// Draw (or clear) the analysis best-move arrow from an eval-bar search reply.
// Off, out of analysis, or a terminal position (no move) all clear it. When the
// best move is a promotion, drop a ghost piece (Lichess-style) on the landing
// square so it's clear which piece the pawn should become.
function drawBestArrow(move) {
  if (!showBestArrow || !analysisBarVisible()) return;
  if (!move || move.from == null) { cg.setAutoShapes([]); return; }
  const dest = squareName(move.to);
  const shapes = [{ orig: squareName(move.from), dest, brush: 'paleBlue' }];
  if (move.promotion) {
    shapes.push({ orig: dest, piece: { role: CG_ROLE[move.promotion], color: evalBar.turn, scale: 0.6 } });
  }
  cg.setAutoShapes(shapes);
}

// The AI-vs-AI duel bars: each engine's own score at the VIEWED ply (recorded
// per history entry by onSearchResult in White's view), so stepping back shows
// what the engines thought at the time, not their latest opinion — bottom
// engine left, top engine right, so a board flip swaps which bar shows which
// engine.
function paintDuelBars() {
  const bottom = viewColor();
  const evals = history[viewIndex].evals;
  const name = (c) => `${c === 'white' ? 'White' : 'Black'} engine`;
  paintEvalBarEl($('eval-bar-left'), evals[bottom], name(bottom));
  paintEvalBarEl($('eval-bar-right'), evals[opponent(bottom)], name(opponent(bottom)));
}

// Show/hide the bars for the current mode+phase and keep their values current.
// Called from render() and applyModeVisibility(), so every path that changes
// the position, the view, the phase, or the option lands here; the analysis-bar
// fen check makes repeat calls free.
function syncEvalBar() {
  const left = $('eval-bar-left'), right = $('eval-bar-right');
  const analysisBar = analysisBarVisible();
  const duel = !analysisBar && duelBarsVisible();
  // The board cedes 26px per visible bar (see styles.css); resizing it means
  // chessground must re-measure, so redraw only on a real layout change.
  const row = left.parentElement;
  const prevLayout = row.className;
  row.classList.toggle('bars-1', analysisBar);
  row.classList.toggle('bars-2', duel);
  if (analysisBar && left.hidden) paintEvalBarEl(left, null, 'Eval'); // neutral until the first eval lands
  left.hidden = !(analysisBar || duel);
  right.hidden = !duel;
  if (row.className !== prevLayout) cg.redrawAll();
  if (!analysisBar) {
    // The dedicated worker only serves the analysis bar; drop it (and any
    // in-flight reply) whenever that bar isn't showing.
    if (evalBar.worker) { evalBar.worker.terminate(); evalBar.worker = null; }
    evalBar.seq++;
    evalBar.pending = evalBar.dirty = false;
    evalBar.fen = null;
  }
  if (!analysisBar && !duel) return;
  const flip = viewColor() === 'black'; // white fill sits at White's end
  left.classList.toggle('flipped', flip);
  right.classList.toggle('flipped', flip);
  if (duel) paintDuelBars();
  else requestEvalBar();
}

function requestEvalBar() {
  // The analysis eval runs the nn (loop champion), with the weights resolved from the
  // catalog at request time. On a deep-link/restore boot the catalog loads asynchronously,
  // so netUrl is briefly null; firing the search then would run the wasm nn eval with no
  // weights loaded — a "memory access out of bounds" trap that kills the worker (no bar, no
  // arrow). Defer until loadNetCatalog resolves and kicks us again with the net in hand.
  if (EVAL_BAR.engine === 'nn' && !netUrl(EVAL_BAR.net)) return;
  const fen = repFens[viewIndex];
  if (fen === evalBar.fen) return; // already showing (or searching) this position
  if (evalBar.pending) { evalBar.dirty = true; return; }
  evalBar.fen = fen;
  const st = history[viewIndex].state;
  // Terminal positions need no search (and have no moves to search).
  const here = viewIndex === history.length - 1 ? status : gameStatus(st);
  if (here.over) {
    paintEvalBar(here.result === 'checkmate' ? (here.winner === 'white' ? EVAL_BAR_MATE : -EVAL_BAR_MATE) : 0);
    drawBestArrow(null); // no move in a finished position
    updateStatusText();  // nothing to search here — no "thinking…" line
    return;
  }
  drawBestArrow(null); // clear the arrow for the old position until the new search lands
  if (!evalBar.worker) evalBar.worker = createEvalBarWorker();
  evalBar.pending = true;
  evalBar.turn = st.turn;
  evalBar.worker.postMessage({
    type: 'search', seq: ++evalBar.seq, state: st,
    depth: EVAL_BAR.depth, maxMs: EVAL_BAR.maxMs, engine: EVAL_BAR.engine, net: netUrl(EVAL_BAR.net),
    // The repetition window up to the VIEWED ply (same trimming as repWindow).
    posHistory: repFens.slice(0, viewIndex + 1).slice(-(st.halfmove + 1)),
    wasmUrl: WASM_URL,
  });
  updateStatusText(); // a search is now in flight — show the "thinking…" line
}

// --- promotion picker ---
const GLYPH = {
  white: { q: '♕', r: '♖', b: '♗', n: '♘' },
  black: { q: '♛', r: '♜', b: '♝', n: '♞' },
};

function askPromotion(color) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'promo';
    const card = document.createElement('div');
    card.className = 'promo-card';
    card.innerHTML = '<p>Promote to</p>';
    const row = document.createElement('div');
    row.className = 'promo-buttons';
    for (const role of ['q', 'r', 'b', 'n']) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = GLYPH[color][role];
      btn.addEventListener('click', () => { overlay.remove(); resolve(role); });
      row.appendChild(btn);
    }
    card.appendChild(row);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
  });
}

// --- online play ---
// Drive the host/join panel through its phases by toggling element visibility and
// the status line. `detail` carries the share code (hosting), the connected-colour
// message, or an error/idle hint.
function setOnlinePhase(phase, detail = '') {
  const idle = phase === 'idle';
  // Lock the mode switch while a session is live (hosting, connecting, connected)
  // so you can't tear it down by accident — you must Cancel/Leave first, which
  // returns to 'idle' and re-enables it.
  $('mode').disabled = !idle;
  $('online-setup').hidden = !idle;
  $('online-share').hidden = phase !== 'hosting';
  $('online-leave').hidden = idle;
  $('online-leave').textContent = phase === 'connected' ? 'Leave' : 'Cancel';
  $('online-swap').hidden = !(phase === 'connected' && isHost); // host-only, while connected
  if (phase === 'hosting') $('online-code-out').textContent = detail;
  const msg = {
    idle: detail,
    searching: 'Searching for an opponent…',
    hosting: 'Waiting for an opponent to join…',
    connecting: detail || 'Connecting…',
    connected: detail,
  }[phase];
  $('online-status').textContent = msg ?? detail;
}

const connectedMsg = (color) => `Connected — you play ${color === 'white' ? 'White' : 'Black'}.`;

// The share code lives in the URL hash (`#code=…`) so a host can just copy the
// address bar and a joiner who opens that link auto-joins (see startup below).
// A hash (not a `?query`) keeps the code out of what browsers persist/send and
// out of the path the way some browsers save query strings on the bare URL.
const hashParams = () => new URLSearchParams((window.location.hash || '').replace(/^#/, ''));
const shareUrl = (code) => {
  const u = new URL(window.location.href);
  const p = hashParams();
  p.set('code', code);
  u.hash = p.toString();
  return u.href;
};
function setUrlCode(code) {
  const u = new URL(window.location.href);
  const p = hashParams();
  if (code) p.set('code', code);
  else p.delete('code');
  const s = p.toString();
  u.hash = s ? s : '';
  window.history.replaceState(null, '', u.href); // window.history — `history` is the local move list
}
const getUrlCode = () => normalizeCode(hashParams().get('code') || '');

// Deep-linkable modes: the chosen mode rides in the same hash (`#mode=…`) so a link
// can open straight into Puzzles/AI-vs-AI/etc., and switching modes updates the URL
// (replaceState, so no history spam and no `hashchange` for our own writes). The
// default two-player mode is left out to keep the bare URL clean; an online share
// link already carries `code=`, which implies online on its own.
const MODES = new Set(['human-human', 'human-ai', 'ai-ai', 'online', 'puzzle', 'analysis', 'editor']);
const getUrlMode = () => { const m = hashParams().get('mode'); return MODES.has(m) ? m : null; };
function setUrlMode(mode) {
  const u = new URL(window.location.href);
  const p = hashParams();
  if (mode && mode !== 'human-human') p.set('mode', mode);
  else p.delete('mode');
  const s = p.toString();
  u.hash = s ? s : '';
  window.history.replaceState(null, '', u.href); // window.history — `history` is the local move list
}

// Tear down the current session (if any) and clear connection state. The caller
// is responsible for the resulting UI phase.
function leaveOnline() {
  if (matchSession) { matchSession.cancel(); matchSession = null; }
  if (online) { online.close(); online = null; }
  onlineConnected = false;
  onlineColor = null;
  isHost = false;
  matchmade = false;
}

function startHost() {
  leaveOnline();
  isHost = true;
  const choice = $('online-color').value;
  const color = choice === 'random' ? (Math.random() < 0.5 ? 'white' : 'black') : choice;
  setOnlinePhase('connecting');
  online = hostGame({
    onCode: (code) => { setUrlCode(code); setOnlinePhase('hosting', code); },
    onConnected: () => onHostConnected(color),
    onData: onOnlineData,
    onClosed: onOnlineClosed,
    onError: onOnlineError,
  });
}

function startJoin() {
  const code = normalizeCode($('online-code').value);
  if (code.length < 4) { setOnlinePhase('idle', 'Enter the code your opponent shared.'); return; }
  leaveOnline();
  setOnlinePhase('connecting');
  online = joinGame(code, {
    // The host assigns colours, so the joiner just waits for the `hello` message.
    onConnected: () => { if (!onlineConnected) setOnlinePhase('connecting', 'Connected — waiting for host…'); },
    onData: onOnlineData,
    onClosed: onOnlineClosed,
    onError: onOnlineError,
  });
}

// Auto-matchmaking: wait in the shared queue room until paired with someone, then
// drop straight into a private game with them. The pair agree a code and which side
// hosts (see queue.js); we just route the result into the normal host/join path.
function startFindMatch() {
  leaveOnline();
  setOnlinePhase('searching');
  matchSession = findMatch({
    onMatched: ({ code, isHost: host }) => {
      matchSession = null; // the queue room is already left inside findMatch
      if (host) startMatchedHost(code);
      else startMatchedJoin(code);
    },
    onError: (m) => { matchSession = null; setOnlinePhase('idle', m || 'Matchmaking failed.'); },
  });
}

// The matched host: like startHost but with the agreed code and a random colour
// (there's no colour picker in matchmaking). `matchmade` marks the throwaway code.
function startMatchedHost(code) {
  leaveOnline();
  isHost = true;
  matchmade = true;
  const color = Math.random() < 0.5 ? 'white' : 'black';
  setOnlinePhase('connecting');
  online = hostGame({
    onCode: (c) => setUrlCode(c), // keep the URL shareable, but skip the "hosting" UI
    onConnected: () => onHostConnected(color),
    onData: onOnlineData,
    onClosed: onOnlineClosed,
    onError: onOnlineError,
  }, code);
}

// The matched joiner: identical to startJoin, but with the agreed code.
function startMatchedJoin(code) {
  leaveOnline();
  matchmade = true;
  setOnlinePhase('connecting');
  online = joinGame(code, {
    onConnected: () => { if (!onlineConnected) setOnlinePhase('connecting', 'Connected — waiting for host…'); },
    onData: onOnlineData,
    onClosed: onOnlineClosed,
    onError: onOnlineError,
  });
}

// The host's chosen colour is decided locally; on connect it tells the joiner the
// opposite colour via `hello`, then both sides start from the same clean position.
function onHostConnected(color) {
  onlineColor = color;
  onlineConnected = true;
  online.send({ t: 'hello', color: opponent(color) });
  setOnlinePhase('connected', connectedMsg(color));
  playConnectSound();
  newGame();
}

// Host-only: swap colours and start a fresh game. Reuses the `hello` message —
// the joiner reassigns its colour and resets the same way it does on connect.
function onlineSwap() {
  if (!onlineConnected || !isHost) return;
  onlineColor = opponent(onlineColor);
  online.send({ t: 'hello', color: opponent(onlineColor) });
  setOnlinePhase('connected', connectedMsg(onlineColor));
  newGame();
}

// A matchmade game's opponent left (or its connection errored out) — the private
// code was throwaway, so go straight back into the queue for a new opponent. A peer
// leaving can reach us as a clean close OR a connection error depending on the
// WebRTC offerer/answerer role (that asymmetry is why a host leaving and a joiner
// leaving don't surface the same way), so BOTH handlers funnel through here.
function requeueMatchmade() {
  if (!matchmade) return false;
  setUrlCode(''); // drop the finished match's code from the URL
  startFindMatch(); // its leaveOnline() tears down the old session, then re-queues
  render(); // lock the board while searching again
  return true;
}

function onOnlineClosed() {
  if (matchSession) return; // already re-queued — ignore a duplicate close/error event
  if (requeueMatchmade()) return;
  // Host of a deliberately-shared lobby: keep the room (and code) alive so a new
  // opponent can still join the same code.
  if (isHost && online) {
    onlineConnected = false;
    setOnlinePhase('hosting', online.getCode());
    render(); // lock the board until someone joins
    return;
  }
  // Joiner: the host owned the lobby, so there's nothing left to wait on — go idle.
  leaveOnline();
  setUrlCode('');
  setOnlinePhase('idle', 'Opponent disconnected.');
  render(); // lock the board
}

function onOnlineError(message) {
  if (matchSession) return; // already re-queued — ignore a trailing error event
  if (requeueMatchmade()) return;
  leaveOnline();
  setUrlCode('');
  setOnlinePhase('idle', message);
  render();
}

// A message from the peer. Moves are re-derived from our own legalMoves so both
// clients apply the identical engine move; a reset restarts the local game.
function onOnlineData(msg) {
  if (!msg || typeof msg !== 'object') return;
  if (msg.t === 'hello') {
    // The host tells us (the joiner) which colour to play — on connect, and again
    // on a side swap. Either way: take the colour, go live, and reset.
    const justConnected = !onlineConnected; // first hello = the actual connect (not a swap)
    onlineColor = msg.color === 'black' ? 'black' : 'white';
    onlineConnected = true;
    setOnlinePhase('connected', connectedMsg(onlineColor));
    if (justConnected) playConnectSound();
    newGame();
  } else if (msg.t === 'move') {
    if (!onlineConnected || status.over) return;
    if (state.turn === onlineColor) return; // not the opponent's turn — ignore strays
    const m = status.legal.find((x) => x.from === msg.from && x.to === msg.to
      && (msg.promotion ? x.promotion === msg.promotion : !x.promotion));
    if (m) commit(m); // remote move: commit but do NOT echo it back
  } else if (msg.t === 'reset') {
    newGame();
  }
}

// --- controls ---
function newGame() {
  cancelAi();
  ui.running = false;
  ui.started = false;
  syncToggleLabel();
  state = newGameState();
  status = gameStatus(state);
  history = [startEntry(state)];
  repFens = [toFen(state)];
  viewIndex = 0;
  posCounts = new Map();
  countPosition(state);
  lastCommitAt = performance.now();
  if (ui.mode === 'editor') { enterEditor(); return; } // reset to an editable start position
  render();
  driveAi();
}

// Replace the live game with an imported one: reset from `start`, replay `moves`,
// then show the final position. Keeps the current mode (so you can load a line and
// let the AIs continue from it), but leaves AI-vs-AI paused until you press Start.
function loadGame(start, moves) {
  // Importing a game while in puzzle mode replaces the puzzle — clear it so solver
  // moves on the imported position aren't checked against a stale solution.
  if (ui.mode === 'puzzle') {
    clearTimeout(puzzleTimer);
    puzzle = null;
    puzzlePhase = 'idle';
    cg.setAutoShapes([]);
  }
  cancelAi();
  ui.running = false;
  ui.started = false;
  syncToggleLabel();
  state = start;
  status = gameStatus(state);
  history = [startEntry(state)];
  repFens = [toFen(state)];
  posCounts = new Map();
  countPosition(state);
  for (const mv of moves) recordMove(mv);
  viewIndex = history.length - 1;
  lastCommitAt = performance.now();
  render();
  driveAi();
}

function syncToggleLabel() {
  // Start (fresh game) → Pause (while running) → Resume (paused).
  $('ai-toggle').textContent = ui.running ? 'Pause' : ui.started ? 'Resume' : 'Start';
}

const strengthOf = (slot) =>
  slot === 'ai' ? ui.strengthAi : slot === 'white' ? ui.strengthWhite : ui.strengthBlack;
const engineOf = (slot) =>
  slot === 'ai' ? ui.engineAi : slot === 'white' ? ui.engineWhite : ui.engineBlack;

// The engine picker is a segmented radio toggle (`engine-<slot>-hc|-nn`), not a
// <select>; these read/write the checked option by slot.
const ENGINE_UI_KEY = { ai: 'engineAi', white: 'engineWhite', black: 'engineBlack' };
function engineValue(slot) {
  const checked = document.querySelector(`input[name="engine-${slot}"]:checked`);
  return checked ? checked.value : 'handcrafted';
}
function setEngineValue(slot, v) {
  const suffix = v === 'nn' ? 'nn' : v === 'handcrafted3' ? 'hc3' : 'hc';
  const el = $(`engine-${slot}-${suffix}`);
  if (el) el.checked = true;
}

// --- neural-net catalog (web/public/nn) --------------------------------------
// The selectable nets are listed in public/nn/manifest.json; fetch it once, fill the
// per-slot <select>s, and hand the worker the chosen net's full URL.
let netCatalog = [];   // [{ name, file, arch, note }]
let netDefault = null; // manifest default name
const NET_UI_KEY = { ai: 'netAi', white: 'netWhite', black: 'netBlack' };
const netOf = (slot) =>
  slot === 'ai' ? ui.netAi : slot === 'white' ? ui.netWhite : ui.netBlack;

// Absolute URL of a named net's weights file, so it resolves identically from the
// worker (whose relative URLs would point under assets/) and the page. null if the
// name isn't in the catalog.
function netUrl(name) {
  const entry = netCatalog.find((n) => n.name === name);
  return entry ? new URL(`${import.meta.env.BASE_URL}nn/${entry.file}`, location.href).href : null;
}

// The wasm search engine (web/engine → public/apos.wasm). Resolved here against the
// document base and passed to the worker, since a worker's own relative URL would resolve
// under assets/ (same reason net URLs are resolved on the page). Built by `npm run
// build:wasm`; if it's missing the worker falls back to the JS search.
const WASM_URL = new URL(`${import.meta.env.BASE_URL}apos.wasm`, location.href).href;

async function loadNetCatalog() {
  try {
    const url = new URL(`${import.meta.env.BASE_URL}nn/manifest.json`, location.href).href;
    const man = await fetch(url).then((r) => r.json());
    netCatalog = Array.isArray(man.nets) ? man.nets : [];
    netDefault = man.default || (netCatalog[0] && netCatalog[0].name) || null;
  } catch {
    netCatalog = []; netDefault = null; // no catalog -> nn falls back to material
  }
  for (const slot of ['ai', 'white', 'black']) {
    const sel = $(`net-${slot}`);
    sel.innerHTML = '';
    for (const n of netCatalog) {
      const opt = document.createElement('option');
      opt.value = n.name;
      opt.textContent = n.name;       // keep the option short so the <select> stays narrow
      if (n.note) opt.title = n.note; // full description on hover
      sel.appendChild(opt);
    }
    sel.value = netDefault || '';
    ui[NET_UI_KEY[slot]] = sel.value || null;
  }
  // A deep-linked/restored analysis mode may have deferred its eval-bar search because the
  // net couldn't be resolved before this loaded (see requestEvalBar). Now that the catalog
  // is in, kick it so the bar + best-move arrow appear, using the current loop champion.
  if (analysisBarVisible()) { evalBar.fen = null; requestEvalBar(); }
}

function applyModeVisibility() {
  const m = ui.mode;
  $('side-control').hidden = m !== 'human-ai';
  $('ai-toggle').hidden = m !== 'ai-ai';
  // Puzzles have their own Retry/Next buttons (in the mode row); "New game" would be
  // ambiguous there. Next is enabled/disabled by phase in renderPuzzleMeta.
  $('new-game').hidden = m === 'puzzle';
  $('puzzle-prev').hidden = m !== 'puzzle';
  $('puzzle-retry').hidden = m !== 'puzzle';
  $('puzzle-next').hidden = m !== 'puzzle';
  $('row-puzzle').hidden = m !== 'puzzle';
  // Analysis is its own mode. Its panel lives in the actions area (so it stays
  // reachable on mobile, where the .controls column is hidden during analysis); the
  // "Back to puzzle" button shows only when analysis was reached from a puzzle.
  const analysis = m === 'analysis';
  $('row-analysis').hidden = !analysis;
  $('analysis-back').hidden = !(analysis && puzzleSession);
  // Leaving analysis turns the best-move arrow off AND clears it from the board (otherwise
  // the drawn arrow lingers into the next mode); entering turns it on (see enterAnalysis).
  if (!analysis && showBestArrow) { showBestArrow = false; $('analysis-arrows').checked = false; cg.setAutoShapes([]); }
  // Mobile analysis layout: lift the move list + nav directly under the board (see
  // styles.css); inert on desktop where the media-query rules don't apply.
  document.body.dataset.analysis = analysis ? '1' : '';
  $('row-ai-swap').hidden = m !== 'ai-ai';
  $('row-online').hidden = m !== 'online';
  // human-ai: one colour-agnostic AI row. ai-ai: separate White/Black rows.
  $('row-ai').hidden = m !== 'human-ai';
  $('row-white').hidden = m !== 'ai-ai';
  $('row-black').hidden = m !== 'ai-ai';
  $('row-editor').hidden = m !== 'editor';
  // A row's custom depth/timeout inputs appear only when its strength is "Custom".
  toggleCustom('ai', m === 'human-ai' && ui.strengthAi === 'custom');
  toggleCustom('white', m === 'ai-ai' && ui.strengthWhite === 'custom');
  toggleCustom('black', m === 'ai-ai' && ui.strengthBlack === 'custom');
  // The net picker appears only when a slot's engine is the neural net.
  for (const slot of ['ai', 'white', 'black']) $(`net-field-${slot}`).hidden = engineOf(slot) !== 'nn';
  // The eval bar is mode-dependent (analysis only); this covers the paths that change
  // mode without a render (e.g. switching into the editor).
  syncEvalBar();
  syncMobileTabs(); // show/hide the mobile Settings tab to match the mode
}

function toggleCustom(slot, show) {
  $(`custom-depth-${slot}`).closest('.custom-fields').hidden = !show;
}

// While an AI-vs-AI match is actively running, lock both engines' settings and
// the swap button so they can't change mid-match; they unlock when the match is
// paused/stopped/over. Called from render() (covers game-over) and on toggle.
function applyAiLock() {
  const locked = ui.mode === 'ai-ai' && ui.running && !status.over;
  for (const id of ['depth-white', 'depth-black',
    'custom-depth-white', 'custom-ms-white', 'custom-depth-black', 'custom-ms-black', 'ai-swap']) {
    $(id).disabled = locked;
  }
  for (const slot of ['white', 'black']) {
    for (const r of document.querySelectorAll(`input[name="engine-${slot}"]`)) r.disabled = locked;
    $(`net-${slot}`).disabled = locked;
  }
}

// Search params for the AI playing `turn`: a preset depth + the default cap, or
// that slot's custom depth + timeout when its strength is set to "Custom".
function aiParams(turn) {
  const slot = ui.mode === 'ai-ai' ? turn : 'ai';
  const engine = engineOf(slot);
  const net = engine === 'nn' ? netUrl(netOf(slot)) : undefined; // weights URL for the worker
  const v = strengthOf(slot);
  if (v !== 'custom') return { depth: parseInt(v, 10), maxMs: ui.maxMs, engine, net };
  // 0 means "no limit" for either field. Both unlimited would never return, so
  // fall back to the default time cap in that case.
  let { depth, ms } = ui.custom[slot];
  if (depth === 0 && ms === 0) ms = ui.maxMs;
  return { depth: depth === 0 ? Infinity : depth, maxMs: ms === 0 ? Infinity : ms, engine, net };
}

// A PGN player name for one side: "Human" for a human-controlled slot, or a
// description of the engine driving that colour, incl. which eval it used
// ("AI (Neural net, depth 7, 6000ms)"). human-human / online / editor are all
// people on both sides.
function playerName(color) {
  const isAi = ui.mode === 'ai-ai' || (ui.mode === 'human-ai' && color !== ui.humanColor);
  if (!isAi) return 'Human';
  const { depth, maxMs, engine } = aiParams(color);
  const slot = ui.mode === 'ai-ai' ? color : 'ai';
  const e = engine === 'nn' ? `Neural net (${netOf(slot) || '?'})`
    : engine === 'handcrafted3' ? 'Handcrafted v3' : 'Handcrafted';
  const d = depth === Infinity ? 'unlimited' : depth;
  const t = maxMs === Infinity ? 'no time limit' : `${maxMs}ms`;
  return `AI (${e}, depth ${d}, ${t})`;
}

function clampInt(value, min, max, fallback) {
  const n = parseInt(value, 10);
  return Number.isNaN(n) ? fallback : Math.max(min, Math.min(max, n));
}

// Browsers restore control values on reload, so read them into `ui` at startup
// instead of assuming the hardcoded defaults — otherwise the shown selection and
// the actual behaviour disagree until the next change event.
function syncControlsFromDom() {
  ui.mode = $('mode').value;
  ui.humanColor = $('side').value;
  ui.strengthAi = $('depth-ai').value;
  ui.strengthWhite = $('depth-white').value;
  ui.strengthBlack = $('depth-black').value;
  ui.engineAi = engineValue('ai');
  ui.engineWhite = engineValue('white');
  ui.engineBlack = engineValue('black');
  ui.varyOpenings = $('vary-openings').checked;
  ui.showEvalBars = $('show-evals').checked;
  for (const slot of ['ai', 'white', 'black']) {
    ui.custom[slot].depth = clampInt($(`custom-depth-${slot}`).value, 0, 40, 8);
    ui.custom[slot].ms = clampInt($(`custom-ms-${slot}`).value, 0, 60000, 6000);
  }
}

// --- board editor ---
const CG_ROLE = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };

// Set the board up for free editing, starting from the current position. No game
// bookkeeping happens while editing; the edited board is turned into a real game
// only when you switch to a play mode (see the mode change handler).
function enterEditor() {
  cancelAi();
  ui.running = false;
  ui.started = false;
  syncToggleLabel();
  editorTurn = state.turn;
  $('editor-turn').value = editorTurn;
  cg.set({
    fen: toFen(state),
    orientation: 'white',
    turnColor: 'white',
    lastMove: undefined,
    check: false,
    viewOnly: false,
    movable: { free: true, color: 'both', showDests: false, dests: new Map() },
    draggable: { enabled: true, deleteOnDropOff: true },
  });
  renderTrays({ state });
  updateStatusText();
}

// Castling is available for a side when its king and the relevant rook still sit on
// their home squares (a1/e1/h1, a8/e8/h8) — derived so the editor needs no toggles.
function deriveCastling(board) {
  const at = (i, role, color) => board[i] && board[i].role === role && board[i].color === color;
  return {
    K: at(4, 'k', 'white') && at(7, 'r', 'white'),
    Q: at(4, 'k', 'white') && at(0, 'r', 'white'),
    k: at(60, 'k', 'black') && at(63, 'r', 'black'),
    q: at(60, 'k', 'black') && at(56, 'r', 'black'),
  };
}

// Turn the edited board into a game state, or null if it isn't playable (a side is
// missing its king). chessground's FEN is placement-only, so we splice in the chosen
// side to move and derived castling.
function readEditorState() {
  const placement = cg.getFen();
  const st = parseFen(`${placement} ${editorTurn === 'white' ? 'w' : 'b'} - - 0 1`);
  st.castling = deriveCastling(st.board);
  let wk = 0, bk = 0;
  for (const p of st.board) if (p && p.role === 'k') (p.color === 'white' ? wk++ : bk++);
  return wk && bk ? st : null;
}

// Re-evaluate who controls the game after a mode/side change, *keeping* the current
// position so it can be continued — handed between a human and the AI from wherever
// the game stands — instead of being reset. Aborts any in-flight search and stops
// autonomous AI-vs-AI play; the new controller (if any) takes over via driveAi().
function handoffControl() {
  cancelAi();
  ui.running = false;
  ui.started = false;
  syncToggleLabel();
  render();
  driveAi();
}

// Switch to a new mode — from the dropdown, or routed programmatically (a `#mode=`
// deep link / hashchange). Keeps the dropdown + URL in sync and, on mobile, surfaces
// the relevant tab. A no-op if `next` is already the mode, or if leaving the editor
// with an unplayable position (snaps the dropdown back so edits aren't lost).
function selectMode(next) {
  const prev = ui.mode;
  if (next === prev) return;
  const prevView = viewColor(); // the orientation showing now, to carry into analysis

  // Leaving the editor: the edited board becomes a fresh game. Reject an unplayable
  // position (a side with no king) and snap back to the editor so edits aren't lost.
  let editedBoard = null;
  if (prev === 'editor') {
    editedBoard = readEditorState();
    if (!editedBoard) {
      alert('Give each side a king before leaving the board editor.');
      $('mode').value = 'editor';
      return;
    }
  }

  ui.mode = next;
  $('mode').value = next; // keep the dropdown in sync when routed programmatically
  ui.flipped = false; // each mode auto-orients; don't carry a stale flip over
  if (next === 'analysis') analysisOrient = prevView; // keep the board pointing the same way
  setUrlMode(next);
  if (prev === 'analysis') puzzleSession = null; // the saved session is consumed/abandoned on leaving

  if (prev === 'online' && next !== 'online') { leaveOnline(); setUrlCode(''); }
  // Leaving puzzles keeps the position (so it can be analyzed in another mode), but
  // drops the puzzle itself: no pending reply, no hint arrow, no stale solution.
  // Heading into analysis is the exception: snapshot the live session first so
  // "Back to puzzle" can resume it exactly where it stands.
  if (prev === 'puzzle' && next !== 'puzzle') {
    if (next === 'analysis' && puzzle) puzzleSession = snapshotPuzzleSession();
    clearTimeout(puzzleTimer);
    puzzle = null;
    puzzlePhase = 'idle';
    cg.setAutoShapes([]);
    renderPuzzleMeta();
  }

  applyModeVisibility();
  // On mobile, jump to the Settings tab for modes whose primary controls live there
  // (online's host/join panel, the board editor) so the panel shows without a tap.
  if (next === 'online' || next === 'editor') setMobileTab('settings');

  if (next === 'editor') { enterEditor(); return; }
  // Online needs a clean handshake, so it starts a fresh game; every other mode
  // change continues the current position (there's a New game button per mode).
  if (next === 'online') { setOnlinePhase('idle'); newGame(); return; }
  if (next === 'puzzle') { enterPuzzleMode(); return; } // a puzzle replaces the edited board
  // From the editor, adopt the edited board first; analysis then sets up over it.
  if (editedBoard) loadGame(editedBoard, []);
  if (next === 'analysis') { enterAnalysis(); return; }
  if (editedBoard) return; // already loaded above for a non-analysis target
  handoffControl();
}

// Set up analysis over the current position: stop any AI, turn the best-move arrow on
// by default, and render (which shows the eval bar and kicks off its search — the reply
// draws the arrow). The position is whatever the previous mode left (handoff semantics).
function enterAnalysis() {
  cancelAi();
  ui.running = false;
  ui.started = false;
  syncToggleLabel();
  showBestArrow = true;
  $('analysis-arrows').checked = true;
  cg.setAutoShapes([]); // drop any leftover hint/arrow from the prior mode
  applyModeVisibility();
  render();
}
$('mode').addEventListener('change', (e) => selectMode(e.target.value));
$('side').addEventListener('change', (e) => {
  ui.humanColor = e.target.value;
  handoffControl(); // swap which side you play without losing the game
});
$('depth-ai').addEventListener('change', (e) => {
  ui.strengthAi = e.target.value;
  applyModeVisibility(); // reveal/hide custom inputs
});
$('depth-white').addEventListener('change', (e) => {
  ui.strengthWhite = e.target.value;
  applyModeVisibility();
});
$('depth-black').addEventListener('change', (e) => {
  ui.strengthBlack = e.target.value;
  applyModeVisibility();
});
for (const slot of ['ai', 'white', 'black']) {
  for (const r of document.querySelectorAll(`input[name="engine-${slot}"]`)) {
    r.addEventListener('change', () => {
      ui[ENGINE_UI_KEY[slot]] = engineValue(slot);
      $(`net-field-${slot}`).hidden = engineValue(slot) !== 'nn'; // reveal the net picker for nn
    });
  }
  $(`net-${slot}`).addEventListener('change', (e) => { ui[NET_UI_KEY[slot]] = e.target.value || null; });
}
for (const slot of ['ai', 'white', 'black']) {
  $(`custom-depth-${slot}`).addEventListener('change', (e) => {
    ui.custom[slot].depth = clampInt(e.target.value, 0, 40, 8);
    e.target.value = ui.custom[slot].depth; // reflect the clamped value
  });
  $(`custom-ms-${slot}`).addEventListener('change', (e) => {
    ui.custom[slot].ms = clampInt(e.target.value, 0, 60000, 6000);
    e.target.value = ui.custom[slot].ms;
  });
}
$('vary-openings').addEventListener('change', (e) => {
  ui.varyOpenings = e.target.checked;
  ui.recentOpenings = []; // start a fresh rotation when the option is toggled
});
$('show-evals').addEventListener('change', (e) => {
  ui.showEvalBars = e.target.checked;
  syncEvalBar(); // show/hide the duel bars without waiting for the next move
});
$('new-game').addEventListener('click', () => {
  if (ui.mode === 'online' && onlineConnected) online.send({ t: 'reset' });
  newGame();
});
// AI-vs-AI: swap White's and Black's settings (preset strength + custom
// depth/timeout) so the stronger engine plays the other colour. Takes effect on
// the next move; the position is left as-is so you can swap mid-game.
$('ai-swap').addEventListener('click', () => {
  [ui.strengthWhite, ui.strengthBlack] = [ui.strengthBlack, ui.strengthWhite];
  [ui.engineWhite, ui.engineBlack] = [ui.engineBlack, ui.engineWhite];
  [ui.netWhite, ui.netBlack] = [ui.netBlack, ui.netWhite];
  [ui.custom.white, ui.custom.black] = [ui.custom.black, ui.custom.white];
  $('depth-white').value = ui.strengthWhite;
  $('depth-black').value = ui.strengthBlack;
  setEngineValue('white', ui.engineWhite);
  setEngineValue('black', ui.engineBlack);
  $('net-white').value = ui.netWhite || '';
  $('net-black').value = ui.netBlack || '';
  $('custom-depth-white').value = ui.custom.white.depth;
  $('custom-ms-white').value = ui.custom.white.ms;
  $('custom-depth-black').value = ui.custom.black.depth;
  $('custom-ms-black').value = ui.custom.black.ms;
  applyModeVisibility(); // reveal/hide each row's custom inputs to match
});

// Puzzle panel controls. A filter change redeals from the newly-filtered list.
$('puzzle-prev').addEventListener('click', prevPuzzle);
$('puzzle-retry').addEventListener('click', () => { if (puzzle) loadPuzzle(puzzle); });
$('puzzle-hint').addEventListener('click', puzzleHint);
$('puzzle-solution').addEventListener('click', revealPuzzleSolution);
$('puzzle-next').addEventListener('click', nextPuzzle);
// "Analysis": leave for the standalone analysis mode to explore freely — it's often
// not obvious how the game unfolds after the tactic lands. selectMode snapshots the
// puzzle session on the way out so "Back to puzzle" can resume it.
$('puzzle-continue').addEventListener('click', () => {
  if (ui.mode !== 'puzzle') return;
  selectMode('analysis');
  // Scroll back up so the board is in focus — the puzzle panel may have left the
  // page scrolled down past it.
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Analysis panel controls.
// "Back to puzzle": resume the puzzle session we entered analysis from, exactly where
// it stood. selectMode('puzzle') consumes puzzleResume in enterPuzzleMode.
$('analysis-back').addEventListener('click', () => {
  if (!puzzleSession) return;
  puzzleResume = puzzleSession; // enterPuzzleMode restores this instead of dealing anew
  selectMode('puzzle');
});

// Best-move arrow: on, force a fresh eval-bar search so its reply carries the move
// to draw; off, clear the arrow immediately.
$('analysis-arrows').addEventListener('change', (e) => {
  showBestArrow = e.target.checked;
  if (!showBestArrow) { cg.setAutoShapes([]); return; }
  evalBar.fen = null; // bypass the "same position" short-circuit so the move comes back
  requestEvalBar();
});
for (const id of ['puzzle-difficulty', 'puzzle-theme']) {
  $(id).addEventListener('change', () => {
    if (puzzleCatalog === null) return; // catalog still loading; enterPuzzleMode will filter
    refilterPuzzles();
    nextPuzzle();
  });
}

// Online panel controls.
$('online-host').addEventListener('click', startHost);
$('online-join').addEventListener('click', startJoin);
$('online-find').addEventListener('click', startFindMatch);
$('online-code').addEventListener('keydown', (e) => { if (e.key === 'Enter') startJoin(); });
// Auto-join the moment a complete code is entered — typing the last character or
// pasting a fresh code over a stale one connects without clicking Join. 'input'
// fires on paste (keydown doesn't); programmatic value sets (the #code= launch /
// hashchange path) don't fire it, so they can't double-trigger a join here.
$('online-code').addEventListener('input', () => {
  if (normalizeCode($('online-code').value).length >= CODE_LENGTH) startJoin();
});
$('online-leave').addEventListener('click', () => { leaveOnline(); setUrlCode(''); setOnlinePhase('idle'); newGame(); });
$('online-swap').addEventListener('click', onlineSwap);
$('online-copy').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(shareUrl($('online-code-out').textContent));
    const hint = $('online-copy-hint');
    hint.textContent = 'Copied';
    setTimeout(() => { hint.textContent = 'Copy link'; }, 1200);
  } catch { /* clipboard may be blocked; the code is shown for manual copy */ }
});

// Flip board: a pure view toggle (orientation, trays, eval-bar sides). The
// editor owns the board directly (render() skips it), so flip it in place.
$('flip-board').addEventListener('click', () => {
  ui.flipped = !ui.flipped;
  if (ui.mode === 'editor') {
    cg.set({ orientation: viewColor() });
    renderTrays({ state });
    return;
  }
  render();
});

// Review navigation: buttons, clicking a move, and arrow/Home/End keys.
$('nav-first').addEventListener('click', () => goTo(0));
$('nav-prev').addEventListener('click', () => goTo(viewIndex - 1));
$('nav-next').addEventListener('click', () => goTo(viewIndex + 1));
$('nav-last').addEventListener('click', () => goTo(history.length - 1));
$('moves').addEventListener('click', (e) => {
  const t = e.target.closest('[data-i]');
  if (t) goTo(parseInt(t.dataset.i, 10));
});

// PGN import/export — a save/replay format for sharing games and reproducing bugs
// (load a line, then let the AIs continue from it). See pgn.js for the format.
function downloadPgn(text) {
  const blob = new Blob([text], { type: 'application/x-chess-pgn' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `aposchess-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.pgn`;
  a.click();
  URL.revokeObjectURL(url);
}
$('export-pgn').addEventListener('click', async (e) => {
  const text = exportPgn(history, status, { white: playerName('white'), black: playerName('black') });
  const b = e.currentTarget;
  try {
    // Copy to the clipboard — quickest for the debug round-trip (paste it back via
    // Import). Fall back to a file download if the clipboard is unavailable/blocked.
    await navigator.clipboard.writeText(text);
    b.textContent = 'Copied!';
    setTimeout(() => { b.textContent = 'Export PGN'; }, 1200);
  } catch {
    downloadPgn(text);
  }
});
function loadPgnText(text) {
  try {
    const { start, moves } = importPgn(text);
    loadGame(start, moves);
  } catch (err) {
    alert('Could not import PGN: ' + err.message);
  }
}
$('import-pgn').addEventListener('click', async () => {
  // Pair with the clipboard export: read the pasted PGN directly. If the browser
  // won't grant clipboard read (Firefox blocks it for pages, or it's denied), fall
  // back to the file picker so import still works.
  try {
    const text = await navigator.clipboard.readText();
    if (text && text.trim()) { loadPgnText(text); return; }
  } catch { /* fall through to the file picker */ }
  $('import-pgn-file').click();
});
$('import-pgn-file').addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  e.target.value = ''; // allow re-importing the same file later
  if (!file) return;
  loadPgnText(await file.text());
});

// Board editor: drag a spare piece from the palette onto the board (chessground's
// dragNewPiece), drag a piece off the board to delete it (deleteOnDropOff). The
// side-to-move select and the clear/start buttons round out the editor.
for (const el of document.querySelectorAll('#editor-palette .sparepiece')) {
  const piece = { color: el.dataset.color, role: CG_ROLE[el.dataset.role] };
  const startDrag = (e) => {
    if (ui.mode !== 'editor') return;
    e.preventDefault(); // don't start a text selection / scroll
    cg.dragNewPiece(piece, e);
  };
  el.addEventListener('mousedown', startDrag);
  el.addEventListener('touchstart', startDrag, { passive: false });
}
$('editor-turn').addEventListener('change', (e) => { editorTurn = e.target.value; updateStatusText(); });
$('editor-clear').addEventListener('click', () => cg.set({ fen: '8/8/8/8/8/8/8/8' }));
$('editor-start').addEventListener('click', () => cg.set({ fen: toFen(newGameState()) }));
document.addEventListener('keydown', (e) => {
  const tag = e.target.tagName;
  if (tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft') goTo(viewIndex - 1);
  else if (e.key === 'ArrowRight') goTo(viewIndex + 1);
  else if (e.key === 'Home') goTo(0);
  else if (e.key === 'End') goTo(history.length - 1);
  else return;
  e.preventDefault();
});

$('ai-toggle').addEventListener('click', () => {
  ui.running = !ui.running;
  if (ui.running) ui.started = true;
  syncToggleLabel();
  applyAiLock();
  if (ui.running) {
    lastCommitAt = performance.now();
    driveAi();
  } else {
    cancelAi(); // stop the loop and abort any in-flight search
    render();
  }
});

// Set the app up for whatever ui.mode currently says — used at startup (where the
// mode may have been restored by the browser) and again if a late form restoration
// changes it (see resyncRestoredControls).
function enterStartupMode() {
  if (ui.mode === 'online') setOnlinePhase('idle');
  if (ui.mode === 'editor') {
    enterEditor(); // a restored editor mode needs the editable board set up
  } else if (ui.mode === 'puzzle') {
    enterPuzzleMode(); // a restored puzzle mode fetches the catalog and deals a puzzle
  } else if (ui.mode === 'analysis') {
    enterAnalysis(); // a restored/deep-linked analysis mode (no puzzle to return to)
  } else {
    render();
    driveAi(); // if a restored mode has the AI to move first
  }
}

// --- mobile section tabs ---
// On narrow screens the below-board content (move list, settings, rules) is split
// into tabs so you don't scroll past one to reach another; the mode picker + action
// buttons stay pinned under the board. body[data-mobile-tab] drives the CSS — it does
// nothing on desktop, where the media-query rules don't apply and every section shows.
const MOBILE_TABS = ['moves', 'settings', 'rules'];
function setMobileTab(tab) {
  if (!MOBILE_TABS.includes(tab)) tab = 'moves';
  document.body.dataset.mobileTab = tab;
  for (const b of document.querySelectorAll('.mobile-tab')) {
    b.setAttribute('aria-selected', String(b.dataset.mobileTab === tab));
  }
}
// Modes whose settings live in the .controls section (the Settings tab). The rest
// either have no settings or keep them pinned in the actions bar (puzzle's panel),
// so the Settings tab would be empty — hide it, and fall back to Moves if it's open.
const SETTINGS_MODES = new Set(['human-ai', 'ai-ai', 'online', 'editor']);
function syncMobileTabs() {
  const hasSettings = SETTINGS_MODES.has(ui.mode);
  $('tab-settings').hidden = !hasSettings;
  if (!hasSettings && document.body.dataset.mobileTab === 'settings') setMobileTab('moves');
}
for (const b of document.querySelectorAll('.mobile-tab')) {
  b.addEventListener('click', () => setMobileTab(b.dataset.mobileTab));
}
setMobileTab('moves');

syncControlsFromDom();
// A `#mode=…` deep link opens straight into that mode (unless a `#code=` is also
// present — that means "join this game", handled by joinFromUrlCode below, which
// forces online regardless).
const urlMode = getUrlMode();
if (urlMode && getUrlCode().length < CODE_LENGTH) {
  ui.mode = urlMode;
  $('mode').value = urlMode;
  if (urlMode === 'online' || urlMode === 'editor') setMobileTab('settings');
}
loadNetCatalog(); // async: fills the per-slot net pickers from public/nn/manifest.json
applyModeVisibility();
enterStartupMode();

// Reopening a CLOSED tab also restores the controls' previous values — but unlike
// a plain reload (restored before scripts run, which syncControlsFromDom above
// handles), session restore lands them asynchronously, after startup already read
// the markup defaults, and fires no change events. The visible controls then
// disagree with the app (the mode dropdown says "Puzzles", the app is in
// two-player) until the next manual change. No event marks the restoration, so
// re-check the controls a few times shortly after load and re-enter the mode if
// they changed under us.
function controlsOutOfSync() {
  if ($('mode').value !== ui.mode
    || $('side').value !== ui.humanColor
    || $('depth-ai').value !== ui.strengthAi
    || $('depth-white').value !== ui.strengthWhite
    || $('depth-black').value !== ui.strengthBlack
    || engineValue('ai') !== ui.engineAi
    || engineValue('white') !== ui.engineWhite
    || engineValue('black') !== ui.engineBlack
    || $('vary-openings').checked !== ui.varyOpenings
    || $('show-evals').checked !== ui.showEvalBars) return true;
  for (const slot of ['ai', 'white', 'black']) {
    if (clampInt($(`custom-depth-${slot}`).value, 0, 40, 8) !== ui.custom[slot].depth
      || clampInt($(`custom-ms-${slot}`).value, 0, 60000, 6000) !== ui.custom[slot].ms) return true;
  }
  return false;
}
function resyncRestoredControls() {
  if (!controlsOutOfSync()) return; // user-made changes keep ui in sync via the handlers
  const prevMode = ui.mode;
  syncControlsFromDom();
  applyModeVisibility();
  if (ui.mode !== prevMode) enterStartupMode();
}
for (const ms of [200, 600, 1500, 3000]) setTimeout(resyncRestoredControls, ms);

// A `#code=…` in the URL means "join this game". The hash can arrive two ways:
// at load (a shared link opened cold) or later, when someone pastes a new share
// URL into the address bar of an already-open tab — that only changes the hash, so
// the page never reloads. Handle both by switching to online mode and joining the
// code. Our own hash writes go through history.replaceState (see setUrlCode), which
// fires no `hashchange`, so this only ever runs for genuine user navigation.
function joinFromUrlCode() {
  const code = getUrlCode();
  if (code.length < CODE_LENGTH) return; // no/partial code in the hash
  // If we're hosting, never abandon our lobby for an incoming code — tearing it
  // down would close the room and strand anyone trying to join our code. A host who
  // genuinely wants to join elsewhere must Leave first. Joiners/idle tabs do switch.
  if (isHost) return;
  if (online && normalizeCode(online.getCode()) === code) return; // already on it
  $('mode').value = 'online';
  ui.mode = 'online';
  applyModeVisibility();
  setOnlinePhase('idle');
  $('online-code').value = code;
  startJoin();
}
// A hash change is either a shared `#code=` (join) or a `#mode=` route (switch mode).
// A code wins — it implies online. Our own writes use replaceState and fire no
// hashchange, so this only runs for genuine navigation (a pasted/edited URL).
function onHashChange() {
  if (getUrlCode().length >= CODE_LENGTH) { joinFromUrlCode(); return; }
  const m = getUrlMode();
  if (m && m !== ui.mode) selectMode(m);
}
window.addEventListener('hashchange', onHashChange);
joinFromUrlCode(); // opened via a shared link (cold load)

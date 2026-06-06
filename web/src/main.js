// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
import { Chessground } from 'chessground';
import 'chessground/assets/chessground.base.css';
import 'chessground/assets/chessground.brown.css';
import 'chessground/assets/chessground.cburnett.css';
import './styles.css';

import { newGameState, toFen, parseSquare, squareName, opponent } from './board.js';
import { applyMove, gameStatus, destsMap } from './engine.js';
import { hostGame, joinGame, normalizeCode } from './online.js';

const boardEl = document.getElementById('board');
const $ = (id) => document.getElementById(id);

const ui = {
  mode: 'human-human', // 'human-human' | 'human-ai' | 'ai-ai' | 'online'
  humanColor: 'white', // which side the human controls in 'human-ai'
  // AI strength per slot: a preset depth string '1'..'7', or 'custom'.
  strengthAi: '2',     // opponent strength in 'human-ai'
  strengthWhite: '2',  // per-colour strength in 'ai-ai'
  strengthBlack: '2',
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
};

let state = newGameState();
let status = gameStatus(state);
let aiTimer = null;

// Move/capture/check sounds: one reusable Audio element each, rewound before play
// so rapid consecutive moves (e.g. AI vs AI) still click. Public assets, so they
// resolve against Vite's base URL.
const moveSound = new Audio(import.meta.env.BASE_URL + 'sound/standard/Move.mp3');
const captureSound = new Audio(import.meta.env.BASE_URL + 'sound/standard/Capture.mp3');
const checkSound = new Audio(import.meta.env.BASE_URL + 'sound/standard/Check.mp3');
// Check takes priority over capture (and covers checkmate, which is also a check).
function playMoveSound(capture, check) {
  const a = check ? checkSound : capture ? captureSound : moveSound;
  a.currentTime = 0;
  a.play().catch(() => {}); // ignore autoplay blocks before first interaction
}
// captured[color] = list of opponent pieces that `color` has captured (live game).
let captured = { white: [], black: [] };

// Move history for review. Each entry is a snapshot after a ply; index 0 is the
// start position. `viewIndex` is the position shown on the board, which may lag
// the live game while the user steps back through earlier moves.
let history = [startEntry(state)];
let viewIndex = 0;

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

function startEntry(s) {
  return { state: s, captured: { white: [], black: [] }, lastMove: null, san: null, check: false };
}

// The AI search runs in a worker. `aiSeq` tags each real-move request and
// `ponderSeq` each ponder request, so a reply that arrives after the position
// changed (new game, opponent moved, stop) is discarded by sequence mismatch.
let aiSeq = 0;
let ponderSeq = 0;
let aiThinking = false;
let lastCommitAt = performance.now(); // for pacing AI moves (the watch delay)
// While the human thinks, the AI ponders the position after its predicted reply
// (`predictedReply`, from the last search). `ponderState` is that position and
// `ponderBest` tracks the deepest iteration reached so we stop once it converges.
let predictedReply = null;
let ponderState = null;
let ponderBest = 0;
const PONDER_STEP_MS = 700; // ponder in short bursts so the worker stays responsive
let aiWorker = createAiWorker();

function createAiWorker() {
  const w = new Worker(new URL('./aiWorker.js', import.meta.url), { type: 'module' });
  w.onmessage = ({ data }) => {
    if (data.type === 'ponder') onPonderResult(data);
    else onSearchResult(data);
  };
  w.onerror = (e) => {
    console.error('AI worker error:', e.message);
    aiThinking = false;
    updateStatusText();
  };
  return w;
}

// Cancel any pending/in-flight AI work. Terminating is the only way to stop a
// deep search mid-think, so we replace the worker outright; that also discards
// the persistent transposition table, which is correct for a hard reset.
function cancelAi() {
  clearTimeout(aiTimer);
  aiSeq++;
  ponderSeq++;
  aiThinking = false;
  predictedReply = null;
  aiWorker.terminate();
  aiWorker = createAiWorker();
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
});

// Which colour the human may move right now (undefined = board locked).
function controllableColor() {
  if (status.over) return undefined;
  if (ui.mode === 'human-human') return state.turn;
  if (ui.mode === 'human-ai' && state.turn === ui.humanColor) return state.turn;
  if (ui.mode === 'online' && onlineConnected && state.turn === onlineColor) return state.turn;
  return undefined;
}

function aiToMove() {
  if (status.over) return false;
  if (ui.mode === 'human-ai') return state.turn !== ui.humanColor;
  if (ui.mode === 'ai-ai') return ui.running;
  return false;
}

// The colour shown at the bottom of the board (board orientation + bottom tray):
// the human's side in you-vs-AI, this client's side online, else White.
function viewColor() {
  if (ui.mode === 'human-ai') return ui.humanColor;
  if (ui.mode === 'online') return onlineColor || 'white';
  return 'white';
}

function render() {
  const entry = history[viewIndex];
  const atLive = viewIndex === history.length - 1;
  const color = atLive ? controllableColor() : undefined;
  cg.set({
    fen: toFen(entry.state),
    turnColor: entry.state.turn,
    orientation: viewColor(),
    lastMove: entry.lastMove ? [squareName(entry.lastMove.from), squareName(entry.lastMove.to)] : undefined,
    check: entry.check ? entry.state.turn : false,
    viewOnly: !atLive, // lock the board while reviewing an earlier position
    movable: { color, dests: (atLive && color) ? destsMap(status.legal) : new Map() },
  });
  renderTrays(entry);
  renderMoveList();
  updateStatusText();
}

// --- captured pieces / material advantage (Lichess-style) ---
const SHAPE = { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚' };
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

function renderTray(el, color, board, caps) {
  const pieces = [...caps[color]].sort((a, b) => POINTS[b.role] - POINTS[a.role]);
  el.querySelector('.caps').innerHTML = pieces
    .map((p) => `<span class="cap-${p.color}">${SHAPE[p.role]}</span>`)
    .join('');
  const adv = pointAdvantage(board, color);
  el.querySelector('.adv').textContent = adv > 0 ? `+${adv}` : '';
}

function renderTrays(entry) {
  // Bottom player matches board orientation; top player is the opponent.
  const bottom = viewColor();
  renderTray($('tray-bottom'), bottom, entry.state.board, entry.captured);
  renderTray($('tray-top'), opponent(bottom), entry.state.board, entry.captured);
}

function updateStatusText() {
  const el = $('status');
  el.classList.toggle('over', status.over);
  if (status.result === 'checkmate') {
    el.textContent = `Checkmate — ${status.winner === 'white' ? 'White' : 'Black'} wins.`;
  } else if (status.result === 'stalemate') {
    el.textContent = 'Stalemate — draw.';
  } else if (status.result === 'fifty-move') {
    el.textContent = 'Draw — fifty-move rule.';
  } else if (status.result === 'repetition') {
    el.textContent = 'Draw — threefold repetition.';
  } else if (ui.mode === 'online' && !onlineConnected) {
    el.textContent = 'Not connected — host or join a game.';
  } else if (aiThinking) {
    const side = state.turn === 'white' ? 'White' : 'Black';
    el.textContent = `${side} is thinking…`;
  } else {
    const side = state.turn === 'white' ? 'White' : 'Black';
    el.textContent = `${side} to move${status.check ? ' — check!' : ''}`;
  }
}

function commit(move) {
  // A new position supersedes any pending search or ponder reply.
  clearTimeout(aiTimer);
  aiSeq++;
  ponderSeq++;
  aiThinking = false;
  if (move.capture) {
    const taken = state.board[move.to]; // occupant of the landing square (incl. jumps)
    if (taken) captured[state.turn].push(taken);
  }
  const pre = state;
  state = applyMove(pre, move);
  status = gameStatus(state);
  // Threefold repetition is a draw. Override only if the game isn't already over
  // (checkmate/stalemate/fifty-move take precedence and may share the position).
  if (!status.over && countPosition(state) >= 3) {
    status = { over: true, check: status.check, legal: status.legal, result: 'repetition', winner: null };
  }
  const wasLive = viewIndex === history.length - 1;
  history.push({
    state,
    captured: { white: [...captured.white], black: [...captured.black] },
    lastMove: move,
    san: toSan(pre, move, status),
    check: status.check,
  });
  if (wasLive) viewIndex = history.length - 1; // follow the game unless reviewing
  lastCommitAt = performance.now();
  playMoveSound(move.capture, status.check);
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
  viewIndex = clamped;
  render();
}

// chessground reports a legal (from, to); resolve it to an engine move,
// asking for a promotion piece when several moves share that destination.
function onUserMove(orig, dest) {
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
  commit(move);
  if (ui.mode === 'online' && online && onlineConnected) {
    online.send({ t: 'move', from: move.from, to: move.to, promotion: move.promotion || null });
  }
}

// After every position change, decide what the worker should be doing: search
// the real move when it's an AI's turn, or — in you-vs-AI while it's your turn —
// ponder its predicted reply so the next real search starts warm.
function driveAi() {
  clearTimeout(aiTimer);
  updateWakeLock();
  if (status.over) { aiThinking = false; updateStatusText(); return; }
  if (aiToMove()) startSearch();
  else if (canPonder()) startPonder();
  else { aiThinking = false; updateStatusText(); }
}

// Think immediately (no pre-delay): the search overlaps the watch pause, so the
// AI is never idle. The result is held until at least `ui.delay` has elapsed
// since the last move, keeping AI-vs-AI watchable.
function startSearch() {
  const seq = ++aiSeq;
  ponderSeq++; // a real search ends any ponder chain
  aiThinking = true;
  updateStatusText();
  const { depth, maxMs } = aiParams(state.turn);
  aiWorker.postMessage({ type: 'search', seq, state, depth, maxMs });
}

function onSearchResult(data) {
  if (data.seq !== aiSeq) return; // stale result for a superseded position
  predictedReply = data.ponder || null;
  if (!aiToMove() || !data.move) { aiThinking = false; updateStatusText(); return; }
  const move = data.move;
  const seq = data.seq;
  const wait = Math.max(0, lastCommitAt + ui.delay - performance.now());
  aiTimer = setTimeout(() => {
    if (aiSeq !== seq || !aiToMove()) return;
    commit(move);
  }, wait);
}

// Ponder = think about the position we'd reach if the human plays the move the
// AI predicted for them. On a hit, the real search reuses this work; on a miss,
// the table still holds useful overlap. Done in repeated short bursts so a real
// move request (the human moving) is picked up within one burst.
function canPonder() {
  return ui.mode === 'human-ai' && !status.over
    && state.turn === ui.humanColor && !!predictedReply;
}

function startPonder() {
  const pm = status.legal.find((m) => m.from === predictedReply.from && m.to === predictedReply.to);
  if (!pm) { aiThinking = false; return; }
  ponderState = applyMove(state, pm);
  ponderBest = 0;
  aiThinking = false; // pondering is background; the status still reads "your move"
  updateStatusText();
  const seq = ++ponderSeq;
  const { depth } = aiParams(state.turn);
  aiWorker.postMessage({ type: 'ponder', seq, state: ponderState, depth, maxMs: PONDER_STEP_MS });
}

function onPonderResult(data) {
  if (data.seq !== ponderSeq || !canPonder()) return;
  const { depth } = aiParams(state.turn);
  // Stop once we've searched to full strength or stopped making progress (e.g. a
  // forced line resolved), so we don't spin firing instant bursts.
  if (data.reached >= depth || data.reached <= ponderBest) return;
  ponderBest = data.reached;
  aiWorker.postMessage({ type: 'ponder', seq: data.seq, state: ponderState, depth, maxMs: PONDER_STEP_MS });
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
  $('online-host').hidden = !idle;
  $('online-join').hidden = !idle;
  $('online-code').hidden = !idle;
  $('online-color-label').hidden = !idle;
  $('online-share').hidden = phase !== 'hosting';
  $('online-leave').hidden = idle;
  $('online-leave').textContent = phase === 'connected' ? 'Leave' : 'Cancel';
  $('online-swap').hidden = !(phase === 'connected' && isHost); // host-only, while connected
  if (phase === 'hosting') $('online-code-out').textContent = detail;
  const msg = {
    idle: detail,
    hosting: 'Waiting for an opponent to join…',
    connecting: detail || 'Connecting…',
    connected: detail,
  }[phase];
  $('online-status').textContent = msg ?? detail;
}

const connectedMsg = (color) => `Connected — you play ${color === 'white' ? 'White' : 'Black'}.`;

// The share code lives in the page URL (`?code=…`) so a host can just copy the
// address bar and a joiner who opens that link auto-joins (see startup below).
const shareUrl = (code) => {
  const u = new URL(window.location.href);
  u.searchParams.set('code', code);
  return u.href;
};
function setUrlCode(code) {
  const u = new URL(window.location.href);
  if (code) u.searchParams.set('code', code);
  else u.searchParams.delete('code');
  history.replaceState(null, '', u.href);
}
const getUrlCode = () => normalizeCode(new URL(window.location.href).searchParams.get('code') || '');

// Tear down the current session (if any) and clear connection state. The caller
// is responsible for the resulting UI phase.
function leaveOnline() {
  if (online) { online.close(); online = null; }
  onlineConnected = false;
  onlineColor = null;
  isHost = false;
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

// The host's chosen colour is decided locally; on connect it tells the joiner the
// opposite colour via `hello`, then both sides start from the same clean position.
function onHostConnected(color) {
  onlineColor = color;
  onlineConnected = true;
  online.send({ t: 'hello', color: opponent(color) });
  setOnlinePhase('connected', connectedMsg(color));
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

function onOnlineClosed() {
  leaveOnline();
  setUrlCode('');
  setOnlinePhase('idle', 'Opponent disconnected.');
  render(); // lock the board
}

function onOnlineError(message) {
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
    onlineColor = msg.color === 'black' ? 'black' : 'white';
    onlineConnected = true;
    setOnlinePhase('connected', connectedMsg(onlineColor));
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
  captured = { white: [], black: [] };
  history = [startEntry(state)];
  viewIndex = 0;
  posCounts = new Map();
  countPosition(state);
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

function applyModeVisibility() {
  const m = ui.mode;
  $('side-control').hidden = m !== 'human-ai';
  $('ai-toggle').hidden = m !== 'ai-ai';
  $('row-ai-swap').hidden = m !== 'ai-ai';
  $('row-online').hidden = m !== 'online';
  // human-ai: one colour-agnostic AI row. ai-ai: separate White/Black rows.
  $('row-ai').hidden = m !== 'human-ai';
  $('row-white').hidden = m !== 'ai-ai';
  $('row-black').hidden = m !== 'ai-ai';
  // A row's custom depth/timeout inputs appear only when its strength is "Custom".
  toggleCustom('ai', m === 'human-ai' && ui.strengthAi === 'custom');
  toggleCustom('white', m === 'ai-ai' && ui.strengthWhite === 'custom');
  toggleCustom('black', m === 'ai-ai' && ui.strengthBlack === 'custom');
}

function toggleCustom(slot, show) {
  $(`custom-depth-${slot}`).closest('label').hidden = !show;
  $(`custom-ms-${slot}`).closest('label').hidden = !show;
}

// Search params for the AI playing `turn`: a preset depth + the default cap, or
// that slot's custom depth + timeout when its strength is set to "Custom".
function aiParams(turn) {
  const slot = ui.mode === 'ai-ai' ? turn : 'ai';
  const v = strengthOf(slot);
  if (v !== 'custom') return { depth: parseInt(v, 10), maxMs: ui.maxMs };
  // 0 means "no limit" for either field. Both unlimited would never return, so
  // fall back to the default time cap in that case.
  let { depth, ms } = ui.custom[slot];
  if (depth === 0 && ms === 0) ms = ui.maxMs;
  return { depth: depth === 0 ? Infinity : depth, maxMs: ms === 0 ? Infinity : ms };
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
  for (const slot of ['ai', 'white', 'black']) {
    ui.custom[slot].depth = clampInt($(`custom-depth-${slot}`).value, 0, 40, 8);
    ui.custom[slot].ms = clampInt($(`custom-ms-${slot}`).value, 0, 60000, 6000);
  }
}

$('mode').addEventListener('change', (e) => {
  const prev = ui.mode;
  ui.mode = e.target.value;
  if (prev === 'online' && ui.mode !== 'online') { leaveOnline(); setUrlCode(''); }
  applyModeVisibility();
  if (ui.mode === 'online') setOnlinePhase('idle');
  newGame();
});
$('side').addEventListener('change', (e) => {
  ui.humanColor = e.target.value;
  newGame();
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
  $(`custom-depth-${slot}`).addEventListener('change', (e) => {
    ui.custom[slot].depth = clampInt(e.target.value, 0, 40, 8);
    e.target.value = ui.custom[slot].depth; // reflect the clamped value
  });
  $(`custom-ms-${slot}`).addEventListener('change', (e) => {
    ui.custom[slot].ms = clampInt(e.target.value, 0, 60000, 6000);
    e.target.value = ui.custom[slot].ms;
  });
}
$('new-game').addEventListener('click', () => {
  if (ui.mode === 'online' && onlineConnected) online.send({ t: 'reset' });
  newGame();
});
// AI-vs-AI: swap White's and Black's settings (preset strength + custom
// depth/timeout) so the stronger engine plays the other colour. Takes effect on
// the next move; the position is left as-is so you can swap mid-game.
$('ai-swap').addEventListener('click', () => {
  [ui.strengthWhite, ui.strengthBlack] = [ui.strengthBlack, ui.strengthWhite];
  [ui.custom.white, ui.custom.black] = [ui.custom.black, ui.custom.white];
  $('depth-white').value = ui.strengthWhite;
  $('depth-black').value = ui.strengthBlack;
  $('custom-depth-white').value = ui.custom.white.depth;
  $('custom-ms-white').value = ui.custom.white.ms;
  $('custom-depth-black').value = ui.custom.black.depth;
  $('custom-ms-black').value = ui.custom.black.ms;
  applyModeVisibility(); // reveal/hide each row's custom inputs to match
});

// Online panel controls.
$('online-host').addEventListener('click', startHost);
$('online-join').addEventListener('click', startJoin);
$('online-code').addEventListener('keydown', (e) => { if (e.key === 'Enter') startJoin(); });
$('online-leave').addEventListener('click', () => { leaveOnline(); setUrlCode(''); setOnlinePhase('idle'); newGame(); });
$('online-swap').addEventListener('click', onlineSwap);
$('online-copy').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(shareUrl($('online-code-out').textContent));
    const b = $('online-copy');
    b.textContent = 'Copied';
    setTimeout(() => { b.textContent = 'Copy link'; }, 1200);
  } catch { /* clipboard may be blocked; the code is shown for manual copy */ }
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
  if (ui.running) {
    lastCommitAt = performance.now();
    driveAi();
  } else {
    cancelAi(); // stop the loop and abort any in-flight search
    render();
  }
});

syncControlsFromDom();
applyModeVisibility();
if (ui.mode === 'online') setOnlinePhase('idle');
render();
driveAi(); // if a restored mode has the AI to move first

// Opened via a shared link (`?code=…`): switch to online mode and auto-join.
const launchCode = getUrlCode();
if (launchCode.length >= 4) {
  $('mode').value = 'online';
  ui.mode = 'online';
  applyModeVisibility();
  setOnlinePhase('idle');
  $('online-code').value = launchCode;
  startJoin();
}

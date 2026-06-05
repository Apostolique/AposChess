// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
import { Chessground } from 'chessground';
import 'chessground/assets/chessground.base.css';
import 'chessground/assets/chessground.brown.css';
import 'chessground/assets/chessground.cburnett.css';
import './styles.css';

import { newGameState, toFen, parseSquare, squareName, opponent } from './board.js';
import { applyMove, gameStatus, destsMap } from './engine.js';

const boardEl = document.getElementById('board');
const $ = (id) => document.getElementById(id);

const ui = {
  mode: 'human-human', // 'human-human' | 'human-ai' | 'ai-ai'
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
};

let state = newGameState();
let status = gameStatus(state);
let aiTimer = null;

// Move/capture sounds: one reusable Audio element each, rewound before play so
// rapid consecutive moves (e.g. AI vs AI) still click. Public assets, so they
// resolve against Vite's base URL.
const moveSound = new Audio(import.meta.env.BASE_URL + 'sound/standard/Move.mp3');
const captureSound = new Audio(import.meta.env.BASE_URL + 'sound/standard/Capture.mp3');
function playMoveSound(capture) {
  const a = capture ? captureSound : moveSound;
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
  return undefined;
}

function aiToMove() {
  if (status.over) return false;
  if (ui.mode === 'human-ai') return state.turn !== ui.humanColor;
  if (ui.mode === 'ai-ai') return ui.running;
  return false;
}

function render() {
  const entry = history[viewIndex];
  const atLive = viewIndex === history.length - 1;
  const color = atLive ? controllableColor() : undefined;
  cg.set({
    fen: toFen(entry.state),
    turnColor: entry.state.turn,
    orientation: ui.mode === 'human-ai' ? ui.humanColor : 'white',
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
  const bottom = ui.mode === 'human-ai' ? ui.humanColor : 'white';
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
  playMoveSound(move.capture);
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
      commit(matches.find((m) => m.promotion === role) || matches[0]);
    });
  } else {
    commit(matches[0]);
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

// --- controls ---
function newGame() {
  cancelAi();
  ui.running = false;
  syncToggleLabel();
  state = newGameState();
  status = gameStatus(state);
  captured = { white: [], black: [] };
  history = [startEntry(state)];
  viewIndex = 0;
  lastCommitAt = performance.now();
  render();
  driveAi();
}

function syncToggleLabel() {
  $('ai-toggle').textContent = ui.running ? 'Stop' : 'Start';
}

const strengthOf = (slot) =>
  slot === 'ai' ? ui.strengthAi : slot === 'white' ? ui.strengthWhite : ui.strengthBlack;

function applyModeVisibility() {
  const m = ui.mode;
  $('side-control').hidden = m !== 'human-ai';
  $('ai-toggle').hidden = m !== 'ai-ai';
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
  ui.mode = e.target.value;
  applyModeVisibility();
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
$('new-game').addEventListener('click', newGame);

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
render();
driveAi(); // if a restored mode has the AI to move first

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
  depth: 2,
  maxMs: 6000,         // hard cap on AI thinking time (deep settings stay bounded)
  delay: 450,          // ms pause before an AI move, so play is watchable
  running: false,      // AI-vs-AI loop active
};

let state = newGameState();
let status = gameStatus(state);
let aiTimer = null;
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

// The AI search runs in a worker. `aiSeq` tags each request so a result that
// arrives after the position changed (new game, undo, stop) is discarded.
let aiSeq = 0;
let aiThinking = false;
let aiWorker = createAiWorker();

function createAiWorker() {
  const w = new Worker(new URL('./aiWorker.js', import.meta.url), { type: 'module' });
  w.onmessage = ({ data }) => {
    if (data.seq !== aiSeq) return; // stale result for a superseded position
    aiThinking = false;
    if (aiToMove() && data.move) commit(data.move);
  };
  w.onerror = (e) => {
    console.error('AI worker error:', e.message);
    aiThinking = false;
    updateStatusText();
  };
  return w;
}

// Cancel any pending/in-flight AI move. Terminating is the only way to stop a
// deep search mid-think, so we replace the worker outright.
function cancelAi() {
  clearTimeout(aiTimer);
  aiSeq++;
  aiThinking = false;
  aiWorker.terminate();
  aiWorker = createAiWorker();
}

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
  aiSeq++; // a new position supersedes any pending AI result
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
  render();
  scheduleAiIfNeeded();
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
  if (cur) cur.scrollIntoView({ block: 'nearest' });
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

function scheduleAiIfNeeded() {
  clearTimeout(aiTimer);
  if (!aiToMove()) return;
  const seq = aiSeq;
  aiTimer = setTimeout(() => {
    if (!aiToMove() || aiSeq !== seq) return;
    aiThinking = true;
    updateStatusText();
    aiWorker.postMessage({ seq, state, depth: ui.depth, maxMs: ui.maxMs });
  }, ui.delay);
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
  render();
  scheduleAiIfNeeded();
}

function syncToggleLabel() {
  $('ai-toggle').textContent = ui.running ? 'Stop' : 'Start';
}

function applyModeVisibility() {
  $('side-control').hidden = ui.mode !== 'human-ai';
  $('depth-control').hidden = ui.mode === 'human-human';
  $('ai-toggle').hidden = ui.mode !== 'ai-ai';
}

// Browsers restore <select> values on reload, so read them into `ui` at startup
// instead of assuming the hardcoded defaults — otherwise the shown selection and
// the actual behaviour disagree until the next change event.
function syncControlsFromDom() {
  ui.mode = $('mode').value;
  ui.humanColor = $('side').value;
  ui.depth = parseInt($('depth').value, 10);
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
$('depth').addEventListener('change', (e) => {
  ui.depth = parseInt(e.target.value, 10);
});
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
    scheduleAiIfNeeded();
  } else {
    cancelAi(); // stop the loop and abort any in-flight search
    render();
  }
});

syncControlsFromDom();
applyModeVisibility();
render();
scheduleAiIfNeeded(); // if a restored mode has the AI to move first

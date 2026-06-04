// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
import { Chessground } from 'chessground';
import 'chessground/assets/chessground.base.css';
import 'chessground/assets/chessground.brown.css';
import 'chessground/assets/chessground.cburnett.css';
import './styles.css';

import { newGameState, toFen, parseSquare, squareName, opponent } from './board.js';
import { applyMove, gameStatus, destsMap } from './engine.js';
import { chooseMove } from './ai.js';

const boardEl = document.getElementById('board');
const $ = (id) => document.getElementById(id);

const ui = {
  mode: 'human-human', // 'human-human' | 'human-ai' | 'ai-ai'
  humanColor: 'white', // which side the human controls in 'human-ai'
  depth: 2,
  delay: 450,          // ms pause before an AI move, so play is watchable
  running: false,      // AI-vs-AI loop active
};

let state = newGameState();
let status = gameStatus(state);
let lastMove = null;
let aiTimer = null;
// captured[color] = list of opponent pieces that `color` has captured.
let captured = { white: [], black: [] };

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
  status = gameStatus(state);
  const color = controllableColor();
  cg.set({
    fen: toFen(state),
    turnColor: state.turn,
    orientation: ui.mode === 'human-ai' ? ui.humanColor : 'white',
    lastMove: lastMove ? [squareName(lastMove.from), squareName(lastMove.to)] : undefined,
    check: status.check ? state.turn : false,
    movable: { color, dests: color ? destsMap(status.legal) : new Map() },
  });
  renderTrays();
  updateStatusText();
}

// --- captured pieces / material advantage (Lichess-style) ---
const SHAPE = { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚' };
const POINTS = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

function pointAdvantage(color) {
  let w = 0, b = 0;
  for (const p of state.board) {
    if (!p) continue;
    if (p.color === 'white') w += POINTS[p.role]; else b += POINTS[p.role];
  }
  const diff = w - b;
  return color === 'white' ? diff : -diff;
}

function renderTray(el, color) {
  const pieces = [...captured[color]].sort((a, b) => POINTS[b.role] - POINTS[a.role]);
  el.querySelector('.caps').innerHTML = pieces
    .map((p) => `<span class="cap-${p.color}">${SHAPE[p.role]}</span>`)
    .join('');
  const adv = pointAdvantage(color);
  el.querySelector('.adv').textContent = adv > 0 ? `+${adv}` : '';
}

function renderTrays() {
  // Bottom player matches board orientation; top player is the opponent.
  const bottom = ui.mode === 'human-ai' ? ui.humanColor : 'white';
  renderTray($('tray-bottom'), bottom);
  renderTray($('tray-top'), opponent(bottom));
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
  } else {
    const side = state.turn === 'white' ? 'White' : 'Black';
    el.textContent = `${side} to move${status.check ? ' — check!' : ''}`;
  }
}

function commit(move) {
  if (move.capture) {
    const taken = state.board[move.to]; // occupant of the landing square (incl. jumps)
    if (taken) captured[state.turn].push(taken);
  }
  state = applyMove(state, move);
  lastMove = move;
  render();
  scheduleAiIfNeeded();
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
  aiTimer = setTimeout(() => {
    if (!aiToMove()) return;
    const move = chooseMove(state, ui.depth);
    if (move) commit(move);
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
  clearTimeout(aiTimer);
  ui.running = false;
  syncToggleLabel();
  state = newGameState();
  lastMove = null;
  captured = { white: [], black: [] };
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
$('ai-toggle').addEventListener('click', () => {
  ui.running = !ui.running;
  syncToggleLabel();
  scheduleAiIfNeeded();
});

applyModeVisibility();
render();

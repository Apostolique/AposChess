// AposChess ladder visualizer — client. Vanilla ES module, hand-rolled SVG
// charts (no chart lib, no CDN). Everything is driven by the local server's
// /api/* endpoints and refreshes live over SSE.
//
// The game viewer imports the app's own board and rules (served from web/src by
// the tool server) instead of carrying a second copy: replaying a recorded game
// through the real move generator is what makes the check sound and the material
// trays agree with the site.

import { parseFen, parseSquare, opponent, START_FEN } from '/src/board.js';
import { legalMoves, applyMove, kingAttacked } from '/src/engine.js';

// ------------------------------------------------------------------ utilities
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
function el(tag, attrs = {}, ...kids) {
  const n = document.createElementNS(tag === 'svg' || SVG_TAGS.has(tag) ? SVGNS : HTMLNS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === 'class') n.setAttribute('class', v);
    else if (k === 'html') n.innerHTML = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
}
const HTMLNS = 'http://www.w3.org/1999/xhtml';
const SVGNS = 'http://www.w3.org/2000/svg';
const SVG_TAGS = new Set(['svg', 'g', 'path', 'line', 'rect', 'circle', 'text', 'polyline', 'polygon', 'defs', 'clipPath', 'tspan']);

const CAT = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'].map((v) => `var(${v})`);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? '–' : Number(n).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d }));
const pct = (n) => (n == null ? '–' : `${(n * 100).toFixed(1)}%`);

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
function parseTag(tag) {
  const m = /^([a-z]+)(\d+)@(.+)$/.exec(tag || '');
  if (!m) return { eng: '?', depth: 0, version: tag };
  return { eng: m[1], depth: Number(m[2]), version: m[3] };
}
function familyColor(eng, version) {
  if (eng === 'hc') return css('--hc');
  if (version === '?') return css('--mat');
  return css('--s1');
}
function nodeLabel(eng, version, champs) {
  if (eng === 'hc') return version === '?' ? 'material?' : `handcrafted v${version}`;
  const c = champs?.byHash?.[version];
  return c?.name || `nn ${version}`;
}

// ------------------------------------------------------------- shared tooltip
const TT = $('#tooltip');
function showTip(html, clientX, clientY) {
  TT.innerHTML = html;
  TT.classList.add('show');
  TT.style.left = `${clientX + window.scrollX}px`;
  TT.style.top = `${clientY + window.scrollY}px`;
}
function hideTip() { TT.classList.remove('show'); }

// --------------------------------------------------------------- SVG charting
// A minimal but honest line/scatter chart with axes, gridlines, a crosshair and
// a shared HTML tooltip. One linear y-axis only (never dual-axis).
function linScale(d0, d1, r0, r1) {
  const s = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  s.invert = (px) => d0 + ((px - r0) / (r1 - r0 || 1)) * (d1 - d0);
  s.domain = [d0, d1]; s.range = [r0, r1];
  return s;
}
function niceTicks(min, max, count = 6) {
  const span = (max - min) || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const norm = step0 / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const t = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) t.push(Math.round(v * 1e6) / 1e6);
  return t;
}

function lineChart(opts) {
  const W = 940, H = opts.height || 400;
  const m = { t: 16, r: 18, b: 44, l: 58, ...(opts.margin || {}) };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const series = opts.series.filter((s) => s.points.length);
  const allX = series.flatMap((s) => s.points.map((p) => p.x));
  // A band is data, so it sets the scale like any other value. One unmeasured point stretches the
  // axis and flattens the rest of the curve, which is the honest reading of that point.
  const allY = series.flatMap((s) => s.points.flatMap((p) => (s.band && p.lo != null ? [p.y, p.lo, p.hi] : [p.y])));
  const xd = opts.xDomain || [Math.min(...allX), Math.max(...allX)];
  let yd = opts.yDomain || [Math.min(...allY), Math.max(...allY)];
  if (yd[0] === yd[1]) yd = [yd[0] - 1, yd[1] + 1];
  if (!opts.yDomain) { const pad = (yd[1] - yd[0]) * 0.08; yd = [yd[0] - pad, yd[1] + pad]; }
  const x = linScale(xd[0], xd[1], m.l, m.l + iw);
  const y = linScale(yd[0], yd[1], m.t + ih, m.t);

  const svg = el('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'xMinYMin meet' });
  const g = el('g'); svg.append(g);

  // y grid + ticks
  for (const t of niceTicks(yd[0], yd[1], 6)) {
    const yy = y(t);
    g.append(el('line', { class: 'gridline', x1: m.l, x2: m.l + iw, y1: yy, y2: yy }));
    g.append(el('text', { x: m.l - 8, y: yy + 3, 'text-anchor': 'end' }, opts.formatY ? opts.formatY(t) : fmt(t)));
  }
  // x ticks
  const xticks = opts.xTicks || niceTicks(xd[0], xd[1], Math.min(10, allX.length));
  for (const t of xticks) {
    const xx = x(t);
    g.append(el('text', { x: xx, y: m.t + ih + 18, 'text-anchor': 'middle' }, opts.formatX ? opts.formatX(t) : fmt(t)));
  }
  g.append(el('line', { class: 'baseline', x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
  if (opts.yLabel) g.append(el('text', { class: 'axis-title', x: 14, y: m.t + ih / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${m.t + ih / 2})` }, opts.yLabel));
  if (opts.xLabel) g.append(el('text', { class: 'axis-title', x: m.l + iw / 2, y: H - 6, 'text-anchor': 'middle' }, opts.xLabel));

  // reference lines
  for (const ref of opts.refs || []) {
    const yy = y(ref.y);
    g.append(el('line', { class: 'crosshair', x1: m.l, x2: m.l + iw, y1: yy, y2: yy, style: `stroke:${ref.color || css('--muted')}` }));
    g.append(el('text', { x: m.l + iw - 2, y: yy - 4, 'text-anchor': 'end', style: `fill:${ref.color || css('--muted')}` }, ref.label));
  }
  for (const ref of opts.vrefs || []) {
    if (ref.x < xd[0] || ref.x > xd[1]) continue;
    const xx = x(ref.x);
    g.append(el('line', { class: 'crosshair', x1: xx, x2: xx, y1: m.t, y2: m.t + ih, style: `stroke:${ref.color || css('--muted')}` }));
    if (ref.label) g.append(el('text', { x: xx + 4, y: m.t + 10, style: `fill:${ref.color || css('--muted')}` }, ref.label));
  }

  // Uncertainty bands, all of them before any line, so a later series' wash never sits on top of an
  // earlier series' line. The fill is the series hue at 10% with the two bounds drawn as hairlines:
  // the estimate keeps the only full-weight mark, and the bounds stay readable as bounds.
  const sortX = (pts) => pts.slice().sort((a, b) => a.x - b.x);
  for (const s of series) {
    if (!s.band) continue;
    const bp = sortX(s.points).filter((p) => p.lo != null && p.hi != null);
    if (!bp.length) continue;
    const at = (p, k) => `${x(p.x).toFixed(1)},${y(clampY(p[k], yd)).toFixed(1)}`;
    const top = bp.map((p, i) => `${i ? 'L' : 'M'}${at(p, 'hi')}`).join(' ');
    const bottom = bp.slice().reverse().map((p) => `L${at(p, 'lo')}`).join(' ');
    g.append(el('path', { class: 'series-band', d: `${top} ${bottom} Z`, style: `fill:${s.color}` }));
    for (const edge of ['hi', 'lo']) {
      g.append(el('path', { class: 'series-band-edge', d: bp.map((p, i) => `${i ? 'L' : 'M'}${at(p, edge)}`).join(' '), style: `stroke:${s.color}` }));
    }
  }

  // series
  for (const s of series) {
    const pts = sortX(s.points);
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.x).toFixed(1)},${y(clampY(p.y, yd)).toFixed(1)}`).join(' ');
    g.append(el('path', { class: 'series-line', d, style: `stroke:${s.color};${s.dash ? `stroke-dasharray:${s.dash}` : ''}` }));
    if (s.dots !== false) for (const p of pts) g.append(el('circle', { class: 'dot', cx: x(p.x), cy: y(clampY(p.y, yd)), r: p.mate ? 5 : 3.5, style: `fill:${p.mate ? css('--warning') : s.color}` }));
    if (s.label && opts.directLabels) {
      const last = pts[pts.length - 1];
      g.append(el('text', { x: x(last.x) + 6, y: y(clampY(last.y, yd)) + 3, style: `fill:${s.color};font-weight:600` }, s.label));
    }
  }

  // hover overlay + crosshair
  const cross = el('line', { class: 'crosshair', y1: m.t, y2: m.t + ih, style: 'display:none' });
  g.append(cross);
  const overlay = el('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent', style: 'cursor:crosshair' });
  g.append(overlay);
  const flatPts = series.flatMap((s) => s.points.map((p) => ({ ...p, s })));
  overlay.addEventListener('mousemove', (ev) => {
    const rect = svg.getBoundingClientRect();
    const sx = W / rect.width;
    const localX = (ev.clientX - rect.left) * sx;
    const xv = x.invert(localX);
    // nearest x column
    const cols = [...new Set(flatPts.map((p) => p.x))].sort((a, b) => Math.abs(a - xv) - Math.abs(b - xv));
    const cx = cols[0];
    if (cx == null) return;
    cross.style.display = ''; cross.setAttribute('x1', x(cx)); cross.setAttribute('x2', x(cx));
    const here = flatPts.filter((p) => p.x === cx).sort((a, b) => b.y - a.y);
    const fy = (v) => (opts.formatY ? opts.formatY(v) : fmt(v, 1));
    const rows = here.map((p) => {
      const band = p.s.band && p.lo != null ? `<span class="tt-band">${fy(p.lo)} – ${fy(p.hi)}</span>` : '';
      return `<div class="tt-row"><span class="k"><span class="swatch" style="background:${p.s.color}"></span>${p.s.label}</span><span class="v">${fy(p.y)}${p.mate ? ' ⚑' : ''}${band}</span></div>`;
    }).join('');
    showTip(`<div class="tt-title">${opts.formatX ? opts.formatX(cx) : fmt(cx)}${opts.xLabel ? '' : ''}</div>${rows}`, ev.clientX, ev.clientY);
  });
  overlay.addEventListener('mouseleave', () => { hideTip(); cross.style.display = 'none'; });
  return svg;
}
function clampY(v, yd) { return Math.max(yd[0], Math.min(yd[1], v)); }

function barChart(opts) {
  const W = 940, H = opts.height || 300;
  const m = { t: 14, r: 18, b: 60, l: 58, ...(opts.margin || {}) };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const data = opts.data;
  const maxV = Math.max(0, ...data.map((d) => d.value));
  const minV = Math.min(0, ...data.map((d) => d.value));
  const y = linScale(minV, maxV, m.t + ih, m.t);
  const bw = iw / data.length;
  const svg = el('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}` });
  const g = el('g'); svg.append(g);
  for (const t of niceTicks(minV, maxV, 5)) {
    const yy = y(t);
    g.append(el('line', { class: 'gridline', x1: m.l, x2: m.l + iw, y1: yy, y2: yy }));
    g.append(el('text', { x: m.l - 8, y: yy + 3, 'text-anchor': 'end' }, opts.formatY ? opts.formatY(t) : fmt(t)));
  }
  const y0 = y(0);
  data.forEach((d, i) => {
    const bx = m.l + i * bw + bw * 0.16;
    const w = bw * 0.68;
    const top = Math.min(y(d.value), y0), h = Math.abs(y(d.value) - y0);
    const rect = el('rect', { x: bx, y: top, width: w, height: Math.max(1, h), rx: 4, style: `fill:${d.color || css('--s1')}` });
    rect.addEventListener('mousemove', (ev) => showTip(`<div class="tt-title">${d.label}</div><div class="tt-row"><span class="k">${opts.valueLabel || 'value'}</span><span class="v">${opts.formatY ? opts.formatY(d.value) : fmt(d.value, 1)}</span></div>`, ev.clientX, ev.clientY));
    rect.addEventListener('mouseleave', hideTip);
    g.append(rect);
    g.append(el('text', { x: bx + w / 2, y: m.t + ih + 14, 'text-anchor': 'end', transform: `rotate(-40 ${bx + w / 2} ${m.t + ih + 14})`, style: 'font-size:10px' }, d.label));
  });
  g.append(el('line', { class: 'baseline', x1: m.l, x2: m.l + iw, y1: y0, y2: y0 }));
  return svg;
}

function legend(items, onToggle) {
  const box = el('div', { class: 'legend' });
  for (const it of items) {
    const item = el('div', { class: `item${it.off ? ' off' : ''}` },
      el('span', { class: 'swatch', style: `background:${it.color}` }), it.label);
    if (onToggle) item.addEventListener('click', () => onToggle(it));
    box.append(item);
  }
  return box;
}

// ------------------------------------------------------------------- state
const State = {
  ladder: null, champs: null, summary: null, pool: null,
  // ONE depth for the whole dashboard — ladder, generations and matchups all read it, so a depth
  // picked on any of them carries to the others and the numbers stay comparable. It starts unset
  // and is resolved from the data on first render (see resolveDepth): a hardcoded default filters
  // every view down to nothing when a run doesn't cover that depth. 'all' means "don't pin a
  // depth", and each view spells out what that means for it.
  depth: null,
  depthSel: new Set(),   // depth×Elo: which engines are plotted
  mFam: 'all',           // matchups: engine family filter
  mCross: 'unlinked',    // matchups: which rows get cross-depth columns — none | unlinked | all
  gamesFilter: null,     // games: {a, b} tag filter
  openGame: null,        // games: {g, ply} currently in the viewer
  pendingGame: null,     // games: {g, ply} asked for by the URL, honoured once
  trackId: null,         // training: open track
};
// Controls for the game currently open in the viewer; driven by keyboard hotkeys.
let activeViewer = null;
// One playback timer for the whole page. A viewer gets replaced under a running interval every
// time the user picks another game or a live refresh rebuilds the list, and an orphaned interval
// keeps stepping — repainting the new viewer's eval chart and dragging State.openGame back to the
// game it belongs to, with no button left that can stop it. Every teardown path calls stopPlayback.
let playback = null; // { timer, onStop }
const isPlaying = () => playback !== null;
function stopPlayback() {
  if (!playback) return;
  clearInterval(playback.timer);
  const { onStop } = playback;
  playback = null;
  onStop();
}

async function loadCore() {
  const [summary, ladder] = await Promise.all([api('/api/summary'), api('/api/ladder')]);
  State.summary = summary; State.champs = summary.champions; State.ladder = ladder;
}

// group ranking entries by version → {name, eng, gen, arch, byDepth}
function groupVersions() {
  const map = new Map();
  for (const r of State.ladder.ranking || []) {
    const key = `${r.eng}@${r.version}`;
    if (!map.has(key)) {
      const c = State.champs?.byHash?.[r.version];
      map.set(key, { key, eng: r.eng, version: r.version, name: r.name || nodeLabel(r.eng, r.version, State.champs), gen: c?.gen ?? null, arch: c?.arch ?? null, byDepth: new Map(), anchor: false });
    }
    const v = map.get(key);
    v.byDepth.set(Number(r.depth), r);
    if (r.anchor) v.anchor = true;
  }
  return map;
}
function bestNode(v) { let best = null; for (const r of v.byDepth.values()) if (!best || r.elo > best.elo) best = r; return best; }

// ---------------------------------------------------------------- depth pickers
// The depths the RANKING actually contains, not `ladder.depths` (which is only what the
// current run schedules). A `--depths=1-3` run still carries the depth-6 anchor row, so the
// two lists differ and the pin has to stay reachable in the dropdowns.
function availableDepths() {
  const seen = new Set();
  for (const r of State.ladder?.ranking || []) { const d = Number(r.depth); if (Number.isFinite(d)) seen.add(d); }
  if (!seen.size) for (const d of State.ladder?.depths || [1, 2, 3, 4, 5, 6, 7, 8]) seen.add(Number(d));
  return [...seen].sort((a, b) => a - b);
}
// Deepest depth carrying at least `min` rows that match `pred`, or null.
function deepestWith(pred = () => true, min = 2) {
  const count = new Map();
  for (const r of State.ladder?.ranking || []) {
    if (!pred(r)) continue;
    const d = Number(r.depth);
    if (Number.isFinite(d)) count.set(d, (count.get(d) || 0) + 1);
  }
  const usable = [...count.entries()].filter(([, n]) => n >= min).map(([d]) => d);
  return usable.length ? Math.max(...usable) : null;
}
// Keep the depth the user picked as long as the data still has a node at it, otherwise fall back
// to a sensible one. Runs on every render, not once — State survives live reloads, and a run that
// changes --depths must not leave the views pinned to a depth that no longer exists.
// A single explicit pick survives (the anchor's own depth is worth a look); only the auto-picked
// default insists on company, and it prefers a depth the CHAMPIONS were rated at — the anchor's
// depth often holds nothing else, which leaves the generations curve empty.
function resolveDepth() {
  if (State.depth === 'all') return 'all';
  if (State.depth != null && availableDepths().includes(Number(State.depth))) return Number(State.depth);
  const isChampRow = (r) => State.champs?.byHash?.[r.version]?.gen != null;
  return deepestWith(isChampRow, 2) ?? deepestWith(() => true, 2) ?? availableDepths().at(-1) ?? null;
}
// The one depth control, rendered by every view that filters by depth. Picking here re-renders the
// current view and writes the URL; the other views pick the value up when you switch to them.
function depthSelect() {
  return el('select', { 'data-ctl': 'depth', onchange: (e) => setDepth(e.target.value === 'all' ? 'all' : Number(e.target.value)) },
    ...['all', ...availableDepths()].map((d) => el('option', { value: d, selected: String(d) === String(State.depth) ? '' : null },
      d === 'all' ? 'all depths' : `depth ${d}`)));
}
function setDepth(v) { State.depth = v; renderView(currentView); syncUrl(); }
const depthNote = 'Depth is shared by every view.';

// ================================================================= LADDER view
function renderLadder() {
  const root = $('#view-ladder'); root.innerHTML = '';
  const S = State.summary;
  const versions = groupVersions();
  const champs = State.champs;

  // ---- hero cards
  const current = [...versions.values()].find((v) => champs?.byHash?.[v.version]?.current);
  const conv = S.convergence;
  const match = S.match;
  const cards = el('div', { class: 'grid cards' });

  if (current) {
    const bn = bestNode(current);
    cards.append(el('div', { class: 'card' },
      el('h3', {}, 'Current champion'),
      el('div', { class: 'stat' }, current.name, el('span', { class: 'unit' }, `gen ${current.gen ?? '?'}`)),
      el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Peak Elo'), el('span', { class: 'v' }, `${fmt(bn?.elo)}  (d${bn?.depth})`)),
      el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Architecture'), el('span', { class: 'v mono arch' }, archStr(current.arch))),
      el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Hash'), el('span', { class: 'v mono' }, current.version)),
    ));
  }
  cards.append(el('div', { class: 'card' },
    el('h3', {}, 'Convergence'),
    convBadge(conv),
    el('div', { class: 'kv mt' }, el('span', { class: 'k' }, 'Pairs'), el('span', { class: 'v' }, fmt(conv?.pairs))),
    el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Never met'), el('span', { class: 'v' }, fmt(conv?.adjacentUnlinked))),
    el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Worst mis-order'), el('span', { class: 'v' }, `${fmt(conv?.misorderCost, 0)} Elo`)),
    el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Depth inversions'), el('span', { class: 'v' }, fmt(conv?.confidentInversions))),
    el('div', { class: 'pill-note mt', text: conv?.verdict || '' }),
  ));
  if (match) {
    const edge = match.elo;
    cards.append(el('div', { class: 'card' },
      el('h3', {}, 'Latest gate match'),
      el('div', { class: 'stat small' }, `${fmt(match.wins)}–${fmt(match.draws)}–${fmt(match.losses)}`, el('span', { class: 'unit' }, `${fmt(match.games)} games`)),
      el('div', { class: 'kv mt' }, el('span', { class: 'k' }, 'Score'), el('span', { class: 'v' }, pct(match.score))),
      el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Edge'), el('span', { class: 'v' }, `${edge >= 0 ? '+' : ''}${fmt(edge, 1)} Elo`)),
      el('div', { class: 'kv' }, el('span', { class: 'k' }, 'SPRT'), el('span', { class: 'v' }, el('span', { class: `badge ${match.sprt === 'accept' ? 'good' : match.sprt === 'reject' ? 'crit' : 'neutral'}` }, match.sprt || '–'))),
      match.div ? el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Eval corr'), el('span', { class: 'v' }, fmt(match.div.corr, 3))) : null,
    ));
  }
  cards.append(el('div', { class: 'card' },
    el('h3', {}, 'Pool'),
    el('div', { class: 'stat small' }, fmt(State.ladder.ranking?.length), el('span', { class: 'unit' }, 'nodes')),
    el('div', { class: 'kv mt' }, el('span', { class: 'k' }, 'Champions'), el('span', { class: 'v' }, fmt([...versions.values()].filter((v) => v.gen != null).length))),
    el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Ladder games'), el('span', { class: 'v', id: 'statLadderGames' }, fmt(S.counts?.games))),
    el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Dataset positions'), el('span', { class: 'v' }, fmt(S.meta?.dataset?.totalLines))),
    el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Ledger updated'), el('span', { class: 'v' }, S.meta?.generated ? new Date(S.meta.generated).toLocaleString() : '–')),
  ));
  root.append(cards);

  // ---- toolbar
  const tb = el('div', { class: 'toolbar mt' });
  tb.append(el('label', {}, 'Depth', depthSelect()));
  const anchorTag = State.ladder.anchor || 'the anchor';
  tb.append(el('span', { class: 'sub' }, State.depth === 'all'
    ? `Each engine at its strongest depth. ${depthNote}`
    : `All engines searching to depth ${State.depth}. Anchor ${anchorTag} ≡ 1500. ${depthNote}`));
  root.append(tb);

  // ---- leaderboard rows
  const rows = [...versions.values()]
    .map((v) => ({ v, r: State.depth === 'all' ? bestNode(v) : v.byDepth.get(State.depth) }))
    .filter((x) => x.r);
  rows.sort((a, b) => b.r.elo - a.r.elo);
  if (!rows.length) {
    root.append(el('div', { class: 'empty-state' }, `No rated node searches to depth ${State.depth}. Pick another depth.`));
    return;
  }
  const maxElo = Math.max(...rows.map((x) => x.r.elo));
  const minElo = Math.min(...rows.map((x) => x.r.elo));

  const table = el('table', { class: 'data mt' });
  table.append(el('thead', {}, el('tr', {},
    el('th', {}, '#'), el('th', { class: 'name' }, 'Engine'), el('th', {}, 'Gen'),
    el('th', { class: 'name' }, 'Arch'), el('th', {}, 'Depth'), el('th', {}, 'Elo'),
    el('th', {}, '±95'), el('th', {}, 'Games'), el('th', {}, 'Labels'))));
  const tbody = el('tbody');
  rows.forEach((x, i) => {
    const { v, r } = x;
    const col = familyColor(v.eng, v.version);
    const barW = 4 + 90 * ((r.elo - minElo) / ((maxElo - minElo) || 1));
    const tr = el('tr', { class: v.gen != null && champs?.byHash?.[v.version]?.current ? 'row-click' : 'row-click' });
    tr.append(
      el('td', { class: 'num' }, String(i + 1)),
      el('td', { class: 'name' }, el('span', { class: 'chip' }, el('span', { class: 'swatch', style: `background:${col}` }), v.name)),
      el('td', { class: 'num' }, v.gen != null ? String(v.gen) : '–'),
      el('td', { class: 'name mono', style: { color: 'var(--muted)', fontSize: '11px' } }, archStr(v.arch)),
      el('td', { class: 'num' }, String(r.depth)),
      el('td', { class: 'num' }, el('span', {}, fmt(r.elo), el('span', { class: 'elo-bar', style: { width: `${barW}px`, background: col } }))),
      el('td', { class: 'num', style: { color: 'var(--muted)' } }, `±${fmt(r.margin)}`),
      el('td', { class: 'num' }, fmt(r.games)),
      el('td', { class: 'num', style: { color: 'var(--muted)' } }, fmt(r.records)),
    );
    tr.addEventListener('click', () => { openDepthFor(v.key); });
    tbody.append(tr);
  });
  table.append(tbody);
  root.append(table);
}
function archStr(arch) { return arch ? `[${arch.join(',')}]` : '–'; }
function convBadge(c) {
  if (!c) return el('span', { class: 'badge neutral' }, 'no data');
  if (c.converged) return el('span', { class: 'badge good' }, '✓ converged');
  if (c.linked === false) return el('span', { class: 'badge warn' }, 'unlinked pairs');
  if (c.resolved) return el('span', { class: 'badge warn' }, '⚠ inversions');
  return el('span', { class: 'badge warn' }, 'not converged');
}

// ============================================================ GENERATIONS view
function renderGenerations() {
  const root = $('#view-generations'); root.innerHTML = '';
  const versions = [...groupVersions().values()].filter((v) => v.gen != null).sort((a, b) => a.gen - b.gen);
  const atDepth = (v) => (State.depth === 'all' ? bestNode(v) : v.byDepth.get(State.depth));

  const tb = el('div', { class: 'toolbar' });
  tb.append(el('label', {}, 'At depth', depthSelect()));
  tb.append(el('span', { class: 'sub' }, `Champion strength across train:loop generations — the climb. ${State.depth === 'all' ? 'Each champion at its strongest depth, so the curve mixes depths. ' : ''}${depthNote}`));
  root.append(el('h2', { class: 'section' }, 'Generational progress ', el('span', { class: 'sub' }, '· Elo of each loop champion, banded by its ±95 against the pin')));
  root.append(tb);

  // lo/hi are the ladder's ±95 against the hc6 pin. It's the interval the fit publishes per node, so
  // it answers "how well do we know this champion's place on the scale" — not "is gen N+1 above gen
  // N", which needs the pairwise contrast variance the ladder doesn't persist (depth-ladder.mjs:794
  // on why subtracting two vs-pin margins overstates it). Two overlapping bands are not a verdict.
  const pts = versions.map((v) => {
    const r = atDepth(v);
    if (!r) return null;
    // A row with no margin drops out of the band rather than poisoning the y-domain with NaN.
    const ci = Number.isFinite(r.margin) ? r.margin : null;
    return { x: v.gen, y: r.elo, lo: ci == null ? null : r.elo - ci, hi: ci == null ? null : r.elo + ci, meta: v, name: v.name };
  }).filter(Boolean);
  if (!pts.length) {
    root.append(el('div', { class: 'empty-state' }, `No champion is rated at depth ${State.depth}. Pick a depth this run covers.`));
    return;
  }
  // hc reference at the same depth
  const hc = [...groupVersions().values()].find((v) => v.eng === 'hc' && v.version === '2');
  const hcNode = hc ? atDepth(hc) : null;
  const refs = hcNode ? [{ y: hcNode.elo, label: `handcrafted v2 (d${hcNode.depth})`, color: css('--hc') }] : [];

  const card = el('div', { class: 'card pad0' });
  const wrap = el('div', { class: 'chart-wrap', style: { padding: '10px' } });
  wrap.append(lineChart({
    height: 420,
    series: [{ key: 'gen', label: 'champion Elo', color: css('--s1'), points: pts, band: true }],
    xDomain: [Math.min(...pts.map((p) => p.x)) - 0.4, Math.max(...pts.map((p) => p.x)) + 0.4],
    xTicks: versions.map((v) => v.gen),
    xLabel: 'generation', yLabel: 'Elo', refs,
    formatX: (g) => { const v = versions.find((z) => z.gen === g); return v ? `${g} ${v.name}` : fmt(g); },
    formatY: (v) => fmt(v),
  }));
  card.append(wrap);
  root.append(card);

  // delta bars (Elo gained per generation)
  const deltas = [];
  for (let i = 1; i < pts.length; i++) deltas.push({ label: pts[i].name, value: pts[i].y - pts[i - 1].y, color: (pts[i].y - pts[i - 1].y) >= 0 ? css('--s3') : css('--s2') });
  root.append(el('h2', { class: 'section mt' }, 'Elo gained per generation ', el('span', { class: 'sub' }, State.depth === 'all' ? '· at each champion\'s best depth' : `· at depth ${State.depth}`)));
  const card2 = el('div', { class: 'card pad0' });
  const wrap2 = el('div', { class: 'chart-wrap', style: { padding: '10px' } });
  wrap2.append(barChart({ height: 300, data: deltas, valueLabel: 'Δ Elo', formatY: (v) => `${v >= 0 ? '+' : ''}${fmt(v, 0)}` }));
  card2.append(wrap2);
  root.append(card2);
}

// ============================================================== DEPTH×ELO view
function openDepthFor(key) {
  State.depthSel = new Set([key]);
  const hc = [...groupVersions().values()].find((v) => v.eng === 'hc' && v.version === '2');
  if (hc) State.depthSel.add(hc.key);
  switchView('depth');
}
function renderDepth() {
  const root = $('#view-depth'); root.innerHTML = '';
  const versions = groupVersions();
  const all = [...versions.values()];
  // default selection: current champion + prev champion + handcrafted v2
  if (!State.depthSel.size) {
    const champs = all.filter((v) => v.gen != null).sort((a, b) => b.gen - a.gen);
    for (const v of champs.slice(0, 3)) State.depthSel.add(v.key);
    const hc = all.find((v) => v.eng === 'hc' && v.version === '2'); if (hc) State.depthSel.add(hc.key);
  }
  root.append(el('h2', { class: 'section' }, 'Search depth × Elo ', el('span', { class: 'sub' }, '· how much strength each extra ply buys')));
  // Depth is the x-axis here, so there is nothing to filter — but the dashboard's shared depth is
  // marked, so you can see where the other views are reading these curves.
  const marker = State.depth === 'all' ? [] : [{ x: Number(State.depth), label: `depth ${State.depth}`, color: css('--s4') }];
  if (marker.length) root.append(el('div', { class: 'sub', style: { marginBottom: '8px' } }, `Marker at depth ${State.depth} — the depth the other views are showing.`));

  const selected = [...State.depthSel].map((k) => versions.get(k)).filter(Boolean);
  selected.sort((a, b) => (b.gen ?? -1) - (a.gen ?? -1));
  const series = selected.map((v, i) => ({
    key: v.key, label: v.name, color: CAT[i % CAT.length],
    points: [...v.byDepth.values()].map((r) => ({ x: Number(r.depth), y: r.elo })).sort((a, b) => a.x - b.x),
  }));

  // x-axis spans the depths actually plotted — a run at --depths=1-3 shouldn't leave half the
  // chart empty, and a node rated only at the anchor depth should show its gap honestly.
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const xLo = xs.length ? Math.min(...xs) : 1;
  const xHi = Math.max(xLo + 1, xs.length ? Math.max(...xs) : 8);
  const xTicks = [];
  for (let d = Math.floor(xLo); d <= Math.ceil(xHi); d++) xTicks.push(d);

  const card = el('div', { class: 'card pad0' });
  const wrap = el('div', { class: 'chart-wrap', style: { padding: '10px' } });
  if (series.length) {
    wrap.append(lineChart({
      height: 440, series, directLabels: true, vrefs: marker,
      xDomain: [xLo - 0.3, xHi + 0.3], xTicks,
      xLabel: 'search depth (plies)', yLabel: 'Elo',
      formatX: (d) => `depth ${fmt(d)}`, formatY: (v) => fmt(v),
      margin: { r: 120 },
    }));
  } else wrap.append(el('div', { class: 'empty-state' }, 'Select engines below to plot their depth curves.'));
  card.append(wrap);
  root.append(card);

  // slope readout — Elo per ply between deepest two levels
  if (series.length) {
    const info = el('div', { class: 'flex wrap mt', style: { gap: '18px' } });
    for (const s of series) {
      const p = s.points; if (p.length < 2) continue;
      const slope = (p[p.length - 1].y - p[0].y) / (p[p.length - 1].x - p[0].x);
      info.append(el('span', { class: 'chip' }, el('span', { class: 'swatch', style: `background:${s.color}` }),
        el('span', { class: 'sub' }, `${s.label}: `), el('b', {}, `${fmt(slope, 0)} Elo/ply`)));
    }
    root.append(info);
  }

  // engine picker (toggle chips)
  root.append(el('hr', { class: 'sep' }));
  root.append(el('div', { class: 'sub', style: { marginBottom: '8px' } }, 'Toggle engines (newest first):'));
  const picks = all.filter((v) => v.byDepth.size).sort((a, b) => (b.gen ?? -1) - (a.gen ?? -1) || (a.eng < b.eng ? 1 : -1));
  const colorByKey = new Map(series.map((s) => [s.key, s.color]));
  const items = picks.map((v) => ({ key: v.key, label: `${v.name}${v.gen != null ? ` · g${v.gen}` : ''}`, off: !State.depthSel.has(v.key), color: colorByKey.get(v.key) || css('--muted') }));
  root.append(legend(items, (it) => {
    if (State.depthSel.has(it.key)) State.depthSel.delete(it.key); else State.depthSel.add(it.key);
    renderDepth();
    syncUrl();
  }));
}

// =============================================================== MATCHUPS view
async function renderMatchups() {
  const root = $('#view-matchups'); root.innerHTML = '';
  root.append(el('h2', { class: 'section' }, 'Head-to-head ', el('span', { class: 'sub' }, '· every game these engines have played')));
  if (!State.pool) State.pool = await api('/api/pool');
  const pairs = State.pool.pairs || {};
  const eloOf = new Map(); const labelOf = new Map(); const parsedOf = new Map();
  for (const r of State.ladder.ranking) { eloOf.set(r.tag, r.elo); labelOf.set(r.tag, `${r.name || nodeLabel(r.eng, r.version, State.champs)} d${r.depth}`); parsedOf.set(r.tag, r); }

  // tags present in pool (zero-game entries don't count — see pairScore)
  const present = new Set();
  for (const [k, v] of Object.entries(pairs)) {
    if (!(v.games > 0)) continue;
    const [a, b] = k.split('|'); present.add(a); present.add(b);
  }

  const tb = el('div', { class: 'toolbar' });
  tb.append(el('label', {}, 'Depth', depthSelect()));
  tb.append(el('label', {}, 'Family', el('select', { 'data-ctl': 'fam', onchange: (e) => { State.mFam = e.target.value; keepFocus(renderMatchups); syncUrl(); } },
    ...['all', 'nn', 'hc'].map((f) => el('option', { value: f, selected: f === State.mFam ? '' : null }, f === 'all' ? 'all engines' : f)))));
  const CROSS = { none: 'hide', unlinked: 'for blank rows', all: 'all opponents' };
  tb.append(el('label', {}, 'Cross-depth', el('select', { 'data-ctl': 'cross', onchange: (e) => { State.mCross = e.target.value; keepFocus(renderMatchups); syncUrl(); } },
    ...Object.entries(CROSS).map(([v, lbl]) => el('option', { value: v, selected: v === State.mCross ? '' : null }, lbl)))));

  // Rows are the nodes at the picked depth. Columns are those plus every off-depth opponent a row
  // actually played, because a node's first games are cross-depth by construction: the ladder
  // calibrates a new net against itself at depth 6, and the direct-link floor pairs it with a rated
  // champion. Filtering both axes to one depth hid exactly the evidence a new engine has — Tara's
  // depth-1 row was blank the day she arrived while carrying 128 games.
  const byElo = (a, b) => (eloOf.get(a) ?? 0) - (eloOf.get(b) ?? 0);
  const inFam = (p) => State.mFam === 'all' || p.eng === State.mFam;
  const rows = [...present].filter((t) => {
    const p = parsedOf.get(t); if (!p) return false;
    if (State.depth !== 'all' && Number(p.depth) !== Number(State.depth)) return false;
    return inFam(p);
  }).sort(byElo);
  const rowSet = new Set(rows);
  // Every node keeps a calibration link to itself at depth 6 and to the handcrafted anchors, so
  // pulling in every off-depth opponent quadruples the columns for a very sparse block — 23 rows
  // become 115 columns at depth 6. Default to rescuing only the rows the depth filter left with no
  // opponent at all, which is the case that reads as missing data; 'all opponents' is the full view.
  const linked = new Set();
  for (const [k, v] of Object.entries(pairs)) {
    if (!(v.games > 0)) continue;
    const [a, b] = k.split('|');
    if (rowSet.has(a) && rowSet.has(b)) { linked.add(a); linked.add(b); }
  }
  const wantsCross = (t) => (State.mCross === 'all' ? true : State.mCross === 'unlinked' ? !linked.has(t) : false);
  const off = new Set();
  for (const [k, v] of Object.entries(pairs)) {
    if (!(v.games > 0)) continue;
    const [a, b] = k.split('|');
    for (const [x, y] of [[a, b], [b, a]]) {
      if (!rowSet.has(x) || rowSet.has(y) || off.has(y) || !wantsCross(x)) continue;
      const p = parsedOf.get(y);
      if (p && inFam(p)) off.add(y);
    }
  }
  // At 'all depths' every opponent that passes the family filter is already a row, so this is empty
  // and the grid stays square without a special case.
  const offCols = [...off].sort(byElo);
  const cols = [...rows, ...offCols];
  tb.append(el('span', { class: 'sub' }, `${rows.length} nodes · cell = row's score vs column · click a played cell for the games`));
  if (offCols.length) tb.append(el('span', { class: 'sub' }, `· ${offCols.length} cross-depth opponent${offCols.length === 1 ? '' : 's'} past the divider`));
  else {
    // With cross-depth hidden, a node whose every game was played against another depth reads as a
    // node with no games at all. Say which, rather than leaving a blank row to be read as a bug.
    const blank = rows.filter((t) => !linked.has(t));
    if (blank.length) tb.append(el('span', { class: 'badge warn' }, `${blank.map((t) => shortLabel(t, parsedOf)).join(', ')} played only at other depths`));
  }
  // The corpus half of a cell is only as current as the last rank:pool scan (it caches on the
  // dataset's size+mtime), so a cell can trail the dataset until the next run.
  if (State.pool.stale) tb.append(el('span', { class: 'badge warn' }, 'corpus scan behind dataset'));
  root.append(tb);

  if (!rows.length || cols.length < 2) { root.append(el('div', { class: 'empty-state' }, 'Not enough directly-played nodes at this filter. Try “all depths”.')); return; }

  // The first off-depth column carries the divider, so the same-depth block stays readable as a block.
  const colExtra = (i) => (i < rows.length ? '' : i === rows.length ? ' off sep' : ' off');
  const scroll = el('div', { class: 'heat-scroll' });
  const table = el('table', { class: 'heat' });
  const head = el('tr', {}, el('th', { class: 'corner' }, ''));
  cols.forEach((c, i) => head.append(el('th', { class: `collab${colExtra(i)}` }, el('div', {}, shortLabel(c, parsedOf)))));
  table.append(head);
  for (const rTag of rows) {
    const tr = el('tr', {}, el('th', { class: 'rowlab' }, shortLabel(rTag, parsedOf)));
    for (let i = 0; i < cols.length; i++) {
      const cTag = cols[i];
      if (rTag === cTag) { tr.append(el('td', { class: `cell diag${colExtra(i)}` }, '')); continue; }
      const info = pairScore(pairs, rTag, cTag);
      if (!info) { tr.append(el('td', { class: `cell empty${colExtra(i)}` }, '')); continue; }
      const bg = divergingColor(info.score);
      const td = el('td', { class: `cell${colExtra(i)}`, style: `background:${bg}` }, Math.round(info.score * 100));
      td.addEventListener('mousemove', (ev) => showTip(
        `<div class="tt-title">${labelOf.get(rTag)} vs ${labelOf.get(cTag)}</div>`
        + `<div class="tt-row"><span class="k">Score</span><span class="v">${pct(info.score)}</span></div>`
        + `<div class="tt-row"><span class="k">Games</span><span class="v">${info.games}</span></div>`
        // Where the evidence came from. A pair the ranker never scheduled can still be the
        // best-measured on the board: a promotion gate is thousands of games at depth 6.
        + (info.store < info.games
          ? `<div class="tt-row"><span class="k">of which gate/self-play</span><span class="v">${info.games - info.store}</span></div>` : ''), ev.clientX, ev.clientY));
      td.addEventListener('mouseleave', hideTip);
      td.addEventListener('click', () => openGamesFor(rTag, cTag));
      tr.append(td);
    }
    table.append(tr);
  }
  scroll.append(table);
  root.append(scroll);

  // legend for diverging scale
  const lg = el('div', { class: 'flex mt', style: { gap: '8px' } });
  lg.append(el('span', { class: 'sub' }, 'row loses'));
  for (let s = 0; s <= 1.0001; s += 0.1) lg.append(el('span', { style: { width: '26px', height: '14px', borderRadius: '3px', background: divergingColor(s), display: 'inline-block' } }));
  lg.append(el('span', { class: 'sub' }, 'row wins'));
  root.append(lg);
}
function shortLabel(tag, parsedOf) { const p = parsedOf.get(tag); if (!p) return tag; const nm = (p.name || nodeLabel(p.eng, p.version, State.champs)); return `${nm.length > 10 ? nm.slice(0, 10) : nm} d${p.depth}`; }
// A pool entry can exist with zero games: a matchup the orchestrator stopped before any game
// finished still gets recorded, as {games:0,sumA:0}. That's a pair the two nodes never actually
// played, so it reads as never-met (empty cell) rather than 0/0 = NaN.
function pairScore(pairs, a, b) {
  const played = (e) => (e && e.games > 0 ? e : null);
  const ab = played(pairs[`${a}|${b}`]); if (ab) return { score: ab.sumA / ab.games, games: ab.games, store: ab.store ?? ab.games };
  const ba = played(pairs[`${b}|${a}`]); if (ba) return { score: 1 - ba.sumA / ba.games, games: ba.games, store: ba.store ?? ba.games };
  return null;
}
function divergingColor(score) {
  // 0 → neg pole, .5 → neutral, 1 → pos pole
  const neg = [217, 89, 38], mid = [56, 56, 53], pos = [57, 135, 229];
  const t = Math.max(0, Math.min(1, score));
  const lerp = (a, b, u) => a.map((c, i) => Math.round(c + (b[i] - c) * u));
  const rgb = t < 0.5 ? lerp(neg, mid, t / 0.5) : lerp(mid, pos, (t - 0.5) / 0.5);
  return `rgb(${rgb.join(',')})`;
}

// ================================================================= GAMES view
function openGamesFor(a, b) { State.gamesFilter = { a, b }; switchView('games'); }
async function renderGames() {
  const root = $('#view-games'); root.innerHTML = ''; activeViewer = null;
  const versions = groupVersions();
  const allTags = State.ladder.ranking.map((r) => r.tag);

  const tb = el('div', { class: 'toolbar' });
  const f = State.gamesFilter || {};
  const mkSel = (which, val) => el('select', { 'data-ctl': `games-${which}`, onchange: (e) => { State.gamesFilter = { ...State.gamesFilter, [which]: e.target.value || null }; syncUrl(); loadGameList(); } },
    el('option', { value: '' }, which === 'a' ? 'any engine' : 'any opponent'),
    ...allTags.map((t) => el('option', { value: t, selected: t === val ? '' : null }, tagLabel(t))));
  tb.append(el('label', {}, 'Player', mkSel('a', f.a)));
  tb.append(el('label', {}, 'vs', mkSel('b', f.b)));
  tb.append(el('span', { class: 'sub', id: 'gamesCount' }, ''));
  root.append(tb);

  const layout = el('div', { class: 'viewer' });
  const left = el('div', { id: 'gamesListWrap' }, el('div', { class: 'card pad0' }, el('div', { class: 'games-list', id: 'gamesList' })));
  const right = el('div', { id: 'gameViewerWrap' }, el('div', { class: 'empty-state' }, 'Select a game to replay it.'));
  layout.append(left, right);
  root.append(layout);
  loadGameList();
}
function tagLabel(t) { const p = parseTag(t); return `${nodeLabel(p.eng, p.version, State.champs)} d${p.depth}`; }
async function loadGameList() {
  const f = State.gamesFilter || {};
  const qs = new URLSearchParams();
  if (f.a && f.b) { qs.set('a', f.a); qs.set('b', f.b); } else if (f.a) qs.set('player', f.a); else if (f.b) qs.set('player', f.b);
  qs.set('limit', '120');
  const data = await api(`/api/games?${qs}`);
  const list = $('#gamesList'); if (!list) return;
  list.innerHTML = '';
  const cnt = $('#gamesCount'); if (cnt) cnt.textContent = `${fmt(data.total)} games${(f.a || f.b) ? ' match this filter' : ' total'} · showing ${data.games.length}`;
  if (!data.games.length) { list.append(el('div', { class: 'empty-state' }, 'No games.')); stopPlayback(); activeViewer = null; State.openGame = null; return; }
  // A live run rebuilds this list under the user, so keep the open game and the ply it was on
  // when it is still in the page. Only fall back to the newest game when it isn't.
  const keep = State.openGame && data.games.some((g) => g.g === State.openGame.g) ? State.openGame : null;
  // A game named by the URL is honoured once even when it falls outside this page of the list —
  // that's a link someone pasted or a refresh, and the game it names is the point of the link.
  const want = State.pendingGame; State.pendingGame = null;
  const openId = want ? want.g : keep ? keep.g : data.games[0].g;
  const openPly = want ? want.ply : keep ? keep.ply : null;
  const resume = !want && keep ? isPlaying() : false; // a rebuild mid-replay shouldn't stop the replay
  for (const g of data.games) {
    const resChar = g.r === 1 ? 'W' : g.r === -1 ? 'B' : '½';
    const resClass = g.r === 1 ? 'w' : g.r === -1 ? 'l' : 'd';
    const row = el('div', { class: 'game-row' },
      el('div', { class: `res ${resClass}` }, resChar),
      el('div', {}, el('div', {}, `${tagLabel(g.w)}  vs  ${tagLabel(g.b)}`), el('div', { class: 'sub mono' }, g.g)),
      el('div', { class: 'sub num' }, `${g.n} plies`));
    if (g.g === openId) row.style.background = 'var(--surface-2)';
    row.addEventListener('click', () => { $$('.game-row', list).forEach((r) => r.style.background = ''); row.style.background = 'var(--surface-2)'; openGame(g.g); });
    list.append(row);
  }
  openGame(openId, openPly, resume);
}

// ---- board + eval game viewer (blue2 board + Merida pieces, matching the app)
// The viewer writes the URL from async callbacks (a fetch that lands late, a replay timer), so
// every one of them checks it still owns the page — otherwise leaving mid-load stamps a games
// URL over the view you switched to.
const syncGamesUrl = () => { if (currentView === 'games') syncUrl(); };
let viewerSeq = 0; // bumped per open; a slower fetch checks it before touching the DOM
const pieceUrl = (color, role) => `/assets/pieces/merida/${color === 'white' ? 'w' : 'b'}${role.toUpperCase()}.svg`;

// A recorded move is UCI text; the engine wants the move object it generated, which also
// carries the castle and promotion flags applyMove needs. Look it up among the legal moves
// rather than reconstructing it, so a record that disagrees with the rules is caught here
// instead of quietly producing a board that never existed.
function moveFromUci(state, uci) {
  const from = parseSquare(uci.slice(0, 2));
  const to = parseSquare(uci.slice(2, 4));
  const promotion = uci[4] ? uci[4].toLowerCase() : null;
  return legalMoves(state).find((m) => m.from === from && m.to === to && (m.promotion || null) === promotion) || null;
}

// Replay the game once when it opens, into one entry per ply. Each carries what the board,
// the trays and the sounds need: the squares the move touched, whether it captured, and
// whether it left the side to move in check. Precomputed rather than re-derived per ply —
// a running replay redraws twice a second, and the check flag costs a move generation.
//
// A ply the rules reject ends the replay, and the viewer says where it stopped. These
// records are written by the engines themselves, so that would be a bug worth seeing.
function replay(game) {
  let state = parseFen(game.start || START_FEN);
  const plies = [{ state, last: null, capture: false, check: false }];
  for (const uci of game.moves) {
    const m = moveFromUci(state, uci);
    if (!m) break;
    const capture = !!state.board[m.to];
    state = applyMove(state, { ...m, capture });
    plies.push({ state, last: { from: m.from, to: m.to }, capture, check: kingAttacked(state.board, state.turn) });
  }
  return plies;
}

// --- material difference (the app's trays, same rules) -----------------------
// Read off the BOARD rather than from capture history: per role, this side's surplus of
// pieces still standing. Equal trades cancel for free and a promotion reads correctly.
// The point values are the app's, which are the familiar chess ones rather than the
// engine's own (in this variant a knight is worth about a rook).
const POINTS = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
function materialDiff(board, color) {
  const counts = { white: {}, black: {} };
  let pts = 0;
  for (const p of board) {
    if (!p) continue;
    counts[p.color][p.role] = (counts[p.color][p.role] || 0) + 1;
    pts += (p.color === 'white' ? 1 : -1) * POINTS[p.role];
  }
  const mine = counts[color], theirs = counts[opponent(color)];
  const surplus = [];
  for (const role of ['q', 'r', 'b', 'n', 'p']) { // descending value
    for (let i = (mine[role] || 0) - (theirs[role] || 0); i > 0; i--) surplus.push(role);
  }
  return { surplus, adv: color === 'white' ? pts : -pts };
}
// One tray, built once per open and refilled per ply. Mono (single-colour) glyphs like the
// app's — whose pieces they were is already said by the tray they sit in.
function makeTray(who) {
  const caps = el('div', { class: 'caps' });
  const adv = el('span', { class: 'adv' });
  const box = el('div', { class: 'tray' }, caps, adv, el('span', { class: 'who' }, who));
  box.fill = (board, color) => {
    const { surplus, adv: n } = materialDiff(board, color);
    caps.innerHTML = surplus.map((role) => `<span class="cap cap-${role}"></span>`).join('');
    adv.textContent = n > 0 ? `+${n}` : '';
  };
  return box;
}

// --- sounds ------------------------------------------------------------------
// The app's three, with the app's priority: check beats capture (and covers checkmate,
// which is also a check). One reusable element each, rewound before playing, so a fast
// replay still clicks. Muting persists — this page sits open for hours next to a loop.
const SOUNDS = {
  move: new Audio('/sound/standard/Move.mp3'),
  capture: new Audio('/sound/standard/Capture.mp3'),
  check: new Audio('/sound/standard/Check.mp3'),
};
let muted = localStorage.getItem('viz-muted') === '1';
function playMoveSound(capture, check) {
  if (muted) return;
  const a = check ? SOUNDS.check : capture ? SOUNDS.capture : SOUNDS.move;
  a.currentTime = 0;
  a.play().catch(() => {}); // ignore autoplay blocks before the first click on the page
}
function whitePovEval(game) {
  const startWhite = !(game.start && game.start.split(' ')[1] === 'b');
  return (game.v || []).map((v, i) => {
    const sideWhite = startWhite ? i % 2 === 0 : i % 2 === 1;
    const val = sideWhite ? v : -v;
    return { i, mate: Math.abs(v) > 50000, v: val };
  });
}
async function openGame(g, startPly = null, autoplay = false) {
  const wrap = $('#gameViewerWrap'); if (!wrap) return;
  stopPlayback();
  const seq = ++viewerSeq;
  wrap.innerHTML = '<div class="empty-state">Loading…</div>';
  const game = await api(`/api/game?g=${encodeURIComponent(g)}`);
  if (seq !== viewerSeq) return; // a newer pick won the race — don't overwrite its viewer
  if (game.error) { wrap.innerHTML = '<div class="empty-state">Game not found.</div>'; return; }
  // Everything below indexes plies, so `lastPly` — not the record's move count — is the end of
  // the board: they differ only when the replay hit a move the rules reject.
  const plies = replay(game);
  const lastPly = plies.length - 1;
  const state = { ply: startPly == null ? lastPly : Math.max(0, Math.min(lastPly, startPly)) };
  State.openGame = { g, ply: state.ply };
  syncGamesUrl();
  wrap.innerHTML = '';

  const info = el('div', { class: 'flex wrap', style: { justifyContent: 'space-between', marginBottom: '10px' } },
    el('div', {}, el('b', {}, tagLabel(game.players.w)), el('span', { class: 'sub' }, '  (white)  vs  '), el('b', {}, tagLabel(game.players.b)), el('span', { class: 'sub' }, '  (black)')),
    el('span', { class: `badge ${game.r === 1 ? 'good' : game.r === -1 ? 'crit' : 'neutral'}` }, game.r === 1 ? 'white won' : game.r === -1 ? 'black won' : 'draw'));

  const boardEl = el('div', { class: 'board' });
  // Material trays flank the board the way the app's do: bottom is White, since the viewer
  // always draws from White's side. Each names its engine, so the board says who is who.
  const trayTop = makeTray(tagLabel(game.players.b));
  const trayBottom = makeTray(tagLabel(game.players.w));
  const transport = el('div', { class: 'transport' });
  const readout = el('div', { class: 'ply-readout' });
  const range = el('input', { type: 'range', min: '0', max: String(lastPly), value: String(state.ply) });
  const btn = (t, on) => el('button', { onclick: on }, t);
  const stop = stopPlayback;
  // The URL carries the ply, but not while a replay is running: that would be a history write
  // twice a second, and the ply worth remembering is the one you stopped on.
  // Forward moves are sounded, like the app's move tree: it sounds the ply you land on, and
  // stepping back is silent. Dragging the slider is silent too — a scrub across a game would
  // be a hundred sounds cutting each other off.
  const setPly = (p, { silent = false } = {}) => {
    const next = Math.max(0, Math.min(lastPly, p));
    const forward = next > state.ply;
    state.ply = next;
    State.openGame = { g, ply: state.ply };
    range.value = String(state.ply);
    draw();
    if (forward && !silent) playMoveSound(plies[next].capture, plies[next].check);
    if (!isPlaying()) syncGamesUrl();
  };
  const play = () => {
    playBtn.textContent = '❚❚';
    playback = {
      timer: setInterval(() => {
        if (state.ply >= lastPly) setPly(0);
        setPly(state.ply + 1);
        if (state.ply >= lastPly) stopPlayback();
      }, 550),
      onStop: () => { playBtn.textContent = '▶'; syncGamesUrl(); },
    };
  };
  const playBtn = btn('▶', () => { if (isPlaying()) stopPlayback(); else play(); });
  transport.append(
    btn('⏮', () => { stop(); setPly(0); }),
    btn('‹', () => { stop(); setPly(state.ply - 1); }),
    playBtn,
    btn('›', () => { stop(); setPly(state.ply + 1); }),
    btn('⏭', () => { stop(); setPly(lastPly); }),
    range, readout);
  range.addEventListener('input', () => { stop(); setPly(Number(range.value), { silent: true }); });

  // expose to the global hotkey handler (← → step, Home/End jump, Space play)
  activeViewer = {
    prev: () => { stop(); setPly(state.ply - 1); },
    next: () => { stop(); setPly(state.ply + 1); },
    first: () => { stop(); setPly(0); },
    last: () => { stop(); setPly(lastPly); },
    toggle: () => playBtn.click(),
  };

  const hint = el('div', { class: 'pill-note', style: { marginTop: '6px' } }, '← → step · ↑↓ / Home End jump · Space play');
  const left = el('div', {}, trayTop, boardEl, trayBottom, transport, hint);
  // Only when the replay stopped early — the record and the rules disagree from here on.
  if (lastPly < game.moves.length) {
    left.append(el('div', { class: 'pill-note', style: { color: 'var(--warning)' } },
      `Replay stops at ply ${lastPly}: ${game.moves[lastPly]} is not legal here. The record has ${game.moves.length}.`));
  }
  const movelist = el('div', { class: 'movelist' });
  // Held by reference, not looked up by id: a stale viewer must never be able to find this one.
  const evalBox = el('div', { class: 'chart-wrap', style: { padding: '8px' } });
  const evalWrap = el('div', { class: 'card pad0 mt' }, evalBox);
  const right = el('div', {}, info, evalWrap, movelist);

  function draw() {
    const pos = plies[state.ply];
    const board = pos.state.board, last = pos.last;
    boardEl.innerHTML = '';
    for (let r = 7; r >= 0; r--) for (let f = 0; f < 8; f++) {
      const idx = r * 8 + f;
      const sq = el('div', { class: `sq${last && (idx === last.from || idx === last.to) ? ' hl' : ''}` });
      const p = board[idx];
      if (p) sq.append(el('div', { class: 'piece', style: `background-image:url('${pieceUrl(p.color, p.role)}')` }));
      boardEl.append(sq);
    }
    trayTop.fill(board, 'black');
    trayBottom.fill(board, 'white');
    readout.textContent = `${state.ply} / ${lastPly}`;
    $$('.mv', movelist).forEach((m) => m.classList.toggle('on', Number(m.dataset.ply) === state.ply));
    drawEval();
  }
  const evalPts = whitePovEval(game);
  function drawEval() {
    const w = evalBox; w.innerHTML = '';
    const pts = evalPts.map((e) => ({ x: e.i, y: Math.max(-1000, Math.min(1000, e.v)), mate: e.mate }));
    const chart = lineChart({
      height: 220, margin: { l: 48, b: 30, t: 10 },
      series: [{ key: 'eval', label: 'eval (white)', color: css('--s1'), points: pts, dots: false }],
      xDomain: [0, game.moves.length], yDomain: [-1000, 1000],
      xLabel: 'ply', formatY: (v) => (v === 0 ? '0' : `${v > 0 ? '+' : ''}${(v / 100).toFixed(0)}`),
      refs: [{ y: 0, label: '', color: css('--axis') }, { y: evalPts[state.ply] ? Math.max(-1000, Math.min(1000, evalPts[state.ply].v)) : 0, label: '', color: 'transparent' }],
    });
    w.append(chart);
    // current-ply marker
    const svg = chart; const W = 940;
    const m = { l: 48, r: 18, t: 10, b: 30 }; const iw = W - m.l - m.r;
    const xx = m.l + (state.ply / (game.moves.length || 1)) * iw;
    svg.append(el('line', { class: 'crosshair', x1: xx, x2: xx, y1: m.t, y2: 220 - m.b, style: `stroke:${css('--s4')}` }));
  }
  // build movelist — only the plies that replayed, so every entry has a position to jump to
  for (let i = 0; i < lastPly; i++) {
    if (i % 2 === 0) movelist.append(el('span', { class: 'mvno' }, `${i / 2 + 1}.`));
    const mv = el('span', { class: 'mv', 'data-ply': String(i + 1) }, game.moves[i]);
    mv.addEventListener('click', () => { stop(); setPly(i + 1); });
    movelist.append(mv);
  }

  wrap.append(el('div', { class: 'viewer' }, left, right));
  draw();
  if (autoplay) play();
}

// =============================================================== TRAINING view
async function renderTraining() {
  const root = $('#view-training'); root.innerHTML = '';
  root.append(el('h2', { class: 'section' }, 'Training tracks ', el('span', { class: 'sub' }, '· experiment progress from train:loop')));
  const data = await api('/api/experiments');
  const tracks = (data.tracks || []).filter((t) => t.state).sort((a, b) => (b.state?.best?.absElo ?? 0) - (a.state?.best?.absElo ?? 0));
  if (!tracks.length) { root.append(el('div', { class: 'empty-state' }, 'No experiment tracks found.')); return; }

  const table = el('table', { class: 'data' });
  table.append(el('thead', {}, el('tr', {},
    el('th', { class: 'name' }, 'Track'), el('th', {}, 'Cycles'), el('th', {}, 'Runs'),
    el('th', {}, 'Promotions'), el('th', {}, 'Best absElo'), el('th', {}, 'Best score'), el('th', {}, 'SPRT'))));
  const tbody = el('tbody');
  for (const t of tracks) {
    const s = t.state; const best = s.best || {};
    const tr = el('tr', { class: 'row-click' },
      el('td', { class: 'name' }, el('div', {}, t.recipe?.slug || t.id), el('div', { class: 'sub mono' }, t.id)),
      el('td', { class: 'num' }, fmt(s.cycles)), el('td', { class: 'num' }, fmt(s.runs)),
      el('td', { class: 'num' }, fmt(s.promotions)),
      el('td', { class: 'num' }, fmt(best.absElo, 0)),
      el('td', { class: 'num' }, pct(best.score)),
      el('td', { class: 'num' }, el('span', { class: `badge ${best.sprt === 'accept' ? 'good' : 'neutral'}` }, best.sprt || '–')));
    tr.addEventListener('click', () => openTrack(t.id));
    tbody.append(tr);
  }
  table.append(tbody);
  root.append(table);
  root.append(el('div', { id: 'trackChart', class: 'mt' }));
  // The track from the URL wins, as long as it is still a track this loop knows.
  openTrack(tracks.some((t) => t.id === State.trackId) ? State.trackId : tracks[0].id);
}
async function openTrack(id) {
  const box = $('#trackChart'); if (!box) return;
  State.trackId = id;
  if (currentView === 'training') syncUrl();
  box.innerHTML = '<div class="empty-state">Loading…</div>';
  const data = await api(`/api/experiment?id=${encodeURIComponent(id)}`);
  const rows = (data.history || []).map((r, i) => ({ ...r, seq: i + 1 }));
  box.innerHTML = '';
  if (!rows.length) { box.append(el('div', { class: 'empty-state' }, 'No cycle history.')); return; }
  box.append(el('h2', { class: 'section' }, `Track ${id} `, el('span', { class: 'sub' }, `· ${rows.length} cycles`)));
  const absPts = rows.filter((r) => r.absElo != null).map((r) => ({ x: r.seq, y: r.absElo, meta: r }));
  const promoPts = rows.filter((r) => r.promoted && r.absElo != null).map((r) => ({ x: r.seq, y: r.absElo }));
  const card = el('div', { class: 'card pad0' }, el('div', { class: 'chart-wrap', style: { padding: '10px' } },
    lineChart({
      height: 300,
      series: [
        { key: 'abs', label: 'absElo', color: css('--s1'), points: absPts },
        promoPts.length ? { key: 'promo', label: 'promotion', color: css('--good'), points: promoPts, dots: true, dash: '0' } : null,
      ].filter(Boolean),
      xLabel: 'cycle', yLabel: 'absElo', formatX: (v) => fmt(v),
    })));
  box.append(card);
  const scorePts = rows.filter((r) => r.score != null).map((r) => ({ x: r.seq, y: r.score }));
  box.append(el('h2', { class: 'section mt' }, 'Gate score per cycle'));
  box.append(el('div', { class: 'card pad0' }, el('div', { class: 'chart-wrap', style: { padding: '10px' } },
    lineChart({
      height: 240, series: [{ key: 'score', label: 'gate score', color: css('--s2'), points: scorePts }],
      xLabel: 'cycle', yDomain: [0.3, 0.7], refs: [{ y: 0.5, label: 'even', color: css('--muted') }],
      formatX: (v) => fmt(v), formatY: (v) => pct(v),
    }))));
}

// =============================================================== router / live
const VIEWS = { ladder: renderLadder, generations: renderGenerations, depth: renderDepth, matchups: renderMatchups, games: renderGames, training: renderTraining };
let currentView = 'ladder';
// A rebuild throws away the toolbar the user is standing in, so a dropdown that triggers one loses
// focus on its first pick and the arrow keys stop cycling. Controls that survive a rebuild carry a
// stable `data-ctl` name, and whatever held focus gets it back once the new DOM is in place —
// async views (matchups, games, training) restore after their fetch, not before it.
function keepFocus(render) {
  const ctl = document.activeElement?.dataset?.ctl;
  const restore = () => {
    if (!ctl) return;
    // A view keeps its DOM when you switch away, so an older copy of the same control can still be
    // sitting in a hidden section. Only the active view's copy can take focus.
    const sel = `[data-ctl="${ctl}"]`;
    ($(`.view.active ${sel}`) || $(sel))?.focus({ preventScroll: true });
  };
  const out = render();
  if (out && typeof out.then === 'function') return out.then((v) => { restore(); return v; }, (e) => { restore(); throw e; });
  restore();
  return out;
}
function renderView(name) {
  // Every view reads State.depth, so the depth is re-checked against the data here rather than
  // in each view — a run that drops a depth must not leave any of them pinned to a dead one.
  State.depth = resolveDepth();
  try { keepFocus(VIEWS[name]); } catch (e) { $(`#view-${name}`).innerHTML = `<div class="empty-state">Error: ${e.message}</div>`; console.error(e); }
}
function switchView(name, { push = true } = {}) {
  if (currentView !== 'games' || name !== 'games') stopPlayback(); // leaving the viewer, or a full rebuild of it
  currentView = name;
  $$('nav.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
  $$('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${name}`));
  renderView(name);
  syncUrl(push);
}
$('#tabs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) switchView(b.dataset.view); });

// ------------------------------------------------------------------- URL state
// The whole view state lives in the hash, so a reload lands back where you were — this page sits
// open for hours next to a running train:loop and gets refreshed constantly. Hash rather than
// path: the server stays a plain static file handler with no catch-all route.
// The shared depth rides along on every view, including the two that don't filter by it, so it
// survives a refresh taken from the depth curve or the game viewer.
function urlParams() {
  const q = new URLSearchParams();
  if (State.depth != null) q.set('depth', String(State.depth));
  if (currentView === 'depth' && State.depthSel.size) q.set('sel', [...State.depthSel].join(','));
  if (currentView === 'matchups' && State.mFam !== 'all') q.set('fam', State.mFam);
  if (currentView === 'matchups' && State.mCross !== 'unlinked') q.set('cross', State.mCross);
  if (currentView === 'games') {
    const f = State.gamesFilter || {};
    if (f.a) q.set('a', f.a);
    if (f.b) q.set('b', f.b);
    if (State.openGame) { q.set('g', State.openGame.g); q.set('ply', String(State.openGame.ply)); }
  }
  if (currentView === 'training' && State.trackId) q.set('track', State.trackId);
  return q;
}
function syncUrl(push = false) {
  const q = urlParams().toString();
  const hash = `#/${currentView}${q ? `?${q}` : ''}`;
  if (hash === location.hash) return;
  // pushState/replaceState don't fire hashchange, so writing here never re-enters applyUrl.
  if (push) history.pushState(null, '', hash); else history.replaceState(null, '', hash);
}
function applyUrl() {
  const [pathPart, qs] = location.hash.replace(/^#\/?/, '').split('?');
  const view = Object.hasOwn(VIEWS, pathPart) ? pathPart : 'ladder';
  const p = new URLSearchParams(qs || '');
  const depth = p.get('depth');
  if (depth) State.depth = depth === 'all' ? 'all' : Number(depth);
  const sel = p.get('sel');
  if (sel) State.depthSel = new Set(sel.split(',').filter(Boolean));
  State.mFam = p.get('fam') || 'all';
  State.mCross = ['none', 'unlinked', 'all'].includes(p.get('cross')) ? p.get('cross') : 'unlinked';
  const a = p.get('a'), b = p.get('b');
  State.gamesFilter = a || b ? { a: a || null, b: b || null } : null;
  const g = p.get('g');
  const ply = p.has('ply') ? Number(p.get('ply')) : NaN;
  State.pendingGame = g ? { g, ply: Number.isFinite(ply) ? ply : null } : null; // null opens at the end
  State.openGame = null;
  State.trackId = p.get('track') || null;
  switchView(view, { push: false });
}
// Back/forward and a hand-edited hash both land here (every entry we write is a hash, so
// hashchange covers popstate); re-applying is idempotent.
window.addEventListener('hashchange', applyUrl);

// keyboard hotkeys — drive the open game viewer (ignored while typing in a control)
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (currentView !== 'games' || !activeViewer) return;
  const map = {
    ArrowLeft: 'prev', ArrowRight: 'next',
    ArrowUp: 'first', Home: 'first', ArrowDown: 'last', End: 'last',
    ' ': 'toggle', Spacebar: 'toggle',
  };
  const action = map[e.key];
  if (!action) return;
  e.preventDefault();
  activeViewer[action]();
});

// sound toggle — the replay is the only thing that makes noise, but this page is often open
// beside other work, so the mute has to be one click away and has to survive a refresh.
const soundBtn = $('#soundBtn');
function paintSoundBtn() {
  soundBtn.textContent = muted ? '🔇' : '🔊';
  soundBtn.title = muted ? 'Unmute move sounds' : 'Mute move sounds';
}
soundBtn.addEventListener('click', () => {
  muted = !muted;
  localStorage.setItem('viz-muted', muted ? '1' : '0');
  paintSoundBtn();
});
paintSoundBtn();

// theme
const themeBtn = $('#themeBtn');
themeBtn.addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('viz-theme', next);
  renderView(currentView); // charts read the CSS variables at build time
});
if (localStorage.getItem('viz-theme')) document.documentElement.setAttribute('data-theme', localStorage.getItem('viz-theme'));

// live SSE
// A running rank:pool appends a finished game every second or two but only rewrites the ledger
// and the pool once a whole matchup lands. Rebuilding the DOM per appended game would reset the
// scroll position, the heatmap and the open replay every second, so those files only refresh the
// counters — a view rebuild waits for data that actually changes the view.
const VOLATILE_FILES = /(ladder-games\.jsonl|match-timing\.json|loop\.log)$/;
function refreshCounters() {
  const n = $('#statLadderGames');
  if (n) n.textContent = fmt(State.summary?.counts?.games);
}
function connectLive() {
  const pill = $('#livePill'); const txt = $('#liveText');
  const es = new EventSource('/api/events');
  es.onopen = () => { pill.classList.add('on'); txt.textContent = 'live'; };
  es.onerror = () => { pill.classList.remove('on'); txt.textContent = 'reconnecting…'; };
  let reloadTimer = null;
  let structural = false; // sticky across the events the timer coalesces
  es.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type !== 'change') return;
    const files = msg.files || [];
    structural = structural || !files.length || files.some((f) => !VOLATILE_FILES.test(f));
    txt.textContent = 'updating…';
    clearTimeout(reloadTimer);
    reloadTimer = setTimeout(async () => {
      const rebuild = structural;
      structural = false;
      State.pool = null;
      await loadCore();
      if (rebuild) { renderView(currentView); syncUrl(); } else refreshCounters();
      txt.textContent = 'live';
    }, 300);
  };
}

// boot
(async function main() {
  try {
    await loadCore();
    applyUrl();
    connectLive();
  } catch (e) {
    $('#view-ladder').innerHTML = `<div class="empty-state">Could not load data: ${e.message}<br><span class="sub">Is training/data/loop present?</span></div>`;
    console.error(e);
  }
})();

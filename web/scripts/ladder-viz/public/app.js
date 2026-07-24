// AposChess ladder visualizer — client. Vanilla ES module, hand-rolled SVG
// charts (no chart lib, no CDN). Everything is driven by the local server's
// /api/* endpoints and refreshes live over SSE.

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
  const allY = series.flatMap((s) => s.points.map((p) => p.y));
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

  // series
  for (const s of series) {
    const pts = s.points.slice().sort((a, b) => a.x - b.x);
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
    const rows = here.map((p) => `<div class="tt-row"><span class="k"><span class="swatch" style="background:${p.s.color}"></span>${p.s.label}</span><span class="v">${opts.formatY ? opts.formatY(p.y) : fmt(p.y, 1)}${p.mate ? ' ⚑' : ''}</span></div>`).join('');
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
  depthSel: new Set(), depthMode: 'fixed', ladderDepth: 6, ladderMode: 'depth',
};
// Controls for the game currently open in the viewer; driven by keyboard hotkeys.
let activeViewer = null;

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
    el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Ladder games'), el('span', { class: 'v' }, fmt(S.counts?.games))),
    el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Dataset positions'), el('span', { class: 'v' }, fmt(S.meta?.dataset?.totalLines))),
    el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Ledger updated'), el('span', { class: 'v' }, S.meta?.generated ? new Date(S.meta.generated).toLocaleString() : '–')),
  ));
  root.append(cards);

  // ---- toolbar
  const tb = el('div', { class: 'toolbar mt' });
  const depths = State.ladder.depths || [1, 2, 3, 4, 5, 6, 7, 8];
  const modeSeg = el('div', { class: 'seg' },
    segBtn('Best depth', State.ladderMode === 'best', () => { State.ladderMode = 'best'; renderLadder(); }),
    segBtn('By depth', State.ladderMode === 'depth', () => { State.ladderMode = 'depth'; renderLadder(); }));
  tb.append(el('label', {}, 'View', modeSeg));
  if (State.ladderMode === 'depth') {
    const sel = el('select', { onchange: (e) => { State.ladderDepth = Number(e.target.value); renderLadder(); } },
      ...depths.map((d) => el('option', { value: d, selected: d === State.ladderDepth ? '' : null }, `depth ${d}`)));
    tb.append(el('label', {}, 'Depth', sel));
  }
  tb.append(el('span', { class: 'sub' }, State.ladderMode === 'best' ? 'Each engine at its strongest depth.' : `All engines searching to depth ${State.ladderDepth}. Anchor hc6@2 ≡ 1500.`));
  root.append(tb);

  // ---- leaderboard rows
  let rows;
  if (State.ladderMode === 'best') {
    rows = [...versions.values()].map((v) => ({ v, r: bestNode(v) })).filter((x) => x.r);
  } else {
    rows = [...versions.values()].map((v) => ({ v, r: v.byDepth.get(State.ladderDepth) })).filter((x) => x.r);
  }
  rows.sort((a, b) => b.r.elo - a.r.elo);
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
function segBtn(label, active, on) { return el('button', { class: active ? 'active' : '', onclick: on }, label); }
function archStr(arch) { return arch ? `[${arch.join(',')}]` : '–'; }
function convBadge(c) {
  if (!c) return el('span', { class: 'badge neutral' }, 'no data');
  if (c.converged) return el('span', { class: 'badge good' }, '✓ converged');
  if (c.resolved) return el('span', { class: 'badge warn' }, '⚠ inversions');
  return el('span', { class: 'badge warn' }, 'not converged');
}

// ============================================================ GENERATIONS view
function renderGenerations() {
  const root = $('#view-generations'); root.innerHTML = '';
  const versions = [...groupVersions().values()].filter((v) => v.gen != null).sort((a, b) => a.gen - b.gen);
  const depths = State.ladder.depths || [1, 2, 3, 4, 5, 6, 7, 8];
  if (!State.genDepth) State.genDepth = 6;

  const tb = el('div', { class: 'toolbar' });
  tb.append(el('label', {}, 'At depth',
    el('select', { onchange: (e) => { State.genDepth = Number(e.target.value); renderGenerations(); } },
      ...depths.map((d) => el('option', { value: d, selected: d === State.genDepth ? '' : null }, `depth ${d}`)))));
  tb.append(el('span', { class: 'sub' }, 'Champion strength across train:loop generations — the climb.'));
  root.append(el('h2', { class: 'section' }, 'Generational progress ', el('span', { class: 'sub' }, '· Elo of each loop champion')));
  root.append(tb);

  const pts = versions.map((v) => { const r = v.byDepth.get(State.genDepth); return r ? { x: v.gen, y: r.elo, meta: v, name: v.name } : null; }).filter(Boolean);
  // hc reference at same depth
  const hc = [...groupVersions().values()].find((v) => v.eng === 'hc' && v.version === '2');
  const hcNode = hc?.byDepth.get(State.genDepth);
  const refs = hcNode ? [{ y: hcNode.elo, label: `handcrafted v2 (d${State.genDepth})`, color: css('--hc') }] : [];

  const card = el('div', { class: 'card pad0' });
  const wrap = el('div', { class: 'chart-wrap', style: { padding: '10px' } });
  wrap.append(lineChart({
    height: 420,
    series: [{ key: 'gen', label: 'champion Elo', color: css('--s1'), points: pts }],
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
  root.append(el('h2', { class: 'section mt' }, 'Elo gained per generation ', el('span', { class: 'sub' }, `· at depth ${State.genDepth}`)));
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

  const selected = [...State.depthSel].map((k) => versions.get(k)).filter(Boolean);
  selected.sort((a, b) => (b.gen ?? -1) - (a.gen ?? -1));
  const series = selected.map((v, i) => ({
    key: v.key, label: v.name, color: CAT[i % CAT.length],
    points: [...v.byDepth.values()].map((r) => ({ x: Number(r.depth), y: r.elo })).sort((a, b) => a.x - b.x),
  }));

  const card = el('div', { class: 'card pad0' });
  const wrap = el('div', { class: 'chart-wrap', style: { padding: '10px' } });
  if (series.length) {
    wrap.append(lineChart({
      height: 440, series, directLabels: true,
      xDomain: [0.7, 8.3], xTicks: [1, 2, 3, 4, 5, 6, 7, 8],
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
  }));
}

// =============================================================== MATCHUPS view
async function renderMatchups() {
  const root = $('#view-matchups'); root.innerHTML = '';
  root.append(el('h2', { class: 'section' }, 'Head-to-head ', el('span', { class: 'sub' }, '· games actually played in the ranking pool')));
  if (!State.pool) State.pool = await api('/api/pool');
  const pairs = State.pool.pairs || {};
  const eloOf = new Map(); const labelOf = new Map(); const parsedOf = new Map();
  for (const r of State.ladder.ranking) { eloOf.set(r.tag, r.elo); labelOf.set(r.tag, `${r.name || nodeLabel(r.eng, r.version, State.champs)} d${r.depth}`); parsedOf.set(r.tag, r); }

  // tags present in pool
  const present = new Set();
  for (const k of Object.keys(pairs)) { const [a, b] = k.split('|'); present.add(a); present.add(b); }

  const depths = ['all', ...(State.ladder.depths || [1, 2, 3, 4, 5, 6, 7, 8])];
  if (State.mDepth === undefined) State.mDepth = 6;
  if (State.mFam === undefined) State.mFam = 'all';
  const tb = el('div', { class: 'toolbar' });
  tb.append(el('label', {}, 'Depth', el('select', { onchange: (e) => { State.mDepth = e.target.value === 'all' ? 'all' : Number(e.target.value); renderMatchups(); } },
    ...depths.map((d) => el('option', { value: d, selected: String(d) === String(State.mDepth) ? '' : null }, d === 'all' ? 'all depths' : `depth ${d}`)))));
  tb.append(el('label', {}, 'Family', el('select', { onchange: (e) => { State.mFam = e.target.value; renderMatchups(); } },
    ...['all', 'nn', 'hc'].map((f) => el('option', { value: f, selected: f === State.mFam ? '' : null }, f === 'all' ? 'all engines' : f)))));

  let tags = [...present].filter((t) => {
    const p = parsedOf.get(t); if (!p) return false;
    if (State.mDepth !== 'all' && Number(p.depth) !== Number(State.mDepth)) return false;
    if (State.mFam !== 'all' && p.eng !== State.mFam) return false;
    return true;
  });
  tags.sort((a, b) => (eloOf.get(a) ?? 0) - (eloOf.get(b) ?? 0));
  tb.append(el('span', { class: 'sub' }, `${tags.length} nodes · cell = row's score vs column · click a played cell for the games`));
  root.append(tb);

  if (tags.length < 2) { root.append(el('div', { class: 'empty-state' }, 'Not enough directly-played nodes at this filter. Try “all depths”.')); return; }

  const scroll = el('div', { class: 'heat-scroll' });
  const table = el('table', { class: 'heat' });
  const head = el('tr', {}, el('th', { class: 'corner' }, ''));
  for (const c of tags) head.append(el('th', { class: 'collab' }, el('div', {}, shortLabel(c, parsedOf))));
  table.append(head);
  for (const rTag of tags) {
    const tr = el('tr', {}, el('th', { class: 'rowlab' }, shortLabel(rTag, parsedOf)));
    for (const cTag of tags) {
      if (rTag === cTag) { tr.append(el('td', { class: 'cell diag' }, '')); continue; }
      const info = pairScore(pairs, rTag, cTag);
      if (!info) { tr.append(el('td', { class: 'cell empty' }, '')); continue; }
      const bg = divergingColor(info.score);
      const td = el('td', { class: 'cell', style: `background:${bg}` }, Math.round(info.score * 100));
      td.addEventListener('mousemove', (ev) => showTip(
        `<div class="tt-title">${labelOf.get(rTag)} vs ${labelOf.get(cTag)}</div>`
        + `<div class="tt-row"><span class="k">Score</span><span class="v">${pct(info.score)}</span></div>`
        + `<div class="tt-row"><span class="k">Games</span><span class="v">${info.games}</span></div>`, ev.clientX, ev.clientY));
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
function pairScore(pairs, a, b) {
  const kab = `${a}|${b}`, kba = `${b}|${a}`;
  if (pairs[kab]) return { score: pairs[kab].sumA / pairs[kab].games, games: pairs[kab].games };
  if (pairs[kba]) return { score: 1 - pairs[kba].sumA / pairs[kba].games, games: pairs[kba].games };
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
  const mkSel = (which, val) => el('select', { onchange: (e) => { State.gamesFilter = { ...State.gamesFilter, [which]: e.target.value || null }; loadGameList(); } },
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
  if (!data.games.length) { list.append(el('div', { class: 'empty-state' }, 'No games.')); activeViewer = null; return; }
  for (const g of data.games) {
    const resChar = g.r === 1 ? 'W' : g.r === -1 ? 'B' : '½';
    const resClass = g.r === 1 ? 'w' : g.r === -1 ? 'l' : 'd';
    const row = el('div', { class: 'game-row' },
      el('div', { class: `res ${resClass}` }, resChar),
      el('div', {}, el('div', {}, `${tagLabel(g.w)}  vs  ${tagLabel(g.b)}`), el('div', { class: 'sub mono' }, g.g)),
      el('div', { class: 'sub num' }, `${g.n} plies`));
    row.addEventListener('click', () => { $$('.game-row', list).forEach((r) => r.style.background = ''); row.style.background = 'var(--surface-2)'; openGame(g.g); });
    list.append(row);
  }
  // auto-open first
  openGame(data.games[0].g);
}

// ---- board + eval game viewer (blue2 board + Merida pieces, matching the app)
const pieceUrl = (color, role) => `/assets/pieces/merida/${color}${role.toUpperCase()}.svg`;
function parseFen(fen) {
  const [bp, turn] = fen.split(' ');
  const board = new Array(64).fill(null);
  const rows = bp.split('/');
  for (let r = 0; r < 8; r++) {
    const row = rows[7 - r]; let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) { f += +ch; continue; }
      const role = ch.toLowerCase(); const color = ch === role ? 'b' : 'w';
      board[r * 8 + f] = { role, color }; f++;
    }
  }
  return { board, whiteToMove: turn !== 'b' };
}
function sqIdx(name) { return (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97); }
function applyMoveTo(board, mv) {
  const from = sqIdx(mv.slice(0, 2)), to = sqIdx(mv.slice(2, 4)), promo = mv[4];
  const p = board[from]; if (!p) return { from, to };
  board[from] = null;
  const np = promo ? { role: promo.toLowerCase(), color: p.color } : p;
  board[to] = np;
  // castling: king two files → move rook alongside
  if (p.role === 'k' && Math.abs((to % 8) - (from % 8)) === 2) {
    const rank = Math.floor(to / 8);
    if (to % 8 === 6) { board[rank * 8 + 5] = board[rank * 8 + 7]; board[rank * 8 + 7] = null; }
    else if (to % 8 === 2) { board[rank * 8 + 3] = board[rank * 8 + 0]; board[rank * 8 + 0] = null; }
  }
  return { from, to };
}
function boardAtPly(game, ply) {
  const { board, whiteToMove } = parseFen(game.start || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  let last = null;
  for (let i = 0; i < ply; i++) last = applyMoveTo(board, game.moves[i]);
  return { board, last, whiteToMove };
}
function whitePovEval(game) {
  const startWhite = !(game.start && game.start.split(' ')[1] === 'b');
  return (game.v || []).map((v, i) => {
    const sideWhite = startWhite ? i % 2 === 0 : i % 2 === 1;
    const val = sideWhite ? v : -v;
    return { i, mate: Math.abs(v) > 50000, v: val };
  });
}
async function openGame(g) {
  const wrap = $('#gameViewerWrap'); if (!wrap) return;
  wrap.innerHTML = '<div class="empty-state">Loading…</div>';
  const game = await api(`/api/game?g=${encodeURIComponent(g)}`);
  if (game.error) { wrap.innerHTML = '<div class="empty-state">Game not found.</div>'; return; }
  const state = { ply: game.moves.length };
  wrap.innerHTML = '';

  const info = el('div', { class: 'flex wrap', style: { justifyContent: 'space-between', marginBottom: '10px' } },
    el('div', {}, el('b', {}, tagLabel(game.players.w)), el('span', { class: 'sub' }, '  (white)  vs  '), el('b', {}, tagLabel(game.players.b)), el('span', { class: 'sub' }, '  (black)')),
    el('span', { class: `badge ${game.r === 1 ? 'good' : game.r === -1 ? 'crit' : 'neutral'}` }, game.r === 1 ? 'white won' : game.r === -1 ? 'black won' : 'draw'));

  const boardEl = el('div', { class: 'board' });
  const transport = el('div', { class: 'transport' });
  const readout = el('div', { class: 'ply-readout' });
  const range = el('input', { type: 'range', min: '0', max: String(game.moves.length), value: String(state.ply) });
  const btn = (t, on) => el('button', { onclick: on }, t);
  let playTimer = null;
  const stop = () => { if (playTimer) { clearInterval(playTimer); playTimer = null; playBtn.textContent = '▶'; } };
  const setPly = (p) => { state.ply = Math.max(0, Math.min(game.moves.length, p)); range.value = String(state.ply); draw(); };
  const playBtn = btn('▶', () => {
    if (playTimer) { stop(); return; }
    playBtn.textContent = '❚❚';
    playTimer = setInterval(() => { if (state.ply >= game.moves.length) { setPly(0); } setPly(state.ply + 1); if (state.ply >= game.moves.length) stop(); }, 550);
  });
  transport.append(
    btn('⏮', () => { stop(); setPly(0); }),
    btn('‹', () => { stop(); setPly(state.ply - 1); }),
    playBtn,
    btn('›', () => { stop(); setPly(state.ply + 1); }),
    btn('⏭', () => { stop(); setPly(game.moves.length); }),
    range, readout);
  range.addEventListener('input', () => { stop(); setPly(Number(range.value)); });

  // expose to the global hotkey handler (← → step, Home/End jump, Space play)
  activeViewer = {
    prev: () => { stop(); setPly(state.ply - 1); },
    next: () => { stop(); setPly(state.ply + 1); },
    first: () => { stop(); setPly(0); },
    last: () => { stop(); setPly(game.moves.length); },
    toggle: () => playBtn.click(),
  };

  const hint = el('div', { class: 'pill-note', style: { marginTop: '6px' } }, '← → step · ↑↓ / Home End jump · Space play');
  const left = el('div', {}, boardEl, transport, hint);
  const movelist = el('div', { class: 'movelist' });
  const evalWrap = el('div', { class: 'card pad0 mt' }, el('div', { class: 'chart-wrap', id: 'evalChartWrap', style: { padding: '8px' } }));
  const right = el('div', {}, info, evalWrap, movelist);

  function draw() {
    const { board, last } = boardAtPly(game, state.ply);
    boardEl.innerHTML = '';
    for (let r = 7; r >= 0; r--) for (let f = 0; f < 8; f++) {
      const idx = r * 8 + f;
      const dark = (r + f) % 2 === 0;
      const sq = el('div', { class: `sq${last && (idx === last.from || idx === last.to) ? ' hl' : ''}` });
      const p = board[idx];
      if (p) sq.append(el('div', { class: 'piece', style: `background-image:url('${pieceUrl(p.color, p.role)}')` }));
      boardEl.append(sq);
    }
    readout.textContent = `${state.ply} / ${game.moves.length}`;
    $$('.mv', movelist).forEach((m) => m.classList.toggle('on', Number(m.dataset.ply) === state.ply));
    drawEval();
  }
  const evalPts = whitePovEval(game);
  function drawEval() {
    const w = $('#evalChartWrap'); if (!w) return; w.innerHTML = '';
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
  // build movelist
  for (let i = 0; i < game.moves.length; i++) {
    if (i % 2 === 0) movelist.append(el('span', { class: 'mvno' }, `${i / 2 + 1}.`));
    const mv = el('span', { class: 'mv', 'data-ply': String(i + 1) }, game.moves[i]);
    mv.addEventListener('click', () => { stop(); setPly(i + 1); });
    movelist.append(mv);
  }

  wrap.append(el('div', { class: 'viewer' }, left, right));
  draw();
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
  openTrack(tracks[0].id);
}
async function openTrack(id) {
  const box = $('#trackChart'); if (!box) return;
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
function switchView(name) {
  currentView = name;
  $$('nav.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
  $$('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${name}`));
  try { VIEWS[name](); } catch (e) { $(`#view-${name}`).innerHTML = `<div class="empty-state">Error: ${e.message}</div>`; console.error(e); }
}
$('#tabs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) switchView(b.dataset.view); });

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

// theme
const themeBtn = $('#themeBtn');
themeBtn.addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('viz-theme', next);
  switchView(currentView);
});
if (localStorage.getItem('viz-theme')) document.documentElement.setAttribute('data-theme', localStorage.getItem('viz-theme'));

// live SSE
function connectLive() {
  const pill = $('#livePill'); const txt = $('#liveText');
  const es = new EventSource('/api/events');
  es.onopen = () => { pill.classList.add('on'); txt.textContent = 'live'; };
  es.onerror = () => { pill.classList.remove('on'); txt.textContent = 'reconnecting…'; };
  let reloadTimer = null;
  es.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type !== 'change') return;
    txt.textContent = 'updating…';
    clearTimeout(reloadTimer);
    reloadTimer = setTimeout(async () => {
      State.pool = null;
      await loadCore();
      switchView(currentView);
      txt.textContent = 'live';
    }, 300);
  };
}

// boot
(async function main() {
  try {
    await loadCore();
    switchView('ladder');
    connectLive();
  } catch (e) {
    $('#view-ladder').innerHTML = `<div class="empty-state">Could not load data: ${e.message}<br><span class="sub">Is training/data/loop present?</span></div>`;
    console.error(e);
  }
})();

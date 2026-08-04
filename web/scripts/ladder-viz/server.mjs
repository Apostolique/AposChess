// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Ladder visualizer — a local, zero-dependency dashboard for the AposChess
// strength data. It reads the git-ignored training/data/loop artifacts straight
// off disk (the Elo ladder, the pairwise pool, the harvested games, the training
// tracks) and serves a single-page UI with live updates.
//
// This is a DEV TOOL. It lives under web/scripts/ so it never lands in the Vite
// build / GitHub Pages deploy (only web/index.html + web/public/ ship). Run it
// from web/:  npm run viz   (then open http://localhost:5178)
//
// No npm install, no bundler: pure node:http + node:fs. All data is re-read from
// disk per request (so it is always fresh); the big games file is byte-indexed
// once at startup and appended incrementally. A single fs.watch on the loop dir
// feeds a Server-Sent-Events stream so open browsers refresh themselves live.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '../../..');           // repo root
const DATA = path.join(ROOT, 'training', 'data');
const LOOP = path.join(DATA, 'loop');
const NN = path.join(ROOT, 'web', 'public', 'nn');
const PUBLIC = path.join(__dirname, 'public');
const ASSETS = path.join(ROOT, 'web', 'src', 'assets'); // blue2 board + merida pieces (shared with the app)
const SRC = path.join(ROOT, 'web', 'src');              // board.js + engine.js, imported by the game viewer
const SOUND = path.join(ROOT, 'web', 'public', 'sound'); // the app's move / capture / check sounds

const F = {
  ledger: path.join(LOOP, 'engine-elo.ladder.json'),
  // The pairwise matrix rank:pool last fitted, which it writes out after every matchup. Derived
  // from the dataset (plus loop/legacy-pairs.json), so it's the whole picture in one file —
  // ladder matchups, gate games and all.
  pool: path.join(LOOP, 'ladder-pool.json'),
  selfplay: path.join(DATA, 'selfplay.jsonl'),
  match: path.join(LOOP, 'match.json'),
  timing: path.join(LOOP, 'match-timing.json'),
  manifest: path.join(NN, 'manifest.json'),
  nameHistory: path.join(NN, 'name-history.json'),
  log: path.join(LOOP, 'loop.log'),
  experiments: path.join(LOOP, 'experiments'),
};

const argPort = (() => {
  const a = process.argv.find((x) => x.startsWith('--port='));
  return a ? parseInt(a.slice(7), 10) : Number(process.env.PORT) || 5178;
})();

// ---------------------------------------------------------------- small helpers
function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch { return null; }
}
function statSafe(file) {
  try { return fs.statSync(file); } catch { return null; }
}

// -------------------------------------------------- champions: name/gen/arch map
// Merge the shipped catalog (manifest.json, live nets) with name-history.json
// (pruned champions keep their name forever). Keyed by 6-hex weights hash.
function championIndex() {
  const manifest = readJson(F.manifest) || { nets: [] };
  const hist = readJson(F.nameHistory) || { names: [] };
  const byHash = new Map();
  const byName = new Map();
  const add = (e, source) => {
    if (!e.hash) return;
    const prev = byHash.get(e.hash) || {};
    const merged = {
      hash: e.hash,
      name: e.name ?? prev.name ?? null,
      gen: e.gen ?? prev.gen ?? null,
      arch: e.arch ?? prev.arch ?? null,
      file: e.file ?? prev.file ?? null,
      loopChampion: e.loopChampion ?? prev.loopChampion ?? false,
      current: e.current ?? prev.current ?? false,
      note: e.note ?? prev.note ?? null,
      source: prev.source || source,
    };
    byHash.set(e.hash, merged);
    if (merged.name) byName.set(merged.name.toLowerCase(), merged);
  };
  for (const e of hist.names || []) add(e, 'history');
  for (const e of manifest.nets || []) add(e, 'manifest');
  return {
    default: manifest.default || null,
    byHash: Object.fromEntries(byHash),
    list: [...byHash.values()].sort((a, b) => (a.gen ?? 999) - (b.gen ?? 999)),
  };
}

// ------------------------------------------------------------- games byte-index
// The game files are large (100s of MB). We keep a light in-memory index of
// {byteOffset, byteLen, g, w, b, r, plies} per line so a single game body can
// be read by seeking, and matchup filters run over metadata only. Built once,
// then extended incrementally as the files grow (train:loop appends live).
//
// ONE source: every producer — generation, the gate, the ranking pool — harvests into the
// dataset, so it holds every game exactly once. (It used to take two, with the pool's separate
// archive indexed first to claim the ids both files held; the game-id dedup below is what's left
// of that, and it still earns its keep against a torn tail or a re-indexed prefix.)
const SOURCES = [F.selfplay];
const gindex = { entries: [], byG: new Map(), src: SOURCES.map(() => ({ indexedBytes: 0, size: 0 })), size: 0, skipped: 0 };

// A non-promoted gate candidate is never archived and carries an "elo<N>" strength label
// instead of an identity (see vtag.mjs ephemeralVersion) — rank:pool leaves it out of the
// fit, so it has no ladder row and no rating. Its games would be unattributable rows here,
// so they stay out of the viz entirely.
const EPHEMERAL = /@elo-?\d+$/;

function addGameLine(buf, off, len, src) {
  let rec;
  try { rec = JSON.parse(buf.toString('utf8')); } catch { return; }
  if (!rec || !rec.g) return;
  if (gindex.byG.has(rec.g)) return;                      // already indexed
  const w = rec.players?.w ?? null;
  const b = rec.players?.b ?? null;
  if (EPHEMERAL.test(w || '') || EPHEMERAL.test(b || '')) { gindex.skipped++; return; }
  const e = { off, len, src, g: rec.g, w, b, r: rec.r ?? null, n: rec.moves?.length ?? 0 };
  gindex.byG.set(rec.g, e);
  gindex.entries.push(e);
}

function indexSource(src, reset) {
  const file = SOURCES[src];
  const state = gindex.src[src];
  const st = statSafe(file);
  if (!st) { state.indexedBytes = 0; state.size = 0; return; }
  if (reset || st.size < state.indexedBytes) state.indexedBytes = 0;
  if (st.size === state.indexedBytes) { state.size = st.size; return; }

  const fd = fs.openSync(file, 'r');
  try {
    const CHUNK = 1 << 20;
    const buf = Buffer.allocUnsafe(CHUNK);
    let filePos = state.indexedBytes;
    let lineStart = filePos;
    let carry = Buffer.alloc(0);
    while (filePos < st.size) {
      const toRead = Math.min(CHUNK, st.size - filePos);
      const bytes = fs.readSync(fd, buf, 0, toRead, filePos);
      if (bytes <= 0) break;
      let chunkStart = 0;
      for (let i = 0; i < bytes; i++) {
        if (buf[i] === 0x0a) {
          const line = carry.length
            ? Buffer.concat([carry, buf.subarray(chunkStart, i)])
            : buf.subarray(chunkStart, i);
          addGameLine(line, lineStart, line.length, src);
          carry = Buffer.alloc(0);
          chunkStart = i + 1;
          lineStart = filePos + i + 1;
        }
      }
      if (chunkStart < bytes) carry = Buffer.concat([carry, buf.subarray(chunkStart, bytes)]);
      filePos += bytes;
    }
    // Only fully terminated lines are indexed; a partial trailing line (a game
    // being written right now) waits for its newline on the next pass.
    state.indexedBytes = lineStart;
    state.size = st.size;
  } finally {
    fs.closeSync(fd);
  }
}

function indexGames(reset = false) {
  // A truncated/rewritten source invalidates cross-source dedup, so rebuild everything.
  const shrank = SOURCES.some((f, i) => { const st = statSafe(f); return st && st.size < gindex.src[i].indexedBytes; });
  if (reset || shrank) {
    gindex.entries = []; gindex.byG.clear(); gindex.skipped = 0;
    for (const s of gindex.src) { s.indexedBytes = 0; s.size = 0; }
    reset = true;
  }
  for (let i = 0; i < SOURCES.length; i++) indexSource(i, reset);
  gindex.size = gindex.src.reduce((n, s) => n + s.size, 0);
}

function readGameBody(g) {
  const e = gindex.byG.get(g);
  if (!e) return null;
  const fd = fs.openSync(SOURCES[e.src], 'r');
  try {
    const buf = Buffer.allocUnsafe(e.len);
    fs.readSync(fd, buf, 0, e.len, e.off);
    return JSON.parse(buf.toString('utf8'));
  } catch { return null; } finally { fs.closeSync(fd); }
}

// ------------------------------------------------------------- pairwise evidence
// Exactly what rank:pool fitted, read straight from the snapshot it writes after every matchup.
// Nothing to mirror or reconcile: every game lives in the dataset and the ratings are derived
// from it in one pass, so this file already holds ladder matchups and gate games alike.
function poolPairs() {
  const snap = readJson(F.pool) || {};
  const out = {};
  let total = 0;
  for (const [k, v] of Object.entries(snap.pairs || {})) {
    if (k.split('|').some((id) => EPHEMERAL.test(id))) continue;   // unrated candidate — never shown
    out[k] = { games: v.games, sumA: v.sumA };
    total += v.games;
  }
  // The snapshot stamps the dataset it was derived from. Between rank:pool runs the loop keeps
  // appending (gate games, generation), so say when the ratings haven't seen the newest games.
  const ds = statSafe(F.selfplay);
  const stale = !!(ds && snap.data && (snap.data.size !== ds.size || Math.abs((snap.data.mtimeMs ?? 0) - ds.mtimeMs) > 1));
  return { pairs: out, corpusGames: total, stale };
}

// ------------------------------------------------------- experiments / tracks
function experimentTracks() {
  const out = [];
  let dirs = [];
  try { dirs = fs.readdirSync(F.experiments, { withFileTypes: true }); } catch { return out; }
  for (const d of dirs) {
    if (!d.isDirectory()) continue;
    const base = path.join(F.experiments, d.name);
    const state = readJson(path.join(base, 'state.json'));
    const recipe = readJson(path.join(base, 'recipe.json'));
    if (!state && !recipe) continue;
    out.push({ id: d.name, state, recipe });
  }
  return out;
}
function experimentHistory(id) {
  if (!/^[a-z0-9]+$/i.test(id)) return null;
  const file = path.join(F.experiments, id, 'history.jsonl');
  let txt;
  try { txt = fs.readFileSync(file, 'utf8'); } catch { return null; }
  const rows = [];
  for (const line of txt.split('\n')) {
    if (!line.trim()) continue;
    try { rows.push(JSON.parse(line)); } catch { /* skip */ }
  }
  return rows;
}

function tailFile(file, maxBytes = 24 * 1024) {
  const st = statSafe(file);
  if (!st) return '';
  const start = Math.max(0, st.size - maxBytes);
  const fd = fs.openSync(file, 'r');
  try {
    const len = st.size - start;
    const buf = Buffer.allocUnsafe(len);
    fs.readSync(fd, buf, 0, len, start);
    let s = buf.toString('utf8');
    if (start > 0) s = s.slice(s.indexOf('\n') + 1); // drop partial first line
    return s;
  } finally { fs.closeSync(fd); }
}

// ------------------------------------------------------------------ SSE / watch
const sseClients = new Set();
function broadcast(payload) {
  const data = `data: ${JSON.stringify(payload)}\n\n`;
  for (const res of sseClients) { try { res.write(data); } catch { /* ignore */ } }
}
let watchTimer = null;
const pendingChanges = new Set();
function scheduleBroadcast(name) {
  pendingChanges.add(name);
  if (watchTimer) return;
  watchTimer = setTimeout(() => {
    watchTimer = null;
    const files = [...pendingChanges];
    pendingChanges.clear();
    if (files.some((f) => f.includes('selfplay.jsonl'))) indexGames();
    broadcast({ type: 'change', files, ts: Date.now() });
  }, 500);
}
function startWatch() {
  const watchDir = (dir) => {
    try {
      fs.watch(dir, { recursive: true }, (_ev, filename) => {
        if (filename) scheduleBroadcast(String(filename));
      });
    } catch { /* dir may not exist yet */ }
  };
  watchDir(LOOP);
  watchDir(NN);
  // selfplay.jsonl sits beside loop/, not in it — non-recursive so LOOP isn't watched twice.
  try { fs.watch(DATA, (_ev, filename) => { if (filename) scheduleBroadcast(String(filename)); }); } catch { /* may not exist */ }
}

// ----------------------------------------------------------------------- routes
function sendJson(res, obj, code = 200) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(body);
}
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.mp3': 'audio/mpeg' };
function sendFileFrom(res, baseDir, rel) {
  const file = path.join(baseDir, rel);
  if (!file.startsWith(baseDir)) { res.writeHead(403); res.end('forbidden'); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    res.end(data);
  });
}
function sendStatic(res, rel) { sendFileFrom(res, PUBLIC, rel); }

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const p = url.pathname;

  if (p === '/api/events') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store',
      Connection: 'keep-alive', 'X-Accel-Buffering': 'no',
    });
    res.write('retry: 3000\n\n');
    sseClients.add(res);
    const ping = setInterval(() => { try { res.write(': ping\n\n'); } catch { /* */ } }, 25000);
    req.on('close', () => { clearInterval(ping); sseClients.delete(res); });
    return;
  }

  if (p === '/api/summary') {
    const ledger = readJson(F.ledger);
    return sendJson(res, {
      champions: championIndex(),
      convergence: ledger?.convergence ?? null,
      meta: ledger ? { generated: ledger.generated, anchor: ledger.anchor, depths: ledger.depths, dataset: ledger.dataset } : null,
      match: readJson(F.match),
      counts: { games: gindex.entries.length, gamesBytes: gindex.size, gamesSkipped: gindex.skipped },
      files: fileStatus(),
    });
  }
  if (p === '/api/ladder') return sendJson(res, readJson(F.ledger) || { ranking: [] });
  if (p === '/api/pool') return sendJson(res, poolPairs());
  if (p === '/api/timing') return sendJson(res, readJson(F.timing) || {});
  if (p === '/api/log') return sendJson(res, { text: tailFile(F.log) });
  if (p === '/api/experiments') return sendJson(res, { tracks: experimentTracks() });
  if (p === '/api/experiment') {
    const id = url.searchParams.get('id') || '';
    const rows = experimentHistory(id);
    return rows ? sendJson(res, { id, history: rows }) : sendJson(res, { error: 'not found' }, 404);
  }
  if (p === '/api/games') {
    const a = url.searchParams.get('a');
    const b = url.searchParams.get('b');
    const player = url.searchParams.get('player');
    const limit = Math.min(500, parseInt(url.searchParams.get('limit') || '80', 10));
    const offset = parseInt(url.searchParams.get('offset') || '0', 10);
    let list = gindex.entries;
    if (a && b) list = list.filter((e) => (e.w === a && e.b === b) || (e.w === b && e.b === a));
    else if (player) list = list.filter((e) => e.w === player || e.b === player);
    const total = list.length;
    const page = list.slice(Math.max(0, total - offset - limit), total - offset).reverse();
    return sendJson(res, { total, games: page });
  }
  if (p === '/api/game') {
    const g = url.searchParams.get('g') || '';
    const body = readGameBody(g);
    return body ? sendJson(res, body) : sendJson(res, { error: 'not found' }, 404);
  }

  if (p === '/' || p === '/index.html') return sendStatic(res, 'index.html');
  if (p === '/app.js') return sendStatic(res, 'app.js');
  if (p === '/style.css') return sendStatic(res, 'style.css');
  if (p.startsWith('/assets/')) return sendFileFrom(res, ASSETS, decodeURIComponent(p.slice('/assets/'.length)));
  // The app's own sounds, so a replay clicks the way the site does.
  if (p.startsWith('/sound/')) return sendFileFrom(res, SOUND, decodeURIComponent(p.slice('/sound/'.length)));
  // The app's engine, so the viewer replays games through the real variant rules rather than a
  // second copy of them (that is where check detection comes from). board.js and engine.js are
  // plain ES modules with no bundler features, so the browser imports them as they ship — but
  // nothing else in src/ is meant to be reachable, hence the .js-only gate.
  if (p.startsWith('/src/') && p.endsWith('.js')) return sendFileFrom(res, SRC, decodeURIComponent(p.slice('/src/'.length)));

  res.writeHead(404); res.end('not found');
});

function fileStatus() {
  const out = {};
  for (const [k, file] of Object.entries(F)) {
    const st = statSafe(file);
    out[k] = st ? { exists: true, size: st.size, mtime: st.mtimeMs } : { exists: false };
  }
  return out;
}

// ------------------------------------------------------------------------- boot
console.log('AposChess ladder visualizer');
console.log('  repo root :', ROOT);
console.log('  loop data :', LOOP, statSafe(LOOP) ? '' : '  (MISSING — no data yet)');
const t0 = Date.now();
indexGames(true);
console.log(`  indexed   : ${gindex.entries.length} games (${(gindex.size / 1e6).toFixed(0)} MB) in ${Date.now() - t0}ms`);
startWatch();
server.listen(argPort, () => {
  console.log(`\n  ➜  http://localhost:${argPort}\n`);
});

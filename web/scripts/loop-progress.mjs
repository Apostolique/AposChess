// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Read-only progress report for train:loop. It re-reads the persisted loop log
// (training/data/loop/loop.log) and turns the per-cycle verdict lines the loop already
// writes into a trend you can act on: is the latest run's candidate climbing toward the
// gate (let it run) or stuck below 50% (restart with different values)?
//
// LESSON FROM 'Mona' (the [128,64,32] champion, promoted 2026-07-12). A new/larger
// architecture can't warm-start from a differently-shaped champion, so its GATE EDGE sits
// below 50% for DOZENS of cycles even while the net is genuinely, steadily getting stronger.
// Mona's experiment track climbed ~434 → 547 in *absolute* Elo across 50 cycles / 3 runs,
// yet lost the gate (negative edge) on 48 of them — right up until the accumulating net
// finally overtook the champion and promoted at +41. So a below-50% run is NOT automatically
// a failed one: for an experiment track it's the expected early shape of a bootstrapping net.
// The real progress signal is the TRACK's absolute-Elo trajectory (experiment-registry.mjs,
// `npm run train:experiments`), which this report now folds in — not the gate score alone,
// which is measured against a moving (strengthening) champion opponent.
//
// RESTARTS ARE MERGED. Ctrl-C'ing the loop and relaunching it later with the same recipe
// writes a fresh `train:loop start` line and restarts the loop's cycle numbering at 1, but
// it is NOT a new experiment: a warm relaunch resumes the same track's lineage and keeps
// refining the same candidate chain. This report therefore groups consecutive log runs into
// one logical run when the newer one warm-starts the same recipe (same track id; for
// pre-registry logs, same hidden+λ), renumbering cycles cumulatively — so stopping the loop
// overnight no longer makes the trend/read start over from "cycle 1".
//
// ONE RUN IS SEVERAL ARCHITECTURES. `--rotate` (auto by default) moves the loop onto a different
// recipe mid-run — a different SHAPE, its own track, its own lineage and cycle numbering — without
// writing a new `train:loop start` line (train-loop.mjs adoptRecipe). So a launch is parsed here as
// a sequence of SPANS, one per recipe it trained, split at the `↻ Rotating recipe` lines. Every
// per-recipe number (the config line, the absolute-Elo trajectory, the trend, the verdict) is
// scoped to the span it belongs to, because none of them survive an architecture change: a trend
// fitted across a rotation mixes two different nets, and a verdict of "this track has stopped
// improving — consider a different recipe" is stale advice when the loop already rotated away from
// that track on exactly that signal. The run-level totals (cycle count, promotions) stay run-wide,
// but a promotion is credited to the SPAN that earned it, not to the shape the launch started on.
//
//   npm run train:progress           # latest run in detail + a one-line history + a read
//   npm run train:progress -- --runs=12   # show that many (merged) runs in the history
//   npm run train:progress -- --all  # detail EVERY run's cycles, not just the latest
//   npm run train:progress -- --log=PATH  # point at a different loop.log
//
// Nothing here writes or spawns — it only parses the log and the champion weights, so it's
// safe to run against a live loop. The numbers it reports are exactly the loop's own
// `cycle N: ...` lines (candidate score %, Elo vs champion, SPRT verdict), grouped by run.

import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { fmtDur, printWrapped } from './fmt.mjs';
import {
  suggestRecipes, readHistory, trackDir, compactSlug,
  ledgerBestByVersion, reAnchoredAbsElo,
} from './experiment-registry.mjs';
import { weightsHash } from './vtag.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');
const loopDir = resolve(repoDir, 'training', 'data', 'loop');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const logFile = typeof args.log === 'string' ? resolve(args.log) : join(loopDir, 'loop.log');
const histN = args.runs !== undefined ? Number(args.runs) : 8;
const detailAll = !!args.all;

const championFile = resolve(webDir, 'src', 'nn-weights.json');

if (!existsSync(logFile)) {
  console.error(`No loop log at ${logFile}. Run \`npm run train:loop\` first (or pass --log=PATH).`);
  process.exit(1);
}

// --- Parse the log into a list of runs, each holding its config + cycle verdicts. -------
// The loop writes three line shapes we care about (see train-loop.mjs `log(...)`):
//   [ts] train:loop start — batch N @ depth D | gate Gg @ depth D SPRT(0,E) | candidate hidden=[..] λ=L <warm/cold> ...
//   [ts] cycle N: PROMOTED ✓  candidate P% / Elo +E over champion (G games, cycle took T)...
//   [ts] cycle N: kept champion — candidate P% / Elo E (SPRT V, G games, cycle took T). <tail>
//   [ts] train:loop stopped after K promotion(s) in T...
//   [ts] Discarded stale lineage (reason).
//   [ts]   ↻ Rotating recipe (why) → hidden=[..], λ=L, …
//   [ts]     track <id> (new|resuming at cycle N) · features F
// The registry writes a `Recipe <slug> [<id>] — … (track run #N…)` line just before the
// matching `train:loop start` (train-loop.mjs beginRun). Capture it so each run knows which
// experiment TRACK it belongs to — that's how we reach the track's cross-run absolute-Elo
// trajectory (the real progress signal). Runs before the registry existed have no such line.
// The last two shapes are a mid-run rotation: the same launch continuing on a different recipe,
// which starts a new SPAN (its indented `track` line names the track the cycles below it record
// to). They carry no per-track run number, so a span's history entries are matched by time window.
const RECIPE = /^Recipe (\S+) \[([0-9a-f]+)\].*\(track run #(\d+)/;
const ROTATE = /^\s+↻ Rotating recipe \((.*)\) → hidden=\[([^\]]*)\]/;
const ROTATE_TRACK = /^\s+track ([0-9a-f]+)\b/;
const TS = /^\[([\d-]+ [\d:]+)\] (.*)$/;
const runs = [];
let cur = null;
let pendingTrack = null; // {id} of the track awaiting the next `train:loop start`
const curSpan = () => (cur && cur.spans.length ? cur.spans[cur.spans.length - 1] : null);

for (const raw of readFileSync(logFile, 'utf8').split('\n')) {
  const tm = raw.match(TS);
  if (!tm) continue;
  const [, ts, body] = tm;

  const rm = body.match(RECIPE);
  if (rm) { pendingTrack = { id: rm[2] }; continue; }

  if (body.startsWith('train:loop start')) {
    const hidden = (body.match(/hidden=\[([^\]]*)\]/) || [, '?'])[1];
    const lambda = (body.match(/λ=([\d.]+)/) || [, null])[1];
    cur = {
      start: ts, end: null, stopped: null, promotions: 0, cycles: [],
      hidden,
      batch: num(body.match(/batch (\d+)/)),
      gateGames: num(body.match(/gate (\d+)g/)),
      gateDepth: num(body.match(/gate \d+g @ depth (\d+)/)),
      elo1: num(body.match(/SPRT\(\d+,(\d+)\)/)),
      lambda,
      // Whether this launch is allowed to change recipe on its own, and on what rule. It decides
      // whether a stalled track is something to act on or something the loop already handles.
      rotate: (body.match(/\| rotate ([^|]+)/) || [, null])[1]?.trim() ?? null,
      start_kind: /cold start/.test(body) ? 'cold'
        : /cold first cycle/.test(body) ? 'cold→warm'
        : /resuming lineage/.test(body) ? 'warm (resumed lineage)'
        : 'warm',
      lineageDiscarded: null,
      // One span per recipe this launch trains; the launch's own recipe opens the first.
      spans: [{
        kind: 'launch', startTs: ts, endTs: null, hidden, lambda,
        trackId: pendingTrack?.id ?? null, why: null, cycles: [],
      }],
    };
    runs.push(cur);
    pendingTrack = null;
    continue;
  }
  if (!cur) continue;

  // A mid-run recipe change: everything after it belongs to a different net and a different track.
  const rot = body.match(ROTATE);
  if (rot) {
    cur.spans.push({
      kind: 'rotate', startTs: ts, endTs: null,
      hidden: rot[2], lambda: (body.match(/λ=([\d.]+)/) || [, null])[1],
      trackId: null, why: rot[1], cycles: [],
    });
    continue;
  }
  // The rotation's track id arrives on the next line. Only ever fill the rotation span that is
  // still waiting for one, so an unrelated indented `track …` line can't be mistaken for it.
  const rt = body.match(ROTATE_TRACK);
  if (rt && curSpan()?.kind === 'rotate' && !curSpan().trackId) {
    curSpan().trackId = rt[1];
    continue;
  }

  if (body.startsWith('Discarded stale lineage')) {
    cur.lineageDiscarded = (body.match(/\(([^)]+)\)/) || [, '?'])[1];
    continue;
  }
  if (body.startsWith('train:loop stopped')) {
    cur.stopped = (body.match(/after (\d+) promotion/) || [, '?'])[1];
    cur.end = ts;
    continue;
  }

  const cm = body.match(/^cycle (\d+): (PROMOTED|kept champion)/);
  if (cm) {
    const promoted = cm[2] === 'PROMOTED';
    const tail = /kept as lineage/.test(body) ? 'lineage+'
      : /Lineage reset/.test(body) ? 'lineage reset'
      : promoted ? 'PROMOTED' : '';
    const cyc = {
      ts,
      n: Number(cm[1]),
      promoted,
      score: Number((body.match(/candidate ([\d.]+)%/) || [, NaN])[1]),
      elo: Number((body.match(/Elo ([+-]?\d+)/) || [, NaN])[1]),
      // A gate that reached H1 but failed its CONFIRMATION match writes no "SPRT <verdict>"
      // token — the SPRT did decide H1, it just wasn't the last word (train-loop.mjs
      // runConfirm). Reporting that as '?' would hide the one outcome most worth seeing: the
      // gate and the rematch disagreed, which is exactly what the confirmation exists to
      // surface. The track history keeps the raw sprt:"H1" alongside the confirm record.
      sprt: promoted ? 'H1'
        : /the gate reached H1/.test(body) ? 'H1 unconfirmed'
          : (body.match(/SPRT (H0|inconclusive)/) || [, '?'])[1],
      games: num(body.match(/(\d+) games/)),
      dur: (body.match(/cycle took ([^)]+)\)/) || [, '?'])[1],
      tail,
    };
    cur.cycles.push(cyc);       // run-wide, for the totals
    curSpan().cycles.push(cyc); // and to the recipe that actually produced it
    cur.end = ts;
    if (promoted) cur.promotions++;
  }
}

function num(m) { return m ? Number(m[1]) : null; }

// Close every span's time window: a span runs until the next one starts (a rotation, the next
// launch, or — for the last one — the end of the log). The window is how a span's cycles are
// matched to the track-history entries carrying their absElo: a rotation logs the track id but
// not the per-track run number, so the run-number filter used for launches isn't available.
const allSpans = runs.flatMap((r) => r.spans);
for (let i = 0; i < allSpans.length - 1; i++) allSpans[i].endTs = allSpans[i + 1].startTs;

if (!runs.length) {
  console.log('Loop log has no runs yet (no `train:loop start` lines).');
  process.exit(0);
}

// --- Merge warm same-recipe relaunches into logical runs. --------------------------------
// A relaunch continues the previous run's chain when it warm-starts (a cold/cold-first
// relaunch deliberately begins a fresh chain) and trains the same recipe: same experiment
// track id when both runs have one; across the registry boundary we don't guess; for two
// pre-registry runs the best available key is hidden+λ. A legacy lineage-discard note means
// accumulation restarted from scratch, so it breaks the chain too.
// The comparison is against the previous launch's LAST span, not the recipe it started on: if the
// loop rotated mid-run and you then relaunched onto that same rotated-to recipe, the relaunch really
// is picking the chain back up where it left off — and conversely, a relaunch onto the shape the
// previous launch had already rotated AWAY from is a different chain, not a continuation.
function continuesChain(prev, run) {
  if (!run.start_kind.startsWith('warm')) return false;
  if (run.lineageDiscarded) return false;
  const tail = prev.spans[prev.spans.length - 1], head = run.spans[0];
  if (tail.trackId && head.trackId) return tail.trackId === head.trackId;
  if (tail.trackId || head.trackId) return false;
  return tail.hidden === head.hidden && tail.lambda === head.lambda;
}
const chains = [];
for (const run of runs) {
  const tail = chains.length ? chains[chains.length - 1] : null;
  if (tail && continuesChain(tail[tail.length - 1], run)) tail.push(run);
  else chains.push([run]);
}
// Flatten a chain to the shape printCycles/readRun consume: cycles and spans concatenated across
// the segments; start/start_kind from the FIRST launch (how the chain began); config + stopped
// state from the LATEST launch (what's on disk/running now). `hidden`/`lambda` come from the LAST
// SPAN — the recipe the chain is actually training — not from the launch line, which a rotation
// may have left several architectures behind.
function chainView(segs) {
  const first = segs[0], last = segs[segs.length - 1];
  const spans = segs.flatMap((r) => r.spans);
  const tail = spans[spans.length - 1];
  return {
    segs, spans,
    cycles: segs.flatMap((r) => r.cycles),
    start: first.start, end: last.end, stopped: last.stopped,
    promotions: segs.reduce((a, r) => a + r.promotions, 0),
    rotations: spans.filter((s) => s.kind === 'rotate').length,
    hidden: tail.hidden, lambda: tail.lambda, launchHidden: spans[0].hidden,
    batch: last.batch, gateGames: last.gateGames,
    gateDepth: last.gateDepth, elo1: last.elo1, rotate: last.rotate,
    start_kind: first.start_kind,
    lineageDiscarded: first.lineageDiscarded,
  };
}
const views = chains.map(chainView);

// --- Champion shape, to flag a candidate that can't warm-start from it. -----------------
let champHidden = null, champArch = null;
try {
  champArch = JSON.parse(readFileSync(championFile, 'utf8')).arch;
  if (Array.isArray(champArch) && champArch.length >= 3) champHidden = champArch.slice(1, -1).join(',');
} catch { /* placeholder/material champion — no arch */ }

// --- The anchor that makes a stored absElo comparable to "how it would score vs the -------
// champion RIGHT NOW". A cycle's absElo (championLedgerElo + gate edge) was recorded against the
// champion of THAT cycle, but the BT pool is re-fit every cycle so a FIXED champion's ledger rating
// drifts over time (observed ~40 Elo across two days, champion unchanged) — so the stored absElo is
// not comparable across cycles and is re-anchored onto today's ledger at read time. The ledger map,
// the re-anchoring, and the per-track "best cycle" all live in experiment-registry.mjs, shared with
// train:experiments/suggestRecipes so the fix stays in lockstep across the reports.
const ledgerElo = ledgerBestByVersion(loopDir);
const champHashNow = weightsHash(championFile);
const champEloNow = champHashNow !== '?' && ledgerElo.has(champHashNow) ? ledgerElo.get(champHashNow) : null;

// Expected score (0..1) of a candidate at absolute Elo `absElo` vs the CURRENT champion — the
// standard logistic curve, so it reads in win-rate units like a gate score but stays comparable
// across cycles (the moving-champion staleness is gone: both sides sit on the ledger scale).
// Null when the champion isn't placed on the ledger or the candidate has no absElo.
function estScoreVsChamp(absElo) {
  if (champEloNow == null || !Number.isFinite(absElo)) return null;
  return 1 / (1 + 10 ** ((champEloNow - absElo) / 400));
}
const estPct = (absElo) => {
  const s = estScoreVsChamp(absElo);
  return s == null ? null : `${(s * 100).toFixed(0)}%`;
};

// Display shortening of a hidden spec: a run of identical layer sizes collapses, so
// "256,16,16,16,16,16,16,16,16,16,16,16" reads as "256,16×11". The deep repetitive shapes the
// suggester proposes now are long enough to wreck any column they sit in. Same notation as the
// registry's compactSlug (which does this for a slug's `h…` prefix); display only.
function compactHidden(hidden) {
  const runsOf = [];
  for (const n of String(hidden).split(',')) {
    const last = runsOf[runsOf.length - 1];
    if (last && last.n === n) last.k += 1;
    else runsOf.push({ n, k: 1 });
  }
  return runsOf.map(({ n, k }) => (k > 1 ? `${n}×${k}` : n)).join(',');
}
const archLabel = (hidden) => `h[${compactHidden(hidden)}]`;

// The recipes a run trained, in order, as display labels. Consecutive spans that are the same
// recipe — a warm relaunch picking the same track back up — collapse to one entry, so the arrow
// chain shows where the loop actually CHANGED recipe rather than one arrow per launch. Keyed by
// track id (the recipe's identity, not its shape) so a rotation between two same-shaped recipes
// that differ only in λ or filters still shows as a step.
// Names of the recipes that promoted on a run but are NOT the one it ended on — the credit a
// launch-line-only report gives to the wrong shape. A rotation can land on the same architecture
// with different knobs (λ, filters), so a name that would collide with the current recipe's is
// disambiguated by track id. Empty when the run's promotions all belong to its current recipe.
function promotingArchsElsewhere(view) {
  const tail = view.spans[view.spans.length - 1];
  const names = view.spans
    .filter((s) => s !== tail && s.cycles.some((c) => c.promoted))
    .map((s) => (archLabel(s.hidden) === archLabel(tail.hidden) && s.trackId
      ? `${archLabel(s.hidden)} [${s.trackId}]`
      : archLabel(s.hidden)));
  return [...new Set(names)];
}

function archChain(view) {
  const out = [];
  let prevKey = null;
  for (const s of view.spans) {
    const key = s.trackId || `${s.hidden}|${s.lambda}`;
    if (key === prevKey) continue;
    // A rotation can keep the shape and change only the knobs, which would print as "h[64] → h[64]"
    // and read as a no-op. Name the track when the label alone doesn't show the step.
    const label = archLabel(s.hidden);
    out.push(out.length && out[out.length - 1].startsWith(label) && s.trackId
      ? `${label} [${s.trackId}]` : label);
    prevKey = key;
  }
  return out;
}

// A rotated-to span's slug isn't in the log — the rotation line records the track id and the
// recipe label, not the slug — so read it from the track's own immutable recipe.json.
function trackSlug(id) {
  try {
    return JSON.parse(readFileSync(join(trackDir(loopDir, id), 'recipe.json'), 'utf8')).slug || id;
  } catch { return id; }
}

// The track-history entries a span produced. Matched by TIME WINDOW rather than by the per-track
// run number: a rotation doesn't log one, and a span is a contiguous stretch of wall-clock anyway.
// The window is [spanStart, nextSpanStart) and the history `ts` is written seconds after the cycle
// line it belongs to (recordCycle runs at the end of a cycle, the next rotation at the start of the
// following one), so every entry lands in the span that produced it.
function spanHistory(span) {
  if (!span.trackId) return []; // pre-registry launch
  return readHistory(trackDir(loopDir, span.trackId))
    .filter((h) => h?.ts && h.ts >= span.startTs && (!span.endTs || h.ts < span.endTs));
}

// The highest-absElo cycle a span produced, ranked by RE-ANCHORED absElo (today's-ledger scale, not
// the drift-prone stored snapshot) so it stays comparable across cycles. Best-shaped object or null.
function spanBestAbs(span) {
  let best = null;
  for (const h of spanHistory(span)) {
    const a = reAnchoredAbsElo(h, ledgerElo);
    if (!Number.isFinite(a) || (best && a <= best.absElo)) continue;
    best = { absElo: a, score: h.score, sprt: h.sprt, run: h.run, cycle: h.cycle, ts: h.ts, hash: h.hash ?? null };
  }
  return best;
}

// The best cycle a merged run produced, across EVERY architecture it trained — a run that rotated
// mid-way earned its best on whichever span produced it, which is not necessarily the one it
// launched on. Returns { …best, span } or null (pre-registry / no absElo yet).
function bestAbsCycle(view) {
  let best = null;
  for (const span of view.spans) {
    const b = spanBestAbs(span);
    if (b && (!best || b.absElo > best.absElo)) best = { ...b, span };
  }
  return best;
}

// Least-squares slope of a numeric series (Elo points per cycle), for "climbing vs flat".
function slope(ys) {
  const n = ys.length;
  if (n < 2) return 0;
  const xs = ys.map((_, i) => i);
  const mx = (n - 1) / 2, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  return sxx ? sxy / sxx : 0;
}
const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;

// The cross-run absolute-Elo trajectory for ONE experiment track — the one a given span trains.
// Unlike the gate score (which is measured against a champion that strengthens over time, so it
// can't be compared cycle-to-cycle), the track's per-cycle `absElo` — championLedgerElo + gate edge
// — is a stable yardstick: a candidate can be steadily improving in absolute Elo while still losing
// the gate, which is exactly the shape of a new-architecture bootstrap (the 'Mona' story). It is
// per-TRACK and never per-run, because a run that rotated trained several unrelated nets. Returns
// null when the span predates the registry or its track has no absElo yet.
function trackTrajectory(span) {
  if (!span?.trackId) return null;
  const hist = readHistory(trackDir(loopDir, span.trackId));
  const abs = hist.map((h) => reAnchoredAbsElo(h, ledgerElo)).filter(Number.isFinite); // drift removed
  if (abs.length < 2) return null;
  return {
    slug: trackSlug(span.trackId),
    id: span.trackId,
    cycles: hist.length,
    runs: new Set(hist.map((h) => h.run)).size,
    promotions: hist.filter((h) => h.promoted).length,
    firstAbs: abs[0],
    latestAbs: abs[abs.length - 1],
    bestAbs: Math.max(...abs),
    absSlope: slope(abs), // whole-track Elo/cycle
    recentAbsSlope: slope(abs.slice(-8)), // recent-window Elo/cycle
  };
}
// Log timestamps are written in UTC (train-loop's stamp() uses toISOString). Date.parse on a
// no-offset datetime treats it as LOCAL, which on a machine behind UTC pushes the time into the
// future (negative "ago") — so parse it explicitly as UTC, then render local for display.
const parseLogTs = (ts) => new Date(ts.replace(' ', 'T') + 'Z');
const fmtLocal = (d) => {
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
};
const pad = (s, w) => String(s).padEnd(w);
const padL = (s, w) => String(s).padStart(w);
const signed = (n) => (n >= 0 ? '+' : '') + n;

// --- Detailed cycle table for a (merged) run. --------------------------------------------
// One block per SPAN, separated by what started it: a warm relaunch picking the chain back up, or
// a `--rotate` switch onto a different recipe. Cycle numbers are the loop's own track-cumulative
// ones and therefore restart when a rotation moves to another track — that reset is real (it is a
// different net's first cycle) and the separator above it says so. Within a track a logged number
// that moves FORWARD is trusted as-is, matching what the console showed live, while one that resets
// (logs from before the loop numbered cycles per track) is renumbered to continue that track.
function printCycles(view) {
  if (!view.cycles.length) { console.log('  (no completed cycles)'); return; }
  console.log(`  ${pad('cyc', 5)}${pad('score', 8)}${pad('Elo', 6)}${pad('SPRT', 14)}${pad('games', 7)}${pad('took', 9)}lineage`);
  const lastByTrack = new Map();
  view.spans.forEach((span, i) => {
    if (i > 0 && span.kind === 'rotate') {
      console.log(`  ── ↻ rotated to ${archLabel(span.hidden)}${span.trackId ? ` · track ${span.trackId}` : ''}`
        + `${span.cycles.length ? '' : ' (no completed cycles yet)'} ──`);
      if (span.why) console.log(`     ${span.why}`);
    } else if (i > 0) {
      console.log(`  ── resumed ${fmtLocal(parseLogTs(span.startTs))}${span.cycles.length ? '' : ' (no completed cycles)'} ──`);
    }
    for (const c of span.cycles) {
      const key = span.trackId || 'legacy';
      const k = Math.max((lastByTrack.get(key) || 0) + 1, c.n);
      lastByTrack.set(key, k);
      const mark = c.promoted ? '✓ ' : '  ';
      console.log('  ' + mark + pad(k, 3)
        + pad(Number.isFinite(c.score) ? c.score.toFixed(1) + '%' : '?', 8)
        + pad(Number.isFinite(c.elo) ? signed(c.elo) : '?', 6)
        + pad(c.sprt, 14)
        + pad(c.games ?? '?', 7)
        + pad(c.dur, 9)
        + c.tail);
    }
  });
}

// --- Per-architecture summary for a run that rotated. ------------------------------------
// The one thing a single cycle table cannot show: which SHAPE each stretch of cycles belongs to,
// and how each of them did. Ranked by nothing — it's chronological, because the story is "the loop
// tried these in this order, and here is where it is now".
function printSpans(view) {
  const rows = view.spans.map((span) => {
    const best = spanBestAbs(span);
    const est = best ? estPct(best.absElo) : null;
    return {
      arch: archLabel(span.hidden),
      track: span.trackId ? `track ${span.trackId}` : 'pre-registry',
      cyc: `${span.cycles.length} cyc`,
      prom: span.cycles.filter((c) => c.promoted).length,
      best: best ? `best ${best.absElo.toFixed(0)}E${est ? ` ~${est}` : ''}` : 'best —',
    };
  });
  const w = (k) => Math.max(...rows.map((r) => String(r[k]).length));
  const [aw, tw, cw, bw] = [w('arch'), w('track'), w('cyc'), w('best')];
  console.log(`\n  Recipes this run trained (--rotate switched them mid-run, newest last):`);
  rows.forEach((r, i) => {
    const last = i === rows.length - 1;
    console.log(`    ${pad(r.arch, aw)}  ${pad(r.track, tw)}  ${padL(r.cyc, cw)}  ${pad(r.best, bw)}`
      + `${r.prom ? `  ${r.prom} promotion(s) ✓` : ''}${last ? '  ← training now' : ''}`);
  });
}

// --- Output. ----------------------------------------------------------------------------
console.log(`\nAposChess train:loop progress — ${relish(logFile)}`);
if (champHidden) console.log(`Current champion: arch [${champArch.join(',')}]  (hidden [${champHidden}])`
  + (champEloNow != null ? `  — ledger Elo ${champEloNow.toFixed(0)} (the anchor for “% vs champ” below)` : ''));

const latest = views[views.length - 1];
const running = isLatestRunning(latest.segs[latest.segs.length - 1]);

// The span the run is on NOW: what `config` describes, what the trajectory and the read are about.
// Everything the loop is currently doing belongs to this recipe, not to the one on the start line.
const latestSpan = latest.spans[latest.spans.length - 1];

console.log(`\n=== Latest run — started ${fmtLocal(parseLogTs(latest.start))}`
  + `${latest.segs.length > 1 ? `, resumed ${latest.segs.length - 1}× (warm same-recipe relaunches merged)` : ''}`
  + `${latest.rotations ? `, rotated ${latest.rotations}× (recipe changed mid-run)` : ''}`
  + `${running ? '  (RUNNING / not yet stopped)' : `  (stopped: ${latest.promotions} promotion(s))`} ===`);
console.log(`  config: hidden=[${latest.hidden}]  gate ${latest.gateGames}g@d${latest.gateDepth} SPRT(0,${latest.elo1})`
  + `  λ=${latest.lambda ?? '?'}  batch ${latest.batch ?? '?'}  ${latest.start_kind} start`
  + `${latest.rotate ? `  rotate ${latest.rotate}` : ''}`);
if (latest.rotations) {
  printWrapped(`this run has trained ${latest.rotations + 1} recipes — `
    + archChain(latest).join(' → ')
    + `. The config above is the CURRENT one (adopted ${fmtLocal(parseLogTs(latestSpan.startTs))}); `
    + 'every per-recipe number below is scoped to it.', '  note:', '    ');
}
const drifted = ['batch', 'gateGames', 'gateDepth', 'elo1']
  .filter((key) => new Set(latest.segs.map((r) => String(r[key]))).size > 1);
if (drifted.length) console.log(`  note: ${drifted.join(', ')} changed between relaunches — config above is the latest launch's.`);
if (latest.lineageDiscarded) console.log(`  note: lineage discarded at start (${latest.lineageDiscarded}) — accumulation restarted from scratch.`);
console.log('');
printCycles(latest);
if (latest.rotations) printSpans(latest);

// Track trajectory: the cross-run absolute-Elo climb the CURRENT recipe is contributing to. This is
// the real progress yardstick (see the 'Mona' note at the top) — a run can look flat/losing at the
// gate while its track's absolute Elo climbs steadily toward overtaking the champion.
const latestTraj = trackTrajectory(latestSpan);
if (latestTraj) {
  const t = latestTraj;
  const dir = t.absSlope > 1.0 ? 'climbing' : t.absSlope < -1.0 ? 'falling' : 'flat';
  const estBest = estPct(t.bestAbs);
  console.log(`\n  Track [${t.id}] ${compactSlug(t.slug)} — ${t.runs} run(s), ${t.cycles} cycle(s) accumulated:`);
  console.log(`    absolute Elo ${t.firstAbs.toFixed(0)} → ${t.latestAbs.toFixed(0)} `
    + `(best ${t.bestAbs.toFixed(0)}${estBest ? `, ~${estBest} vs current champion` : ''}), `
    + `${signed(+t.absSlope.toFixed(1))} Elo/cycle overall, ${dir}.`);
  console.log('    Absolute Elo is the real signal for a warm-starting track — the gate edge above can stay');
  console.log('    negative for dozens of cycles while this climbs (that is how \'Mona\' promoted).');
}

// Per-run read: is the candidate trending toward the gate, and can it even warm-start?
const reads = readRun(latest);
console.log('\n  Read:');
for (const r of reads) console.log(`    • ${r}`);

// --- Compact history of recent (merged) runs. --------------------------------------------
if (views.length > 1) {
  console.log(`\n=== Run history (last ${Math.min(histN, views.length)} of ${views.length}; warm same-recipe relaunches merged) ===`);
  const shown = views.slice(-histN);
  // The shape column names the recipe the run ENDED on (`↻N` = it got there after N mid-run
  // rotations), because that is the one its lineage and its next relaunch continue from. The
  // architectures it passed through on the way are in the run's own `--all` detail, not here.
  const archCol = (view) => archLabel(view.hidden);
  const rotCol = (view) => (view.rotations ? `↻${view.rotations}` : '');
  // Rank by best absolute Elo (comparable across cycles) and gloss it as an estimated score vs the
  // CURRENT champion (the champion line above spells out what the % is against, so keep it terse
  // here). Searched across every span, so a run that peaked before rotating still shows that peak.
  // Pre-registry runs have no absElo — fall back to the era-local gate %, flagged stale so it isn't
  // read against today's stronger champion.
  const bestCol = (view) => {
    const bc = bestAbsCycle(view);
    if (bc) {
      const est = estPct(bc.absElo);
      return `best ${bc.absElo.toFixed(0)}E${est ? ` ~${est}` : ''}`;
    }
    const scores = view.cycles.map((c) => c.score).filter(Number.isFinite);
    return 'best ' + (scores.length ? Math.max(...scores).toFixed(1) + '% stale' : '—');
  };
  // A promotion belongs to the recipe that won the gate, which a rotation can make a different one
  // from both the launch's and the row's shape — name it rather than letting the row imply it.
  const promCol = (view) => {
    if (!view.promotions) return '0 prom';
    const elsewhere = promotingArchsElsewhere(view);
    return `${view.promotions} prom${elsewhere.length ? ` (on ${elsewhere.join(', ')})` : ''}`;
  };
  const archW = Math.max(...shown.map((v) => archCol(v).length));
  const rotW = Math.max(...shown.map((v) => rotCol(v).length));
  const bestW = Math.max(...shown.map((v) => bestCol(v).length));
  const kindW = Math.max(...shown.map((v) => String(v.start_kind).length));
  for (const view of shown) {
    const state = view === latest && running ? 'running' : promCol(view);
    console.log(`  ${pad(fmtLocal(parseLogTs(view.start)), 20)} ${pad(archCol(view), archW)} ${pad(rotCol(view), rotW)} `
      + `${pad(view.start_kind, kindW)} `
      + `${padL(view.segs.length, 2)}×  ${padL(view.cycles.length, 3)}cyc  ${pad(bestCol(view), bestW)}  ${state}`);
  }
  if (detailAll) {
    for (const view of views.slice(0, -1)) {
      console.log(`\n--- run ${fmtLocal(parseLogTs(view.start))} (`
        + archChain(view).join(' → ')
        + `, ${view.start_kind}${view.segs.length > 1 ? `, ${view.segs.length} launches` : ''}) ---`);
      printCycles(view);
      if (view.rotations) printSpans(view);
    }
  }
}

// --- Overall: promotions and how long since the last one. -------------------------------
const allCycles = runs.flatMap((r) => r.cycles);
const totalProm = runs.reduce((a, r) => a + r.promotions, 0);
let sincePromo = 0, lastPromo = null;
for (let i = allCycles.length - 1; i >= 0; i--) {
  if (allCycles[i].promoted) { lastPromo = allCycles[i]; break; }
  sincePromo++;
}
console.log('\n=== Overall ===');
console.log(`  ${allCycles.length} cycle(s) across ${views.length} run(s)`
  + `${runs.length !== views.length ? ` (${runs.length} loop launches — warm same-recipe relaunches merged)` : ''};  ${totalProm} promotion(s).`);
if (lastPromo) {
  const promoAt = parseLogTs(lastPromo.ts);
  const ago = (Date.now() - promoAt.getTime()) / 1000;
  console.log(`  Last promotion: ${fmtLocal(promoAt)} (Elo ${signed(lastPromo.elo)}) — ${fmtDur(ago)} ago, ${sincePromo} cycle(s) since.`);
} else {
  console.log('  No promotions yet in this log.');
}
console.log('');

// When the champion has stalled (many cycles since the last promotion, or none yet), point at
// the experiment registry for concrete next moves: promising-but-stalled recipes worth reviving
// (they warm-start from a saved best) and architectures with no track yet. Read-only, pulled
// from training/data/loop/experiments — the full view is `npm run train:experiments`.
const stalled = totalProm === 0 ? allCycles.length >= 3 : sincePromo >= 6;
if (stalled) {
  const sugg = suggestRecipes(loopDir, { champHidden });
  const resume = sugg.filter((s) => s.kind === 'resume').slice(0, 2);
  const fresh = sugg.filter((s) => s.kind === 'new').slice(0, 2);
  if (resume.length || fresh.length) {
    console.log('=== Stalled? Ideas to try (full list: `npm run train:experiments`) ===');
    // With --rotate on, the loop draws from this same suggester when it rotates, so these are a
    // preview of what it will pick next rather than a to-do list.
    if (latest.rotate && latest.rotate !== 'off') {
      console.log(`  (--rotate ${latest.rotate} is on — the loop picks its next recipe from this same list by itself.)`);
    }
    // Same shape as `npm run train:experiments`: compacted slug, reason wrapped rather than
    // padded, command on its own line so it stays copy-pasteable.
    for (const s of [...resume, ...fresh]) {
      const verb = s.kind === 'resume' ? 'revive' : 'try';
      printWrapped(s.reason, `  • ${verb} ${compactSlug(s.slug)} ·`, '    ');
      console.log(`      ${s.cmd}`);
    }
    console.log('');
  }
}

// Is the latest run still going? The loop appends a "stopped" line on exit; if the run has
// cycles but no stopped line AND a live pidfile exists, treat it as running.
function isLatestRunning(run) {
  if (run.stopped !== null) return false;
  const pidFile = join(loopDir, 'loop.pid');
  if (!existsSync(pidFile)) return false;
  const pid = Number(readFileSync(pidFile, 'utf8').trim());
  try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; }
}

// Heuristic continue-vs-restart read for a run, from its cycle scores + config. These are
// the same checks worth eyeballing by hand; spelled out so the report is actionable.
//
// Scoped to the run's CURRENT span — the recipe it is training now. A trend fitted across a
// rotation would average two different nets' gate scores into a number describing neither, and the
// advice at the end ("keep it running" / "try something else") is advice about a specific recipe.
function readRun(run) {
  const out = [];
  const span = run.spans[run.spans.length - 1];
  const scores = span.cycles.map((c) => c.score).filter(Number.isFinite);
  const elos = span.cycles.map((c) => c.elo).filter(Number.isFinite);
  const traj = trackTrajectory(span);
  // Is the track's absolute Elo climbing? That's the productivity signal for a below-50% run.
  const trackClimbing = traj && traj.absSlope > 1.0;
  // Does the loop change recipe on its own? If so, a track that has stopped improving is something
  // it already handles — the read should say what it will do, not tell you to do it by hand.
  const autoRotates = run.rotate && run.rotate !== 'off';

  if (run.rotations) {
    out.push(`This run rotated ${run.rotations}× (${archChain(run).join(' → ')}), so the numbers below `
      + `cover only ${archLabel(span.hidden)} — ${span.cycles.length
        ? `its ${span.cycles.length} cycle(s) since ${fmtLocal(parseLogTs(span.startTs))}`
        : `adopted ${fmtLocal(parseLogTs(span.startTs))}, no completed cycles yet`}. `
      + 'Earlier cycles in the table belong to recipes the loop has already left.');
  }

  // Shape mismatch: a candidate whose hidden ≠ champion's can't warm-start FROM THE CHAMPION.
  // Pre-registry that meant relearning from scratch every cycle (a real stall). With the
  // experiment registry it instead warm-starts from its OWN track's lineage/best and accumulates
  // — so sub-50% is the EXPECTED early shape of a bootstrapping new-arch track, not a failure.
  // (This exact case produced 'Mona': [128,64,32] bootstrapped over 50 cycles vs a [64,32,16]
  // champion, losing the gate throughout until its absolute Elo finally overtook.)
  if (champHidden && run.hidden !== '?' && run.hidden !== champHidden) {
    if (span.trackId) {
      out.push(`Candidate hidden=[${run.hidden}] ≠ champion hidden=[${champHidden}]: it can't warm-start from the `
        + `champion, but this is a registered experiment track warm-starting from its own lineage/best — so a `
        + `below-50% gate is expected while it bootstraps. Judge it by the track's absolute-Elo trend, not the gate edge.`);
    } else {
      out.push(`Candidate hidden=[${run.hidden}] ≠ champion hidden=[${champHidden}]: it can't warm-start from the `
        + `champion and has no experiment track to accumulate from, so it relearns from scratch and tends to sit well `
        + `below 50%. Run it via train:loop (which keys a track) so warm cycles accumulate, or match the champion's shape.`);
    }
  }
  if (traj) {
    const dir = traj.absSlope > 1.0 ? 'climbing' : traj.absSlope < -1.0 ? 'falling' : 'flat';
    out.push(`Track [${traj.id}]: absolute Elo ${traj.firstAbs.toFixed(0)} → ${traj.latestAbs.toFixed(0)} `
      + `(best ${traj.bestAbs.toFixed(0)}) over ${traj.cycles} cycle(s) / ${traj.runs} run(s), `
      + `${signed(+traj.absSlope.toFixed(1))} Elo/cycle — ${dir}. This, not the gate score, is the progress signal.`);
  }
  if (run.batch === 0) {
    out.push('batch 0 — no dedicated self-play generation; fresh data comes from the gate harvest '
      + '(~2000 near-champion games/cycle) + the ranked pool\'s strong-engine --play games. This is a '
      + 'supported mode (the pool is the generator), not a stall — watch the promotion trend below. '
      + 'Set --batch (e.g. 200) to add a deep champion self-play batch on top.');
  }
  if (run.start_kind === 'cold→warm') {
    out.push('cold-first start: only cycle 1 is random-init — every later cycle warm-starts from the PREVIOUS '
      + 'cycle\'s candidate, so the chain refines one fresh net rather than relearning each cycle. It still never '
      + 'inherits the champion\'s weights, so it can trail a same/larger-arch champion until it catches up.');
  } else if (run.start_kind === 'cold') {
    out.push('cold start (legacy logs, pre-"only cold on first cycle"): every cycle relearned from random init — expect low scores throughout.');
  }

  if (!scores.length) {
    out.push(run.rotations
      ? `No completed cycles yet on ${archLabel(span.hidden)} (adopted ${fmtLocal(parseLogTs(span.startTs))}) — nothing to read until it gates one.`
      : 'No completed cycles yet.');
    return out;
  }
  const recent = scores.slice(-5);
  const a = avg(recent), best = Math.max(...scores);
  const sl = slope(elos.slice(-6)); // Elo points/cycle over the recent window
  const trend = sl > 1.5 ? 'climbing' : sl < -1.5 ? 'falling' : 'flat';
  out.push(`Recent ${recent.length}-cycle avg ${a.toFixed(1)}%, best ${best.toFixed(1)}%, trend ${trend} `
    + `(${signed(+sl.toFixed(1))} Elo/cycle)${run.rotations ? ` — ${archLabel(span.hidden)} only` : ''}.`);

  const anyLineage = span.cycles.some((c) => c.tail === 'lineage+');
  const spanProms = span.cycles.filter((c) => c.promoted).length;
  if (spanProms === 0 && run.promotions > 0) {
    // The run promoted, but on a recipe it has since rotated away from — the champion it must now
    // beat is the one that earlier shape produced, which is exactly why the current one is behind.
    const from = promotingArchsElsewhere(run);
    out.push(`This run's ${run.promotions} promotion(s) came from ${from.join(', ')}, not from ${archLabel(span.hidden)} — `
      + 'the current recipe is being gated against a champion an earlier one on this same run produced.');
  }
  if (spanProms > 0) {
    out.push('Promoted this run — the champion improved. Expect the next candidates to dip (stronger target) before climbing again.');
  } else if (trackClimbing && a < 49) {
    // Below 50% at the gate BUT the track's absolute Elo is climbing — the 'Mona' pattern. Do
    // NOT call this a failed run: a bootstrapping new-arch track loses the gate for dozens of
    // cycles before overtaking. Keep it running (across restarts — the track resumes its lineage).
    out.push(`Verdict: losing the gate but the track's absolute Elo is CLIMBING (${signed(+traj.absSlope.toFixed(1))} Elo/cycle) `
      + '— productive. This is the new-architecture bootstrap pattern that produced \'Mona\'; keep it running (it '
      + 'resumes its lineage across restarts) — it overtakes the champion only once its absolute Elo passes the champion\'s.');
  } else if (a < 49 && trend !== 'climbing' && !traj) {
    out.push('Verdict: candidates are losing and not climbing — NOT productive. Restart with the champion\'s shape, warm, and a fresh data source (a deep --batch generation, or more strong-engine --rank-minutes pool play).');
  } else if (a < 49 && !trackClimbing && traj) {
    // A track that has stopped improving is a rotation trigger, not a chore: with --rotate on, the
    // loop reads this same absElo trend itself and moves on (train-loop.mjs rotationDue). Telling
    // you to "consider a different recipe" while it is already doing that is stale advice.
    out.push(`Verdict: below 50% AND the track's absolute Elo is flat/falling (${signed(+traj.absSlope.toFixed(1))} Elo/cycle) — `
      + 'this track has stopped improving, not just losing a strong gate. '
      + (autoRotates
        ? `--rotate is ${run.rotate}, so the loop rotates off it on this same signal — nothing to do unless you want to pick the next recipe yourself.`
        : 'Consider a different recipe (see the ideas below) or fresh data.'));
  } else if (best >= run.elo1 / 7 + 50 || (anyLineage && trend === 'climbing')) {
    // best% within reach of the gate, or lineage is accumulating upward
    out.push('Verdict: candidates are at/above 50% and accumulating via lineage — productive, let it keep running toward the gate.');
  } else {
    out.push('Verdict: candidates hover near 50% — marginal. Keep the warm/same-shape chain going a few more cycles, or add fresh data (a deep --batch generation, or more strong-engine --rank-minutes pool play).');
  }
  return out;
}

function relish(p) { try { return p.replace(repoDir + '\\', '').replace(repoDir + '/', ''); } catch { return p; } }

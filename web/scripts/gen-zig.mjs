// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Thin shim: build (cached) and run the native Zig self-play generator (engine/), a
// drop-in for `npm run train:gen` — same flags, one GAME-PRIMARY JSONL line per game
// (scripts/gameRecord.mjs: g/players/r/moves/v/vs) appended to the same default dataset.
// Runs with cwd = web/ so relative paths (the default
// ../training/data/selfplay.jsonl, src/nn-weights.json, --out) resolve like the JS tool.
//   npm run gen:zig -- --games=200 --depth=6 --eval=nn --openings=8 [--opening-topk=N]

import { spawnSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const engineDir = resolve(webDir, 'engine');
const bin = resolve(engineDir, 'zig-out', 'bin',
  process.platform === 'win32' ? 'apos-gen.exe' : 'apos-gen');

// Build all native artifacts (cached — near-instant if unchanged). Needs Zig 0.16.
const build = spawnSync('zig build -Doptimize=ReleaseFast',
  { cwd: engineDir, stdio: 'inherit', shell: true });
if (build.status !== 0) {
  console.error('zig build failed (is Zig 0.16 on PATH?).');
  process.exit(1);
}

const run = spawnSync(bin, process.argv.slice(2), { cwd: webDir, stdio: 'inherit' });
process.exit(run.status ?? 1);

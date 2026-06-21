// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Thin shim: build (cached) and run the native Zig match runner (engine/), which
// is a drop-in for `npm run match` — same flags, same result-file. Runs with cwd =
// web/ so relative paths (weights, --result-file) resolve exactly as the JS tools'.
//   npm run match:zig -- --games=800 --depth=4 --eval-a=nn --eval-b=nn \
//     --weights-a=… --weights-b=… --sprt --elo1=20 --result-file=…

import { spawnSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const engineDir = resolve(webDir, 'engine');
const bin = resolve(engineDir, 'zig-out', 'bin',
  process.platform === 'win32' ? 'apos-match.exe' : 'apos-match');

// Build all native artifacts (cached — near-instant if unchanged). Needs Zig 0.16.
const build = spawnSync('zig build -Doptimize=ReleaseFast',
  { cwd: engineDir, stdio: 'inherit', shell: true });
if (build.status !== 0) {
  console.error('zig build failed (is Zig 0.16 on PATH?).');
  process.exit(1);
}

const run = spawnSync(bin, process.argv.slice(2), { cwd: webDir, stdio: 'inherit' });
process.exit(run.status ?? 1);

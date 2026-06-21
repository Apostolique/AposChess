// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Build the browser search engine (web/engine → apos.wasm) and copy it into web/public/
// so Vite serves it at <base>/apos.wasm and bundles it into dist/. The Web Worker
// (src/aiWorker.js) fetches it at runtime. Commit the resulting public/apos.wasm (like the
// nets under public/nn/), so the GitHub Pages build needs no Zig toolchain — run this
// whenever the engine changes:  npm run build:wasm

import { spawnSync } from 'node:child_process';
import { copyFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const engineDir = resolve(webDir, 'engine');

const r = spawnSync('zig build wasm -Doptimize=ReleaseSmall', { cwd: engineDir, stdio: 'inherit', shell: true });
if (r.status !== 0) {
  console.error('zig build wasm failed (is Zig 0.16 on PATH?).');
  process.exit(1);
}

const src = resolve(engineDir, 'zig-out', 'bin', 'apos.wasm');
const destDir = resolve(webDir, 'public');
const dest = resolve(destDir, 'apos.wasm');
mkdirSync(destDir, { recursive: true });
copyFileSync(src, dest);
console.log(`apos.wasm -> ${dest}  (commit it; the deploy build serves it without Zig)`);

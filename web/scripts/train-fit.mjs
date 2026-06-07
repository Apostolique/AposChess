// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Train-only step: fit the neural net on the already-generated dataset and
// export web/src/nn-weights.json. This skips self-play generation, the strength
// match, and the build that the full `npm run train` pipeline runs — use it when
// you just want to (re)train on existing training/data/selfplay.jsonl.
//
// Usage (run from web/):
//   npm run train:fit -- [train.py options]   e.g. --hidden=256 --patience=20
//
// Any options are forwarded verbatim to training/train.py.

import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoDir = resolve(here, '..', '..');
const trainPy = resolve(repoDir, 'training', 'train.py');

function findPython() {
  for (const c of ['python', 'py', 'python3']) {
    const r = spawnSync(c, ['--version'], { shell: true });
    if (r.status === 0) return c;
  }
  console.error('No Python found on PATH (tried python, py, python3).\n'
    + 'Install Python and the trainer deps: pip install -r training/requirements.txt');
  process.exit(1);
}

const python = findPython();
const cmd = `${python} "${trainPy}" ${process.argv.slice(2).join(' ')}`.trim();
console.log(`> ${cmd}`);
const r = spawnSync(cmd, { stdio: 'inherit', shell: true, cwd: repoDir });
process.exit(r.status ?? 1);

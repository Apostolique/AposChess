// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Pause / resume a running train:loop from ANOTHER terminal. A long run pegs every core
// (gen + the gate default --jobs to all CPUs), so this freezes the loop's whole process
// tree to hand the machine back — then thaws it exactly where it was. Suspended threads
// burn no CPU and keep all in-memory state, so nothing is lost: the in-flight gate or
// generation simply stops counting time until you resume.
//
//   npm run train:pause     # freeze the running loop (frees all CPU)
//   npm run train:resume    # thaw it, continue exactly where it stopped
//   npm run train:status    # is a loop running? paused or active?
//
// The loop (train-loop.mjs) writes its PID to training/data/loop/loop.pid while running;
// this reads that and suspends/resumes that PID and every descendant. On Windows the freeze
// goes through ntdll (scripts/win-suspend.ps1); on POSIX it's SIGSTOP/SIGCONT per PID. A
// PAUSED marker file next to the pidfile records the state so `status` can report it and a
// double-pause / double-resume is a harmless no-op.

import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');
const loopDir = resolve(repoDir, 'training', 'data', 'loop');
const pidFile = join(loopDir, 'loop.pid');
const pauseFlag = join(loopDir, 'PAUSED');
const isWin = process.platform === 'win32';

const action = (process.argv[2] || '').toLowerCase();
if (!['pause', 'resume', 'toggle', 'status'].includes(action)) {
  console.error('Usage: node scripts/loop-ctl.mjs <pause|resume|toggle|status>');
  process.exit(2);
}

// Is `pid` a live process? signal 0 doesn't deliver anything, it just probes existence.
function alive(pid) {
  try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; }
}

// The PID the loop recorded — or null if no loop is running (no/stale pidfile).
function loopPid() {
  if (!existsSync(pidFile)) return null;
  const pid = Number(readFileSync(pidFile, 'utf8').trim());
  return Number.isInteger(pid) && pid > 0 && alive(pid) ? pid : null;
}

// Suspend/resume the whole tree rooted at `pid`. Returns true on success.
function setFrozen(pid, freeze) {
  if (isWin) {
    const ps = resolve(here, 'win-suspend.ps1');
    const arg = ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ps,
      '-RootPid', String(pid), '-Action', freeze ? 'suspend' : 'resume'];
    // Prefer PowerShell 7 (pwsh); fall back to Windows PowerShell (powershell.exe).
    for (const exe of ['pwsh', 'powershell']) {
      const r = spawnSync(exe, arg, { encoding: 'utf8' });
      if (r.error) continue; // exe not found — try the next
      if (r.status !== 0) { console.error(r.stderr || `(${exe} exited ${r.status})`); return false; }
      const n = Number((r.stdout || '').trim());
      console.log(`${freeze ? 'Suspended' : 'Resumed'} ${Number.isFinite(n) ? n : '?'} process(es) in the loop tree.`);
      return true;
    }
    console.error('Neither pwsh nor powershell was found on PATH.');
    return false;
  }
  // POSIX: enumerate descendants via ps, then SIGSTOP/SIGCONT each (parent first to freeze).
  const tree = posixTree(pid);
  const order = freeze ? tree : [...tree].reverse();
  for (const t of order) { try { process.kill(t, freeze ? 'SIGSTOP' : 'SIGCONT'); } catch { /* gone */ } }
  console.log(`${freeze ? 'Suspended' : 'Resumed'} ${tree.length} process(es) in the loop tree.`);
  return true;
}

// POSIX descendant walk: BFS over `ps -eo pid,ppid` (parent first), root included.
function posixTree(root) {
  const r = spawnSync('ps', ['-eo', 'pid=,ppid='], { encoding: 'utf8' });
  const kids = new Map();
  for (const line of (r.stdout || '').trim().split('\n')) {
    const [pid, ppid] = line.trim().split(/\s+/).map(Number);
    if (!kids.has(ppid)) kids.set(ppid, []);
    kids.get(ppid).push(pid);
  }
  const out = [], seen = new Set(), q = [root];
  while (q.length) {
    const cur = q.shift();
    if (seen.has(cur)) continue;
    seen.add(cur); out.push(cur);
    for (const c of kids.get(cur) || []) q.push(c);
  }
  return out;
}

const pid = loopPid();

if (action === 'status') {
  if (!pid) { console.log('No train:loop is running (no live loop.pid).'); process.exit(0); }
  console.log(existsSync(pauseFlag) ? `train:loop (pid ${pid}) is PAUSED.` : `train:loop (pid ${pid}) is running.`);
  process.exit(0);
}

if (!pid) {
  console.error('No train:loop is running (no live loop.pid). Start it with `npm run train:loop`.');
  process.exit(1);
}

const paused = existsSync(pauseFlag);
const wantPause = action === 'pause' || (action === 'toggle' && !paused);

if (wantPause && paused) { console.log(`Already paused (pid ${pid}).`); process.exit(0); }
if (!wantPause && !paused) { console.log(`Not paused (pid ${pid}); nothing to resume.`); process.exit(0); }

if (!setFrozen(pid, wantPause)) process.exit(1);
if (wantPause) writeFileSync(pauseFlag, `${new Date().toISOString()} pid ${pid}\n`);
else rmSync(pauseFlag, { force: true });
console.log(wantPause
  ? 'Paused. CPU is freed; run `npm run train:resume` to continue where it stopped.'
  : 'Resumed.');

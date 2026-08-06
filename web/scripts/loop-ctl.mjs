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
//   npm run train:stop      # end it after the step it's running now
//   npm run train:stop -- --now   # kill the tree immediately, losing that step
//
// The loop (train-loop.mjs) writes its PID to training/data/loop/loop.pid while running;
// this reads that and suspends/resumes that PID and every descendant. On Windows the freeze
// goes through ntdll (scripts/win-suspend.ps1); on POSIX it's SIGSTOP/SIGCONT per PID. A
// PAUSED marker file next to the pidfile records the state so `status` can report it and a
// double-pause / double-resume is a harmless no-op.
//
// `stop` exists because Ctrl-C in the loop's own terminal is not reliable: the loop spends
// nearly all of a cycle blocked in spawnSync, where its SIGINT handler can't run, so it only
// infers the interrupt from how the child died — and a child that finished normally right as
// you pressed it leaves the loop rolling into the next step. The graceful form writes a STOP
// marker the loop checks between steps, which always lands, though it waits out whatever is
// running (a 2000-game gate can be hours). `--now` kills the process tree instead.

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
const stopFlag = join(loopDir, 'STOP');
const isWin = process.platform === 'win32';

const argv = process.argv.slice(2);
const action = (argv[0] || '').toLowerCase();
const killNow = argv.includes('--now');
if (!['pause', 'resume', 'toggle', 'status', 'stop'].includes(action)) {
  console.error('Usage: node scripts/loop-ctl.mjs <pause|resume|toggle|status|stop [--now]>');
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

// Kill `pid` and every descendant. Windows has no process groups to signal, so taskkill /T
// walks the tree; POSIX gets SIGKILL per pid, root first so it can't spawn another step
// while we're working down the list.
function killTree(pid) {
  if (isWin) {
    const r = spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], { encoding: 'utf8' });
    if (r.status !== 0 && alive(pid)) { console.error(r.stderr || r.stdout || '(taskkill failed)'); return false; }
    return true;
  }
  for (const t of posixTree(pid)) { try { process.kill(t, 'SIGKILL'); } catch { /* already gone */ } }
  return true;
}

// Sleep synchronously (this script is a one-shot with no event loop to yield to).
const napMs = (ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);

const pid = loopPid();

if (action === 'status') {
  if (!pid) {
    console.log('No train:loop is running (no live loop.pid).');
    process.exit(0);
  }
  const state = existsSync(pauseFlag) ? 'is PAUSED' : 'is running';
  const pending = existsSync(stopFlag) ? ' — stop requested, it ends after the current step' : '';
  console.log(`train:loop (pid ${pid}) ${state}${pending}.`);
  process.exit(0);
}

if (action === 'stop') {
  if (!pid) {
    // Nothing to stop. Clear a stale marker/pidfile anyway so the next launch starts clean
    // (a hard-killed loop never gets to run its own exit cleanup).
    rmSync(stopFlag, { force: true });
    rmSync(pauseFlag, { force: true });
    rmSync(pidFile, { force: true });
    console.log('No train:loop is running. Cleared any stale loop.pid / PAUSED / STOP markers.');
    process.exit(0);
  }
  if (killNow) {
    if (!killTree(pid)) process.exit(1);
    for (let i = 0; i < 20 && alive(pid); i++) napMs(100);
    if (alive(pid)) { console.error(`pid ${pid} is still alive after taskkill; stop it by hand.`); process.exit(1); }
    rmSync(stopFlag, { force: true });
    rmSync(pauseFlag, { force: true });
    rmSync(pidFile, { force: true });
    console.log(`Killed the loop tree (pid ${pid}). The in-flight step is lost; the dataset and the\n`
      + 'experiment track are unaffected (both are written incrementally / atomically), so a\n'
      + 'relaunch resumes the track at the next cycle.');
    process.exit(0);
  }
  if (existsSync(pauseFlag)) {
    // A suspended loop executes nothing, so it can never notice the marker.
    console.error(`train:loop (pid ${pid}) is PAUSED, so it can't see a stop request.\n`
      + 'Run `npm run train:resume` first, or `npm run train:stop -- --now` to kill it outright.');
    process.exit(1);
  }
  if (existsSync(stopFlag)) { console.log(`Stop already requested (pid ${pid}).`); process.exit(0); }
  writeFileSync(stopFlag, `${new Date().toISOString()} requested\n`);
  console.log(`Stop requested (pid ${pid}). The loop ends after the step it's running now, so a\n`
    + 'gate match still plays itself out. Watch it with `npm run train:status`, or use\n'
    + '`npm run train:stop -- --now` to kill it immediately instead.');
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

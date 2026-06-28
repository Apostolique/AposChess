// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// A graceful "stop early" control shared by the long-running CLI tools. Press a
// key (q / Esc / Enter / Space) or Ctrl-C and the tool stops scheduling NEW work,
// then finalizes what it already has so its output stays VALID — e.g. refresh-v
// copies the remaining lines through and still does its atomic rename, the match
// runner reports the games already played. This is a clean early finish, NOT a
// kill: the data is left consistent.
//
// installStop(onStop) calls onStop() exactly once, the first time a stop is
// requested, and returns { requested, dispose }. Call dispose() when the tool
// finishes normally so the terminal's raw mode is restored and stdin stops
// holding the event loop open. A SECOND Ctrl-C (or ctrl-c keypress) force-exits,
// so a hung finalize can still be escaped.
//
// Keypresses are only captured for a TOP-LEVEL interactive run. When a tool is
// spawned by an orchestrator (train:loop, rank:pool), the orchestrator sets
// APOS_CHILD in the child's env and the keypress capture is disabled there — only
// one process can own the TTY's raw mode, and raw mode would also swallow the
// Ctrl-C the orchestrator relies on. In that case we fall back to a plain SIGINT
// handler (still a graceful stop), so the orchestrator's stop semantics survive.

import readline from 'node:readline';

// Whether a top-level interactive run can capture single keypresses (vs. an
// orchestrated child / non-TTY, which only gets the SIGINT fallback).
const keysAvailable = !process.env.APOS_CHILD && !!(process.stdin.isTTY && process.stdin.setRawMode);

// Print the startup-banner hint — but only when the keypress button is actually
// live, so orchestrated children (train:loop, rank matchups) don't tell the user
// to press a key that won't reach them.
export function printStopHint() {
  if (keysAvailable) console.log('  Press q (or Ctrl-C) to stop early — output is finalized cleanly.');
}

// Exit code an orchestrated child uses to tell its parent "I stopped because of a
// Ctrl-C, after draining cleanly" — distinct from 0 (finished normally) and from a
// real non-zero failure, so the parent (train:loop) can end the loop instead of
// either ignoring the stop or logging a phantom crash. 130 is the conventional
// SIGINT exit code. Set as process.exitCode (not process.exit) so the tool's normal
// finalize still runs and its output stays valid; tools that exit explicitly on the
// stop path use `process.exit()` (no arg) so they inherit it.
export const STOP_EXIT_CODE = 130;

export function installStop(onStop, { keys = true } = {}) {
  let requested = false;
  let disposed = false;
  const stdin = process.stdin;
  // Only grab the keyboard for a top-level interactive TTY run (see header).
  const useKeys = keys && keysAvailable;

  function fire(force) {
    if (requested) { if (force) process.exit(STOP_EXIT_CODE); return; } // second Ctrl-C: force quit
    requested = true;
    // Orchestrated child: flag the eventual clean exit as "stopped by interrupt" so the
    // parent can distinguish a Ctrl-C drain from a normal finish. Top-level interactive
    // runs keep exiting 0 — no spurious "exit code 130" from npm when the user presses q.
    if (process.env.APOS_CHILD) process.exitCode = STOP_EXIT_CODE;
    onStop();
  }

  function onKeypress(_str, key) {
    if (!key) return;
    if (key.ctrl && key.name === 'c') { fire(true); return; } // raw mode suppresses the default SIGINT
    if (['q', 'escape', 'return', 'space'].includes(key.name)) fire(false);
  }
  function onSigint() { fire(true); }

  if (useKeys) {
    readline.emitKeypressEvents(stdin);
    stdin.setRawMode(true);
    stdin.resume();
    stdin.unref?.();           // don't keep the process alive on stdin alone
    stdin.on('keypress', onKeypress);
  }
  // Always handle SIGINT too: it's the path for non-TTY / orchestrated runs, and a
  // platform where Ctrl-C still arrives as a signal. The `requested` guard makes the
  // double-registration harmless. Having a listener also stops Node from hard-exiting
  // on Ctrl-C, so the tool can finish its clean finalize.
  process.on('SIGINT', onSigint);
  // Last-resort terminal restore if the process exits without disposing.
  process.on('exit', () => { if (useKeys && !disposed) { try { stdin.setRawMode(false); } catch { /* not a TTY */ } } });

  function dispose() {
    if (disposed) return;
    disposed = true;
    process.removeListener('SIGINT', onSigint);
    if (useKeys) {
      stdin.removeListener('keypress', onKeypress);
      try { stdin.setRawMode(false); } catch { /* not a TTY anymore */ }
      stdin.pause();
    }
  }

  return { get requested() { return requested; }, dispose };
}

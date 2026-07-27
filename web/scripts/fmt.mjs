// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Shared console formatting for the Node CLI scripts, so every tool reports
// durations, counts, sizes, and live progress the same way.

// "45s", "3m 02s", "1h 04m" — pick the unit that reads naturally at that length.
export function fmtDur(secs) {
  secs = Math.round(secs);
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  if (m < 60) return `${m}m ${String(secs % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

// Thousands separators for counts — "2,284,567" instead of "2284567".
export const fmtNum = (n) => Math.round(n).toLocaleString('en-US');

export const fmtMB = (bytes) => (bytes / 1e6).toFixed(1) + ' MB';

// Usable line width, falling back to 100 columns when stdout isn't a TTY (piped into a pager,
// redirected to a log). Floored so a very narrow terminal can't wrap prose to shreds.
export const termWidth = () => Math.max(72, process.stdout.columns || 100);

// Word-wrap `text`, opening the first line with `first` and indenting continuations by `hang`.
// For a label followed by prose of unpredictable length — padding it into a fixed column only
// aligns until one entry outgrows the column, and then nothing lines up at all.
export function printWrapped(text, first = '', hang = '', width = termWidth()) {
  let line = first;
  let empty = first.trim() === '';
  for (const word of String(text).split(/\s+/).filter(Boolean)) {
    if (!empty && line.length + 1 + word.length > width) {
      console.log(line);
      line = hang;
      empty = true;
    }
    line += (empty ? '' : ' ') + word;
    empty = false;
  }
  if (!empty) console.log(line);
}

// A single in-place status line (carriage-return repaint). update() redraws the
// line (padding over any longer previous content), clear() erases it so permanent
// lines can be printed without leftover characters.
export function liveStatus(stream = process.stdout) {
  let len = 0;
  return {
    update(s) { stream.write('\r' + s.padEnd(len)); len = s.length; },
    clear() { if (len) { stream.write('\r' + ' '.repeat(len) + '\r'); len = 0; } },
  };
}

// Throttle for live updates: returns true at most once per `ms` (and on the
// first call), so per-line progress checks stay free.
export function everyMs(ms) {
  let next = 0;
  return () => {
    const t = Date.now();
    if (t < next) return false;
    next = t + ms;
    return true;
  };
}

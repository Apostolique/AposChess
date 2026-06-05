// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Runs the AI search off the main thread so deeper lookahead never freezes the
// UI. The page posts { seq, state, depth }; we reply with { seq, move }. The
// game state is plain data (board array + flags), so it survives structured
// cloning across the worker boundary unchanged.

import { chooseMove } from './ai.js';

self.onmessage = ({ data }) => {
  const { seq, state, depth, maxMs } = data;
  const move = chooseMove(state, depth, Math.random, maxMs);
  self.postMessage({ seq, move });
};

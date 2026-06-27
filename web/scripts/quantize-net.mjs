// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Post-hoc NNUE quantization: convert a float weights file (the generic {arch,scale,
// layers} layout nn.js reads) into the INTEGER layout ({int:true, quant:{qa,qw}, ...}),
// no retraining. The integer forward pass (nn.js compileInt / nn.zig) uses plain ReLU at
// fixed-point scale QA, so it reproduces the float net up to rounding (a few cp) — and it
// is what lets the Zig search maintain the first layer incrementally (the "U" in NNUE).
//
// Scales match training/train.py's QUANT_QA / QUANT_QW (kept in sync by hand — two short
// constants). Layer 0 (the accumulator) weights+bias scale by QA; dense layers (incl. the
// scalar head) weights by QW, biases by QW*QA (they add to a QW*QA-scaled pre-activation).
//
// Usage (from web/):
//   node scripts/quantize-net.mjs <float-net.json> <out-int-net.json>

import { readFileSync, writeFileSync } from 'node:fs';

// Scales chosen so post-hoc rounding error vs the float net is ~1cp (a coarser int8-style
// 127/64 cost ~30cp on a 4-layer plain-ReLU net feeding tanh); intermediates stay well
// under 2^53 so JS float64 stays exact. Must match training/train.py's QUANT_QA / QUANT_QW.
const QA = 1024; // activation fixed-point scale (== train.py QUANT_QA)
const QW = 1024; // dense-weight scale          (== train.py QUANT_QW)

const [inPath, outPath] = process.argv.slice(2);
if (!inPath || !outPath) {
  console.error('usage: node scripts/quantize-net.mjs <float-net.json> <out-int-net.json>');
  process.exit(2);
}

const net = JSON.parse(readFileSync(inPath, 'utf8'));
if (net.int) { console.error(`${inPath} is already quantized.`); process.exit(2); }
if (!net.arch || !net.layers) { console.error(`${inPath} is not a trained float net.`); process.exit(2); }

const qi = (arr, s) => arr.map((x) => Math.round(x * s));
const layers = net.layers.map((L, i) => (i === 0
  ? { w: qi(L.w, QA), b: qi(L.b, QA) }
  : { w: qi(L.w, QW), b: qi(L.b, QW * QA) }));

const out = { arch: net.arch, scale: net.scale, int: true, quant: { qa: QA, qw: QW }, layers };
writeFileSync(outPath, JSON.stringify(out));
console.log(`Quantized ${inPath} -> ${outPath} (QA=${QA}, QW=${QW}).`);

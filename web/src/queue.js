// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Automatic matchmaking on top of Trystero. Everyone who wants a game joins one
// fixed "queue" room and waits; the moment two unpaired players meet, they pair
// off, leave the queue, and play in a fresh private room (handed back to main.js,
// which drives it through the normal online host/join path).
//
// Why a queue and not a presence lobby: the queue is *self-draining* — it only ever
// holds people currently waiting, and pairs drain out immediately — so the room
// stays tiny and the full WebRTC mesh a shared lobby would create never builds up.
//
// Pairing has to be safe without any server to arbitrate, under eventually-
// consistent presence. Two mechanisms make it race-free:
//   1. Symmetry break — every client has a stable `selfId`; only the LOWER id of a
//      would-be pair sends the proposal, so two clients never propose to each other.
//   2. Confirm-before-leaving — a 2-message propose/accept handshake. You only leave
//      the queue on a confirmed accept, and you accept at most one proposal, so even
//      if two clients' views disagree nobody leaves into a phantom match.
// The proposer mints the private room code and ships it in the proposal, so both
// sides land in the same room with no extra round-trips. Lower id hosts (assigns
// colours), matching the rest of the app's host/joiner split.
import { joinRoom, selfId } from 'trystero';
import { APP_ID, RTC_CONFIG, makeCode } from './online.js';

const QUEUE_ROOM = 'queue'; // the one well-known room everyone waiting shares
const PROPOSE_TIMEOUT_MS = 5000; // give up on a silent target and try someone else
const BUSY_COOLDOWN_MS = 1500;   // after a reject/timeout, skip that peer this long

// handlers: { onSearching(), onMatched({ code, isHost }), onError(message) }
// Returns { cancel() }.
export function findMatch(handlers) {
  let room = null;
  let send = null;
  let done = false;             // matched or cancelled — ignore everything after
  const peers = new Set();      // peer ids currently in the queue room
  const busy = new Set();       // peers to skip for now (rejected us / went silent)
  let outstanding = null;       // { target, code } while awaiting an accept/reject
  let proposeTimer = null;

  // The lowest-id peer we may still pair with (busy ones excluded).
  function lowestCandidate() {
    let lo = null;
    for (const p of peers) if (!busy.has(p) && (lo === null || p < lo)) lo = p;
    return lo;
  }

  // Decide whether to propose. Only the lower id of a pair initiates; everyone else
  // waits to receive a proposal. Re-run on every membership/state change.
  function evaluate() {
    if (done || outstanding) return;
    const target = lowestCandidate();
    if (target === null) return;          // no one to pair with yet
    if (selfId < target) propose(target); // we're the lower id → initiate
  }

  function propose(target) {
    const code = makeCode();
    outstanding = { target, code };
    send({ t: 'propose', code }, { target });
    proposeTimer = setTimeout(() => {     // no reply → assume gone, skip and retry
      if (done || !outstanding) return;
      cooldown(outstanding.target);
      outstanding = null;
      evaluate();
    }, PROPOSE_TIMEOUT_MS);
  }

  // Temporarily exclude a peer, then reconsider them (they may have been mid-match).
  function cooldown(id) {
    busy.add(id);
    setTimeout(() => { if (!done) { busy.delete(id); evaluate(); } }, BUSY_COOLDOWN_MS);
  }

  function matched(otherId, code) {
    if (done) return;
    done = true;
    clearTimeout(proposeTimer);
    try { room?.leave(); } catch { /* ignore */ }
    handlers.onMatched?.({ code, isHost: selfId < otherId });
  }

  function onMsg(data, meta) {
    if (done || !data || typeof data !== 'object') return;
    // Trystero ≥0.25 delivers the sender to onMessage as `{ peerId }` in the second
    // arg (older versions passed the id string directly) — accept either shape.
    const fromId = meta && typeof meta === 'object' ? meta.peerId : meta;
    if (data.t === 'propose') {
      // Already committing to someone else → decline so they retry elsewhere.
      if (outstanding) { send({ t: 'reject' }, { target: fromId }); return; }
      // Commit synchronously (so a second propose arriving in the await window is
      // ignored via the `done` guard above), but only LEAVE the queue once the
      // `accept` has actually flushed to the peer. Leaving first can tear the data
      // channel down before the accept is delivered, stranding the proposer — it
      // keeps "Searching" while we've already moved on to "Connecting".
      done = true;
      clearTimeout(proposeTimer);
      const code = data.code;
      Promise.resolve(send({ t: 'accept', code }, { target: fromId })).finally(() => {
        try { room?.leave(); } catch { /* ignore */ }
        handlers.onMatched?.({ code, isHost: selfId < fromId });
      });
    } else if (data.t === 'accept') {
      if (!outstanding || fromId !== outstanding.target) return;
      matched(fromId, outstanding.code);
    } else if (data.t === 'reject') {
      if (!outstanding || fromId !== outstanding.target) return;
      clearTimeout(proposeTimer);
      cooldown(outstanding.target);
      outstanding = null;
      evaluate();
    }
  }

  try {
    room = joinRoom({ appId: APP_ID, rtcConfig: RTC_CONFIG }, QUEUE_ROOM, {
      onJoinError: (d) => { if (!done) handlers.onError?.((d && d.error) || 'Could not reach matchmaking.'); },
    });
    const action = room.makeAction('mm');
    send = action.send;
    action.onMessage = onMsg;
    room.onPeerJoin = (id) => { if (done) return; peers.add(id); evaluate(); };
    room.onPeerLeave = (id) => {
      if (done) return;
      peers.delete(id);
      busy.delete(id);
      // If our pending partner vanished, drop the proposal and look again at once.
      if (outstanding && outstanding.target === id) { clearTimeout(proposeTimer); outstanding = null; }
      evaluate();
    };
    handlers.onSearching?.();
  } catch (err) {
    handlers.onError?.((err && err.message) || 'Could not start matchmaking.');
  }

  return {
    cancel() {
      if (done) return;
      done = true;
      clearTimeout(proposeTimer);
      try { room?.leave(); } catch { /* ignore */ }
    },
  };
}

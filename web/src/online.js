// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Peer-to-peer online play via Trystero. Trystero needs no server of ours: peers
// find each other through a public signaling network (the default `trystero`
// entry uses Nostr relays; `trystero/torrent`, `trystero/mqtt`, etc. are drop-in
// alternatives) and then talk directly over WebRTC.
//
// Trystero is room-based: both players join the same room id (our share code,
// under a fixed appId namespace) and are introduced automatically — there is no
// central id to wait for. We keep the host/join split only at this layer's API so
// main.js is unchanged: "hosting" just means we minted the code and will assign
// colours; both sides otherwise behave identically. Game messages are opaque here
// — main.js owns the move/reset/hello protocol.
import { joinRoom } from 'trystero';

export const APP_ID = 'aposchess';
// Unambiguous alphabet for the visible code (no 0/O/1/I/L).
const ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';
// Length of a generated share code; also what the join field treats as "complete".
export const CODE_LENGTH = 5;

// WebRTC can't connect on its own once a NAT/firewall is involved (and browsers
// hide local IPs behind mDNS, so even two tabs often need help). STUN lets each
// peer discover a reachable address — enough for same-network and many home
// setups. TURN relays the data when no direct path exists (strict/symmetric NAT,
// or hidden local IPs); there is no reliable zero-signup public TURN any more, so
// for robust cross-network play supply your own (Metered/Twilio/ExpressTURN all
// have free tiers). Trystero forwards `rtcConfig` straight to RTCPeerConnection.
//
// TURN credentials come from build-time env vars so they stay out of the repo —
// set them in `web/.env.local` (see `.env.example`) and rebuild. Note these are
// baked into the client bundle and therefore visible to users; that's expected
// for TURN (use a provider that scopes/expires credentials). Multiple comma-
// separated URLs sharing one credential are supported.
const turnUrls = (import.meta.env.VITE_TURN_URLS || '').split(',').map((u) => u.trim()).filter(Boolean);
export const RTC_CONFIG = {
  iceServers: [
    { urls: ['stun:stun.l.google.com:19302', 'stun:stun1.l.google.com:19302'] },
    ...(turnUrls.length
      ? [{ urls: turnUrls, username: import.meta.env.VITE_TURN_USERNAME, credential: import.meta.env.VITE_TURN_CREDENTIAL }]
      : []),
  ],
};

export function makeCode(n = CODE_LENGTH) {
  let s = '';
  for (let i = 0; i < n; i++) s += ALPHABET[Math.floor(Math.random() * ALPHABET.length)];
  return s;
}

export const normalizeCode = (code) => code.trim().toUpperCase();

// handlers: { onCode(code), onConnected(), onData(msg), onClosed(), onError(message) }
// Returns { send(obj), close(), getCode() }.
//
// Trystero ≥0.25 API: makeAction() returns a { send, onMessage } object (not a
// tuple), and onPeerJoin/onPeerLeave are assignable handlers (not methods).
function start(handlers, code) {
  const session = { room: null, action: null, peerId: null, closed: false };

  try {
    const room = joinRoom({ appId: APP_ID, rtcConfig: RTC_CONFIG }, code, {
      onJoinError: (d) => { if (!session.closed) handlers.onError?.((d && d.error) || 'Could not connect.'); },
    });
    session.room = room;

    const action = room.makeAction('msg');
    session.action = action;
    action.onMessage = (data) => { if (!session.closed) handlers.onData?.(data); };

    room.onPeerJoin = (peerId) => {
      if (session.closed || session.peerId) return; // a 1v1 game: ignore extra peers
      session.peerId = peerId;
      handlers.onConnected?.();
    };
    room.onPeerLeave = (peerId) => {
      if (session.closed || peerId !== session.peerId) return;
      // Free the slot so the room can accept a new peer on the same code. A host keeps
      // the room open (see onOnlineClosed), and onPeerJoin then fires again for the next
      // opponent; a joiner closes the session in response to onClosed instead.
      session.peerId = null;
      handlers.onClosed?.();
    };
  } catch (err) {
    handlers.onError?.((err && err.message) || 'Could not start the connection.');
  }

  return {
    send(obj) {
      if (session.action && session.peerId && !session.closed) {
        // send() returns a Promise; swallow rejections (e.g. peer just left).
        Promise.resolve(session.action.send(obj)).catch(() => {});
      }
    },
    close() { session.closed = true; try { session.room?.leave(); } catch { /* ignore */ } },
    getCode() { return code; },
  };
}

// Host a game. `code` is normally minted here, but matchmaking passes a code the
// two peers already agreed on so both join the same private room.
export function hostGame(handlers, code = makeCode()) {
  const session = start(handlers, code);
  handlers.onCode?.(code); // available immediately — no server round-trip
  return session;
}

export function joinGame(code, handlers) {
  return start(handlers, normalizeCode(code));
}

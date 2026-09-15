# Story 20.10 / Epic 20 F6 — RTX 3060 log analysis: evidence

Source: `installer_output/logs/{myvoice.log, rthook_debug.log, predownload.log}`
copied from the RTX 3060 PC (12 GiB, `device_capability=8.6`) on 2026-09-14.
Two sessions on the same card in the same sitting — a same-hardware A/B the
Commander did not plan but that the log delivered:

| session | build | geometry | inductor cache key |
|---|---|---|---|
| 19:05:06 | 2.2.0.56 (June install) | cs25 + la5 → `decode_window_frames=30` | `a7ea1bcf…` (populated in June) |
| 19:16:45 | **2.2.0.57** (this build) | cs10 + la0 → `decode_window_frames=10` | `8384a5df…` (never seen before) |

No `Task was destroyed`, no streaming-mode fallback, no compile disengage
in either session. `rthook_debug.log`: all three launches found torch libs,
CUDA redist, Triton 3.6.0 with `['amd', 'nvidia']` backends. Progressive
CSV capture was not enabled (env-gated), so the six `ttfa_*` metrics are
absent; the numbers below are log-timestamp deltas.

## 1. Warm generations — the Epic 20 result on a 3060

Request = `Starting TTS generation for:`; session opened = `Progressive
playback session opened` (first chunk has arrived at the consumer); first
dispatch = first `Dispatch chunk` line (audio handed to PyAudio).

| build | text | request → session opened | request → first dispatch |
|---|---|---|---|
| 56 | 16 chars | 2.96 s | (single chunk, no dispatch line) |
| 56 | 57 chars | 4.18 s | 6.29 s |
| 56 | 158 chars | 4.36 s | 6.17 s |
| **57** | 155 chars | **1.88 s** | **2.80 s** |
| **57** | 181 chars | **1.86 s** | **2.82 s** |

First chunk **−57 %** (4.2–4.4 s → 1.86–1.88 s); first audible audio
**−55 %** (6.2 s → 2.8 s). The RTX 5090 same-sitting measurement was
−52.8 %. The 3060's absolute first-chunk time is 2.4× the 5090's 767 ms,
in line with the cards' relative throughput.

The adaptive pre-buffer path (< 16 GiB) was observed for the first time:
`detected_vram=12.0 GiB`, `cushion_budget=2.0s`, `T_a_estimate` 3.6–12.6 s.
First dispatch carried two chunks (`dispatched_bytes=75690` = 37,290 +
38,400 — the first chunk's 555-sample edge loss is visible in the byte
count), i.e. ~1.58 s of audio was held before release, ~0.9 s after the
first chunk landed. Producer cadence was 0.68–1.0 s per 0.8 s chunk (P ≈
0.85–0.95), so τ_gapless ≈ 0.5–1.3 s ≤ budget — consistent with the
feasible-cushion regime, but the regime is not logged (see §4).

## 2. The first generation on build 57 — the defect

```
19:16:46,644  Voice clone prompt cache: hydrated 0/12 CLONED voices for tier small from disk
19:16:51,887  Compile warmup priming skipped (no_priming_prompt): BASE is resident but the
              active profile has no cached voice_clone_prompt; skipping ...; cache stays cold
19:16:58,197  Starting TTS generation for: Testing TTFA on the RTX 30 60 ...
19:16:58,200  Voice clone prompt cache miss for D:\MyVoice\voice_files\Sarira-F.wav (tier=small); computing
19:16:58,201  Unloading model: Base (Clone)                      ← reload for prompt creation
19:17:02,051  Model Base (Clone) loaded successfully in 3.65s
19:17:15,334  Voice clone prompt created successfully             ← 13.3 s
19:17:15,338  Starting TTS generation (TRUE_STREAM) ...
19:17:16,586  [QwenTTS] TRUE_STREAM geometry: chunk_size=10 lookahead=0 (carries_codec_state=True)
19:17:48,432  Progressive playback session opened                 ← ~30 s cold compile inside
```

**50.2 s request-to-audio.** Everything priming exists to absorb landed on
the user's first click because priming had no prompt to prime with.

Sarira-F is a bundled default voice: the installer ships
`Sarira-F.{wav,txt,small.pt,small.pt.meta.json}` from
`src/install_files/default_voices/` (built by
`precompute_default_embeddings.py`). On the build host the meta matches the
wav/txt exactly. The June session hydrated 1/12 — the one prompt that PC
had computed at runtime — so the 11 bundled ones had never hydrated there
either.

### Root cause, proven

`_voice_clone_prompt_meta_is_valid` rejects on `|meta_mtime − st_mtime| >
1e-3`. Inno Setup's default `TimeStampRounding=2` rounds installed files'
mtimes down to an even second. One-file test installer, compiled with the
build host's ISCC and run silently:

| directive | Sarira-F.wav installed − source | Sarira-F.txt installed − source |
|---|---|---|
| default (`TimeStampRounding=2`) | **−0.887 s** | **−1.002 s** |
| `TimeStampRounding=0` | 0.0 | 0.0 |

Both sides of the check fail on a default install; the `.pt` is treated as
stale and the lazy path recomputes.

### Fix

* `build_tools/installer.iss`: `TimeStampRounding=0` — the meta now
  round-trips exactly.
* `qwen_tts_service.py`: `_MTIME_TOLERANCE_SECONDS = 2.0` shared by the meta
  check (wav + txt) and the in-memory fingerprint check. Covers the same
  rounding class through any other copy path without weakening stale
  detection: a re-recorded wav is minutes newer, and size + sidecar + pin
  still gate.
* Tests (`test_voice_clone_prompt_cache.py::TestHydrateVoiceClonePromptCache`):
  `test_hydration_survives_installer_timestamp_rounding` shifts the
  installed files back by 1.999 s (worst case of the rounding) and requires
  a hit — **fails against the 1 ms tolerance** (verified by stashing the
  fix); `test_hydration_still_rejects_mtime_drift_beyond_rounding` shifts by
  2.5 s and requires a miss.

Existing installs: the next installer overwrites the default voices
(`ignoreversion`) with exact timestamps, so hydration recovers on upgrade
without touching user data. A user's own runtime-computed prompts were
never affected (written in place, exact mtimes).

## 3. Other observations

* Generation 4 (19:20:47, 220 chars) went `Starting TTS generation
  (streaming): chunks=3` — the SENTENCE_STREAM path, 8.0 s to first chunk,
  15.3 s to first dispatch. No fallback was logged, so this was a streaming
  mode override in Settings (the text reads "lets see what happens when i
  do sen…"). Not a defect; it does show the size of the cliff between the
  two modes on a 3060.
* Build 57's cold compile at `decode_window_frames=10` took ~30 s versus
  ~9 s for build 56's `=30` — but 56 was hitting an inductor disk cache
  populated in June (`cache=cold` is process state, not disk state). The
  next launch of 57 on that PC will hit `8384a5df…` on disk.
* Model load 4.7 s / 3.65 s on the 3060 vs 13.0 s for build 56 — build 56's
  load included the June-cache inductor warm-up; not comparable.
* Errors in the log are all pre-existing classes unrelated to Epic 20:
  monitor device 11 / virtual device −1 not present on that PC, and the
  `stop_all_playback` / `stop_all_virtual_microphone_playback` attribute
  errors on the Clear Comms stop path (19:07:03, build 56 only in this log
  but the code is unchanged — worth a look in a UI story).

## 4. Observability gap raised

`StreamingChunkBuffer.last_release_reason` is exposed for tests only. One
INFO line per session naming the regime (`producer_keeps_up`,
`gapless_feasible`, `gapless_unreachable`, `max_hold_chunks`,
`max_pre_delay`, `is_final`) would let a user log answer the F6 question
directly instead of by byte-count inference.

---

## 5. Build 58 confirmation (logs copied 20:26, same PC, same sitting)

Two launches of 2.2.0.58. `rthook_debug.log` clean (8 hook completions
across the day's launches), predownload cache-hit as before.

**Story 20.10 confirmed on both launches:**

```
20:17:52,252  Voice clone prompt cache: hydrated 12/12 CLONED voices for tier small from disk
20:17:57,611  Compile-priming Generate gate: ENGAGED
20:17:57,611  Compile priming: dispatching against the resident model Base (Clone)
20:18:07,097  Compile warmup primed cache successfully (duration=9485ms)
20:18:07,097  Compile-priming Generate gate: RELEASED

20:23:52,138  Voice clone prompt cache: hydrated 12/12 CLONED voices for tier small from disk
20:23:56,656  torch.compile + CUDA Graph engaged (decode_window_frames=10, ..., cache=warm)
20:23:57,193  Starting TTS generation (TRUE_STREAM): ... text='Hello world....'      ← priming
20:24:06,619  Compile cache hit; warm-path priming completed (duration=9438ms)
```

The bundled *quality*-tier prompt also hit on disk (2 ms) when the tier was
switched: `Voice clone prompt cache hit on disk: D:\MyVoice\voice_files\Sarira-F.quality.pt`.
The inductor cache key is unchanged between builds 57 and 58
(`8384a5df…`), so the compile artefacts carried over; `cache=cold` on the
first 58 launch is the priming marker, which build 57 never set because
priming never ran there.

**First user generation, request → first chunk → first audible audio:**

| launch | mode | note | first chunk | first audio |
|---|---|---|---|---|
| 20:17 | SENTENCE_STREAM (override left on from the earlier test) | 36 chars | 18.5 s | 18.5 s |
| 20:17 | SENTENCE_STREAM | 56 chars | 8.6 s | 8.6 s |
| 20:17 | TRUE_STREAM, first in process | 64 chars; includes the one-time CodecStateCache self-test (1.48 s) | 3.08 s | 3.98 s |
| 20:23 | TRUE_STREAM, tier just switched to quality (1.7B) | 45 chars; 17.07 s model load + cold compile for the 1.7B key inside the request | 19.2 s | 20.2 s |
| 20:23 | TRUE_STREAM, 1.7B warm | 127 chars | **1.90 s** | **2.94 s** |

Against build 57's 50.2 s first generation: the cold work now lands at
startup behind the priming gate. The 1.7B warm numbers on the 3060 match
the 0.6B ones (1.86–1.90 s first chunk) — the first-chunk floor on this
card is not talker-size-bound at cs10.

**Two follow-ups raised, not acted on:**

* Priming runs once at startup against the resident model. A **tier
  switch** unloads the model and the next generation pays the reload +
  cold compile (19.2 s above). Re-priming after a tier change — same
  `_run_compile_priming` path, same gate — would close it.
* When priming runs in SENTENCE_STREAM mode (a user override), it primes
  the batch decode path and the first TRUE_STREAM generation still pays
  the codec self-test (~1.5 s on a 3060). Priming through TRUE_STREAM
  regardless of the user's mode override would absorb it; whether that is
  correct when the user has deliberately chosen sentence mode is a product
  call.

"""Story 20.4 AC #5 follow-up - evaluate seam fixes offline, on a FIXED take.

``20-4-seam-analysis.py`` established two independent defects at every
decoder chunk boundary:

  1. A SPLICE-ALIGNMENT BUG. ``decode(N frames)`` returns ``1920*N - 555``
     samples: 1920 samples per frame plus a FIXED 555-sample convolution
     edge loss. ``StreamingDecoderWorker._decode_and_post`` computes its
     trim as ``round(lookahead * len(pcm_full)/len(chunk))``, which treats
     that fixed loss as proportional, so every posted chunk is short by
     ``555 * chunk_size/(chunk_size+lookahead)`` samples - 370 at
     chunk_size=10, 463 at 25. Cross-correlation puts the true splice point
     at exactly ``chunk_size * 1920``, so those samples are real speech,
     deleted. Measured RMS of the deleted span is at or above the
     utterance's own RMS.

  2. A CODEC-STATE MISMATCH. Consecutive chunks decode ``lookahead`` frames
     of IDENTICAL tokens, and the two decodes differ by ~35 % NRMSE
     (correlation ~0.93) because each decode starts from a cold codec
     state. Alignment cannot fix this; it is the reference implementations'
     "cache codec state across chunks" technique showing up as a quality
     defect rather than a speed one.

This script evaluates candidate fixes against the captured pcm_full, so a
sweep costs no GPU time and every variant is the SAME take - the only thing
that varies is the stitching.

Crucially, defect 1's discovery also supplies the material to attack defect
2: chunk k's pcm_full extends ``lookahead*1920 - 555`` samples PAST the
splice point (9,045 samples = 377 ms), covering exactly the audio chunk k+1
re-decodes at its head. The shipped code throws that away. A decoder-side
overlap-add can blend the two decodes across it - and unlike widening the
CONSUMER crossfade, it consumes no distinct audio at all, because both
sides of the blend are the same moment in time.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-4-seam-fix-sweep.py

Working file - gitignored under ``_bmad-output/``; force-add per
``memory/git_repo_state.md``.
"""

from __future__ import annotations

import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAWFULL = os.path.join(HERE, "20-4-seam-rawfull")
SR = 24000
SPF = 1920      # samples per codec frame (measured)
EDGE = 555      # fixed per-decode edge loss (measured)


def split(concat, lengths):
    out, i = [], 0
    for n in lengths:
        out.append(concat[i:i + n])
        i += n
    return out


# --------------------------------------------------------------------------- #
# Stitching variants
# --------------------------------------------------------------------------- #


def stitch_shipped(fulls, frames, cs, la):
    """Exactly what StreamingDecoderWorker does today."""
    parts = []
    for f, n in zip(fulls, frames):
        if n >= cs + la and la > 0:
            spt = f.size / n
            trim = int(round(la * spt))
            parts.append(f[:f.size - trim])
        else:
            parts.append(f)
    return np.concatenate(parts), [int(np.sum([p.size for p in parts[:i + 1]]))
                                   for i in range(len(parts) - 1)]


def stitch_aligned(fulls, frames, cs, la, overlap=0, shape="linear"):
    """Corrected splice point, with an optional decoder-side overlap-add.

    ``overlap=0`` is the pure alignment fix (butt splice at ``cs*SPF``).
    ``overlap=w`` blends chunk k's samples ``[cs*SPF, cs*SPF+w)`` with chunk
    k+1's ``[0, w)`` - the same moment in time decoded twice.

    Output length is independent of ``w``: each full chunk still advances
    the stream by exactly ``cs*SPF``. The crossfade costs no duration and
    no content.
    """
    max_ov = la * SPF - EDGE
    w = int(min(overlap, max_ov))
    out = []
    seams = []
    acc = 0
    for k, (f, n) in enumerate(zip(fulls, frames)):
        is_full = n >= cs + la and la > 0
        last = k == len(fulls) - 1
        if not is_full or last:
            body = f
        else:
            body = f[:cs * SPF]
        if k == 0:
            out.append(body.copy())
            acc += body.size
            continue
        prev = out[-1]
        if w > 0 and len(out) and frames[k - 1] >= cs + la:
            # Blend into the tail of what we already emitted, using the
            # PREVIOUS chunk's discarded continuation.
            prev_full = fulls[k - 1]
            tail = prev_full[cs * SPF: cs * SPF + w]
            head = body[:w]
            m = min(tail.size, head.size)
            if m > 0:
                if shape == "linear":
                    ramp = np.linspace(0.0, 1.0, m, dtype=np.float32)
                else:  # equal-power
                    ramp = np.sin(
                        np.linspace(0.0, np.pi / 2, m, dtype=np.float32)
                    )
                blended = tail[:m] * (1.0 - ramp) + head[:m] * ramp
                body = np.concatenate([blended, body[m:]])
        seams.append(acc)
        out.append(body)
        acc += body.size
    return np.concatenate(out), seams


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def seam_step_ratio(sig, pos, win=64):
    if pos <= 0 or pos >= sig.size:
        return float("nan")
    step = abs(float(sig[pos]) - float(sig[pos - 1]))
    lo, hi = max(0, pos - win), min(sig.size, pos + win)
    local = np.abs(np.diff(sig[lo:hi].astype(np.float64)))
    med = float(np.median(local)) if local.size else 0.0
    return step / med if med > 0 else float("inf")


def spectral_jump(sig, pos, n=1024):
    """Log-spectral distance between the frames either side of ``pos``.

    Insensitive to a pure amplitude step, sensitive to the timbral /
    fine-structure mismatch defect 2 produces - which is what
    ``tonal_distortion`` describes.
    """
    if pos - n < 0 or pos + n > sig.size:
        return float("nan")
    win = np.hanning(n).astype(np.float32)
    A = np.abs(np.fft.rfft(sig[pos - n:pos] * win)) + 1e-8
    B = np.abs(np.fft.rfft(sig[pos:pos + n] * win)) + 1e-8
    return float(np.mean(np.abs(20.0 * np.log10(A / B))))


def evaluate(sig, seams, rng):
    steps = [seam_step_ratio(sig, p) for p in seams]
    jumps = [spectral_jump(sig, p) for p in seams]
    ctrl_pos = rng.integers(2000, max(2001, sig.size - 2000), 300)
    c_steps = [seam_step_ratio(sig, int(x)) for x in ctrl_pos]
    c_jumps = [spectral_jump(sig, int(x)) for x in ctrl_pos]
    f = lambda v: float(np.nanmedian(v)) if len(v) else float("nan")  # noqa: E731
    return f(steps), f(jumps), f(c_steps), f(c_jumps)


def main() -> int:
    print("Story 20.4 - seam fix sweep (offline, fixed take)")
    print("samples/frame={} fixed edge loss={}  max decoder overlap "
          "available = lookahead*{} - {} = {} samples ({:.0f} ms)".format(
              SPF, EDGE, SPF, EDGE, 5 * SPF - EDGE,
              (5 * SPF - EDGE) / (SR / 1000.0)))
    print()

    widths = [0, 64, 256, 512, 1024, 2048, 4096, 9045]
    agg = {}
    for path in sorted(glob.glob(os.path.join(RAWFULL, "*.npz"))):
        d = np.load(path)
        cs, la = int(d["chunk_size"]), int(d["lookahead"])
        fulls = split(d["full_concat"], d["full_lengths"])
        frames = [int(x) for x in d["frames"]]
        if len(fulls) < 3:
            continue  # need at least two seams for a median to mean anything
        name = os.path.basename(path).replace(".npz", "")
        rng = np.random.default_rng(0)

        sig, seams = stitch_shipped(fulls, frames, cs, la)
        s, j, cs_, cj = evaluate(sig, seams, rng)
        print("{:<14} SHIPPED           step={:>7.2f} (ctrl {:>4.2f})   "
              "spectral jump={:>6.2f} dB (ctrl {:>5.2f})".format(
                  name, s, cs_, j, cj))
        agg.setdefault(("shipped", cs), []).append((s / cs_, j - cj))

        for w in widths:
            sig, seams = stitch_aligned(fulls, frames, cs, la, overlap=w)
            s, j, cs_, cj = evaluate(sig, seams, rng)
            label = "ALIGNED+OLA {:>5}".format(w) if w else "ALIGNED (no OLA)"
            print("{:<14} {:<17} step={:>7.2f} (ctrl {:>4.2f})   "
                  "spectral jump={:>6.2f} dB (ctrl {:>5.2f})".format(
                      "", label, s, cs_, j, cj))
            agg.setdefault((w, cs), []).append((s / cs_, j - cj))
        print()

    print("=" * 78)
    print("AGGREGATE - seam metric as a MULTIPLE of the same metric at "
          "non-seam positions")
    print("(1.00 = a seam is statistically indistinguishable from any other "
          "point in the audio)")
    print()
    print("{:<20} {:>22} {:>22}".format(
        "variant", "step ratio (x ctrl)", "excess spectral (dB)"))
    for cs in (25, 10):
        rows = [k for k in agg if k[1] == cs]
        if not rows:
            continue
        print("  chunk_size = {}".format(cs))
        order = ["shipped"] + widths
        for key in order:
            if (key, cs) not in agg:
                continue
            vals = agg[(key, cs)]
            st = float(np.median([v[0] for v in vals]))
            sp = float(np.median([v[1] for v in vals]))
            label = ("shipped" if key == "shipped"
                     else "aligned, no OLA" if key == 0
                     else "aligned + OLA {}".format(key))
            print("    {:<18} {:>20.2f} {:>22.2f}".format(label, st, sp))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Story 20.4 AC #5 follow-up - which seam mechanism is it, (a) or (b)?

Consecutive decoder chunks overlap by ``lookahead`` frames of TOKENS, so
their pcm_full arrays contain the same audio twice. That redundancy is what
makes the question answerable without guessing:

  1. Cross-correlate chunk k's pcm_full against chunk k+1's pcm_full. The
     lag is the true alignment. Both decodes lose the same fixed edge
     amount, so the lag should be exactly ``chunk_size * 1920`` regardless
     of how that loss is split between the head and the tail.
  2. Compare that against where the shipped code actually splices. The
     difference is DROPPED (or duplicated) audio at every seam.
  3. Measure the energy in the dropped span - silence would be harmless,
     speech is not.
  4. In the region where both chunks decode the SAME tokens, measure how
     far apart the two decodes are. That is mechanism (b) in isolation:
     the part of the seam that alignment cannot fix, because it is the
     codec producing different audio from different context.
  5. Re-stitch with the corrected trim and measure the residual seam
     discontinuity, to size what would be left for a crossfade to do.

Usage:
    python310\\python.exe _bmad-output\\implementation-artifacts\\20-4-seam-analysis.py

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
SAMPLES_PER_FRAME = 1920          # measured, not assumed - see 20-4-seam-capture.py
EDGE_LOSS = 555                   # measured fixed per-decode loss


def split(concat, lengths):
    out, i = [], 0
    for n in lengths:
        out.append(concat[i:i + n])
        i += n
    return out


def best_lag(a, b, centre, search=400):
    """Lag maximising normalised cross-correlation of b against a.

    ``a`` is chunk k's pcm_full, ``b`` is chunk k+1's. Alignments around
    ``centre`` are scored on the overlap that both cover.
    """
    # The window must fit inside ``a`` at the LARGEST lag we test, or the
    # search silently returns None for the tighter geometries (at
    # chunk_size=10 pcm_full is only 28,245 samples and the centre is
    # 19,200, leaving 8,645).
    n = min(12000, b.size, a.size - (centre + search))
    if n <= 0:
        return None, float("nan")
    seg_b = b[:n]
    nb = np.linalg.norm(seg_b)
    best, best_lag_ = -2.0, None
    for lag in range(centre - search, centre + search + 1):
        if lag < 0 or lag + n > a.size:
            continue
        seg_a = a[lag:lag + n]
        na = np.linalg.norm(seg_a)
        if na == 0 or nb == 0:
            continue
        c = float(np.dot(seg_a, seg_b) / (na * nb))
        if c > best:
            best, best_lag_ = c, lag
    return best_lag_, best


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def seam_step(sig, pos, win=64):
    """Amplitude discontinuity at ``pos`` relative to local sample-to-sample
    motion. Much greater than 1 means a step a listener hears as a click."""
    if pos <= 0 or pos >= sig.size:
        return float("nan")
    step = abs(float(sig[pos]) - float(sig[pos - 1]))
    lo, hi = max(0, pos - win), min(sig.size, pos + win)
    local = np.abs(np.diff(sig[lo:hi].astype(np.float64)))
    med = float(np.median(local)) if local.size else 0.0
    return step / med if med > 0 else float("inf")


def analyse(path):
    d = np.load(path)
    cs, la = int(d["chunk_size"]), int(d["lookahead"])
    fulls = split(d["full_concat"], d["full_lengths"])
    posted = split(d["posted_concat"], d["posted_lengths"])
    frames = d["frames"]
    name = os.path.basename(path).replace(".npz", "")

    print("\n" + "=" * 78)
    print("{}   chunk_size={} lookahead={}  {} chunks (frames {})".format(
        name, cs, la, len(fulls), list(map(int, frames))))

    expected_lag = cs * SAMPLES_PER_FRAME
    shipped_posted = posted[0].size if len(posted) > 1 else None
    if shipped_posted is not None:
        print("  shipped posts {} samples per full chunk; the correct splice "
              "point is {} -> shipped drops {} samples ({:.2f} ms) per "
              "seam".format(
                  shipped_posted, expected_lag, expected_lag - shipped_posted,
                  (expected_lag - shipped_posted) / (SR / 1000.0)))

    n_seams = 0
    dropped_rms, dropped_peak = [], []
    overlap_nrmse, overlap_corr, lag_deltas = [], [], []
    for k in range(len(fulls) - 1):
        a, b = fulls[k], fulls[k + 1]
        if frames[k] < cs + la:
            continue  # residual chunk; no following full window
        lag, corr = best_lag(a, b, expected_lag)
        n_seams += 1
        lag_deltas.append(lag - expected_lag)

        # (3) the audio the shipped splice throws away
        if shipped_posted is not None:
            gap = a[shipped_posted:expected_lag]
            dropped_rms.append(rms(gap))
            dropped_peak.append(float(np.max(np.abs(gap))) if gap.size else 0.0)

        # (4) same tokens, two decodes: how different are they?
        n = min(12000, b.size, a.size - lag)
        seg_a, seg_b = a[lag:lag + n], b[:n]
        denom = np.linalg.norm(seg_a)
        nrmse = float(np.linalg.norm(seg_a - seg_b) / denom) if denom else float("nan")
        overlap_nrmse.append(nrmse)
        overlap_corr.append(corr)

        if k < 3:
            print("    seam {}: measured lag={} (expected {}, delta {})  "
                  "peak corr={:.4f}  overlap NRMSE={:.4f}".format(
                      k, lag, expected_lag, lag - expected_lag, corr, nrmse))

    if not n_seams:
        print("  (single-chunk utterance - no seam)")
        return None

    print("  --- across {} seam(s) ---".format(n_seams))
    print("  alignment: lag delta from cs*1920 -> min {} max {}".format(
        min(lag_deltas), max(lag_deltas)))
    if dropped_rms:
        print("  dropped span: RMS {:.4f} (median), peak {:.4f} (max); "
              "utterance RMS {:.4f}".format(
                  float(np.median(dropped_rms)), float(np.max(dropped_peak)),
                  rms(d["posted_concat"])))
    print("  same-token overlap: peak corr {:.4f} (median), NRMSE {:.4f} "
          "(median)".format(float(np.median(overlap_corr)),
                            float(np.median(overlap_nrmse))))

    # (5) re-stitch two ways and compare seam discontinuity
    shipped_sig = np.concatenate(posted)
    fixed_parts = []
    for k, f in enumerate(fulls):
        fixed_parts.append(f if frames[k] < cs + la else f[:expected_lag])
    fixed_sig = np.concatenate(fixed_parts)

    def seam_positions(parts):
        pos, acc = [], 0
        for p in parts[:-1]:
            acc += p.size
            pos.append(acc)
        return pos

    sh_steps = [seam_step(shipped_sig, p) for p in seam_positions(posted)]
    fx_steps = [seam_step(fixed_sig, p) for p in seam_positions(fixed_parts)]
    ctrl_rng = np.random.default_rng(0)
    ctrl = [seam_step(shipped_sig, int(x)) for x in
            ctrl_rng.integers(2000, max(2001, shipped_sig.size - 2000), 200)]
    print("  seam step / local median |dx|:  shipped={:.2f}   "
          "corrected-trim={:.2f}   non-seam control={:.2f}".format(
              float(np.median(sh_steps)), float(np.median(fx_steps)),
              float(np.median(ctrl))))
    return {
        "name": name, "cs": cs, "seams": n_seams,
        "drop_samples": (expected_lag - shipped_posted) if shipped_posted else 0,
        "overlap_nrmse": float(np.median(overlap_nrmse)),
        "overlap_corr": float(np.median(overlap_corr)),
        "shipped_step": float(np.median(sh_steps)),
        "fixed_step": float(np.median(fx_steps)),
        "ctrl_step": float(np.median(ctrl)),
    }


def main() -> int:
    print("Story 20.4 - seam mechanism analysis")
    print("samples/frame = {} (measured), fixed per-decode edge loss = {}"
          .format(SAMPLES_PER_FRAME, EDGE_LOSS))
    results = []
    for p in sorted(glob.glob(os.path.join(RAWFULL, "*.npz"))):
        r = analyse(p)
        if r:
            results.append(r)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("{:<14} {:>3} {:>6} {:>10} {:>9} {:>10} {:>9} {:>9} {:>8}".format(
        "fixture", "cs", "seams", "drop", "ovl corr", "ovl NRMSE",
        "step now", "step fix", "ctrl"))
    for r in results:
        print("{:<14} {:>3} {:>6} {:>7} sm {:>9.4f} {:>10.4f} {:>9.2f} "
              "{:>9.2f} {:>8.2f}".format(
                  r["name"], r["cs"], r["seams"], r["drop_samples"],
                  r["overlap_corr"], r["overlap_nrmse"],
                  r["shipped_step"], r["fixed_step"], r["ctrl_step"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

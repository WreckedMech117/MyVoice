"""Story 20.4 AC #5 round-3 prep - WHY does the seam fix produce clicks?

Round 2 failed worse than round 1: the candidate defect class changed from
tonal_distortion to click_or_discontinuity, and s-022 - a SHORT fixture clean
on both arms in round 1 - became blocking. The seam-step metric that said
"0.85x the non-seam baseline" did not predict any of that, so this file does
two things and neither of them is another width sweep:

  PART 1 - a click detector, VALIDATED against the listener actual calls.
    Runs on the exact WAVs auditioned in rounds 1 and 2, at exactly known
    seam positions, across three conditions:
        A  cs25 pre-fix   (the reference arm in both rounds)
        B  cs10 pre-fix   (round-1 candidate: tonal_distortion on m-020)
        C  cs10 + fix     (round-2 candidate: clicks on l-021/m-020/s-022)
    If a metric cannot reproduce the ranking the ear produced, it is not
    evidence about audibility, and it is reported as such rather than used.

  PART 2 - the mechanism, from the captured pcm_full.
    Two hypotheses a step-discontinuity metric is structurally blind to:
      H1 COLD-START EDGE TRANSIENT. The blend ramps INTO next_full[0:1024] -
         the region with no left context, i.e. the worst-decoded part of the
         next chunk - while ramping OUT of prev_full[splice:], which sits in
         the middle of a well-supported decode. If the decode error is
         concentrated at the head and decays, the fix hands over to the bad
         copy as early as possible, and does it 2.5x more often at cs10.
      H2 TRANSIENT DOUBLING. If the two decodes place a transient at slightly
         different times, cross-fading renders it TWICE at partial amplitude.
         On a plosive-dense fixture ("Bit, bat, bot, but, bet") that is a
         click, and a wider fade makes it worse, not better.

Usage:
    python310/python.exe _bmad-output/implementation-artifacts/20-4-click-mechanism.py

Working file - gitignored under _bmad-output/; force-add per
memory/git_repo_state.md.
"""

from __future__ import annotations

import glob
import os
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.join(HERE, "20-4-perceptual-fixtures")
R2 = os.path.join(HERE, "20-4-perceptual-fixtures-r2")
RAWFULL = os.path.join(HERE, "20-4-seam-rawfull")
SR = 24000
SPF, EDGE = 1920, 555
OLA = 1024

UTTS = ["l-020", "l-021", "m-020", "m-021", "s-020", "s-021", "s-022"]

# The listener calls, verbatim from the two audition CSVs. 1 = flagged.
# Column A/B/C matches CONDITIONS below. A and C are round 2 (A is the same
# files as round 1, and 6 of 7 calls matched across rounds); B is round 1
# cs10 arm.
LISTENER = {
    #          A cs25  B cs10  C cs10+fix
    "l-020":   (1,     1,      1),
    "l-021":   (0,     1,      1),
    "m-020":   (0,     1,      1),
    "m-021":   (0,     0,      0),
    "s-020":   (0,     0,      0),
    "s-021":   (0,     0,      0),
    "s-022":   (0,     0,      1),
}


def read_wav(path):
    with wave.open(path, "rb") as fh:
        n = fh.getnframes()
        raw = fh.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def seam_positions(total, chunk_posted):
    """Seam offsets in a released stream of fixed-size posted chunks.

    The consumer releases chunk 1 alone (it clears the 500 ms watermark at
    every geometry here), then passes each later chunk through, so offsets
    are the cumulative sums. The leftover is the residual; it is validated
    against the codec identity so a wrong geometry assumption cannot pass
    silently.
    """
    pos, acc = [], 0
    while acc + chunk_posted < total:
        acc += chunk_posted
        pos.append(acc)
    residual = total - acc
    frames = (residual + EDGE) / float(SPF)
    ok = abs(frames - round(frames)) < 1e-6
    return pos, residual, ok


def lpc_coeffs(x, order):
    x = x - x.mean()
    r = np.correlate(x, x, mode="full")[len(x) - 1: len(x) + order]
    if r[0] <= 0:
        return None
    R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
    try:
        return np.linalg.solve(R + np.eye(order) * r[0] * 1e-6, r[1:order + 1])
    except np.linalg.LinAlgError:
        return None


def click_score(sig, pos, order=32, fit_len=3500, gap=2500, half=2048):
    """Peak LPC prediction-error spike near pos, in units of the local median.

    LPC is fitted on clean audio BEFORE the region of interest and applied
    forward - the standard de-clicking arrangement. Fitting across the click
    would partly absorb it. A click is an isolated broadband prediction
    failure, which is much closer to what an ear flags than a single-sample
    amplitude step.
    """
    fs = pos - gap - fit_len
    if fs < 0 or pos + half + 1 > sig.size:
        return np.nan
    a = lpc_coeffs(sig[fs:fs + fit_len], order)
    if a is None:
        return np.nan
    lo, hi = pos - half, pos + half
    seg = sig[lo - order:hi]
    if seg.size < order + 2:
        return np.nan
    pred = np.zeros(seg.size - order)
    for k in range(order):
        pred += a[k] * seg[order - 1 - k: seg.size - 1 - k]
    err = np.abs(seg[order:] - pred)
    med = np.median(err)
    if med <= 0:
        return np.nan
    return float(err.max() / med)


def hf_burst(sig, pos, half=2048, frame=256):
    """Short-time high-frequency energy peak near pos, relative to the file.

    Clicks are broadband and brief; sustained speech is not. Second
    difference is a crude but adequate HF emphasis, and needs no filter
    design that could itself ring at a discontinuity.
    """
    d = np.diff(sig, n=2)
    lo, hi = max(0, pos - half), min(d.size, pos + half)
    if hi - lo < frame * 2:
        return np.nan
    n_all = (d.size // frame) * frame
    if n_all == 0:
        return np.nan
    ref = float(np.median((d[:n_all].reshape(-1, frame) ** 2).mean(axis=1)))
    if ref <= 0:
        return np.nan
    a = d[lo:hi]
    n = (a.size // frame) * frame
    if n == 0:
        return np.nan
    return float(np.max((a[:n].reshape(-1, frame) ** 2).mean(axis=1)) / ref)


# Posted chunk length per condition. A and B use the pre-fix proportional
# trim; C uses the exact splice.
CONDITIONS = [
    ("A cs25 pre-fix", R1, "{}-cs25.wav",
     (SPF * 30 - EDGE) - int(round(5 * (SPF * 30 - EDGE) / 30.0))),
    ("B cs10 pre-fix", R1, "{}-cs10.wav",
     (SPF * 15 - EDGE) - int(round(5 * (SPF * 15 - EDGE) / 15.0))),
    ("C cs10 + fix", R2, "{}-cs10fix.wav", 10 * SPF),
]


def part1():
    print("=" * 78)
    print("PART 1 - click detector vs the listener actual calls")
    print("=" * 78)
    print("posted chunk length per condition: A={} B={} C={}".format(
        CONDITIONS[0][3], CONDITIONS[1][3], CONDITIONS[2][3]))
    print()
    print("{:<8} {:<17} {:>6} {:>10} {:>11} {:>9} {:>9}".format(
        "utt", "condition", "seams", "click max", "click ctrl", "hf max",
        "listener"))
    table = {}
    for utt in UTTS:
        for ci, (label, root, pat, chunk) in enumerate(CONDITIONS):
            path = os.path.join(root, pat.format(utt))
            if not os.path.exists(path):
                continue
            sig = read_wav(path)
            pos, residual, ok = seam_positions(sig.size, chunk)
            rng = np.random.default_rng(7)
            hi_ = max(6501, sig.size - 2100)
            ctrl = [int(x) for x in rng.integers(6500, hi_, 120)]
            cs_ = [v for v in (click_score(sig, p) for p in pos) if np.isfinite(v)]
            cc = [v for v in (click_score(sig, p) for p in ctrl) if np.isfinite(v)]
            hf = [v for v in (hf_burst(sig, p) for p in pos) if np.isfinite(v)]
            flagged = LISTENER[utt][ci]
            cmax = max(cs_) if cs_ else float("nan")
            table[(utt, ci)] = cmax
            print("{:<8} {:<17} {:>6} {:>10.1f} {:>11.1f} {:>9.1f} {:>9}".format(
                utt, label + ("" if ok else " !GEOM"), len(pos), cmax,
                float(np.median(cc)) if cc else float("nan"),
                max(hf) if hf else float("nan"),
                "FLAG" if flagged else "-"))
        print()

    print("-" * 78)
    print("Does the detector reproduce the ear?  (max click score per file)")
    print()
    flagged_vals, clean_vals = [], []
    for utt in UTTS:
        for ci in range(3):
            v = table.get((utt, ci))
            if v is None or not np.isfinite(v):
                continue
            row = (v, utt, ci)
            (flagged_vals if LISTENER[utt][ci] else clean_vals).append(row)
    flagged_vals.sort(reverse=True)
    clean_vals.sort(reverse=True)
    print("  FLAGGED by listener (n={}): ".format(len(flagged_vals)) + ", ".join(
        "{:.1f}".format(v) for v, _, _ in flagged_vals))
    print("  CLEAN per listener  (n={}): ".format(len(clean_vals)) + ", ".join(
        "{:.1f}".format(v) for v, _, _ in clean_vals))
    if flagged_vals and clean_vals:
        fmin = min(v for v, _, _ in flagged_vals)
        cmx = max(v for v, _, _ in clean_vals)
        print()
        print("  lowest FLAGGED = {:.1f}   highest CLEAN = {:.1f}".format(fmin, cmx))
        print("  separable by one threshold: {}".format(
            "YES" if fmin > cmx else "NO - this metric does not track the ear"))
    print()


def split(concat, lengths):
    out, i = [], 0
    for n in lengths:
        out.append(concat[i:i + n])
        i += n
    return out


def part2():
    print("=" * 78)
    print("PART 2 - mechanism, from captured pcm_full")
    print("=" * 78)
    print()
    print("H1 - COLD-START EDGE TRANSIENT")
    print("  RMS error between the two decodes of the SAME audio, by position")
    print("  into the next chunk decode, normalised by local RMS. If the head")
    print("  is worse, the blend ramps INTO the bad copy over its worst part.")
    print()
    bands = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048),
             (2048, 4096), (4096, 8192)]
    header = "  {:<16}".format("fixture") + "".join(
        "{:>8}".format(b[0]) for b in bands)
    print(header)
    for path in sorted(glob.glob(os.path.join(RAWFULL, "*.npz"))):
        z = np.load(path)
        cs, la = int(z["chunk_size"]), int(z["lookahead"])
        fulls = split(z["full_concat"], z["full_lengths"])
        frames = [int(x) for x in z["frames"]]
        splice = cs * SPF
        prof = {b: [] for b in bands}
        for k in range(len(fulls) - 1):
            if frames[k] < cs + la:
                continue
            a, b = fulls[k], fulls[k + 1]
            n = min(8192, b.size, a.size - splice)
            if n <= 0:
                continue
            aa = a[splice:splice + n].astype(np.float64)
            bb = b[:n].astype(np.float64)
            rms = float(np.sqrt(np.mean(aa ** 2))) or 1e-9
            for lo, hi in bands:
                if hi <= n:
                    prof[(lo, hi)].append(float(
                        np.sqrt(np.mean((aa[lo:hi] - bb[lo:hi]) ** 2)) / rms))
        name = os.path.basename(path).replace(".npz", "")
        vals = "".join(
            "{:>8.3f}".format(float(np.median(prof[b]))) if prof[b] else "     n/a"
            for b in bands)
        print("  {:<16}{}".format(name, vals))
    print()
    print("  (columns are the START sample of each band; the 1024-sample")
    print("   blend covers the first four)")
    print()

    print("H2 - TIMING MISALIGNMENT per seam, measured on the BLEND REGION")
    print("  best lag within +/-64 samples using only the first 1024 samples")
    print("  (what the blend actually mixes), skipping near-silent seams.")
    print()
    for path in sorted(glob.glob(os.path.join(RAWFULL, "*.npz"))):
        z = np.load(path)
        cs, la = int(z["chunk_size"]), int(z["lookahead"])
        fulls = split(z["full_concat"], z["full_lengths"])
        frames = [int(x) for x in z["frames"]]
        splice = cs * SPF
        lags, corrs = [], []
        for k in range(len(fulls) - 1):
            if frames[k] < cs + la:
                continue
            a, b = fulls[k], fulls[k + 1]
            if a.size < splice + OLA + 64 or b.size < OLA:
                continue
            bb = b[:OLA].astype(np.float64)
            if float(np.sqrt(np.mean(bb ** 2))) < 1e-3:
                continue
            best, bl = -2.0, 0
            nb = np.linalg.norm(bb)
            for lag in range(-64, 65):
                s = splice + lag
                if s < 0 or s + OLA > a.size:
                    continue
                aa = a[s:s + OLA].astype(np.float64)
                na = np.linalg.norm(aa)
                if na == 0 or nb == 0:
                    continue
                c = float(np.dot(aa, bb) / (na * nb))
                if c > best:
                    best, bl = c, lag
            lags.append(bl)
            corrs.append(best)
        name = os.path.basename(path).replace(".npz", "")
        if lags:
            print("  {:<16} n={:<3} lag med={:>4} min={:>4} max={:>4}   "
                  "corr med={:.3f} min={:.3f}".format(
                      name, len(lags), int(np.median(lags)), min(lags),
                      max(lags), float(np.median(corrs)), float(np.min(corrs))))
    print()


if __name__ == "__main__":
    part1()
    part2()

"""Summarise a durations JSONL (from durations_jsonl.py).

Per-test total = setup + call + teardown, which is what pytest-timeout's
default (func_only = False) budget covers. Prints the distribution of
*passing* tests, the slowest N, any test with an incomplete phase set (the
test in flight when os._exit fired), and every non-pass outcome.
"""
import json
import sys
from collections import defaultdict

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
phases = defaultdict(dict)
order = []
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r["event"] == "start":
        if r["nodeid"] not in phases:
            order.append(r["nodeid"])
        phases[r["nodeid"]].setdefault("_started", r["t"])
    else:
        phases[r["nodeid"]][r["when"]] = (r["outcome"], r["duration"])

totals = {}
incomplete = []
nonpass = []
for nid in order:
    ph = phases[nid]
    have = [w for w in ("setup", "call", "teardown") if w in ph]
    if "teardown" not in ph:
        incomplete.append((nid, have))
        continue
    total = sum(ph[w][1] for w in have)
    outcomes = {ph[w][0] for w in have}
    totals[nid] = (total, outcomes)
    if outcomes != {"passed"}:
        nonpass.append((nid, outcomes, total))

passing = sorted(((t, n) for n, (t, o) in totals.items() if o == {"passed"}), reverse=True)
vals = [t for t, _ in passing]

def pct(p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[k]

print(f"tests with complete phase data: {len(totals)}  passing: {len(passing)}  non-pass: {len(nonpass)}  in-flight/incomplete: {len(incomplete)}")
if vals:
    print(f"passing total-duration: max={max(vals):.2f}s  p99={pct(99):.2f}s  p95={pct(95):.2f}s  p90={pct(90):.2f}s  median={pct(50):.3f}s  sum={sum(vals):.1f}s")
    for th in (1, 5, 10, 30, 60, 120, 180, 300):
        print(f"  passing tests over {th:>3}s: {sum(1 for v in vals if v > th)}")
print(f"\n--- slowest {top} PASSING tests (setup+call+teardown) ---")
for t, n in passing[:top]:
    ph = phases[n]
    print(f"{t:8.2f}s  {n}   [setup {ph['setup'][1]:.2f} / call {ph.get('call',('',0))[1]:.2f} / teardown {ph['teardown'][1]:.2f}]")
if incomplete:
    print("\n--- INCOMPLETE (in flight when the process ended) ---")
    for n, have in incomplete:
        print(f"  {n}  phases seen: {have}")
if nonpass:
    print(f"\n--- non-pass outcomes ({len(nonpass)}) ---")
    for n, o, t in sorted(nonpass):
        print(f"  {sorted(o)} {t:7.2f}s  {n}")

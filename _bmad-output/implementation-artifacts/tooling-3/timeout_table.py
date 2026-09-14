"""Build the AC #4 table: every whole-suite timeout, its stall location (from
the iteration log's MainThread stack), and its outcome in the per-directory
measurement run (measure-M1.jsonl). Usage: timeout_table.py <label>"""
import json
import re
import sys
from collections import defaultdict

label = sys.argv[1]
hung = [l.split(" ", 2)[1] for l in open(f"suite-{label}.hung", encoding="utf-8") if l.startswith("TIMEOUT: ")]
iters = [int(re.search(r"iteration (\d+)", l).group(1)) for l in open(f"suite-{label}.hung", encoding="utf-8") if l.startswith("TIMEOUT: ")]

# per-directory outcomes
m1 = defaultdict(dict)
for line in open("measure-M1.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["event"] == "phase":
        m1[r["nodeid"]][r["when"]] = r["outcome"]
m1_timeouts = set()
for line in open("measure-M1.log", encoding="utf-8", errors="replace"):
    m = re.match(r"^(tests/.*?) +\+{3,} Timeout \+{3,}", line)
    if m:
        m1_timeouts.add(m.group(1))

def m1_outcome(nid):
    if nid in m1_timeouts:
        return "TIMED OUT (900 s guard)"
    ph = m1.get(nid)
    if not ph:
        return "not reached (file aborted earlier)"
    if "teardown" not in ph:
        return "in flight at abort"
    oc = set(ph.values())
    return "passed" if oc == {"passed"} else "/".join(sorted(oc))

def stall(nid, it):
    lines = open(f"suite-{label}-iter{it}.log", encoding="utf-8", errors="replace").read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(nid + " ") and "Timeout" in l)
    j = next(k for k in range(i, len(lines)) if "Stack of MainThread" in lines[k])
    frames = []
    k = j + 1
    while k < len(lines) and not lines[k].startswith("~~~") and not lines[k].startswith("+++"):
        fm = re.match(r'^\s+File "(.*?)", line (\d+), in (.*)$', lines[k])
        if fm and "site-packages" not in fm.group(1):
            rel = re.sub(r"^.*?(MyVoiceV2|MyVoicePublicInst)\\", "", fm.group(1)).replace("\\", "/")
            frames.append((rel, fm.group(2), fm.group(3), lines[k + 1].strip()))
        k += 1
    inner = frames[-1]
    return f"{inner[0]}:{inner[1]} in {inner[2]} -> {inner[3]}", frames

print("| # | iter | test id | stalled at (innermost project frame) | per-directory run (measure-M1) |")
print("|---|------|---------|--------------------------------------|--------------------------------|")
chains = {}
for n, (nid, it) in enumerate(zip(hung, iters), 1):
    where, frames = stall(nid, it)
    chains[nid] = frames
    print(f"| {n} | {it} | `{nid}` | `{where}` | {m1_outcome(nid)} |")

print("\n\nDistinct stall chains (project frames, outermost test frame first):")
seen = {}
for nid, frames in chains.items():
    key = tuple(f[:3] for f in frames if f[0].startswith("src/"))
    seen.setdefault(key, []).append(nid)
for key, ids in seen.items():
    print(f"\n{len(ids)} test(s):")
    for f in key:
        print(f"    {f[0]}:{f[1]} in {f[2]}")
    for i in ids:
        print(f"      - {i}")

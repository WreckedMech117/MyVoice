"""Print, for every '+++ Timeout +++' block in a pytest -v log, the hung test id
and the project-code frames (src/ and tests/, site-packages elided) of the
MainThread stack, innermost last. Usage: stall_frames.py <log> [<log>...]"""
import re
import sys

for path in sys.argv[1:]:
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(tests/.*?) +\+{3,} Timeout \+{3,}", lines[i])
        if not m:
            i += 1
            continue
        nodeid = m.group(1)
        # find MainThread stack
        j = i + 1
        while j < len(lines) and "Stack of MainThread" not in lines[j]:
            j += 1
        frames = []
        k = j + 1
        while k < len(lines) and not lines[k].startswith("~~~") and not lines[k].startswith("+++"):
            fm = re.match(r'^\s+File "(.*?)", line (\d+), in (.*)$', lines[k])
            if fm and ("\src\\" in fm.group(1) or "\tests\\" in fm.group(1)):
                code = lines[k + 1].strip() if k + 1 < len(lines) else ""
                rel = fm.group(1).split("MyVoiceV2\\", 1)[-1]
                frames.append(f"{rel}:{fm.group(2)} in {fm.group(3)}  ->  {code}")
            k += 1
        print(f"\n{path}: TIMEOUT {nodeid}")
        for f in frames:
            print("    " + f)
        i = k

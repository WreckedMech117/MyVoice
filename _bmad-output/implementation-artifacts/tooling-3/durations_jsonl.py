"""Streaming per-test duration logger for Story tooling-3.

Why this exists: pytest-timeout's ``thread`` method ends the pytest process
with ``os._exit(1)`` when a test exceeds its budget. Anything pytest prints at
session end (``--durations``, ``--junitxml``) is therefore LOST for a run that
hits a timeout. This plugin writes one JSON line per (test, phase) as each
phase finishes and flushes immediately, so the durations of every test that
completed before the abort survive, and the test in flight at the abort is
identifiable as the last ``setup``/``call`` line without a matching
``teardown``.

Load with:  -p durations_jsonl  (with this directory on PYTHONPATH)
Output:     $TOOLING3_DURATIONS (path), default tooling-3-durations.jsonl
"""
import json
import os
import time

_path = os.environ.get("TOOLING3_DURATIONS", "tooling-3-durations.jsonl")
_fh = None


def _out():
    global _fh
    if _fh is None:
        _fh = open(_path, "a", encoding="utf-8")
    return _fh


def pytest_runtest_logstart(nodeid, location):
    f = _out()
    f.write(json.dumps({"event": "start", "nodeid": nodeid, "t": time.time()}) + "\n")
    f.flush()


def pytest_runtest_logreport(report):
    f = _out()
    f.write(json.dumps({
        "event": "phase",
        "nodeid": report.nodeid,
        "when": report.when,
        "outcome": report.outcome,
        "duration": round(report.duration, 4),
        "t": time.time(),
    }) + "\n")
    f.flush()

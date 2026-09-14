"""Run pytest with the durations_jsonl plugin registered.

The bundled python310 is an embeddable distribution (python310._pth ->
isolated mode), so PYTHONPATH is ignored and `-p durations_jsonl` cannot find
the plugin. Register it programmatically instead. Nothing else changes: no
torch import, no DLL preamble -- tests/conftest.py still runs first, exactly
as under `python -m pytest` (AC #5 is verified against this wrapper).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import durations_jsonl  # noqa: E402
import pytest  # noqa: E402

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:], plugins=[durations_jsonl]))

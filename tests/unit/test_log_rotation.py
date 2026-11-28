"""A tester's myvoice.log reached 153 MB: the log was append-only forever."""
import logging
from logging.handlers import RotatingFileHandler

import torch  # noqa: F401  (torch before PyQt6, per the project's DLL rule)

from myvoice import main as main_module


def test_oversized_log_is_cut_to_its_newest_part(tmp_path):
    log = tmp_path / "myvoice.log"
    line = b"2026-01-01 00:00:00,000 - x - INFO - old line\n"
    newest = b"2026-11-27 19:46:45,781 - x - INFO - newest line\n"
    log.write_bytes(line * (3 * main_module._LOG_MAX_BYTES // len(line)) + newest)

    main_module._trim_oversized_log(log)

    data = log.read_bytes()
    assert len(data) <= main_module._LOG_MAX_BYTES
    assert data.endswith(newest)
    assert data.startswith(b"2026-")  # cut on a line boundary


def test_normal_sized_log_is_left_alone(tmp_path):
    log = tmp_path / "myvoice.log"
    log.write_bytes(b"short\n")
    main_module._trim_oversized_log(log)
    assert log.read_bytes() == b"short\n"


def test_missing_log_does_not_raise(tmp_path):
    main_module._trim_oversized_log(tmp_path / "absent.log")


def test_setup_logging_installs_a_bounded_rotating_handler(tmp_path, monkeypatch):
    import myvoice.utils.portable_paths as paths

    monkeypatch.setattr(paths, "get_logs_path", lambda: tmp_path)
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = []
    try:
        main_module.setup_logging()
        rotating = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating) == 1
        assert rotating[0].maxBytes == main_module._LOG_MAX_BYTES
        assert rotating[0].backupCount == main_module._LOG_BACKUP_COUNT
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers = saved

"""quota_tracker 단위 테스트."""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    """격리된 quota.json을 사용하는 quota_tracker 인스턴스."""
    import importlib

    import quota_tracker as qt

    fake_file = tmp_path / "quota.json"
    fake_lock = tmp_path / "quota.json.lock"
    monkeypatch.setattr(qt, "QUOTA_FILE", fake_file)
    monkeypatch.setattr(qt, "LOCK_FILE", fake_lock)
    from filelock import FileLock

    monkeypatch.setattr(qt, "_LOCK", FileLock(str(fake_lock), timeout=5))
    importlib.reload  # no-op, just to keep the import used
    return qt


def test_initial_state_empty(tracker):
    usage = tracker.get_usage("gemini-2.0-flash")
    assert usage["rpd_used"] == 0
    assert usage["rpd_limit"] == 1500
    assert usage["rpm_used"] == 0
    assert usage["rpm_limit"] == 15


def test_can_call_fresh(tracker):
    allowed, reason = tracker.can_call("gemini-2.0-flash")
    assert allowed is True
    assert reason == ""


def test_record_call_increments(tracker):
    tracker.record_call("gemini-2.0-flash")
    usage = tracker.get_usage("gemini-2.0-flash")
    assert usage["rpd_used"] == 1
    assert usage["rpm_used"] == 1


def test_rpd_limit_blocks(tracker):
    for _ in range(1500):
        tracker.record_call("gemini-2.0-flash")
    allowed, reason = tracker.can_call("gemini-2.0-flash")
    assert allowed is False
    assert "일일 한도" in reason


def test_rpm_limit_blocks(tracker):
    for _ in range(15):
        tracker.record_call("gemini-2.0-flash")
    allowed, reason = tracker.can_call("gemini-2.0-flash")
    assert allowed is False
    assert "분당 한도" in reason


def test_unknown_model_blocked(tracker):
    allowed, reason = tracker.can_call("gemini-99-ultra")
    assert allowed is False


def test_day_rollover_resets_counts(tracker):
    tracker.record_call("gemini-2.0-flash")
    assert tracker.get_usage("gemini-2.0-flash")["rpd_used"] == 1

    raw = json.loads(tracker.QUOTA_FILE.read_text(encoding="utf-8"))
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    raw["date"] = yesterday
    tracker.QUOTA_FILE.write_text(json.dumps(raw), encoding="utf-8")

    usage = tracker.get_usage("gemini-2.0-flash")
    assert usage["rpd_used"] == 0


def test_minute_window_gc(tracker):
    stale_ts = time.time() - 120
    raw = {
        "date": date.today().isoformat(),
        "daily": {m: 0 for m in tracker.FREE_TIER_LIMITS},
        "minute_window": {m: [] for m in tracker.FREE_TIER_LIMITS},
    }
    raw["minute_window"]["gemini-2.0-flash"] = [stale_ts, stale_ts + 1, stale_ts + 2]
    tracker.QUOTA_FILE.write_text(json.dumps(raw), encoding="utf-8")

    usage = tracker.get_usage("gemini-2.0-flash")
    assert usage["rpm_used"] == 0


def test_per_model_counters_isolated(tracker):
    tracker.record_call("gemini-2.0-flash")
    tracker.record_call("gemini-2.0-flash")
    tracker.record_call("gemini-1.5-flash")

    u20 = tracker.get_usage("gemini-2.0-flash")
    u15 = tracker.get_usage("gemini-1.5-flash")
    u25 = tracker.get_usage("gemini-2.5-flash")

    assert u20["rpd_used"] == 2
    assert u15["rpd_used"] == 1
    assert u25["rpd_used"] == 0


def test_reset_all(tracker):
    tracker.record_call("gemini-2.0-flash")
    tracker.record_call("gemini-1.5-flash")
    tracker.reset_all()
    assert tracker.get_usage("gemini-2.0-flash")["rpd_used"] == 0
    assert tracker.get_usage("gemini-1.5-flash")["rpd_used"] == 0


def test_25_flash_lower_limits(tracker):
    u = tracker.get_usage("gemini-2.5-flash")
    assert u["rpd_limit"] == 500
    assert u["rpm_limit"] == 10


def test_corrupted_json_recovered(tracker):
    tracker.QUOTA_FILE.write_text("{ not valid json", encoding="utf-8")
    usage = tracker.get_usage("gemini-2.0-flash")
    assert usage["rpd_used"] == 0

"""Gemini 무료 티어 사용량 트래킹 + 하드 리밋.

streamlit_app.py의 Gemini 호출 전후에 can_call/record_call을 호출해
무료 티어(RPM/RPD)를 초과하기 직전에 요청을 차단한다.

저장: 같은 디렉토리의 quota.json (filelock으로 동시성 보호)
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Tuple

from filelock import FileLock

FREE_TIER_LIMITS: dict[str, dict[str, int]] = {
    "gemini-2.0-flash": {"rpm": 15, "rpd": 1500},
    "gemini-1.5-flash": {"rpm": 15, "rpd": 1500},
    "gemini-2.5-flash": {"rpm": 10, "rpd": 500},
}

QUOTA_FILE = Path(__file__).parent / "quota.json"
LOCK_FILE = Path(__file__).parent / "quota.json.lock"
_LOCK = FileLock(str(LOCK_FILE), timeout=5)

RPM_WINDOW_SECONDS = 60


def _empty_state() -> dict:
    return {
        "date": date.today().isoformat(),
        "daily": {m: 0 for m in FREE_TIER_LIMITS},
        "minute_window": {m: [] for m in FREE_TIER_LIMITS},
    }


def _load_raw() -> dict:
    if not QUOTA_FILE.exists():
        return _empty_state()
    try:
        data = json.loads(QUOTA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_state()
    data.setdefault("date", date.today().isoformat())
    data.setdefault("daily", {})
    data.setdefault("minute_window", {})
    for m in FREE_TIER_LIMITS:
        data["daily"].setdefault(m, 0)
        data["minute_window"].setdefault(m, [])
    return data


def _save_raw(data: dict) -> None:
    QUOTA_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _rollover_if_new_day(data: dict) -> dict:
    today = date.today().isoformat()
    if data.get("date") != today:
        data["date"] = today
        data["daily"] = {m: 0 for m in FREE_TIER_LIMITS}
        data["minute_window"] = {m: [] for m in FREE_TIER_LIMITS}
    return data


def _gc_minute_window(data: dict, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    cutoff = now - RPM_WINDOW_SECONDS
    for m in FREE_TIER_LIMITS:
        data["minute_window"][m] = [
            ts for ts in data["minute_window"].get(m, []) if ts > cutoff
        ]
    return data


def _read_state() -> dict:
    with _LOCK:
        data = _load_raw()
        data = _rollover_if_new_day(data)
        data = _gc_minute_window(data)
    return data


def get_usage(model: str) -> dict:
    """UI용: 현재 사용량/한도를 반환."""
    if model not in FREE_TIER_LIMITS:
        return {"rpd_used": 0, "rpd_limit": 0, "rpm_used": 0, "rpm_limit": 0}
    data = _read_state()
    limits = FREE_TIER_LIMITS[model]
    return {
        "rpd_used": data["daily"].get(model, 0),
        "rpd_limit": limits["rpd"],
        "rpm_used": len(data["minute_window"].get(model, [])),
        "rpm_limit": limits["rpm"],
    }


def can_call(model: str) -> Tuple[bool, str]:
    """호출 가능 여부 + 차단 사유."""
    if model not in FREE_TIER_LIMITS:
        return False, f"알 수 없는 모델: {model}"
    usage = get_usage(model)
    if usage["rpd_used"] >= usage["rpd_limit"]:
        return False, f"일일 한도 소진 ({usage['rpd_used']}/{usage['rpd_limit']})"
    if usage["rpm_used"] >= usage["rpm_limit"]:
        return (
            False,
            f"분당 한도 초과 ({usage['rpm_used']}/{usage['rpm_limit']}) — 60초 후 재시도",
        )
    return True, ""


def record_call(model: str) -> None:
    """성공 호출 기록. 캐시 히트 시에는 호출하지 않아야 함."""
    if model not in FREE_TIER_LIMITS:
        return
    with _LOCK:
        data = _load_raw()
        data = _rollover_if_new_day(data)
        data = _gc_minute_window(data)
        data["daily"][model] = data["daily"].get(model, 0) + 1
        data["minute_window"].setdefault(model, []).append(time.time())
        _save_raw(data)


def reset_all() -> None:
    """관리자용: 카운터 전체 리셋."""
    with _LOCK:
        _save_raw(_empty_state())


def snapshot() -> dict:
    """디버깅용: 현재 상태 스냅샷."""
    return _read_state()

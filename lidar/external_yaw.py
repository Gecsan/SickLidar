from __future__ import annotations

import json
import math
from pathlib import Path
import time


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class ExternalYawSource:
    """Read yaw from a text or JSON file written by an external sensor process."""

    def __init__(
        self,
        path: str,
        unit: str = 'deg',
        max_age_sec: float = 0.75,
    ) -> None:
        self.path = Path(path)
        self.unit = unit
        self.max_age_sec = max_age_sec
        self._last_value_rad: float | None = None
        self._last_update_ts: float | None = None

    @property
    def is_live(self) -> bool:
        if self._last_update_ts is None:
            return False
        return (time.time() - self._last_update_ts) <= self.max_age_sec

    def read_yaw_rad(self) -> tuple[float | None, bool]:
        try:
            stat = self.path.stat()
            text = self.path.read_text(encoding='utf-8').strip()
        except FileNotFoundError:
            return self._last_value_rad, False
        except OSError:
            return self._last_value_rad, False

        if not text:
            return self._last_value_rad, False

        try:
            yaw_rad, is_fresh = self._parse_text(text, stat.st_mtime)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return self._last_value_rad, False

        self._last_value_rad = _wrap_angle(yaw_rad)
        self._last_update_ts = stat.st_mtime
        return self._last_value_rad, is_fresh

    def _parse_text(self, text: str, file_mtime: float) -> tuple[float, bool]:
        if text.startswith('{'):
            payload = json.loads(text)
            return self._parse_json_payload(payload, file_mtime)

        value = float(text)
        is_fresh = (time.time() - file_mtime) <= self.max_age_sec
        return self._to_rad(value, self.unit), is_fresh

    def _parse_json_payload(self, payload: dict, file_mtime: float) -> tuple[float, bool]:
        if 'yaw_rad' in payload:
            yaw_rad = float(payload['yaw_rad'])
        elif 'yaw_deg' in payload:
            yaw_rad = math.radians(float(payload['yaw_deg']))
        elif 'yaw' in payload:
            yaw_rad = self._to_rad(float(payload['yaw']), self.unit)
        else:
            raise KeyError('Missing yaw field')

        timestamp = payload.get('timestamp')
        now = time.time()
        if timestamp is not None:
            is_fresh = (now - float(timestamp)) <= self.max_age_sec
        else:
            is_fresh = (now - file_mtime) <= self.max_age_sec

        return yaw_rad, is_fresh

    @staticmethod
    def _to_rad(value: float, unit: str) -> float:
        if unit == 'rad':
            return value
        return math.radians(value)
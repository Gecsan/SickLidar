"""
Read live scan points from the TiM881P web UI websocket feed (crownJSON).

This mirrors what the browser scan viewer uses, avoiding SOPAS desktop tools.

Example:
    python webui_live_points.py --host 192.168.0.1 --viewer-id view
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any

import numpy as np
from websocket import WebSocketApp

from sick_tim881p import ScanData

logger = logging.getLogger(__name__)


@dataclass
class CrownScan:
    scan_number: int
    theta_rad: np.ndarray
    dist_mm: np.ndarray


class CrownJSONClient:
    def __init__(self, host: str, viewer_id: str = "view") -> None:
        self.host = host
        self.viewer_id = viewer_id
        self._ws: WebSocketApp | None = None
        self._handle_id: int | None = None
        self._registered = False

    def run(self) -> None:
        self._handle_id = None
        self._registered = False
        self._ws = WebSocketApp(
            f"ws://{self.host}/crownJSON",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        # Some embedded websocket stacks do not reliably answer ping/pong during
        # heavy load or motion; disabling keepalive avoids false disconnects.
        self._ws.run_forever(ping_interval=0)

    def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            return
        try:
            sock = getattr(self._ws, "sock", None)
            if sock is None or not getattr(sock, "connected", False):
                return
            self._ws.send(json.dumps(payload))
        except Exception:
            return

    def _on_open(self, _ws: WebSocketApp) -> None:
        self._send(
            {
                "header": {"type": "FunctionCall", "clientId": 0, "function": "View/Present/create"},
                "data": {},
                "options": {},
            }
        )

    def _on_message(self, _ws: WebSocketApp, message: str) -> None:
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return

        # First reply: handle allocation for the current "present" session.
        if (
            self._handle_id is None
            and msg.get("header", {}).get("type") == "FunctionReturn"
            and isinstance(msg.get("data", {}).get("handle", {}).get("id"), int)
        ):
            self._handle_id = msg["data"]["handle"]["id"]
            self._send(
                {
                    "header": {"type": "FunctionCall", "clientId": 0, "function": "View/Present/setID"},
                    "data": {"handle": {"type": "handle", "id": self._handle_id}, "id": self.viewer_id},
                    "options": {},
                }
            )
            self._send(
                {
                    "header": {"type": "FunctionCall", "clientId": 0, "function": "View/Present/register"},
                    "data": {"eventname": "OnPresentLive", "handle": {"type": "handle", "id": self._handle_id}},
                    "options": {"queue": {"priority": "HIGH", "maxSize": 0, "discardIfFull": "OLDEST"}},
                }
            )
            self._registered = True
            return

        if not self._registered:
            return

        scan = self._extract_scan(msg)
        if scan is None:
            return

        x = scan.dist_mm * np.cos(scan.theta_rad)
        y = scan.dist_mm * np.sin(scan.theta_rad)

        valid = np.isfinite(scan.dist_mm) & (scan.dist_mm > 0)
        valid_count = int(np.count_nonzero(valid))

        print(f"scan={scan.scan_number} beams={scan.theta_rad.size} valid={valid_count}")
        if valid_count:
            idx = np.flatnonzero(valid)[:5]
            for i in idx:
                print(f"  ({x[i]:8.1f} mm, {y[i]:8.1f} mm)  r={scan.dist_mm[i]:8.1f} mm")

    def _extract_scan(self, msg: dict[str, Any]) -> CrownScan | None:
        if msg.get("header", {}).get("type") != "Event":
            return None

        view_objects = msg.get("data", {}).get("viewObject", [])
        for obj in view_objects:
            data = obj.get("data", {})
            if data.get("class") != "View.Present.Add":
                continue
            payload = data.get("data", {})
            if payload.get("Type") != "POLAR_SCAN":
                continue

            iconics = payload.get("Iconics", [])
            if not iconics:
                return None

            scan = iconics[0].get("data", {})
            theta_spec = scan.get("ChannelTheta")
            dist_values = scan.get("DistValues", [])
            if not isinstance(theta_spec, dict) or not dist_values:
                return None

            dist_spec = dist_values[0]
            theta = self._decode_float32(theta_spec.get("data", ""))
            dist = self._decode_float32(dist_spec.get("data", ""))
            n = min(theta.size, dist.size)
            if n == 0:
                return None

            return CrownScan(
                scan_number=int(scan.get("ScanNumber", -1)),
                theta_rad=theta[:n],
                dist_mm=dist[:n],
            )

        return None

    @staticmethod
    def to_scan_data(scan: CrownScan) -> ScanData:
        return ScanData(
            angles_deg=np.rad2deg(scan.theta_rad).tolist(),
            distances_mm=scan.dist_mm.tolist(),
            timestamp=0.0,
            scan_counter=scan.scan_number,
        )

    @staticmethod
    def _decode_float32(b64: str) -> np.ndarray:
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype="<f4")

    @staticmethod
    def _on_error(_ws: WebSocketApp, error: Any) -> None:
        print(f"websocket error: {error}")

    @staticmethod
    def _on_close(_ws: WebSocketApp, code: int, reason: str) -> None:
        print(f"websocket closed: code={code} reason={reason}")


class WebUIScanSource(CrownJSONClient):
    def __init__(self, host: str, viewer_id: str = "view") -> None:
        super().__init__(host=host, viewer_id=viewer_id)
        self._queue: Queue[ScanData] = Queue(maxsize=4)
        self._thread: Thread | None = None
        self._running = False
        self._reconnect_delay_s = 0.5
        self._last_scan_ts = 0.0
        self._last_restart_ts = 0.0
        self._last_message_ts = 0.0
        self._connected = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        while self._running:
            try:
                self.run()
            except Exception as exc:
                logger.warning("Web UI websocket loop error: %s", exc)

            if not self._running:
                break

            # Reset session state and reconnect quickly.
            self._handle_id = None
            self._registered = False
            self._connected = False
            time.sleep(self._reconnect_delay_s)

    def _restart_stream(self) -> None:
        now = time.time()
        if now - self._last_restart_ts <= 2.0:
            return
        self._last_restart_ts = now
        self._handle_id = None
        self._registered = False
        self._connected = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread is None or not self._thread.is_alive():
            self.start()

    def read_scan(self, timeout: float = 2.0) -> ScanData:
        try:
            return self._queue.get(timeout=timeout)
        except Empty as exc:
            now = time.time()
            if not self._running:
                raise TimeoutError("Timed out waiting for web UI scan data") from exc

            # If thread died or websocket is stale, force a full restart.
            if self._thread is None or not self._thread.is_alive():
                logger.warning("Web UI reader thread stopped; restarting stream")
                self.start()
            elif (not self._connected) or (now - self._last_message_ts > 3.0) or (now - self._last_scan_ts > 3.0):
                logger.warning("Web UI stream stale; restarting websocket session")
                self._restart_stream()
            raise TimeoutError("Timed out waiting for web UI scan data") from exc

    def close(self) -> None:
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass

    def _on_message(self, ws: WebSocketApp, message: str) -> None:
        self._last_message_ts = time.time()
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return

        if (
            self._handle_id is None
            and msg.get("header", {}).get("type") == "FunctionReturn"
            and isinstance(msg.get("data", {}).get("handle", {}).get("id"), int)
        ):
            self._handle_id = msg["data"]["handle"]["id"]
            self._send(
                {
                    "header": {"type": "FunctionCall", "clientId": 0, "function": "View/Present/setID"},
                    "data": {"handle": {"type": "handle", "id": self._handle_id}, "id": self.viewer_id},
                    "options": {},
                }
            )
            self._send(
                {
                    "header": {"type": "FunctionCall", "clientId": 0, "function": "View/Present/register"},
                    "data": {"eventname": "OnPresentLive", "handle": {"type": "handle", "id": self._handle_id}},
                    "options": {"queue": {"priority": "HIGH", "maxSize": 0, "discardIfFull": "OLDEST"}},
                }
            )
            self._registered = True
            return

        if not self._registered:
            return

        scan = self._extract_scan(msg)
        if scan is None:
            return

        scan_data = self.to_scan_data(scan)
        self._last_scan_ts = time.time()
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Empty:
                pass
        try:
            self._queue.put_nowait(scan_data)
        except Full:
            pass

    def _on_close(self, ws: WebSocketApp, code: int, reason: str) -> None:
        super()._on_close(ws, code, reason)
        self._handle_id = None
        self._registered = False
        self._connected = False

    def _on_open(self, ws: WebSocketApp) -> None:
        self._connected = True
        self._last_message_ts = time.time()
        super()._on_open(ws)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read live points from TiM881P web UI websocket feed")
    parser.add_argument("--host", default="192.168.0.1", help="Sensor web host (default: 192.168.0.1)")
    parser.add_argument("--viewer-id", default="view", help="Viewer ID used by UI (default: view)")
    args = parser.parse_args()

    client = CrownJSONClient(host=args.host, viewer_id=args.viewer_id)
    client.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

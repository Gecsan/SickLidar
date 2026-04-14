"""
SICK TIM881P-2100101 LiDAR driver using SOPAS CoLa A protocol over TCP/IP.

Default network settings:
    IP   : 192.168.0.100
  Port : 2111  (CoLa A / ASCII)

Protocol overview
-----------------
Commands are framed with STX (0x02) ... ETX (0x03).
The driver polls single scans with "sRN LMDscandata" or enables a continuous
stream with "sEN LMDscandata 1" and reads frames as they arrive.
"""

import logging
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScanData:
    """One complete 2-D scan frame from the sensor."""

    angles_deg: List[float]    # Beam angles in degrees (one per sample)
    distances_mm: List[float]  # Measured distances in mm  (0 = no return)
    timestamp: float           # Unix wall-clock when the frame was received
    scan_counter: int          # Monotonically-increasing counter from sensor


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class SickTIM881P:
    """
    Driver for the SICK TIM881P 2-D LiDAR sensor.

    Usage (single-shot polling)::

        with SickTIM881P(host='192.168.0.100') as lidar:
            scan = lidar.poll_scan()

    Usage (continuous stream)::

        with SickTIM881P(host='192.168.0.100') as lidar:
            lidar.start_continuous_scan()
            while True:
                scan = lidar.read_continuous_scan()
    """

    _STX = b'\x02'
    _ETX = b'\x03'

    DEFAULT_HOST = '192.168.0.100'
    DEFAULT_PORT = 2111

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open a TCP connection to the sensor."""
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect((self.host, self.port))
        except (OSError, socket.error) as exc:
            sock.close()
            raise ConnectionError(
                f"Cannot connect to SICK TIM881P at {self.host}:{self.port} – {exc}"
            ) from exc
        self._sock = sock
        logger.info("Connected to SICK TIM881P at %s:%d", self.host, self.port)

    def disconnect(self) -> None:
        """Stop streaming (if active) and close the connection."""
        if self._sock is None:
            return
        try:
            self._send_cmd('sEN LMDscandata 0')
        except Exception:
            pass
        self._sock.close()
        self._sock = None
        logger.info("Disconnected from SICK TIM881P")

    def __enter__(self) -> "SickTIM881P":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Public scan interface
    # ------------------------------------------------------------------

    def poll_scan(self) -> "ScanData":
        """Request and return a single scan (polling mode)."""
        response = self._send_cmd('sRN LMDscandata')
        return self._parse_lmd_scan_data(response)

    def start_continuous_scan(self) -> None:
        """Ask the sensor to push scan frames continuously."""
        self._send_cmd('sEN LMDscandata 1')

    def stop_continuous_scan(self) -> None:
        """Stop the continuous scan stream."""
        self._send_cmd('sEN LMDscandata 0')

    def read_continuous_scan(self) -> "ScanData":
        """Read one scan frame from the continuous stream; blocks until received."""
        response = self._recv_frame()
        return self._parse_lmd_scan_data(response)

    # ------------------------------------------------------------------
    # Low-level communication
    # ------------------------------------------------------------------

    def _send_cmd(self, command: str) -> str:
        """Send a CoLa A command and return the decoded response."""
        if self._sock is None:
            raise RuntimeError("Not connected – call connect() first.")
        frame = self._STX + command.encode('ascii') + self._ETX
        self._sock.sendall(frame)
        return self._recv_frame()

    def _recv_frame(self) -> str:
        """Receive bytes until ETX is found; return the inner ASCII payload."""
        if self._sock is None:
            raise RuntimeError("Not connected.")
        buf = bytearray()
        while True:
            chunk = self._sock.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
            if self._ETX[0] in buf:
                break

        raw = bytes(buf)
        start = raw.find(self._STX[0])
        end = raw.find(self._ETX[0])
        if start != -1 and end != -1 and end > start:
            raw = raw[start + 1 : end]
        return raw.decode('ascii', errors='ignore')

    # ------------------------------------------------------------------
    # SOPAS LMDscandata parser
    # ------------------------------------------------------------------

    @staticmethod
    def _hex_to_float(h: str) -> float:
        """Decode an 8-char hex string as a 32-bit IEEE 754 float."""
        return struct.unpack('!f', bytes.fromhex(h.zfill(8)))[0]

    @staticmethod
    def _hex_to_signed32(h: str) -> int:
        """Decode a hex string as a signed 32-bit integer."""
        v = int(h, 16)
        return v - 0x1_0000_0000 if v > 0x7FFF_FFFF else v

    def _parse_lmd_scan_data(self, response: str) -> ScanData:
        """
        Parse a SOPAS CoLa A 'sRA LMDscandata' response.

        Key fields extracted
        --------------------
        fields[7]          : scanCounter (hex)
        DIST1+1            : scaleFactor  (IEEE 754 hex float)
        DIST1+2            : scaleOffset  (IEEE 754 hex float)
        DIST1+3            : startAngle   (signed hex int, 1/10000 deg)
        DIST1+4            : angleStep    (unsigned hex int, 1/10000 deg)
        DIST1+5            : numData      (unsigned hex int)
        DIST1+6 … +6+N-1  : distance values (unsigned hex int, mm raw)
        """
        fields = response.split()

        if len(fields) < 25:
            raise ValueError(
                f"Response too short to be LMDscandata ({len(fields)} fields). "
                f"Raw (first 200 chars): {response[:200]!r}"
            )

        # Scan counter (field index 7 in standard SOPAS layout)
        try:
            scan_counter = int(fields[7], 16)
        except (IndexError, ValueError):
            scan_counter = 0

        # Locate DIST1 channel
        try:
            di = fields.index('DIST1')
        except ValueError:
            raise ValueError(
                "No DIST1 channel in response. "
                "Ensure the sensor is configured to output distance data."
            )

        scale_factor = self._hex_to_float(fields[di + 1])
        scale_offset = self._hex_to_float(fields[di + 2])
        start_angle  = self._hex_to_signed32(fields[di + 3]) / 10_000.0  # degrees
        angle_step   = int(fields[di + 4], 16)               / 10_000.0  # degrees
        num_data     = int(fields[di + 5], 16)

        distances_mm: List[float] = []
        for i in range(num_data):
            raw = int(fields[di + 6 + i], 16)
            distances_mm.append(raw * scale_factor + scale_offset)

        angles_deg = [start_angle + i * angle_step for i in range(num_data)]

        return ScanData(
            angles_deg=angles_deg,
            distances_mm=distances_mm,
            timestamp=time.time(),
            scan_counter=scan_counter,
        )

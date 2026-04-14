"""
Live 2-D LiDAR mapping application for the SICK TIM881P-2100101.

Usage
-----
# Real hardware (set your sensor's IP):
    python main.py --host 192.168.0.100

# Browser/web UI stream from the device:
    python main.py --webui --host 192.168.0.1

# Demo / simulation (no hardware required):
    python main.py --demo

Options
-------
    --host HOST        Sensor IP address       (default: 192.168.0.100)
  --port PORT        SOPAS TCP port          (default: 2111)
  --demo             Run with simulated data
  --map-size M       Map edge in metres      (default: 12.0)
  --resolution R     Grid cell size (metres) (default: 0.05)
  --max-range D      Max valid range (m)     (default: 10.0)
  --save FILE        Save final map to PNG on exit
  --continuous       Use continuous stream instead of polling
"""

from __future__ import annotations

import argparse
import sys
import time
import logging
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

from sick_tim881p import SickTIM881P, ScanData
from external_yaw import ExternalYawSource
from mapper_2d import Mapper2D
from webui_live_points import WebUIScanSource

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Simulated environment (demo mode)
# ──────────────────────────────────────────────────────────────────────────────

_ROOM_HALF_W: float = 4_500.0   # mm – half-width of simulated room
_ROOM_HALF_H: float = 3_000.0   # mm – half-height
_OBS_RADIUS:  float =   250.0   # mm – radius of moving circular obstacle
_OBS_ORBIT_R: float = 2_000.0   # mm – orbit radius of the obstacle


def _generate_demo_scan(frame: int) -> ScanData:
    """
    Return a synthetic scan of a rectangular room containing a circular
    obstacle that orbits the sensor.
    """
    start_angle = -45.0
    angle_step  = 1.0 / 3.0        # ≈ 0.3333° (270° / 811 points)
    num_points  = 811

    angles_deg = np.array(
        [start_angle + i * angle_step for i in range(num_points)],
        dtype=np.float64,
    )
    angles_rad = np.deg2rad(angles_deg)
    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)

    # Distances to room walls
    with np.errstate(divide='ignore', invalid='ignore'):
        d_px = np.where(cos_a >  1e-9,  _ROOM_HALF_W / cos_a,  np.inf)
        d_nx = np.where(cos_a < -1e-9, -_ROOM_HALF_W / cos_a,  np.inf)
        d_py = np.where(sin_a >  1e-9,  _ROOM_HALF_H / sin_a,  np.inf)
        d_ny = np.where(sin_a < -1e-9, -_ROOM_HALF_H / sin_a,  np.inf)

    d_wall = np.minimum.reduce([d_px, d_nx, d_py, d_ny])

    # Moving circular obstacle: ray–circle intersection
    obs_ang = np.deg2rad(frame * 3.0)
    obs_x   = _OBS_ORBIT_R * np.cos(obs_ang)
    obs_y   = _OBS_ORBIT_R * np.sin(obs_ang)

    b     = obs_x * cos_a + obs_y * sin_a
    c_val = obs_x ** 2 + obs_y ** 2 - _OBS_RADIUS ** 2
    disc  = b * b - c_val
    t_obs = np.where(
        disc >= 0,
        np.maximum(0.0, b - np.sqrt(np.maximum(0.0, disc))),
        np.inf,
    )

    d_mm = np.minimum(d_wall, t_obs)
    d_mm += np.random.normal(0.0, 5.0, size=num_points)   # sensor noise ±5 mm
    d_mm  = np.clip(d_mm, 50.0, 25_000.0)

    return ScanData(
        angles_deg=angles_deg.tolist(),
        distances_mm=d_mm.tolist(),
        timestamp=time.time(),
        scan_counter=frame,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Colourmap: grey = unknown, white = free, black = occupied
# ──────────────────────────────────────────────────────────────────────────────

def _make_occ_cmap() -> ListedColormap:
    """
    Build a colormap where masked (unknown) cells appear as mid-grey and
    0→100 occupancy maps black→white (gray_r).
    """
    gray_r = matplotlib.colormaps.get_cmap('gray_r')(np.linspace(0, 1, 256))
    return ListedColormap(gray_r)


# ──────────────────────────────────────────────────────────────────────────────
# Figure setup
# ──────────────────────────────────────────────────────────────────────────────

def _build_figure(mapper: Mapper2D, max_range_m: float):
    """Create and return the matplotlib figure and its two axes."""
    BG     = '#1e1e1e'
    PANEL  = '#2a2a2a'
    TITLE  = '#e0e0e0'
    TICK   = '#aaaaaa'

    fig = plt.figure(figsize=(15, 7), facecolor=BG)
    fig.suptitle(
        'SICK TIM881P-2100101 – Live 2-D LiDAR Map',
        color=TITLE, fontsize=14, fontweight='bold', y=0.98,
    )

    # ── Left panel: polar scan view ──────────────────────────────────────────
    ax_p = fig.add_subplot(121, projection='polar', facecolor='#0d1117')
    ax_p.set_title('Current Scan', color=TITLE, pad=14, fontsize=11)
    ax_p.set_theta_zero_location('E')   # 0° points right (East), matching sensor X-axis
    ax_p.set_theta_direction(1)         # counter-clockwise
    ax_p.set_rlim(0, max_range_m)
    ax_p.set_rgrids(
        np.arange(1, max_range_m + 1, max(1, int(max_range_m / 5))),
        labels=[f'{r} m' for r in np.arange(1, max_range_m + 1, max(1, int(max_range_m / 5)))],
        color=TICK, fontsize=7,
    )
    ax_p.tick_params(colors=TICK, labelsize=7)
    ax_p.grid(color='white', alpha=0.15)
    ax_p.set_facecolor('#0d1117')

    # ── Right panel: occupancy grid ──────────────────────────────────────────
    ax_m = fig.add_subplot(122, facecolor=PANEL)
    ax_m.set_title('Accumulated Occupancy Grid', color=TITLE, pad=14, fontsize=11)
    ax_m.set_aspect('equal')
    ax_m.tick_params(colors=TICK, labelsize=8)
    for spine in ax_m.spines.values():
        spine.set_edgecolor('#444444')

    # Tick labels in metres
    half_g  = mapper.grid_size / 2
    n_ticks = 5
    tick_px = np.linspace(0, mapper.grid_size, n_ticks)
    tick_m  = (tick_px - half_g) * mapper.resolution
    ax_m.set_xticks(tick_px)
    ax_m.set_yticks(tick_px)
    ax_m.set_xticklabels([f'{v:.1f}' for v in tick_m], color=TICK, fontsize=7)
    ax_m.set_yticklabels([f'{v:.1f}' for v in tick_m], color=TICK, fontsize=7)
    ax_m.set_xlabel('x  (m)', color=TICK, fontsize=9)
    ax_m.set_ylabel('y  (m)', color=TICK, fontsize=9)

    # Sensor position marker
    ox, oy = mapper.origin_gx, mapper.origin_gy
    ax_m.plot(ox, oy, 'r+', markersize=10, markeredgewidth=2, label='Sensor')
    ax_m.legend(facecolor=PANEL, edgecolor='#555', labelcolor=TITLE, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, ax_p, ax_m


# ──────────────────────────────────────────────────────────────────────────────
# Animation state container
# ──────────────────────────────────────────────────────────────────────────────

class _AppState:
    def __init__(
        self,
        mapper: Mapper2D,
        fig,
        ax_polar,
        ax_map,
        lidar: Optional[SickTIM881P],
        args: argparse.Namespace,
    ) -> None:
        self.mapper   = mapper
        self.fig      = fig
        self.ax_polar = ax_polar
        self.ax_map   = ax_map
        self.lidar    = lidar
        self.webui    = None
        self.yaw_source = None
        self.args     = args
        self.frame    = 0
        self._last_webui_warn_ts = 0.0

        cmap = _make_occ_cmap()
        cmap.set_bad(color='#555555')   # colour for unknown (masked) cells

        # Initialise grid image
        display = np.ma.masked_where(mapper.grid < 0, mapper.grid)
        self.img = ax_map.imshow(
            display,
            origin='lower',
            cmap=cmap,
            vmin=0, vmax=100,
            interpolation='nearest',
        )

        # Polar points
        (self.polar_line,) = ax_polar.plot(
            [], [], 'o',
            color='#00d4ff', markersize=1.2, alpha=0.8,
        )

        self.dynamic_points = ax_map.scatter(
            [], [],
            s=14,
            c='#ff4d4d',
            alpha=0.85,
            linewidths=0,
            label='Moving / new returns',
        )

        ax_map.legend(facecolor='#2a2a2a', edgecolor='#555', labelcolor='#e0e0e0', fontsize=8)

        # Status text
        self.status_txt = fig.text(
            0.5, 0.01,
            'Scan 0 | 0 scans accumulated',
            ha='center', color='#888888', fontsize=9,
        )

    def update(self, _frame_num: int) -> list:
        # ── Acquire scan ────────────────────────────────────────────────────
        if self.args.demo:
            scan = _generate_demo_scan(self.frame)
        elif self.args.webui:
            assert self.webui is not None
            try:
                scan = self.webui.read_scan(timeout=max(1.0, self.args.interval / 1000.0 * 4.0))
            except Exception as exc:
                now = time.time()
                # Avoid warning spam during transient websocket reconnect periods.
                if now - self._last_webui_warn_ts > 5.0:
                    logger.warning("Web UI scan error: %s", exc)
                    self._last_webui_warn_ts = now
                return []
        else:
            assert self.lidar is not None
            try:
                if self.args.continuous:
                    scan = self.lidar.read_continuous_scan()
                else:
                    scan = self.lidar.poll_scan()
            except Exception as exc:
                logger.warning("Scan error: %s", exc)
                return []

        self.frame += 1

        if self.yaw_source is not None:
            yaw_rad, yaw_fresh = self.yaw_source.read_yaw_rad()
            if yaw_rad is not None:
                self.mapper.set_external_yaw(yaw_rad, is_fresh=yaw_fresh)

        # ── Update mapper ───────────────────────────────────────────────────
        self.mapper.update(scan)

        dynamic_xy = self._compute_dynamic_overlay(scan)

        # ── Polar plot ──────────────────────────────────────────────────────
        angles_rad = np.deg2rad(scan.angles_deg)
        dists_m    = np.asarray(scan.distances_mm) / 1000.0
        valid      = (dists_m > 0.05) & (dists_m < self.args.max_range)
        self.polar_line.set_data(angles_rad[valid], dists_m[valid])

        # ── Occupancy grid ──────────────────────────────────────────────────
        display = np.ma.masked_where(self.mapper.grid < 0, self.mapper.grid)
        self.img.set_data(display)

        if dynamic_xy.size:
            self.dynamic_points.set_offsets(dynamic_xy)
        else:
            self.dynamic_points.set_offsets(np.empty((0, 2), dtype=np.float32))

        # ── Status ──────────────────────────────────────────────────────────
        map_mode = 'HOLD' if self.mapper.integration_frozen else 'LIVE'
        if self.mapper.external_yaw_active:
            yaw_state = 'EXT/fresh' if self.mapper.external_yaw_fresh else 'EXT/stale'
        else:
            yaw_state = 'SCAN'
        self.status_txt.set_text(
            f'Scan counter: {scan.scan_counter} | '
            f'Accumulated: {self.mapper.scan_count} frames | '
            f'Moving: {len(dynamic_xy)} points | '
            f'Loc score: {self.mapper.localization_score:.1f} | '
            f'Map: {map_mode} | '
            f'Yaw: {yaw_state}'
        )

        return [self.polar_line, self.img, self.dynamic_points, self.status_txt]

    def _compute_dynamic_overlay(self, scan: ScanData) -> np.ndarray:
        angles = np.deg2rad(scan.angles_deg)
        dists = np.asarray(scan.distances_mm, dtype=np.float32) / 1000.0
        valid = (dists > 0.05) & (dists < self.args.max_range)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float32)

        angles = angles[valid]
        dists = dists[valid]

        # Transform current scan from sensor-local into the same world frame
        # used by the occupancy mapper, so overlay points stay map-anchored.
        lx = dists * np.cos(angles)
        ly = dists * np.sin(angles)
        c = np.cos(self.mapper.sensor_yaw_rad)
        s = np.sin(self.mapper.sensor_yaw_rad)
        x_m = self.mapper.sensor_x_m + (c * lx - s * ly)
        y_m = self.mapper.sensor_y_m + (s * lx + c * ly)

        gx = (self.mapper.origin_gx + x_m / self.mapper.resolution).astype(np.int32)
        gy = (self.mapper.origin_gy + y_m / self.mapper.resolution).astype(np.int32)
        in_bounds = (
            (gx >= 0) & (gx < self.mapper.grid_size) &
            (gy >= 0) & (gy < self.mapper.grid_size)
        )
        if not np.any(in_bounds):
            return np.empty((0, 2), dtype=np.float32)

        gx = gx[in_bounds]
        gy = gy[in_bounds]
        x_m = x_m[in_bounds]
        y_m = y_m[in_bounds]

        prior_occ = self.mapper.grid[gy, gx]
        hit_count = self.mapper._hit[gy, gx]

        # Highlight endpoints that do not already belong to a well-established wall.
        dynamic = (self.mapper.scan_count >= 5) & ((prior_occ < 55.0) | (hit_count < 3))
        if not np.any(dynamic):
            return np.empty((0, 2), dtype=np.float32)

        px = gx[dynamic].astype(np.float32)
        py = gy[dynamic].astype(np.float32)
        return np.column_stack((px, py))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Live 2-D LiDAR map for SICK TIM881P-2100101',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--host',       default='192.168.0.100', help='Sensor IP address')
    p.add_argument('--port',       default=2111, type=int, help='SOPAS TCP port')
    p.add_argument('--demo',       action='store_true',    help='Run with simulated data')
    p.add_argument('--webui',      action='store_true',    help='Use the device web UI websocket stream')
    p.add_argument('--continuous', action='store_true',    help='Use continuous scan stream')
    p.add_argument('--viewer-id',  default='view',         help='Web UI viewer ID for websocket mode')
    p.add_argument('--map-size',   default=12.0, type=float, metavar='M',
                   help='Square map edge length (metres)')
    p.add_argument('--resolution', default=0.05, type=float, metavar='R',
                   help='Grid cell size (metres)')
    p.add_argument('--max-range',  default=10.0, type=float, metavar='D',
                   help='Max valid measurement range (metres)')
    p.add_argument('--interval',   default=100, type=int,
                   help='Animation interval in milliseconds')
    p.add_argument('--yaw-file',   default=None,
                   help='Path to a text or JSON file containing external yaw')
    p.add_argument('--yaw-unit',   choices=('deg', 'rad'), default='deg',
                   help='Unit for plain-text yaw files or JSON yaw field')
    p.add_argument('--yaw-max-age', default=0.75, type=float,
                   help='Maximum allowed age in seconds for JSON timestamped yaw data')
    p.add_argument('--save',       default=None, metavar='FILE',
                   help='Save the final map as a PNG when closing')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    mapper = Mapper2D(
        map_size_m=args.map_size,
        resolution_m=args.resolution,
        max_range_m=args.max_range,
    )

    lidar: Optional[SickTIM881P] = None
    webui: Optional[WebUIScanSource] = None
    yaw_source: Optional[ExternalYawSource] = None

    try:
        if args.webui and args.continuous:
            logger.warning("Ignoring --continuous because --webui uses its own live stream.")

        if args.webui and args.demo:
            raise ValueError("Choose either --demo or --webui, not both.")

        if args.webui:
            logger.info("Connecting to TiM881P web UI stream at %s …", args.host)
            webui = WebUIScanSource(host=args.host, viewer_id=args.viewer_id)
            webui.start()
        elif not args.demo:
            logger.info("Connecting to SICK TIM881P at %s:%d …", args.host, args.port)
            lidar = SickTIM881P(host=args.host, port=args.port)
            lidar.connect()
            if args.continuous:
                lidar.start_continuous_scan()

        if args.yaw_file:
            logger.info("Reading external yaw from %s", args.yaw_file)
            yaw_source = ExternalYawSource(
                path=args.yaw_file,
                unit=args.yaw_unit,
                max_age_sec=args.yaw_max_age,
            )

        fig, ax_polar, ax_map = _build_figure(mapper, args.max_range)
        state = _AppState(mapper, fig, ax_polar, ax_map, lidar, args)
        state.webui = webui
        state.yaw_source = yaw_source

        ani = animation.FuncAnimation(
            fig,
            state.update,
            interval=args.interval,
            blit=False,
            cache_frame_data=False,
        )

        def _on_close(_event) -> None:
            if args.save:
                fig.savefig(args.save, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
                logger.info("Map saved to %s", args.save)

        fig.canvas.mpl_connect('close_event', _on_close)

        plt.show()

    except ConnectionError as exc:
        logger.error("%s", exc)
        logger.error(
            "Tip: check the sensor IP/port, or use --demo to run without hardware."
        )
        sys.exit(1)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        if lidar is not None:
            lidar.disconnect()
        if webui is not None:
            webui.close()


if __name__ == '__main__':
    main()

"""
2-D Occupancy Grid Mapper for SICK TIM881P data.

The sensor is placed at the grid origin.
Each scan ray is traced through the grid with a Bresenham-style linspace:
  * cells along the ray (excluding the endpoint) are marked as free
  * the endpoint cell is marked as occupied
Occupancy probability is maintained as a simple hit/(hit+miss) ratio
in the range [0, 100].  Cells that have never been visited stay at -1
(unknown).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from sick_tim881p import ScanData


class Mapper2D:
    """
    Accumulates LiDAR scans into a 2-D occupancy grid.

    Parameters
    ----------
    map_size_m   : Side length of the square map in metres.
    resolution_m : Size of one grid cell in metres.
    max_range_m  : Measurements beyond this distance are discarded.
    """

    def __init__(
        self,
        map_size_m: float = 12.0,
        resolution_m: float = 0.05,
        max_range_m: float = 10.0,
    ) -> None:
        self.resolution   = resolution_m
        self.max_range_m  = max_range_m
        self.grid_size    = int(map_size_m / resolution_m)

        # Sensor placed at the centre of the grid
        self.origin_gx = self.grid_size // 2
        self.origin_gy = self.grid_size // 2

        # Hit/miss counters (int32 avoids overflow on long runs)
        self._hit  = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._miss = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Occupancy probability [-1 unknown, 0..100 occupied %]
        self.grid = np.full(
            (self.grid_size, self.grid_size), -1.0, dtype=np.float32
        )

        # Log-odds occupancy state for faster adaptation to scene changes.
        self._log_odds = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self._candidate = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._lo_occ = 1.2
        self._lo_free = 0.45
        self._lo_min = -4.0
        self._lo_max = 4.0
        self.localization_score = 0.0
        self.integration_frozen = False
        self.external_yaw_active = False
        self.external_yaw_fresh = False
        self._external_yaw_raw_rad: float | None = None
        self._external_yaw_offset_rad: float | None = None
        self._wall_anchor_phi_world_rad: float | None = None
        self._wall_anchor_rho_world_m: float | None = None
        self._wall_anchor_miss_count = 0
        self.wall_anchor_score = 0.0

        self.scan_count = 0

        # Sensor pose in world coordinates (metres, radians).
        self.sensor_x_m = 0.0
        self.sensor_y_m = 0.0
        self.sensor_yaw_rad = 0.0

    def set_external_yaw(self, yaw_rad: float | None, is_fresh: bool = True) -> None:
        if yaw_rad is None:
            self.external_yaw_fresh = False
            return

        wrapped = self._wrap_angle(yaw_rad)
        if self._external_yaw_offset_rad is None:
            self._external_yaw_offset_rad = self._wrap_angle(self.sensor_yaw_rad - wrapped)

        self._external_yaw_raw_rad = wrapped
        self.external_yaw_active = True
        self.external_yaw_fresh = is_fresh

    def clear_external_yaw(self) -> None:
        self.external_yaw_active = False
        self.external_yaw_fresh = False
        self._external_yaw_raw_rad = None
        self._external_yaw_offset_rad = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, scan: ScanData) -> None:
        """Integrate one scan frame into the occupancy grid."""
        self.integration_frozen = False

        angles = np.deg2rad(scan.angles_deg)
        dists  = np.asarray(scan.distances_mm, dtype=np.float32) / 1000.0  # → metres

        # Reject out-of-range or zero readings
        valid = (dists > 0.05) & (dists < self.max_range_m)
        angles = angles[valid]
        dists  = dists[valid]

        if dists.size >= 5:
            d_prev2 = np.roll(dists, 2)
            d_prev1 = np.roll(dists, 1)
            d_next1 = np.roll(dists, -1)
            d_next2 = np.roll(dists, -2)
            window = np.stack((d_prev2, d_prev1, dists, d_next1, d_next2), axis=0)
            dists = np.median(window, axis=0)

        if angles.size == 0:
            return

        local_x = dists * np.cos(angles)
        local_y = dists * np.sin(angles)
        wall_line = self._extract_wall_line(local_x, local_y)
        self.wall_anchor_score = 0.0 if wall_line is None else float(wall_line[2])

        external_yaw = self._get_external_yaw_world_rad()
        if external_yaw is not None:
            self.sensor_yaw_rad = external_yaw

        self._localize_against_map(local_x, local_y, wall_line)

        if not self._should_integrate_scan():
            self.integration_frozen = True
            return

        c = np.cos(self.sensor_yaw_rad)
        s = np.sin(self.sensor_yaw_rad)
        world_x = self.sensor_x_m + (c * local_x - s * local_y)
        world_y = self.sensor_y_m + (s * local_x + c * local_y)

        ox = int(round(self.origin_gx + self.sensor_x_m / self.resolution))
        oy = int(round(self.origin_gy + self.sensor_y_m / self.resolution))

        end_gx = (self.origin_gx + world_x / self.resolution).astype(np.int32)
        end_gy = (self.origin_gy + world_y / self.resolution).astype(np.int32)

        neigh_prev = np.abs(dists - np.roll(dists, 1))
        neigh_next = np.abs(dists - np.roll(dists, -1))
        stable_endpoint = (np.maximum(neigh_prev, neigh_next) < 0.18)
        if stable_endpoint.size:
            stable_endpoint[0] = False
            stable_endpoint[-1] = False

        for tx, ty, is_stable in zip(end_gx, end_gy, stable_endpoint):
            n_steps = max(abs(int(tx) - ox), abs(int(ty) - oy)) + 1
            if n_steps < 2:
                continue

            xs = np.round(np.linspace(ox, tx, n_steps)).astype(np.int32)
            ys = np.round(np.linspace(oy, ty, n_steps)).astype(np.int32)

            # Clip to grid bounds
            in_bounds = (
                (xs >= 0) & (xs < self.grid_size) &
                (ys >= 0) & (ys < self.grid_size)
            )
            xs = xs[in_bounds]
            ys = ys[in_bounds]

            if len(xs) == 0:
                continue

            free_stop = max(0, len(xs) - 2)
            if free_stop > 0:
                # Leave the last couple of cells untouched so endpoint jitter does not
                # repeatedly erase and redraw obstacles.
                np.add.at(self._miss, (ys[:free_stop], xs[:free_stop]), 1)
                np.add.at(self._log_odds, (ys[:free_stop], xs[:free_stop]), -self._lo_free)
                self._visited[ys[:free_stop], xs[:free_stop]] = True
                decay_vals = self._candidate[ys[:free_stop], xs[:free_stop]].astype(np.int16) - 1
                self._candidate[ys[:free_stop], xs[:free_stop]] = np.clip(decay_vals, 0, 255).astype(np.uint8)

            end_x = xs[-1]
            end_y = ys[-1]
            self._candidate[end_y, end_x] = min(255, int(self._candidate[end_y, end_x]) + 1)

            promote = is_stable and (
                self._candidate[end_y, end_x] >= 3 or self._hit[end_y, end_x] >= 2
            )
            if promote:
                self._hit[end_y, end_x] += 1
                self._log_odds[end_y, end_x] += self._lo_occ
                self._visited[end_y, end_x] = True

        np.clip(self._log_odds, self._lo_min, self._lo_max, out=self._log_odds)

        visited = self._visited
        self.grid[visited] = (1.0 / (1.0 + np.exp(-self._log_odds[visited]))) * 100.0

        self._update_wall_anchor(wall_line)
        self.scan_count += 1

    def _localize_against_map(
        self,
        local_x: np.ndarray,
        local_y: np.ndarray,
        wall_line: tuple[float, float, float] | None = None,
    ) -> None:
        """Correlative scan matching against the accumulated map (lightweight 2-D SLAM step)."""
        external_yaw = self._get_external_yaw_world_rad()

        # Need enough stable map memory before localization can lock robustly.
        if self.scan_count < 12:
            self.localization_score = 0.0
            return

        if local_x.size < 80:
            self.localization_score = 0.0
            return

        # Use moderately far points for localization; near points are noisier and dynamic.
        ranges = np.sqrt(local_x * local_x + local_y * local_y)
        keep = (ranges > 0.4) & (ranges < self.max_range_m * 0.95)
        if np.count_nonzero(keep) < 80:
            self.localization_score = 0.0
            return
        px_local = local_x[keep][::4]
        py_local = local_y[keep][::4]
        if px_local.size < 40:
            self.localization_score = 0.0
            return

        occ = self.grid
        hit = self._hit
        size = self.grid_size
        inv_res = 1.0 / self.resolution

        # Coarse-to-fine search around current pose estimate.
        if external_yaw is None:
            yaw_candidates_1 = np.deg2rad(np.arange(-8.0, 8.1, 2.0)) + self.sensor_yaw_rad
        else:
            yaw_candidates_1 = np.deg2rad(np.arange(-1.5, 1.51, 0.5)) + external_yaw
        dx_candidates_1 = np.arange(-0.15, 0.151, 0.05)
        dy_candidates_1 = np.arange(-0.15, 0.151, 0.05)

        best_score, best_pose = self._search_pose(
            px_local,
            py_local,
            yaw_candidates_1,
            dx_candidates_1,
            dy_candidates_1,
            occ,
            hit,
            size,
            self.origin_gx,
            self.origin_gy,
            inv_res,
            wall_line=wall_line,
            wall_anchor=self._get_wall_anchor(),
        )
        if best_pose is None:
            return

        yaw0, x0, y0 = best_pose
        if external_yaw is None:
            yaw_candidates_2 = np.deg2rad(np.arange(-2.0, 2.01, 0.5)) + yaw0
        else:
            yaw_candidates_2 = np.deg2rad(np.arange(-0.5, 0.51, 0.25)) + external_yaw
        dx_candidates_2 = np.arange(-0.04, 0.041, 0.02)
        dy_candidates_2 = np.arange(-0.04, 0.041, 0.02)

        best_score_2, best_pose_2 = self._search_pose(
            px_local,
            py_local,
            yaw_candidates_2,
            dx_candidates_2,
            dy_candidates_2,
            occ,
            hit,
            size,
            self.origin_gx,
            self.origin_gy,
            inv_res,
            anchor_xy=(x0, y0),
            wall_line=wall_line,
            wall_anchor=self._get_wall_anchor(),
        )
        if best_pose_2 is None:
            return

        best_yaw, best_x, best_y = best_pose_2

        # Confidence gate: only accept updates that are meaningfully aligned.
        self.localization_score = float(best_score_2)
        min_score = 10.0 if external_yaw is not None else 14.0
        if best_score_2 < min_score:
            return

        # Clamp per-frame correction so fast turns cannot drag the map.
        dyaw = float(np.arctan2(np.sin(best_yaw - self.sensor_yaw_rad), np.cos(best_yaw - self.sensor_yaw_rad)))
        dx = float(best_x - self.sensor_x_m)
        dy = float(best_y - self.sensor_y_m)
        trans = float(np.hypot(dx, dy))

        max_dyaw = np.deg2rad(6.0)
        max_trans = 0.08 if abs(dyaw) < np.deg2rad(2.0) else 0.025

        dyaw = float(np.clip(dyaw, -max_dyaw, max_dyaw))
        if trans > max_trans and trans > 1e-6:
            scale = max_trans / trans
            dx *= scale
            dy *= scale

        alpha_rot = 0.15 if external_yaw is not None else 0.35
        alpha_xy = 0.30
        if external_yaw is None:
            self.sensor_yaw_rad = self._wrap_angle(self.sensor_yaw_rad + alpha_rot * dyaw)
        else:
            self.sensor_yaw_rad = self._wrap_angle(external_yaw + alpha_rot * dyaw)
        self.sensor_x_m += alpha_xy * dx
        self.sensor_y_m += alpha_xy * dy

    def _get_external_yaw_world_rad(self) -> float | None:
        if not self.external_yaw_active or self._external_yaw_raw_rad is None:
            return None
        if self._external_yaw_offset_rad is None:
            return None
        return self._wrap_angle(self._external_yaw_raw_rad + self._external_yaw_offset_rad)

    def _get_wall_anchor(self) -> tuple[float, float] | None:
        if self._wall_anchor_phi_world_rad is None or self._wall_anchor_rho_world_m is None:
            return None
        return (self._wall_anchor_phi_world_rad, self._wall_anchor_rho_world_m)

    def _update_wall_anchor(self, wall_line: tuple[float, float, float] | None) -> None:
        if wall_line is None:
            self._wall_anchor_miss_count += 1
            if self._wall_anchor_miss_count >= 25:
                self._wall_anchor_phi_world_rad = None
                self._wall_anchor_rho_world_m = None
            return

        phi_local, rho_local, quality = wall_line
        if quality < 20.0:
            self._wall_anchor_miss_count += 1
            return

        phi_world, rho_world = self._line_local_to_world(
            phi_local,
            rho_local,
            self.sensor_yaw_rad,
            self.sensor_x_m,
            self.sensor_y_m,
        )

        if self._wall_anchor_phi_world_rad is None or self._wall_anchor_rho_world_m is None:
            self._wall_anchor_phi_world_rad = phi_world
            self._wall_anchor_rho_world_m = rho_world
            self._wall_anchor_miss_count = 0
            return

        angle_err = self._line_angle_diff(phi_world, self._wall_anchor_phi_world_rad)
        rho_err = abs(rho_world - self._wall_anchor_rho_world_m)
        if angle_err <= math.radians(6.0) and rho_err <= 0.25:
            alpha = 0.25
            self._wall_anchor_phi_world_rad = self._blend_line_angle(
                self._wall_anchor_phi_world_rad,
                phi_world,
                alpha,
            )
            self._wall_anchor_rho_world_m = (
                (1.0 - alpha) * self._wall_anchor_rho_world_m + alpha * rho_world
            )
            self._wall_anchor_miss_count = 0
        else:
            self._wall_anchor_miss_count += 1

    def _should_integrate_scan(self) -> bool:
        if self.scan_count < 12:
            return True
        if self.external_yaw_active:
            if not self.external_yaw_fresh:
                return False
            return self.localization_score >= 10.0
        return self.localization_score >= 14.0

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    @classmethod
    def _wrap_line_angle(cls, angle_rad: float) -> float:
        wrapped = cls._wrap_angle(angle_rad)
        if wrapped < 0.0:
            wrapped += math.pi
        if wrapped >= math.pi:
            wrapped -= math.pi
        return wrapped

    @classmethod
    def _line_angle_diff(cls, a_rad: float, b_rad: float) -> float:
        diff = abs(cls._wrap_line_angle(a_rad) - cls._wrap_line_angle(b_rad))
        return min(diff, math.pi - diff)

    @classmethod
    def _blend_line_angle(cls, base_rad: float, new_rad: float, alpha: float) -> float:
        base = cls._wrap_line_angle(base_rad)
        new = cls._wrap_line_angle(new_rad)
        delta = cls._wrap_angle(new - base)
        if delta > math.pi / 2:
            delta -= math.pi
        elif delta < -math.pi / 2:
            delta += math.pi
        return cls._wrap_line_angle(base + alpha * delta)

    @classmethod
    def _canonical_line(cls, phi_rad: float, rho_m: float) -> tuple[float, float]:
        phi = cls._wrap_angle(phi_rad)
        rho = float(rho_m)
        if rho < 0.0:
            rho = -rho
            phi = cls._wrap_angle(phi + math.pi)
        return cls._wrap_line_angle(phi), rho

    @classmethod
    def _line_local_to_world(
        cls,
        phi_local_rad: float,
        rho_local_m: float,
        yaw_world_rad: float,
        x_world_m: float,
        y_world_m: float,
    ) -> tuple[float, float]:
        phi_world = cls._wrap_angle(phi_local_rad + yaw_world_rad)
        rho_world = rho_local_m + x_world_m * math.cos(phi_world) + y_world_m * math.sin(phi_world)
        return cls._canonical_line(phi_world, rho_world)

    @classmethod
    def _extract_wall_line(
        cls,
        local_x: np.ndarray,
        local_y: np.ndarray,
    ) -> tuple[float, float, float] | None:
        if local_x.size < 24:
            return None

        pts = np.column_stack((local_x, local_y)).astype(np.float32, copy=False)
        if pts.shape[0] < 24:
            return None

        jumps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        split_idx = np.nonzero(jumps > 0.30)[0] + 1
        segments = np.split(pts, split_idx)

        best: tuple[float, float, float] | None = None
        best_score = -1.0
        for seg in segments:
            if seg.shape[0] < 18:
                continue

            window_size = min(36, seg.shape[0])
            step = max(4, window_size // 6)
            for start in range(0, max(1, seg.shape[0] - window_size + 1), step):
                window = seg[start:start + window_size]
                if window.shape[0] < 18:
                    continue

                centroid = np.mean(window, axis=0)
                centered = window - centroid
                cov = centered.T @ centered / max(1, window.shape[0] - 1)
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)
                normal = eigvecs[:, order[0]]

                rho = float(np.dot(centroid, normal))
                if rho < 0.0:
                    rho = -rho
                    normal = -normal

                residuals = np.abs(window @ normal - rho)
                mean_res = float(np.mean(residuals))
                inlier_ratio = float(np.mean(residuals < 0.06))
                span = float(np.linalg.norm(window[-1] - window[0]))
                if span < 0.8 or inlier_ratio < 0.82 or mean_res > 0.07:
                    continue

                score = span * window.shape[0] * inlier_ratio / max(0.02, mean_res)
                if score <= best_score:
                    continue

                phi = math.atan2(float(normal[1]), float(normal[0]))
                best_score = score
                best = cls._canonical_line(phi, rho) + (float(score),)

        return best

    @staticmethod
    def _search_pose(
        px_local: np.ndarray,
        py_local: np.ndarray,
        yaw_candidates: np.ndarray,
        dx_candidates: np.ndarray,
        dy_candidates: np.ndarray,
        occ: np.ndarray,
        hit: np.ndarray,
        grid_size: int,
        origin_gx: int,
        origin_gy: int,
        inv_res: float,
        anchor_xy: tuple[float, float] | None = None,
        wall_line: tuple[float, float, float] | None = None,
        wall_anchor: tuple[float, float] | None = None,
    ) -> tuple[float, tuple[float, float, float] | None]:
        best_score = -1e9
        best_pose = None

        if anchor_xy is None:
            anchor_x, anchor_y = 0.0, 0.0
        else:
            anchor_x, anchor_y = anchor_xy

        for yaw in yaw_candidates:
            c = np.cos(yaw)
            s = np.sin(yaw)
            rot_x = c * px_local - s * py_local
            rot_y = s * px_local + c * py_local

            for ddx in dx_candidates:
                for ddy in dy_candidates:
                    x = anchor_x + ddx
                    y = anchor_y + ddy

                    gx = np.round(origin_gx + (x + rot_x) * inv_res).astype(np.int32)
                    gy = np.round(origin_gy + (y + rot_y) * inv_res).astype(np.int32)

                    inb = (gx >= 0) & (gx < grid_size) & (gy >= 0) & (gy < grid_size)
                    if np.count_nonzero(inb) < 25:
                        continue

                    gxx = gx[inb]
                    gyy = gy[inb]
                    occ_vals = occ[gyy, gxx]
                    hit_vals = hit[gyy, gxx]

                    # Reward overlap with occupied/high-hit map memory.
                    score = float(np.mean(np.clip(occ_vals, 0.0, 100.0)))
                    score += 12.0 * float(np.mean(hit_vals >= 4))
                    score += 6.0 * float(np.mean(hit_vals >= 8))

                    if wall_line is not None and wall_anchor is not None:
                        phi_local, rho_local, quality = wall_line
                        anchor_phi, anchor_rho = wall_anchor
                        phi_world, rho_world = Mapper2D._line_local_to_world(
                            phi_local,
                            rho_local,
                            float(yaw),
                            float(x),
                            float(y),
                        )
                        angle_err = Mapper2D._line_angle_diff(phi_world, anchor_phi)
                        rho_err = abs(rho_world - anchor_rho)
                        angle_gate = math.radians(12.0)
                        rho_gate = 0.45
                        if angle_err <= angle_gate and rho_err <= rho_gate:
                            score += min(36.0, quality * 0.06)
                            score += 18.0 * (1.0 - angle_err / angle_gate)
                            score += 12.0 * (1.0 - rho_err / rho_gate)
                        else:
                            score -= 10.0 + 25.0 * min(1.0, angle_err / math.radians(20.0))
                            score -= 15.0 * min(1.0, rho_err / 0.75)

                    if score > best_score:
                        best_score = score
                        best_pose = (float(yaw), float(x), float(y))

        return best_score, best_pose

    def to_xy(self, scan: ScanData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a scan to Cartesian (x, y) in metres for scatter plotting.
        Returns only valid points within max_range.
        """
        angles = np.deg2rad(scan.angles_deg)
        dists  = np.asarray(scan.distances_mm) / 1000.0
        valid  = (dists > 0.05) & (dists < self.max_range_m)
        x = dists[valid] * np.cos(angles[valid])
        y = dists[valid] * np.sin(angles[valid])
        return x, y

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._hit[:]  = 0
        self._miss[:] = 0
        self.grid[:]  = -1.0
        self._log_odds[:] = 0.0
        self._visited[:] = False
        self._candidate[:] = 0
        self.scan_count = 0
        self.sensor_x_m = 0.0
        self.sensor_y_m = 0.0
        self.sensor_yaw_rad = 0.0
        self.localization_score = 0.0
        self.integration_frozen = False
        self.external_yaw_active = False
        self.external_yaw_fresh = False
        self._external_yaw_raw_rad = None
        self._external_yaw_offset_rad = None
        self._wall_anchor_phi_world_rad = None
        self._wall_anchor_rho_world_m = None
        self._wall_anchor_miss_count = 0
        self.wall_anchor_score = 0.0

#!/usr/bin/env python
"""
Live point coordinates from SICK TIM881P LiDAR.
Prints X, Y metre coordinates in real-time.
"""

import argparse
import sys
import numpy as np
from sick_tim881p import SickTIM881P


def main():
    parser = argparse.ArgumentParser(description='Live LiDAR points')
    parser.add_argument('--host', default='192.168.0.100', help='Sensor IP')
    parser.add_argument('--port', default=2111, type=int, help='SOPAS port')
    parser.add_argument('--max-range', default=10.0, type=float, help='Max range in metres')
    parser.add_argument('--continuous', action='store_true', help='Use continuous stream')
    args = parser.parse_args()

    try:
        with SickTIM881P(host=args.host, port=args.port, timeout=5.0) as lidar:
            if args.continuous:
                lidar.start_continuous_scan()
                reader = lidar.read_continuous_scan
            else:
                reader = lidar.poll_scan

            scan_count = 0
            while True:
                try:
                    scan = reader()
                    scan_count += 1

                    # Convert to Cartesian
                    angles = np.deg2rad(scan.angles_deg)
                    dists = np.asarray(scan.distances_mm, dtype=float) / 1000.0

                    # Filter
                    valid = (dists > 0.05) & (dists < args.max_range)
                    x = dists[valid] * np.cos(angles[valid])
                    y = dists[valid] * np.sin(angles[valid])

                    print(f"\n=== Scan {scan_count} | {len(x)} points ===")
                    for xi, yi in zip(x, y):
                        print(f"  x: {xi:7.3f}  y: {yi:7.3f}")

                except KeyboardInterrupt:
                    print("\nStopped.")
                    break
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    break

    except ConnectionError as e:
        print(f"Failed to connect: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

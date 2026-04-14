# SickLidar

Run the SICK TIM881P live LiDAR viewer from this repository.

The main entrypoint is [lidar/main.py](/Users/gecsanreyes/Documents/SickLidar/SickLidar/lidar/main.py).

## Install

From the repository root:

```bash
cd /Users/gecsanreyes/Documents/SickLidar/SickLidar
python3 -m venv .venv
source .venv/bin/activate
pip install -r lidar/requirements.txt
```

## How We Ran It Before

If you run it from inside the `lidar/` folder, this is the command you were using before:

```bash
cd /Users/gecsanreyes/Documents/SickLidar/SickLidar/lidar
/Users/gecsanreyes/Documents/SickLidar/SickLidar/.venv/bin/python main.py --webui --host 192.168.0.1 --yaw-file /tmp/lidar_yaw.json
```

That command only works if your current directory is `lidar/`.

## Run From Repo Root

Equivalent command from the repository root:

```bash
/Users/gecsanreyes/Documents/SickLidar/SickLidar/.venv/bin/python lidar/main.py --webui --host 192.168.0.1 --yaw-file /tmp/lidar_yaw.json
```

If you want to use a local virtual environment in this repo instead:

```bash
cd /Users/gecsanreyes/Documents/SickLidar/SickLidar
source .venv/bin/activate
python lidar/main.py --webui --host 192.168.0.1 --yaw-file /tmp/lidar_yaw.json
```

## Other Run Modes

Demo mode:

```bash
python lidar/main.py --demo
```

Direct TCP mode:

```bash
python lidar/main.py --host 192.168.0.100
```

## Useful Flags

- `--webui`: use the device Web UI websocket stream
- `--host`: LiDAR IP address
- `--yaw-file`: path to the external yaw JSON or text file
- `--viewer-id`: Web UI viewer ID if needed
- `--continuous`: continuous scan mode for direct TCP
- `--save`: save the final map image when closing

## Requirements

- Python 3.10+ recommended
- Dependencies from [lidar/requirements.txt](/Users/gecsanreyes/Documents/SickLidar/SickLidar/lidar/requirements.txt)

## Files

- [README.md](/Users/gecsanreyes/Documents/SickLidar/SickLidar/README.md)
- [lidar/main.py](/Users/gecsanreyes/Documents/SickLidar/SickLidar/lidar/main.py)
- [lidar/requirements.txt](/Users/gecsanreyes/Documents/SickLidar/SickLidar/lidar/requirements.txt)

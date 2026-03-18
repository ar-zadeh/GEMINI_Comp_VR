import argparse
import json
import math
import socket
import time


def build_message(device: str, yaw_deg: float, pitch_deg: float, roll_deg: float, x: float, y: float, z: float) -> bytes:
    payload = {
        "device": device,
        "pos": [x, y, z],
        "rot": [roll_deg, pitch_deg, yaw_deg],
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def run_server(
    host: str,
    port: int,
    device: str,
    yaw_min: float,
    yaw_max: float,
    yaw_speed_deg_per_sec: float,
    hz: float,
    pitch: float,
    roll: float,
    pos_x: float,
    pos_y: float,
    pos_z: float,
) -> None:
    period = 1.0 / hz
    yaw_center = (yaw_min + yaw_max) * 0.5
    yaw_amplitude = abs(yaw_max - yaw_min) * 0.5

    if yaw_amplitude <= 0.0:
        raise ValueError("yaw_min and yaw_max must differ")

    if yaw_speed_deg_per_sec <= 0.0:
        raise ValueError("yaw_speed_deg_per_sec must be > 0")

    omega = yaw_speed_deg_per_sec / yaw_amplitude

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)

    print(f"[yaw_sweep_server] Listening on {host}:{port} for driver connection...")
    conn, addr = server.accept()
    print(f"[yaw_sweep_server] Connected by {addr}")

    start = time.perf_counter()
    sent = 0

    try:
        while True:
            t = time.perf_counter() - start
            yaw = yaw_center + yaw_amplitude * math.sin(omega * t)
            msg = build_message(device, yaw, pitch, roll, pos_x, pos_y, pos_z)
            conn.sendall(msg)
            sent += 1

            if sent % int(max(hz, 1)) == 0:
                print(f"[yaw_sweep_server] yaw={yaw:7.2f} deg")

            time.sleep(period)
    except (BrokenPipeError, ConnectionResetError):
        print("[yaw_sweep_server] Driver disconnected.")
    except KeyboardInterrupt:
        print("\n[yaw_sweep_server] Stopped by user.")
    finally:
        try:
            conn.close()
        finally:
            server.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCP yaw sweep server for OpenEye SteamVR driver")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (must match tcpHost in vrsettings)")
    parser.add_argument("--port", type=int, default=5555, help="Port to bind (must match tcpPort in vrsettings)")
    parser.add_argument("--device", default="headset", choices=["headset", "controller1", "controller2"], help="Device target")

    parser.add_argument("--yaw-min", type=float, default=-30.0, help="Minimum yaw (degrees)")
    parser.add_argument("--yaw-max", type=float, default=30.0, help="Maximum yaw (degrees)")
    parser.add_argument("--yaw-speed", type=float, default=45.0, help="Sweep speed in deg/s")
    parser.add_argument("--hz", type=float, default=60.0, help="Send rate in Hz")

    parser.add_argument("--pitch", type=float, default=0.0, help="Fixed pitch (degrees)")
    parser.add_argument("--roll", type=float, default=0.0, help="Fixed roll (degrees)")
    parser.add_argument("--x", type=float, default=0.0, help="Fixed X position (meters)")
    parser.add_argument("--y", type=float, default=0.0, help="Fixed Y position (meters)")
    parser.add_argument("--z", type=float, default=0.0, help="Fixed Z position (meters)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_server(
        host=args.host,
        port=args.port,
        device=args.device,
        yaw_min=args.yaw_min,
        yaw_max=args.yaw_max,
        yaw_speed_deg_per_sec=args.yaw_speed,
        hz=args.hz,
        pitch=args.pitch,
        roll=args.roll,
        pos_x=args.x,
        pos_y=args.y,
        pos_z=args.z,
    )

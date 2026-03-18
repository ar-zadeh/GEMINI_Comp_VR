#!/usr/bin/env python3
"""
Companion app that redirects the play space origin using IVRChaperoneSetup.

This adjusts the standing or seated zero pose to raw tracking pose, which
rotates/translates the entire virtual world relative to physical space.
"""

import argparse
import math
import sys
import time

import openvr


def _mat_mul(a, b):
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
    return out


def _rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    # Right-handed coordinate system. Order: yaw (Y), pitch (X), roll (Z).
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)

    ry = [
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ]
    rx = [
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ]
    rz = [
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ]

    # R = Rz * Rx * Ry
    return _mat_mul(rz, _mat_mul(rx, ry))


def _apply_pivot_translation(rot, x, y, z, pivot):
    if pivot is None:
        return x, y, z
    px, py, pz = pivot
    # Rotate around pivot: R*(p) + t' == p + t  => t' = t + p - R*p
    rx = rot[0][0] * px + rot[0][1] * py + rot[0][2] * pz
    ry = rot[1][0] * px + rot[1][1] * py + rot[1][2] * pz
    rz = rot[2][0] * px + rot[2][1] * py + rot[2][2] * pz
    return x + px - rx, y + py - ry, z + pz - rz


def _make_transform(pitch, yaw, roll, x, y, z, pivot=None):
    rot = _rotation_matrix(pitch, yaw, roll)
    x, y, z = _apply_pivot_translation(rot, x, y, z, pivot)
    mat = openvr.HmdMatrix34_t()
    mat.m[0][0] = rot[0][0]
    mat.m[0][1] = rot[0][1]
    mat.m[0][2] = rot[0][2]
    mat.m[0][3] = x
    mat.m[1][0] = rot[1][0]
    mat.m[1][1] = rot[1][1]
    mat.m[1][2] = rot[1][2]
    mat.m[1][3] = y
    mat.m[2][0] = rot[2][0]
    mat.m[2][1] = rot[2][1]
    mat.m[2][2] = rot[2][2]
    mat.m[2][3] = z
    return mat


def _call_method(obj, names, *args):
    for name in names:
        fn = getattr(obj, name, None)
        if fn:
            return fn(*args)
    raise AttributeError("No matching method: " + ", ".join(names))


def _get_hmd_pivot():
    system = openvr.VRSystem()
    poses = system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
    )
    hmd_index = openvr.k_unTrackedDeviceIndex_Hmd
    pose = poses[hmd_index]
    if not pose.bPoseIsValid:
        raise RuntimeError("HMD pose not valid")
    m = pose.mDeviceToAbsoluteTracking
    return (m[0][3], m[1][3], m[2][3])


def apply_chaperone_transform(args):
    openvr.init(openvr.VRApplication_Background)
    try:
        pivot = None
        if args.pivot_hmd:
            pivot = _get_hmd_pivot()
        elif args.pivot_x is not None or args.pivot_y is not None or args.pivot_z is not None:
            pivot = (
                args.pivot_x or 0.0,
                args.pivot_y or 0.0,
                args.pivot_z or 0.0,
            )
        setup = openvr.VRChaperoneSetup()
        mat = _make_transform(
            args.pitch, args.yaw, args.roll, args.x, args.y, args.z, pivot
        )

        if args.seated:
            _call_method(
                setup,
                [
                    "setWorkingSeatedZeroPoseToRawTrackingPose",
                    "SetWorkingSeatedZeroPoseToRawTrackingPose",
                ],
                mat,
            )
            target = "seated"
        else:
            _call_method(
                setup,
                [
                    "setWorkingStandingZeroPoseToRawTrackingPose",
                    "SetWorkingStandingZeroPoseToRawTrackingPose",
                ],
                mat,
            )
            target = "standing"

        _call_method(
            setup,
            ["commitWorkingCopy", "CommitWorkingCopy"],
            openvr.EChaperoneConfigFile_Live,
        )

        if args.save_default:
            _call_method(
                setup,
                ["commitWorkingCopy", "CommitWorkingCopy"],
                openvr.EChaperoneConfigFile_Default,
            )

        pivot_msg = ""
        if pivot is not None:
            pivot_msg = f" pivot=({pivot[0]:.3f},{pivot[1]:.3f},{pivot[2]:.3f})"
        print(
            f"Applied {target} chaperone transform: "
            f"rot(pitch={args.pitch}, yaw={args.yaw}, roll={args.roll}) "
            f"pos(x={args.x}, y={args.y}, z={args.z}){pivot_msg}"
        )
    finally:
        openvr.shutdown()


def _read_key():
    # Windows-only: use msvcrt for non-blocking key input.
    import msvcrt
    if not msvcrt.kbhit():
        return None
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):
        ch2 = msvcrt.getch()
        return (ch, ch2)
    return ch


def run_interactive(args):
    print("Interactive mode:")
    print("  Move: W/A/S/D, Up/Down: Q/E")
    print("  Rotate: Arrow keys (Left/Right = yaw, Up/Down = pitch)")
    print("  Roll: Z/X, Reset: R, Quit: ESC")
    print("  Rotation pivot: agent (HMD) by default")

    # Live state
    x, y, z = args.x, args.y, args.z
    pitch, yaw, roll = args.pitch, args.yaw, args.roll

    openvr.init(openvr.VRApplication_Background)
    try:
        setup = openvr.VRChaperoneSetup()
        while True:
            key = _read_key()
            if key is None:
                time.sleep(0.01)
                continue

            dirty = False

            if key == b"\x1b":  # ESC
                break
            if key in (b"w", b"W"):
                z -= args.step
                dirty = True
            elif key in (b"s", b"S"):
                z += args.step
                dirty = True
            elif key in (b"a", b"A"):
                x -= args.step
                dirty = True
            elif key in (b"d", b"D"):
                x += args.step
                dirty = True
            elif key in (b"q", b"Q"):
                y += args.step
                dirty = True
            elif key in (b"e", b"E"):
                y -= args.step
                dirty = True
            elif key in (b"z", b"Z"):
                roll -= args.rot_step
                dirty = True
            elif key in (b"x", b"X"):
                roll += args.rot_step
                dirty = True
            elif key in (b"r", b"R"):
                x = y = z = 0.0
                pitch = yaw = roll = 0.0
                dirty = True
            elif isinstance(key, tuple):
                # Arrow keys
                _, code = key
                if code == b"H":  # Up
                    pitch += args.rot_step
                    dirty = True
                elif code == b"P":  # Down
                    pitch -= args.rot_step
                    dirty = True
                elif code == b"K":  # Left
                    yaw -= args.rot_step
                    dirty = True
                elif code == b"M":  # Right
                    yaw += args.rot_step
                    dirty = True

            if dirty:
                pivot = None
                if args.pivot_x is not None or args.pivot_y is not None or args.pivot_z is not None:
                    pivot = (
                        args.pivot_x or 0.0,
                        args.pivot_y or 0.0,
                        args.pivot_z or 0.0,
                    )
                else:
                    pivot = _get_hmd_pivot()

                mat = _make_transform(pitch, yaw, roll, x, y, z, pivot)

                if args.seated:
                    _call_method(
                        setup,
                        [
                            "setWorkingSeatedZeroPoseToRawTrackingPose",
                            "SetWorkingSeatedZeroPoseToRawTrackingPose",
                        ],
                        mat,
                    )
                else:
                    _call_method(
                        setup,
                        [
                            "setWorkingStandingZeroPoseToRawTrackingPose",
                            "SetWorkingStandingZeroPoseToRawTrackingPose",
                        ],
                        mat,
                    )
                _call_method(
                    setup,
                    ["commitWorkingCopy", "CommitWorkingCopy"],
                    openvr.EChaperoneConfigFile_Live,
                )

                print(
                    f"pos=({x:.3f},{y:.3f},{z:.3f}) "
                    f"rot=({pitch:.1f},{yaw:.1f},{roll:.1f})"
                )
    finally:
        openvr.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Redirect play space origin via IVRChaperoneSetup"
    )
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch in degrees")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw in degrees")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll in degrees")
    parser.add_argument("--x", type=float, default=0.0, help="Translation X (meters)")
    parser.add_argument("--y", type=float, default=0.0, help="Translation Y (meters)")
    parser.add_argument("--z", type=float, default=0.0, help="Translation Z (meters)")
    parser.add_argument(
        "--pivot-hmd",
        action="store_true",
        help="Rotate around current HMD position",
    )
    parser.add_argument("--pivot-x", type=float, help="Pivot X (meters)")
    parser.add_argument("--pivot-y", type=float, help="Pivot Y (meters)")
    parser.add_argument("--pivot-z", type=float, help="Pivot Z (meters)")
    parser.add_argument(
        "--seated",
        action="store_true",
        help="Apply to seated zero pose (default: standing)",
    )
    parser.add_argument(
        "--save-default",
        action="store_true",
        help="Also save to default chaperone config on disk",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: WASD/QE move, arrows rotate",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Translation step in meters per key press",
    )
    parser.add_argument(
        "--rot-step",
        type=float,
        default=2.0,
        help="Rotation step in degrees per key press",
    )

    args = parser.parse_args()

    try:
        if args.interactive:
            run_interactive(args)
        else:
            apply_chaperone_transform(args)
    except Exception as exc:
        print(f"Failed to apply chaperone transform: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

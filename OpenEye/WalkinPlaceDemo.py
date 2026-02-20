#!/usr/bin/env python3
"""
walk_in_place_demo.py
---------------------
Standalone demo: detects walk-in-place and body rotation via webcam + MediaPipe.
Prints "WALKING" or "STANDING" and the body rotation angle to the console.

Usage:
    python walk_in_place_demo.py

Press 'q' in the video window to quit.
"""

import cv2
import math
import time
import collections
import mediapipe as mp

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ── Detection parameters (tune these) ────────────────────────────────────────
# Walking detection
KNEE_HISTORY_LENGTH = 30        # frames of knee-Y history to keep
WALK_OSCILLATION_THRESHOLD = 0.012  # min peak-to-trough amplitude in normalized coords
MIN_STEP_FREQUENCY = 0.8       # Hz — slower than this = standing
MAX_STEP_FREQUENCY = 5.0       # Hz — faster than this = noise
SMOOTHING_WINDOW = 5           # frames for moving-average smoothing

# Rotation detection
ROTATION_DEADZONE = 3.0        # degrees — ignore small jitter
ROTATION_SMOOTHING = 0.3       # exponential smoothing factor (0-1, lower = smoother)

# ── State variables ───────────────────────────────────────────────────────────
left_knee_history = collections.deque(maxlen=KNEE_HISTORY_LENGTH)
right_knee_history = collections.deque(maxlen=KNEE_HISTORY_LENGTH)
timestamps = collections.deque(maxlen=KNEE_HISTORY_LENGTH)

baseline_yaw = None            # first shoulder angle = reference
smoothed_yaw = None            # exponentially smoothed yaw
last_print_time = 0
PRINT_INTERVAL = 0.15          # seconds between console prints


def _moving_average(data, window):
    """Simple moving average over the last `window` items."""
    if len(data) < window:
        return list(data)
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(list(data)[start:i+1]) / (i - start + 1))
    return result


def _count_zero_crossings(signal):
    """Count zero-crossings in a de-meaned signal."""
    if len(signal) < 3:
        return 0
    mean = sum(signal) / len(signal)
    centered = [s - mean for s in signal]
    crossings = 0
    for i in range(1, len(centered)):
        if centered[i-1] * centered[i] < 0:
            crossings += 1
    return crossings


def detect_walking(landmarks):
    """
    Detect walk-in-place by analyzing vertical oscillation of knees.
    Returns (is_walking: bool, cadence_hz: float, amplitude: float)
    """
    now = time.time()
    
    # Get knee Y positions (normalized 0-1, 0=top of frame)
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y

    left_knee_history.append(left_knee_y)
    right_knee_history.append(right_knee_y)
    timestamps.append(now)

    if len(left_knee_history) < 10:
        return False, 0.0, 0.0

    # Smooth the signals
    left_smooth = _moving_average(left_knee_history, SMOOTHING_WINDOW)
    right_smooth = _moving_average(right_knee_history, SMOOTHING_WINDOW)

    # Use the average of both knees for overall oscillation
    avg_signal = [(l + r) / 2 for l, r in zip(left_smooth, right_smooth)]

    # Amplitude: peak-to-trough range
    amplitude = max(avg_signal) - min(avg_signal)

    # Frequency: estimate from zero-crossings over the time window
    time_span = timestamps[-1] - timestamps[0]
    if time_span < 0.3:
        return False, 0.0, amplitude

    crossings = _count_zero_crossings(avg_signal)
    frequency = (crossings / 2.0) / time_span  # each full cycle = 2 crossings

    is_walking = (
        amplitude > WALK_OSCILLATION_THRESHOLD
        and MIN_STEP_FREQUENCY <= frequency <= MAX_STEP_FREQUENCY
    )

    return is_walking, frequency, amplitude


def detect_rotation(landmarks):
    """
    Detect body rotation from shoulder angle.
    Returns (yaw_delta_degrees: float, raw_yaw_degrees: float)
    
    The angle is computed from the line between left and right shoulders
    projected onto the XZ plane (where X is horizontal in the camera frame
    and Z is depth estimated from shoulder width).
    """
    global baseline_yaw, smoothed_yaw

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Horizontal distance between shoulders in normalized coords
    dx = right_shoulder.x - left_shoulder.x
    # Depth difference (z) gives rotation info — positive z = closer to camera
    dz = right_shoulder.z - left_shoulder.z

    # atan2 gives the angle of the shoulder line in the XZ plane
    raw_yaw = math.degrees(math.atan2(dz, dx))

    # Initialize baseline on first good reading
    if baseline_yaw is None:
        baseline_yaw = raw_yaw
        smoothed_yaw = raw_yaw
        return 0.0, raw_yaw

    # Exponential smoothing
    smoothed_yaw = ROTATION_SMOOTHING * raw_yaw + (1 - ROTATION_SMOOTHING) * smoothed_yaw

    yaw_delta = smoothed_yaw - baseline_yaw

    # Apply deadzone
    if abs(yaw_delta) < ROTATION_DEADZONE:
        yaw_delta = 0.0

    return yaw_delta, raw_yaw


def draw_status(frame, is_walking, cadence, amplitude, yaw_delta, raw_yaw):
    """Draw status text overlay on the frame."""
    h, w = frame.shape[:2]

    # Walking status — big text at top
    walk_text = "WALKING" if is_walking else "STANDING"
    walk_color = (0, 255, 0) if is_walking else (0, 0, 255)
    cv2.putText(frame, walk_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, walk_color, 3, cv2.LINE_AA)

    # Details
    cv2.putText(frame, f"Cadence: {cadence:.1f} Hz", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Amplitude: {amplitude:.4f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Rotation status
    if abs(yaw_delta) > ROTATION_DEADZONE:
        direction = "LEFT" if yaw_delta > 0 else "RIGHT"
        rot_text = f"ROTATING {direction} ({yaw_delta:+.1f} deg)"
        rot_color = (255, 165, 0)
    else:
        rot_text = "NO ROTATION"
        rot_color = (200, 200, 200)

    cv2.putText(frame, rot_text, (20, 170), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, rot_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Raw yaw: {raw_yaw:.1f} deg", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Movement arrows (visual cue)
    cx, cy = w // 2, h - 60
    if is_walking:
        # Forward arrow
        cv2.arrowedLine(frame, (cx, cy), (cx, cy - 40), (0, 255, 0), 3, tipLength=0.4)
    if abs(yaw_delta) > ROTATION_DEADZONE:
        # Rotation arrow
        dx_arrow = int(40 * (-1 if yaw_delta > 0 else 1))
        cv2.arrowedLine(frame, (cx, cy), (cx + dx_arrow, cy), (255, 165, 0), 3, tipLength=0.4)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global last_print_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0). Try a different camera index.")
        return

    print("=" * 60)
    print("  Walk-in-Place Locomotion Demo")
    print("  - Walk in place → detected as WALKING")
    print("  - Rotate body   → detected as ROTATING")
    print("  - Press 'q' in the window to quit")
    print("  - Press 'r' to reset rotation baseline")
    print("=" * 60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)

        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        is_walking = False
        cadence = 0.0
        amplitude = 0.0
        yaw_delta = 0.0
        raw_yaw = 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )

            # Detect walking and rotation
            is_walking, cadence, amplitude = detect_walking(landmarks)
            yaw_delta, raw_yaw = detect_rotation(landmarks)

            # Console output (throttled)
            now = time.time()
            if now - last_print_time >= PRINT_INTERVAL:
                last_print_time = now
                status_parts = []
                if is_walking:
                    status_parts.append(f"WALKING (cadence={cadence:.1f}Hz, amp={amplitude:.4f})")
                else:
                    status_parts.append("STANDING")

                if abs(yaw_delta) > ROTATION_DEADZONE:
                    direction = "LEFT" if yaw_delta > 0 else "RIGHT"
                    status_parts.append(f"ROTATING {direction} ({yaw_delta:+.1f}°)")

                print(" | ".join(status_parts))

        # Draw status overlay
        draw_status(frame, is_walking, cadence, amplitude, yaw_delta, raw_yaw)

        cv2.imshow("Walk-in-Place Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset rotation baseline
            global baseline_yaw, smoothed_yaw
            baseline_yaw = None
            smoothed_yaw = None
            print("[Reset] Rotation baseline reset.")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("\nDemo stopped.")


if __name__ == "__main__":
    main()

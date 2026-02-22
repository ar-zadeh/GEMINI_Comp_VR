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
KNEE_HISTORY_LENGTH = 12        # frames of knee-Y history to keep
WALK_OSCILLATION_THRESHOLD = 0.015  # Lowered from 0.012 for better sensitivity
MIN_STEP_FREQUENCY = 0.35       # Hz — slower than this = standing
MAX_STEP_FREQUENCY = 5.0       # Hz — faster than this = noise
SMOOTHING_WINDOW = 2           # frames for moving-average smoothing

# ── State variables ───────────────────────────────────────────────────────────
left_leg_history = collections.deque(maxlen=KNEE_HISTORY_LENGTH)
right_leg_history = collections.deque(maxlen=KNEE_HISTORY_LENGTH)
timestamps = collections.deque(maxlen=KNEE_HISTORY_LENGTH)

FULL_BODY_VISIBILITY_THRESHOLD = 0.6
FRAME_EDGE_MARGIN = 0.02

def nothing(val):
    pass

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


def _is_landmark_visible_in_frame(landmark, visibility_threshold, margin):
    return (
        landmark.visibility >= visibility_threshold
        and margin <= landmark.x <= (1.0 - margin)
        and margin <= landmark.y <= (1.0 - margin)
    )


def is_full_body_visible(landmarks, visibility_threshold=FULL_BODY_VISIBILITY_THRESHOLD, margin=FRAME_EDGE_MARGIN):
    """Require head/torso/legs landmarks to be confidently visible and inside frame."""
    required = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]
    for idx in required:
        if not _is_landmark_visible_in_frame(landmarks[idx], visibility_threshold, margin):
            return False
    return True


def reset_walking_history():
    left_leg_history.clear()
    right_leg_history.clear()
    timestamps.clear()


def detect_walking(landmarks, joint_idx, threshold, smoothing, min_freq, max_freq, independent_legs=True):
    """
    Detect walk-in-place by analyzing vertical oscillation of legs.
    Returns (is_walking: bool, cadence_hz: float, amplitude: float)
    """
    now = time.time()
    
    # Get joint Y positions (normalized 0-1, 0=top of frame)
    if joint_idx == 1:
        left_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
        right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
    else:
        left_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
        right_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y

    left_leg_history.append(left_y)
    right_leg_history.append(right_y)
    timestamps.append(now)

    if len(left_leg_history) < 10:
        return False, 0.0, 0.0

    # Smooth the signals
    left_smooth = _moving_average(left_leg_history, smoothing)
    right_smooth = _moving_average(right_leg_history, smoothing)

    if independent_legs:
        # Evaluate each leg independently to avoid them canceling each other out
        left_amp = max(left_smooth) - min(left_smooth)
        right_amp = max(right_smooth) - min(right_smooth)
        amplitude = max(left_amp, right_amp)
        
        # Calculate frequency based on the leg moving the most
        active_signal = left_smooth if left_amp > right_amp else right_smooth
        crossings = _count_zero_crossings(active_signal)
    else:
        # Legacy average method
        avg_signal = [(l + r) / 2 for l, r in zip(left_smooth, right_smooth)]
        amplitude = max(avg_signal) - min(avg_signal)
        crossings = _count_zero_crossings(avg_signal)

    # Frequency: estimate from zero-crossings over the time window
    time_span = timestamps[-1] - timestamps[0]
    if time_span < 0.3:
        return False, 0.0, amplitude

    frequency = (crossings / 2.0) / time_span  # each full cycle = 2 crossings

    is_walking = (
        amplitude > threshold
        and min_freq <= frequency <= max_freq
    )

    return is_walking, frequency, amplitude


def draw_status(frame, is_walking, cadence, amplitude, full_body_visible):
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

    if not full_body_visible:
        cv2.putText(frame, "Show full body (head to ankles)", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)

    # Movement arrows (visual cue)
    cx, cy = w // 2, h - 60
    if is_walking:
        # Forward arrow
        cv2.arrowedLine(frame, (cx, cy), (cx, cy - 40), (0, 255, 0), 3, tipLength=0.4)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global last_print_time

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0). Try a different camera index.")
        return

    print("=" * 60)
    print("  Walk-in-Place Locomotion Demo")
    print("  - Walk in place → detected as WALKING")
    print("  - Press 'q' in the window to quit")
    print("=" * 60)
    
    cv2.namedWindow("Walk-in-Place Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Walk-in-Place Demo", 640, 600)
    
    # Trackbars for interactive tuning
    cv2.createTrackbar('Joint (0=Knee, 1=Ankle)', 'Walk-in-Place Demo', 0, 1, nothing)
    cv2.createTrackbar('Mode (0=Avg, 1=Indep)', 'Walk-in-Place Demo', 1, 1, nothing)
    cv2.createTrackbar('Threshold (x1000)', 'Walk-in-Place Demo', int(WALK_OSCILLATION_THRESHOLD * 1000), 100, nothing)
    cv2.createTrackbar('Smoothing Window', 'Walk-in-Place Demo', SMOOTHING_WINDOW, 30, nothing)
    cv2.createTrackbar('Min Freq (x100)', 'Walk-in-Place Demo', int(MIN_STEP_FREQUENCY * 100), 200, nothing)
    cv2.createTrackbar('Max Freq (x10)', 'Walk-in-Place Demo', int(MAX_STEP_FREQUENCY * 10), 100, nothing)

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
        full_body_visible = False

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

            # Read tuner parameters
            tuner_joint = cv2.getTrackbarPos('Joint (0=Knee, 1=Ankle)', 'Walk-in-Place Demo')
            tuner_mode = cv2.getTrackbarPos('Mode (0=Avg, 1=Indep)', 'Walk-in-Place Demo')
            tuner_thresh = cv2.getTrackbarPos('Threshold (x1000)', 'Walk-in-Place Demo') / 1000.0
            tuner_smooth = max(1, cv2.getTrackbarPos('Smoothing Window', 'Walk-in-Place Demo'))
            tuner_min_f = cv2.getTrackbarPos('Min Freq (x100)', 'Walk-in-Place Demo') / 100.0
            tuner_max_f = cv2.getTrackbarPos('Max Freq (x10)', 'Walk-in-Place Demo') / 10.0
            tuner_indep = (tuner_mode == 1)

            full_body_visible = is_full_body_visible(landmarks)

            if full_body_visible:
                # Detect walking
                is_walking, cadence, amplitude = detect_walking(
                    landmarks, tuner_joint, tuner_thresh, tuner_smooth,
                    tuner_min_f, tuner_max_f, tuner_indep
                )
            else:
                reset_walking_history()

            # Console output (throttled)
            now = time.time()
            if now - last_print_time >= PRINT_INTERVAL:
                last_print_time = now
                if full_body_visible:
                    walk_msg = "WALKING" if is_walking else "STANDING"
                    print(f"{walk_msg} (cadence={cadence:.1f}Hz, amp={amplitude:.4f})")
                else:
                    print("WAITING: show full body (head to ankles)")

        # Draw status overlay
        draw_status(frame, is_walking, cadence, amplitude, full_body_visible)

        cv2.imshow("Walk-in-Place Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("\nDemo stopped.")


if __name__ == "__main__":
    main()
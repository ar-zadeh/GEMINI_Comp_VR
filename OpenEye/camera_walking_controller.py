import sys
import time
import math
import collections
import threading
import cv2
import mediapipe as mp

from keyboard_controller import KeyboardVRController

# Settings from Demo.py
KNEE_HISTORY_LENGTH = 30
WALK_OSCILLATION_THRESHOLD = 0.015
MIN_STEP_FREQUENCY = 0.22
MAX_STEP_FREQUENCY = 5.0
SMOOTHING_WINDOW = 2

def _moving_average(data, window):
    if len(data) < window:
        return list(data)
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(list(data)[start:i+1]) / (i - start + 1))
    return result

def _count_zero_crossings(signal):
    if len(signal) < 3:
        return 0
    mean = sum(signal) / len(signal)
    centered = [s - mean for s in signal]
    crossings = 0
    for i in range(1, len(centered)):
        if centered[i-1] * centered[i] < 0:
            crossings += 1
    return crossings


class CameraWalkingController(KeyboardVRController):
    """
    Extends KeyboardVRController but overrides movement to use Webcam+MediaPipe
    Walk-In-Place locomotion instead of WASD/Arrows. 
    Maintains the 'Enter' key feature for voice triggering.
    """
    def __init__(self, mcp_module, move_speed: float = 0.05, rotate_speed: float = 2.0):
        # We reuse move_step and rotate_step semantic as scale factors or per-frame steps
        super().__init__(mcp_module, move_step=move_speed, rotate_step=rotate_speed)
        
        self.camera_thread = None
        self.camera_active = False
        
        # State variables for detection logic
        self.left_leg_history = collections.deque(maxlen=KNEE_HISTORY_LENGTH)
        self.right_leg_history = collections.deque(maxlen=KNEE_HISTORY_LENGTH)
        self.timestamps = collections.deque(maxlen=KNEE_HISTORY_LENGTH)
        
        # Disable trackpad mode by default since we manage movement here
        self.mode = 'headset'

    def activate(self):
        super().activate()
        
        if self.camera_active:
            return
            
        self.camera_active = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        print("\n[Camera Walking Controller] Camera loop started.")

    def deactivate(self):
        super().deactivate()
        if not self.camera_active:
            return
            
        self.camera_active = False
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=2.0)
            self.camera_thread = None
        print("\n[Camera Walking Controller] Camera loop stopped.")

    def _handle_char(self, ch: str):
        # Disable WASD and QE for movement, leaving only utility keys like mode toggle
        ch = ch.lower()
        if ch == 'm':
            self.mode = 'headset' if self.mode == 'trackpad' else 'trackpad'
            print(f"\n[CameraWalking] Mode switched (visual only) to {self.mode.upper()}. Camera still controls movement.")

    def _handle_escape_sequence(self):
        # Disable arrow keys for rotation
        pass

    def _detect_walking(self, landmarks):
        now = time.time()
        
        # Use KNEE logic (joint_idx 0 in demo)
        mp_pose = mp.solutions.pose
        left_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
        right_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y

        self.left_leg_history.append(left_y)
        self.right_leg_history.append(right_y)
        self.timestamps.append(now)

        if len(self.left_leg_history) < 10:
            return False, 0.0, 0.0

        left_smooth = _moving_average(self.left_leg_history, SMOOTHING_WINDOW)
        right_smooth = _moving_average(self.right_leg_history, SMOOTHING_WINDOW)

        # Independent legs mode
        left_amp = max(left_smooth) - min(left_smooth)
        right_amp = max(right_smooth) - min(right_smooth)
        amplitude = max(left_amp, right_amp)
        
        active_signal = left_smooth if left_amp > right_amp else right_smooth
        crossings = _count_zero_crossings(active_signal)

        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span < 0.3:
            return False, 0.0, amplitude

        frequency = (crossings / 2.0) / time_span

        is_walking = (
            amplitude > WALK_OSCILLATION_THRESHOLD
            and MIN_STEP_FREQUENCY <= frequency <= MAX_STEP_FREQUENCY
        )

        return is_walking, frequency, amplitude

    def _camera_loop(self):
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam in CameraWalkingController.")
            return

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        last_print_time = 0
        
        try:
            while self.camera_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    is_walking, cadence, amplitude = self._detect_walking(landmarks)

                    # Output status occasionally to not spam
                    now = time.time()
                    if now - last_print_time >= 0.5:
                        msg = "WALKING" if is_walking else "STANDING"
                        # Only print if something is happening to avoid spam 
                        if is_walking:
                            print(f"\r[Camera] {msg} (cadence={cadence:.1f}Hz, amp={amplitude:.4f})", end="")
                        last_print_time = now

                    # Apply movement
                    if is_walking:
                        # Move forward based on cadence or fixed step
                        move_amt = self.move_step # Adjust based on cadence if desired
                        self._move_headset(forward=move_amt)

                # Not doing cv2.imshow to keep it headless and performant
                cv2.waitKey(1)
                
        except Exception as e:
            print(f"[CameraWalking] Exception in camera thread: {e}")
        finally:
            cap.release()
            pose.close()

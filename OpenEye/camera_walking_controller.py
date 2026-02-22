import time
import collections
import threading
import multiprocessing as mproc
import queue

import cv2
import numpy as np
import mediapipe as mp

from keyboard_controller import KeyboardVRController

# Settings from Demo.py
KNEE_HISTORY_LENGTH = 12
WALK_OSCILLATION_THRESHOLD = 0.015
MIN_STEP_FREQUENCY = 0.35
MAX_STEP_FREQUENCY = 5.0
SMOOTHING_WINDOW = 2
FULL_BODY_VISIBILITY_THRESHOLD = 0.6
FRAME_EDGE_MARGIN = 0.02


def _run_overlay_window(status_queue, stop_event, window_name):
    print(f"[CameraWalking] Overlay process started: {window_name}")
    latest = {
        "is_walking": False,
        "cadence": 0.0,
        "amplitude": 0.0,
        "full_body_visible": False,
        "frame_jpeg": None,
    }
    frame_h, frame_w = 420, 720

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, frame_w, frame_h)
    except Exception as e:
        print(f"[CameraWalking] Overlay process failed to create window: {e}")
        return

    try:
        while not stop_event.is_set():
            try:
                while True:
                    latest = status_queue.get_nowait()
            except queue.Empty:
                pass

            frame = None
            frame_jpeg = latest.get("frame_jpeg")
            if frame_jpeg:
                arr = np.frombuffer(frame_jpeg, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    frame = decoded
            if frame is None:
                frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

            frame_h, frame_w = frame.shape[:2]
            is_walking = bool(latest.get("is_walking", False))
            cadence = float(latest.get("cadence", 0.0))
            amplitude = float(latest.get("amplitude", 0.0))
            full_body_visible = bool(latest.get("full_body_visible", False))

            walk_text = "WALKING" if is_walking else "STANDING"
            walk_color = (0, 255, 0) if is_walking else (0, 0, 255)
            cv2.putText(frame, walk_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1.7, walk_color, 3, cv2.LINE_AA)
            cv2.putText(frame, f"Cadence: {cadence:.1f} Hz", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Amplitude: {amplitude:.4f}", (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            if not full_body_visible:
                cv2.putText(frame, "Show full body (head to ankles)", (20, 225),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 215, 255), 2)

            cx, cy = frame_w // 2, frame_h - 70
            if is_walking:
                cv2.arrowedLine(frame, (cx, cy), (cx, cy - 55), (0, 255, 0), 4, tipLength=0.4)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(16) & 0xFF
            if key == ord('q'):
                break
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        print("[CameraWalking] Overlay process exiting.")

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
        self.show_ui = False
        self.window_name = "White Cane Walk UI"
        self.ui_process = None
        self.ui_queue = None
        self.ui_stop_event = None
        self._ui_frame_counter = 0
        
        # Disable trackpad mode by default since we manage movement here
        self.mode = 'headset'

    def set_overlay_enabled(self, enabled: bool):
        self.show_ui = bool(enabled)
        if self.show_ui:
            self._start_overlay_process()
        else:
            self._stop_overlay_process()

    def _start_overlay_process(self):
        if self.ui_process and self.ui_process.is_alive():
            return
        try:
            ctx = mproc.get_context("spawn")
            self.ui_queue = ctx.Queue(maxsize=8)
            self.ui_stop_event = ctx.Event()
            self.ui_process = ctx.Process(
                target=_run_overlay_window,
                args=(self.ui_queue, self.ui_stop_event, self.window_name),
                daemon=False,
            )
            self.ui_process.start()
            print(f"[CameraWalking] Overlay process pid={self.ui_process.pid}")
        except Exception as e:
            print(f"[CameraWalking] Failed to start overlay process: {e}")
            self.show_ui = False
            self.ui_process = None
            self.ui_queue = None
            self.ui_stop_event = None

    def _stop_overlay_process(self):
        if self.ui_stop_event:
            self.ui_stop_event.set()
        if self.ui_process:
            self.ui_process.join(timeout=1.5)
            if self.ui_process.is_alive():
                self.ui_process.terminate()
            print("[CameraWalking] Overlay process stopped.")
            self.ui_process = None
        if self.ui_queue:
            try:
                self.ui_queue.close()
                self.ui_queue.join_thread()
            except Exception:
                pass
            self.ui_queue = None
        self.ui_stop_event = None

    def _publish_overlay_status(self, is_walking, cadence, amplitude, full_body_visible, frame=None):
        if not self.show_ui or not self.ui_queue:
            return
        frame_jpeg = None
        if frame is not None:
            self._ui_frame_counter += 1
            if self._ui_frame_counter % 2 == 0:
                try:
                    ok, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok:
                        frame_jpeg = enc.tobytes()
                except Exception:
                    frame_jpeg = None
        payload = {
            "is_walking": bool(is_walking),
            "cadence": float(cadence),
            "amplitude": float(amplitude),
            "full_body_visible": bool(full_body_visible),
            "frame_jpeg": frame_jpeg,
        }
        try:
            self.ui_queue.put_nowait(payload)
        except queue.Full:
            try:
                self.ui_queue.get_nowait()
            except Exception:
                pass
            try:
                self.ui_queue.put_nowait(payload)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _is_landmark_visible_in_frame(landmark, visibility_threshold, margin):
        return (
            landmark.visibility >= visibility_threshold
            and margin <= landmark.x <= (1.0 - margin)
            and margin <= landmark.y <= (1.0 - margin)
        )

    def _is_full_body_visible(self, landmarks):
        mp_pose = mp.solutions.pose
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
            if not self._is_landmark_visible_in_frame(
                landmarks[idx], FULL_BODY_VISIBILITY_THRESHOLD, FRAME_EDGE_MARGIN
            ):
                return False
        return True

    def _reset_walking_history(self):
        self.left_leg_history.clear()
        self.right_leg_history.clear()
        self.timestamps.clear()

    def activate(self):
        super().activate()
        
        if self.camera_active:
            return
            
        self.camera_active = True
        if self.show_ui:
            self._start_overlay_process()
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
        self._stop_overlay_process()
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

                is_walking = False
                cadence = 0.0
                amplitude = 0.0
                full_body_visible = False

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    full_body_visible = self._is_full_body_visible(landmarks)

                    if not full_body_visible:
                        self._reset_walking_history()
                        now = time.time()
                        if now - last_print_time >= 0.5:
                            print("\r[Camera] WAITING: show full body (head to ankles)", end="")
                            last_print_time = now
                    else:
                        is_walking, cadence, amplitude = self._detect_walking(landmarks)

                        # Output status occasionally to not spam
                        now = time.time()
                        if now - last_print_time >= 0.5:
                            msg = "WALKING" if is_walking else "STANDING"
                            if is_walking:
                                print(f"\r[Camera] {msg} (cadence={cadence:.1f}Hz, amp={amplitude:.4f})", end="")
                            last_print_time = now

                        if is_walking:
                            move_amt = self.move_step
                            self._move_headset(forward=move_amt)

                self._publish_overlay_status(is_walking, cadence, amplitude, full_body_visible, frame)
                
        except Exception as e:
            print(f"[CameraWalking] Exception in camera thread: {e}")
        finally:
            cap.release()
            pose.close()

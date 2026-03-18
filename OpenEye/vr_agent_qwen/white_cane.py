"""
vr_agent/white_cane.py
----------------------
WhiteCaneAssistant: accessibility mode for blind users.
Silently captures images, describes scenes, and recommends navigation actions.
"""

import io
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import base64
from PIL import Image
from pydantic import BaseModel, Field

import base64

from .config import MODEL_WHITE_CANE
from .logger import get_logger
from .audio import AudioAssistant


class WhiteCaneAssistant:
    """
    Accessibility assistant for blind users.
    - Silently captures images at regular intervals (background loop).
    - On demand: describes the scene and recommends a navigation action.
    - Supports 360° scanning by rotating the headset to 4 directions.
    """

    # ── Description prompt template ───────────────────────────────────────────
    _DESCRIPTION_PROMPT = """\
You are an accessibility assistant helping a blind person navigate a VR environment.
The user's current goal is: {goal}

You are receiving images captured at different times. Each image has a timestamp.
The latest image was captured at {timestamp}.

Your task is to:
1. THINK about what you see and how it relates to the user's goal
2. DESCRIBE the scene VERY CONCISELY (max 1-2 sentences) for someone who cannot see (spoken aloud)
3. RECOMMEND one specific action to take. The actions should be "move forward", "move backward", "rotate X degrees", "turn left", "turn right", or "stop"
4. DETERMINE if the goal has been achieved or if the user wants to change it

IMPORTANT RULES:
- Descriptions MUST be under 20 words. Be super clear and concise. DO NOT use bullet points.
- If you see something that suggests the user has reached their goal, set goal_achieved to true
- If the user's spoken input suggests they want to change their goal, extract the new_goal
- Look at previous images to track progress and give context-aware guidance
- Remember you need to help user avoid obstacles, hitting the wall, glasses, etc. User should always have a clear path. Help user to have the clear path DIRECTLY INFRONT OF THEM. Anything other than direct path should be avoided.
"""

    def __init__(self, client, executor, log_dir: Path):
        self.client = client
        self.executor = executor
        self.model_name = MODEL_WHITE_CANE
        self.log_dir = log_dir / "white_cane"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # State
        self.active = False
        self.current_goal: Optional[str] = None
        self.stop_event = threading.Event()
        self.loop_thread: Optional[threading.Thread] = None

        # Audio
        self.audio = AudioAssistant(log_dir, executor)

        # Image cache: list of (timestamp_str, image_bytes, file_path)
        self.cached_images: List[tuple] = []
        self.conversation_history: List[Dict] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def activate(self, goal: Optional[str] = None) -> str:
        self.active = True
        self.current_goal = goal
        self.stop_event.clear()
        self.cached_images = []
        self.conversation_history = []

        logger = get_logger()
        logger.info(f"White cane activated. Goal: {goal}")

        msg = (
            f"White cane mode activated. "
            f"{'Goal: ' + goal if goal else 'No specific goal set. I will describe what I see.'}"
            "\nSay 'disable white cane' to stop, or tell me your new goal anytime."
        )
        self.audio.speak(msg)
        return msg

    def deactivate(self) -> str:
        self.active = False
        self.stop_event.set()
        if self.loop_thread and self.loop_thread.is_alive():
            if threading.current_thread() != self.loop_thread:
                self.loop_thread.join(timeout=2.0)

        get_logger().info("White cane deactivated.")
        return "White cane mode deactivated."

    def set_goal(self, goal: str) -> str:
        old_goal = self.current_goal
        self.current_goal = goal
        get_logger().info(f"White cane goal changed: '{old_goal}' -> '{goal}'")
        return f"Goal updated from '{old_goal}' to '{goal}'"

    # ── Image capture ─────────────────────────────────────────────────────────

    def capture_with_timestamp(self) -> tuple:
        """Capture a screenshot and save it. Returns (timestamp_str, bytes, path)."""
        import json
        logger = get_logger()
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%H:%M:%S")
        filename = f"image_{timestamp_str.replace(':', '_')}.png"
        file_path = self.log_dir / filename

        res = self.executor.call("inspect_surroundings")
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_img.save(str(file_path), format="PNG")
            logger.info(f"White cane capture: {file_path}")
            return (timestamp_str, img_bytes, str(file_path))
        except Exception as e:
            logger.error(f"White cane capture failed: {e}")
            return (timestamp_str, None, None)

    def cleanup_old_images(self, max_age_seconds: int = 10):
        """Delete PNG files in the log directory older than max_age_seconds."""
        logger = get_logger()
        now = time.time()
        count = 0
        try:
            for file_path in self.log_dir.glob("*.png"):
                if file_path.is_file() and now - file_path.stat().st_mtime > max_age_seconds:
                    file_path.unlink()
                    count += 1
            if count > 0:
                logger.info(f"Cleaned up {count} old white cane images (> {max_age_seconds}s).")
        except Exception as e:
            logger.error(f"Error cleaning up old images: {e}")

    # ── Description / recommendation ──────────────────────────────────────────

    def describe_and_recommend(
        self,
        timestamp_str: str,
        img_bytes: Optional[bytes],
        user_input: Optional[str] = None
    ) -> Dict:
        """
        Call Gemini with the current image (+ history) to get a structured response:
        thought, description, action, goal_achieved, new_goal.
        """
        logger = get_logger()

        if not img_bytes:
            return {
                "thought": "Failed to capture image",
                "description": "I'm sorry, I couldn't capture an image. Please try again.",
                "action": "Wait a moment and try again",
                "goal_achieved": False,
                "new_goal": None
            }

        class WhiteCaneResponse(BaseModel):
            thought: str = Field(
                description="Internal reasoning. NOT read to the user."
            )
            description: str = Field(
                description="Natural, extremely concise description (max 20 words). No bullet points. Read aloud."
            )
            action: str = Field(
                description="One specific physical action recommendation."
            )
            goal_achieved: bool = Field(
                description="True if the user's goal has been achieved."
            )
            new_goal: Optional[str] = Field(
                default=None,
                description="New goal if the user expressed a desire to change it."
            )

        parts = []
        prompt = self._DESCRIPTION_PROMPT.format(
            goal=self.current_goal or "exploring the environment",
            timestamp=timestamp_str
        )
        parts.append(types.Part(text=prompt))

        if user_input:
            parts.append(types.Part(text=f'\n[User just said]: "{user_input}"\n'))

        # Historical images (up to 4)
        history_images = self.cached_images[-4:]
        total_images = len(history_images) + 1
        ordinal_labels = ["first", "second", "third", "fourth", "fifth"]

        parts.append(types.Part(text="\n--- IMAGE TIMELINE (chronological order) ---\n"))
        for idx, (ts, img_data, path) in enumerate(history_images):
            if img_data:
                if idx == 0:
                    label = f"At the beginning (image 1 of {total_images}), captured at {ts}"
                else:
                    ordinal = ordinal_labels[idx] if idx < len(ordinal_labels) else f"{idx+1}th"
                    label = f"In the {ordinal} capture (image {idx+1} of {total_images}), at {ts}"
                parts.append(types.Part(text=f"\n[{label}]:"))
                parts.append(types.Part(inline_data=types.Blob(mime_type="image/png", data=img_data)))

        current_position = len(history_images) + 1
        parts.append(types.Part(
            text=f"\n[NOW - Current view (image {current_position} of {total_images}), captured at {timestamp_str}]:"
        ))
        parts.append(types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)))

        if self.conversation_history:
            history_text = "\nPrevious observations:\n"
            for entry in self.conversation_history[-4:]:
                history_text += f"- At {entry['timestamp']}: {entry['summary']}\n"
            parts.append(types.Part(text=history_text))

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=parts)],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": WhiteCaneResponse
                }
            )

            try:
                parsed = response.parsed
                if not parsed:
                    parsed = WhiteCaneResponse.model_validate_json(response_text)
                result = {
                    "thought": parsed.thought,
                    "description": parsed.description,
                    "action": parsed.action,
                    "goal_achieved": parsed.goal_achieved,
                    "new_goal": parsed.new_goal
                }
            except Exception as parse_error:
                logger.warning(f"Structured parse failed, using fallback: {parse_error}")
                result = {
                    "thought": "Processing...",
                    "description": response_text,
                    "action": "Continue as you were",
                    "goal_achieved": False,
                    "new_goal": None
                }

            self.conversation_history.append({
                "timestamp": timestamp_str,
                "summary": (
                    result["description"][:80] + "..."
                    if len(result["description"]) > 80
                    else result["description"]
                )
            })
            return result

        except Exception as e:
            logger.error(f"White cane description failed: {e}")
            return {
                "thought": f"Error: {e}",
                "description": "I encountered an error while analyzing the scene.",
                "action": "Please wait while I try again",
                "goal_achieved": False,
                "new_goal": None
            }

    def format_for_speech(self, result: Dict) -> str:
        """Format structured result into a spoken sentence."""
        speech = result["description"]
        if result["action"]:
            speech += f" I recommend: {result['action']}"
        if result["goal_achieved"]:
            speech += " It looks like you've reached your goal!"
        return speech

    # ── High-level helpers ────────────────────────────────────────────────────

    def get_immediate_help(self, user_input: Optional[str] = None) -> str:
        """Capture now and describe immediately."""
        timestamp_str, img_bytes, file_path = self.capture_with_timestamp()
        if img_bytes:
            self.cached_images.append((timestamp_str, img_bytes, file_path))
        result = self.describe_and_recommend(timestamp_str, img_bytes, user_input)
        if result.get("new_goal"):
            self.set_goal(result["new_goal"])
            print(f"[Goal Changed]: {result['new_goal']}")
        return self.format_for_speech(result)

    def listen_command(self) -> Optional[str]:
        """Listen for a voice command (manual start/stop)."""
        return self.audio.listen_manual_stop()

    # ── 360° scan ─────────────────────────────────────────────────────────────

    def get_headset_rotation(self) -> tuple:
        """Return current headset (pitch, yaw, roll)."""
        logger = get_logger()
        try:
            res = self.executor.call("get_current_pose", device="headset")
            if "Rotation: [" in res:
                rot_str = res.split("Rotation: [")[1].split("]")[0]
                rot_str = rot_str.replace("np.float64(", "").replace(")", "")
                pitch, yaw, roll = map(float, rot_str.split(","))
                return pitch, yaw, roll
        except Exception as e:
            logger.error(f"Failed to get headset rotation: {e}")
        return 0.0, 0.0, 0.0

    def perform_360_scan(self, user_input: Optional[str] = None) -> str:
        """
        Rotate headset to Front / Right / Back / Left, capture one image each,
        then ask Gemini to summarize and recommend an action.
        """
        logger = get_logger()
        logger.info(f"Starting 360 scan for request: {user_input}")
        self.audio.speak("Scanning surroundings...")

        self.cleanup_old_images(max_age_seconds=10)
        start_pitch, start_yaw, start_roll = self.get_headset_rotation()

        captured_images = []
        offsets = [0, -90, -180, -270]
        labels = ["Front", "Right", "Back", "Left"]

        try:
            for i, offset in enumerate(offsets):
                target_yaw = start_yaw + offset
                self.executor.call(
                    "rotate_device", device="headset",
                    pitch=start_pitch, yaw=target_yaw, roll=0
                )
                time.sleep(0.5)
                timestamp_str, img_bytes, _ = self.capture_with_timestamp()
                if img_bytes:
                    captured_images.append((labels[i], img_bytes))

            # Restore original rotation
            self.executor.call(
                "rotate_device", device="headset",
                pitch=start_pitch, yaw=start_yaw, roll=start_roll
            )

            if not captured_images:
                return "Failed to capture images for scan."

            parts = []
            prompt = f"""\
You are an accessibility assistant for a blind user.
The user's goal is: "{self.current_goal or 'Explore'}"
User just asked: "{user_input or 'What should I do?'}"

I have rotated the user to look in 4 directions (Front, Right, Back, Left).
Analyze these 4 images to understand the FULL environment.

Task:
1. Briefly summarize the most important things around the user (hazards, paths, objects of interest).
2. Recommend a clear action (e.g., "Turn right and move towards the door", "Move forward", "Stop, obstacle ahead").
3. Descriptions must be CONCISE (under 30 words).
"""
            parts.append(types.Part(text=prompt))
            for label, img_data in captured_images:
                parts.append(types.Part(text=f"\n[{label} View]:"))
                parts.append(types.Part(inline_data=types.Blob(mime_type="image/png", data=img_data)))

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=parts)]
            )
            result_text = response_text.strip()
            self.cleanup_old_images(max_age_seconds=10)
            return result_text

        except Exception as e:
            logger.error(f"360 Scan failed: {e}")
            try:
                self.executor.call(
                    "rotate_device", device="headset",
                    pitch=start_pitch, yaw=start_yaw, roll=start_roll
                )
            except Exception:
                pass
            return f"Error during scan: {e}"

    # ── Background monitoring loop ────────────────────────────────────────────

    def run_loop(self, interval: float = 2.0, status_interval: float = 30.0):
        """
        Silently capture images every `interval` seconds.
        Announce status every `status_interval` seconds.
        """
        logger = get_logger()
        last_announcement_time = time.time()

        while self.active and not self.stop_event.is_set():
            if time.time() - last_announcement_time > status_interval:
                goal_msg = f"Goal: {self.current_goal}." if self.current_goal else "No specific goal."
                self.audio.speak(f"White cane active. {goal_msg} Say 'help' for details.")
                last_announcement_time = time.time()

            timestamp_str, img_bytes, file_path = self.capture_with_timestamp()
            if img_bytes:
                self.cached_images.append((timestamp_str, img_bytes, file_path))
                if len(self.cached_images) > 20:
                    self.cached_images.pop(0)
                logger.info(f"[White Cane] Silent Monitor: Captured {timestamp_str}")
                print(f"[White Cane] Monitoring... ({timestamp_str})", end='\r')

            self.stop_event.wait(timeout=interval)

        logger.info("White cane loop ended.")

    def start_background_loop(self, interval: float = 2.0) -> str:
        """Start the silent capture loop in a daemon thread."""
        if self.loop_thread and self.loop_thread.is_alive():
            return "White cane loop is already running."
        self.stop_event.clear()
        self.loop_thread = threading.Thread(
            target=self.run_loop, args=(interval,), daemon=True
        )
        self.loop_thread.start()
        return "White cane background loop started (every 2 seconds)."

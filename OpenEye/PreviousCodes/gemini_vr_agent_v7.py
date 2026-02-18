#!/usr/bin/env python3
"""
Gemini VR Agent - Multi-Model Architecture (v2)
Changes from v1:
- Implements specific Gemini models for distinct phases:
  - Planning: gemini-3-flash-preview (Fast, structured plans)
  - Grounding: gemini-robotics-er-1.5-preview (Specialized 2D spatial reasoning)
  - Verification: gemini-2.5-pro (High-reasoning validation)
  - Description: gemini-2.5-flash-lite-preview-09-2025 (Visual description)
- Introduced ActionPlanner and Verifier classes.
- Updated VisualGrounder to strictly use Robotics-ER 1.5 formats.
"""

import os
import shlex
import json
import time
import base64
import traceback
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFile
import io
import math
import shutil
import cv2
import numpy as np
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import Object Tracker
try:
    from object_tracker import ObjectTracker
except ImportError:
    ObjectTracker = None
    print("Warning: ObjectTracker import failed. Tracking will be disabled.")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install google-genai: pip install google-genai")
    exit(1)

# OpenCV availability
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    print("Please install opencv-python and numpy")
    exit(1)

# Audio availability
try:
    from gtts import gTTS
    import whisper
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: gTTS or openai-whisper not found. Audio features disabled.")

# Standard Library
import os
import time
import json
import base64
import threading
import logging
import traceback
import io
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from PIL import Image
import queue

# Remaining original imports
import shlex
from typing import Union
from dataclasses import dataclass, field
from PIL import ImageDraw, ImageFont, ImageFile
import math
import shutil


# --- CONFIG ---
MODEL_PLANNER = "gemini-3-flash-preview"
MODEL_GROUNDING = "gemini-3-flash-preview"
MODEL_VERIFICATION = "gemini-2.5-flash" 
MODEL_DESCRIPTION = "gemini-2.5-flash-lite-preview-09-2025"
MODEL_WHITE_CANE = "gemini-3-flash-preview"
WHISPER_MODEL = "small.en"

LOG_DIR = Path("agent_logs_v2")
SHOW_VISION_PREVIEW = False

# ============================================================================
# UX CONSTANTS
# ============================================================================

class VoiceMenuState:
    IDLE = "idle"
    MAIN_MENU = "main_menu"
    WHITE_CANE_MENU = "white_cane_menu"
    CONFIRMATION = "confirmation"

# ============================================================================
# UTILS
# ============================================================================

class AgentLogger:
    """Simple logger setup."""
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{self.session_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("VRAgent")
        
    def info(self, msg): self.logger.info(msg)
    def error(self, msg): self.logger.error(msg) 
    def action(self, tool, args, result):
        self.info(f"[ACTION] {tool}({json.dumps(args)}) -> {str(result)[:200]}")

_logger = None
def get_logger():
    global _logger
    if not _logger: _logger = AgentLogger(LOG_DIR)
    return _logger

# ============================================================================
# VISUAL GROUNDING (Gemini 3 Flash)
# ============================================================================

class VisualGrounder:
    """Handles object detection using Gemini 3 Flash."""
    def __init__(self, client, log_dir: Path):
        self.client = client
        self.model_name = MODEL_GROUNDING
        self.log_dir = log_dir / "grounding"
        self.log_dir.mkdir(exist_ok=True, parents=True)

    def ground_multiple_objects(self, image_data: bytes, object_names: List[str]) -> Dict[str, List[float]]:
        logger = get_logger()
        objects_str = ", ".join(object_names)
        logger.info(f"Grounding Multiple (Gemini 3 Flash): '{objects_str}'")
        
        # Specific Prompt Format for Gemini 3 Flash (Bounding Box)
        prompt = f"""
        Find the following objects in the image: {objects_str}.
        
        You MUST return the answer in the following JSON format:
        {{
            "thinking": "Reasoning about the scene...",
            "detections": [
                {{
                    "label": "exact_object_name_from_list",
                    "coordinates": [ymin, xmin, ymax, xmax]
                }}
            ]
        }}
        
        1. ymin, xmin, ymax, xmax must be normalized coordinates (0 to 1).
        2. Only return objects you are confident you see.
        """

        class Detection(BaseModel):
            label: str = Field(description="The exact name of the object found.")
            coordinates: List[float] = Field(description="Normalized coordinates [ymin, xmin, ymax, xmax].")

        class GroundingResponse(BaseModel):
            thinking: str = Field(description="Reasoning about the scene and object locations.")
            detections: List[Detection] = Field(description="List of detected objects.")
        
        try:
            # 1. robust image loading
            try:
                pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
                out_buffer = io.BytesIO()
                pil_img.save(out_buffer, format="JPEG", quality=100)
                clean_image_data = out_buffer.getvalue()
            except Exception as e:
                logger.warning(f"Image cleaning failed: {e}")
                clean_image_data = image_data

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=clean_image_data))
                    ])
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GroundingResponse,
                }
            )
            
            # Robust JSON Parsing (Structured Output)
            if not response.candidates:
                logger.error(f"Gemini returned NO candidates. Finish reason: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'Unknown'}")
                # Log full response for debugging
                logger.debug(f"Full Response: {response}")
                return {}

            try:
                # Use Pydantic to validate
                parsed_response = response.parsed
                if not parsed_response:
                     # Fallback to text parsing if .parsed is empty for some reason (rare with config)
                     parsed_response = GroundingResponse.model_validate_json(response.text)

                results = {}
                valid_boxes_for_draw = []

                for det in parsed_response.detections:
                    label = det.label
                    coords = det.coordinates
                    
                    if label and coords and len(coords) == 4:
                        # Handle 0-1000 scale correction
                        if any(c > 1.0 for c in coords):
                            coords = [c / 1000.0 for c in coords]
                        
                        results[label] = coords
                        valid_boxes_for_draw.append({"label": label, "box_2d": coords})

                if valid_boxes_for_draw:
                    self._draw_and_save(clean_image_data, valid_boxes_for_draw, f"multi_{len(results)}_objs")
                else:
                    logger.warning(f"Gemini returned no detections. Raw text: {response.text[:100]}...")
                    
                return results

            except Exception as e:
                logger.error(f"Failed to parse JSON with Pydantic: {e}. Text: {response.text}")
                return {}
             
            return results

        except Exception as e:
            logger.error(f"Multi-Grounding failed: {e}")
            return {}

    def ground_object(self, image_data: bytes, object_description: str) -> List[Dict]:
        # Update single grounding to use the same robust cleaning/parsing logic if you wish
        # For now, we can just wrap the new multi-grounder for simplicity:
        res_dict = self.ground_multiple_objects(image_data, [object_description])
        if object_description in res_dict:
            return [{"box_2d": res_dict[object_description], "label": object_description}]
        return []

    def _draw_and_save(self, image_data: bytes, boxes: List[Dict], description: str):
        logger = get_logger()
        timestamp = datetime.now().strftime("%H%M%S")
        filename = self.log_dir / f"ground_{timestamp}_{description.replace(' ','_')}.jpg"
        
        if CV2_AVAILABLE:
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None: return 
                
                h, w = img.shape[:2]
                colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]

                for i, box in enumerate(boxes):
                    y1, x1, y2, x2 = box['box_2d']
                    label = box.get('label', description)
                    color = colors[i % len(colors)]
                    
                    p1 = (int(x1*w), int(y1*h))
                    p2 = (int(x2*w), int(y2*h))
                    cv2.rectangle(img, p1, p2, color, 2)
                    cv2.putText(img, label, (p1[0], max(20, p1[1]-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.imwrite(str(filename), img)
                logger.info(f"Saved grounding to {filename}")
            except Exception as e:
                logger.error(f"CV2 draw failed: {e}")

# ============================================================================
# AUDIO (TTS & STT)
# ============================================================================

class AudioAssistant:
    """
    Handles Text-to-Speech (gTTS) and Speech-to-Text (Whisper).
    Uses a queue to ensure TTS messages are played sequentially.
    """
    def __init__(self, log_dir: Path, executor=None):
        self.log_dir = log_dir / "audio"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.whisper_model = None
        self.executor = executor
        
        # TTS Queue
        self.speech_queue = queue.Queue()
        threading.Thread(target=self._speech_worker, daemon=True).start()
        
        if AUDIO_AVAILABLE:
            try:
                print(f"Loading Whisper model ({WHISPER_MODEL})... this may take a moment.")
                self.whisper_model = whisper.load_model(WHISPER_MODEL)
                print("Whisper model loaded.")
            except Exception as e:
                print(f"Failed to load Whisper model: {e}")

        self.last_spoken = None

    def _speech_worker(self):
        """Worker thread that processes the speech queue sequentially."""
        while True:
            text = self.speech_queue.get()
            if text is None: break # Sentinel to stop if needed
            
            try:
                if not AUDIO_AVAILABLE:
                    self.speech_queue.task_done()
                    continue

                # Mute mic while speaking
                if self.executor:
                    try:
                        self.executor.call("mute_microphone")
                    except Exception as e:
                        print(f"Failed to mute mic: {e}")

                try:
                    # Create temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        temp_path = fp.name
                    
                    # Generate speech
                    tts = gTTS(text=text, lang='en')
                    tts.save(temp_path)
                    
                    # Speed up audio 1.5x using ffmpeg if available
                    if subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL) == 0:
                        try:
                            fast_path = temp_path.replace(".mp3", "_fast.mp3")
                            subprocess.run([
                                "ffmpeg", "-y", "-i", temp_path, "-filter:a", "atempo=1.5", "-vn", fast_path
                            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                            os.replace(fast_path, temp_path)
                        except Exception as e:
                            print(f"Audio speed adjustment failed: {e}")

                    # Play audio (try mpg123 first, then ffplay)
                    if subprocess.call(["which", "mpg123"], stdout=subprocess.DEVNULL) == 0:
                         subprocess.run(["mpg123", "-q", temp_path], check=False, stdin=subprocess.DEVNULL)
                    elif subprocess.call(["which", "ffplay"], stdout=subprocess.DEVNULL) == 0:
                         subprocess.run(["ffplay", "-nodisp", "-autoexit", "-hide_banner", temp_path], check=False, stdin=subprocess.DEVNULL)
                    elif subprocess.call(["which", "aplay"], stdout=subprocess.DEVNULL) == 0:
                         # gTTS saves as mp3, aplay plays wav. Needs conversion or just fail.
                         print("Warning: Only 'aplay' found, but gTTS outputs MP3. Please install mpg123 or ffmpeg.")
                    else:
                        print("Error: No audio player found (install mpg123 or ffmpeg).")
                    
                    # Cleanup
                    os.remove(temp_path)

                except Exception as e:
                    print(f"TTS Error: {e}")
                finally:
                    # Unmute mic after speaking
                    if self.executor:
                        try:
                            self.executor.call("unmute_microphone")
                        except Exception as e:
                            print(f"Failed to unmute mic: {e}")

            except Exception as e:
                print(f"Speech worker error: {e}")
            finally:
                self.speech_queue.task_done()

    def speak(self, text: str):
        """Add text to the speech queue."""
        if not text: return
        self.last_spoken = text
        self.speech_queue.put(text)

    def repeat_last(self):
        """Repeat the last spoken text."""
        if hasattr(self, 'last_spoken') and self.last_spoken:
            self.speak(self.last_spoken)
        else:
            self.speak("I haven't said anything yet.")

    def listen(self, duration: int = 5) -> Optional[str]:
        """
        Record audio for `duration` seconds and transcribe.
        Returns transcribed text or None.
        """
        if not AUDIO_AVAILABLE or not self.whisper_model:
            print("Audio tools not available.")
            return None
            
        print(f"Listening for {duration} seconds... (Speak now)")
        
        timestamp = datetime.now().strftime("%H%M%S")
        wav_path = self.log_dir / f"rec_{timestamp}.wav"
        
        # 1. Record with arecord
        try:
            # -f cd sets 16bit little endian, 44100Hz, stereo
            # -d duration
            cmd = ["arecord", "-f", "cd", "-d", str(duration), str(wav_path)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Recording check failed (arecord): {e}")
            return None
            
        if not wav_path.exists() or wav_path.stat().st_size < 100:
            print("Recording failed or empty.")
            return None
            
        # 2. Transcribe
        print("Transcribing...")
        try:
            result = self.whisper_model.transcribe(str(wav_path))
            text = result["text"].strip()
            print(f"You said: {text}")
            return text
        except Exception as e:
            print(f"Transcription failed: {e}")
            return None
    def listen_manual_stop(self) -> Optional[str]:
        """
        Starts recording audio and waits for the user to press Enter to stop.
        Returns transcribed text or None.
        """
        if not AUDIO_AVAILABLE or not self.whisper_model:
            print("Audio tools not available.")
            return None
        
        timestamp = datetime.now().strftime("%H%M%S")
        wav_path = self.log_dir / f"rec_{timestamp}.wav"
        
        # 1. Start Recording (Non-blocking)
        print(f"\n[Recording started] Speak now... (Press Enter to stop)")
        
        # Mute mic while recording? Wait, user wants to TALK.
        # "whenever the user presses enter to talk, it mutes the SteamVR mic, and once they press enter again, unmutes it."
        # This means while RECORDING for the Agent, the user should be MUTED in SteamVR (so they don't broadcast to the game).
        
        if self.executor:
            try:
                self.executor.call("mute_microphone")
            except Exception as e:
                print(f"Failed to mute mic: {e}")

        try:
            # -f cd sets 16bit little endian, 44100Hz, stereo
            cmd = ["arecord", "-f", "cd", str(wav_path)]
            # Start process
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 2. Wait for User Input to Stop
            try:
                input() 
            except EOFError:
                pass # Handle cases where stdin is closed
            
            # 3. Stop Recording
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                
            print("[Recording stopped]")
            
        except Exception as e:
            print(f"Recording failed: {e}")
            return None
        finally:
            # Unmute mic after recording
            if self.executor:
                try:
                    self.executor.call("unmute_microphone")
                except Exception as e:
                    print(f"Failed to unmute mic: {e}")
        if not wav_path.exists() or wav_path.stat().st_size < 100:
            print("Recording failed or empty.")
            return None
            
        # 4. Transcribe
        print("Transcribing...")
        try:
            result = self.whisper_model.transcribe(str(wav_path))
            text = result["text"].strip()
            print(f"You said: {text}")
            return text
        except Exception as e:
            print(f"Transcription failed: {e}")
            return None


# ============================================================================
# PLANNING (Gemini 3 Flash)
# ============================================================================

@dataclass
class ActionPlanItem:
    tool: str
    args: Dict[str, Any]
    description: str

class ActionPlanner:
    """Uses Gemini 3 Flash to create a structured action plan."""
    def __init__(self, client):
        self.client = client
        self.model_name = MODEL_PLANNER

    def create_plan(self, user_request: str) -> List[ActionPlanItem]:
        logger = get_logger()
        logger.info(f"Planning for request: {user_request}")
        
        prompt = f"""
        User Request: "{user_request}"
        
        You are a VR Agent Planner. Create a sequential list of tools to execute. In this environment (which is called VRChat), the movements are done using press of the track pad. You need to push it all the way on the direction you want to move. 
        
        AVAILABLE TOOLS:
        1. GROUNDING & TRACKING:
           - locate_object(object_description) -> Use for single items.
           - visual_servo_to_object(object_description, controller, ray_description) -> Use to ALIGN or POINT. Note: BE AS DESCRIPTIVE AS POSSIBLE WITH THE OBJECT DESCRIPTION. Make sure to give the side of the controller as well. For exmaple, if the user says "align the left controller blue ray with the button that says ASMR", you should use the following arguments: object_description="button that says ASMR", controller="controller1", ray_description="left controller blue ray".
           - track_multiple_items(object_names) -> Use when specific multiple items are requested.
           - type_text(text, controller) -> Use to type on virtual keyboard.
           - inspect_surroundings() -> Take a picture.
           - describe_view(question) -> Describe what is seen.
           - verify_action(action_description) -> Check if action succeeded.
        
        2. CONTROLLER INPUTS (Low-level interaction):
           - click_button(controller, button) -> Quick press & release.
           - press_button(controller, button) -> Hold button down.
           - release_button(controller, button) -> Release held button.
             * Buttons: "trigger", "grip", "menu", "system", "trackpad", "a", "b"
           - set_trigger(controller, value) -> Analog trigger (0.0-1.0).
           - set_joystick(controller, x, y) -> Joystick pos (-1.0 to 1.0).
           - move_joystick_direction(controller, direction) -> "up", "down", "left", "right", "forward", "backward".
           - click_trackpad_direction(controller, direction, duration) -> Move & Click trackpad (default 1s).
           - perform_grab(controller) -> Grab object (grip+trigger).
           - perform_release(controller) -> Release object.
           - release_all_inputs(controller) -> Reset all.
           - get_controller_state(controller) -> Check state.
           - reset_controller_positions() -> Reset virtual hands (natural).
           - reset_controller_orientation() -> Reset to Front/Up/Down pose (Left DOWN, Right UP).
           - open_menu_sequence() -> Set positions and open menu.
        
        3. WHITE CANE ACCESSIBILITY (for blind users):
           - white_cane_describe() -> Immediate capture and description for blind user.
           - white_cane_set_goal(goal) -> Set/update navigation goal for white cane mode.
        
        CONTROLLER DEFINITIONS:
        - controller1: LEFT
        - controller2: RIGHT
        
        Return STRICT JSON format:
        {{
            "plan": [
                {{ "tool": "tool_name", "args": {{ "arg1": "val1" }}, "description": "Why this step?" }}
            ]
        }}
        """

        class PlanItem(BaseModel):
            tool: str = Field(description="The name of the tool to call.")
            args: str = Field(description="The arguments for the tool as a valid JSON object string (e.g. '{\"arg1\": \"value\"}').")
            description: str = Field(description="Description of why this step is being taken.")

        class PlanResponse(BaseModel):
            plan: List[PlanItem] = Field(description="Sequential list of actions.")

        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PlanResponse
                }
            )
            
            # Pydantic Parsing
            try:
                parsed = response.parsed
                if not parsed:
                    parsed = PlanResponse.model_validate_json(response.text)
            except Exception as e:
                logger.error(f"Plan validation failed: {e}. Text: {response.text}")
                return []
            
            plan = []
            for item in parsed.plan:
                # Parse the JSON string args back to dict
                try:
                    args_dict = json.loads(item.args)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse args json: {item.args}. Using empty dict.")
                    args_dict = {}

                plan.append(ActionPlanItem(
                    tool=item.tool,
                    args=args_dict,
                    description=item.description
                ))
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return []

class WhiteCaneAssistant:
    """
    Accessibility assistant for blind users.
    Captures images with timestamps, describes scenes naturally,
    and recommends physical actions to navigate toward goals.
    Uses structured output for clear separation of thought/description/action.
    """
    def __init__(self, client, executor, log_dir: Path):
        self.client = client
        self.executor = executor
        self.model_name = MODEL_WHITE_CANE
        self.log_dir = log_dir / "white_cane"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # State
        self.active = False
        self.current_goal = None
        self.stop_event = threading.Event()
        self.loop_thread = None
        self.loop_thread = None
        self.audio = AudioAssistant(log_dir, executor)
        
        # Caching - stores (timestamp_str, image_bytes, file_path)
        self.cached_images: List[tuple] = []
        self.conversation_history: List[Dict] = []
        
        # Description prompt template
        self.description_prompt = """
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

    def capture_with_timestamp(self) -> tuple:
        """
        Captures an image and saves it with timestamp in filename.
        Returns (timestamp_str, image_bytes, file_path)
        """
        logger = get_logger()
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%H:%M:%S")
        filename = f"image_{timestamp_str.replace(':', '_')}.png"
        file_path = self.log_dir / filename
        
        # Capture image
        res = self.executor.call("inspect_surroundings")
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            
            # Save as PNG with timestamp name
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_img.save(str(file_path), format="PNG")
            
            logger.info(f"White cane capture: {file_path}")
            return (timestamp_str, img_bytes, str(file_path))
            
        except Exception as e:
            logger.error(f"White cane capture failed: {e}")
            return (timestamp_str, None, None)

    def describe_and_recommend(self, timestamp_str: str, img_bytes: bytes, user_input: str = None) -> Dict:
        """
        Generates a structured response with thought, description, action for a blind user.
        Returns a dict with keys: thought, description, action, goal_achieved, new_goal
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
        
        # Define structured output schema
        class WhiteCaneResponse(BaseModel):
            thought: str = Field(description="Your internal reasoning about what you see and what to do next. This is NOT read to the user.")
            description: str = Field(description="Natural, extremely concise description (max 20 words). NO bullet points. This WILL be read aloud.")
            action: str = Field(description="One specific physical action recommendation, e.g. 'Turn your head 30 degrees to the right' or 'Take two steps forward'")
            goal_achieved: bool = Field(description="Set to true if the user's goal has been achieved based on what you see")
            new_goal: Optional[str] = Field(default=None, description="If the user expressed a desire to change their goal in their input, extract the new goal here")
        
        # Build conversation parts with history
        parts = []
        
        # Add prompt with current context
        prompt = self.description_prompt.format(
            goal=self.current_goal or "exploring the environment",
            timestamp=timestamp_str
        )
        parts.append(types.Part(text=prompt))
        
        # Add user input if provided (for goal change detection)
        if user_input:
            parts.append(types.Part(text=f"\n[User just said]: \"{user_input}\"\n"))
        
        # Add images in chronological order (limit to 4 previous + 1 current = 5 total)
        # Get last 4 cached images for history
        history_images = self.cached_images[-4:]
        total_images = len(history_images) + 1  # +1 for current image
        
        # Ordinal labels for chronological presentation
        ordinal_labels = ["first", "second", "third", "fourth", "fifth"]
        
        parts.append(types.Part(text="\n--- IMAGE TIMELINE (chronological order) ---\n"))
        
        # Add historical images in chronological order
        for idx, (ts, img_data, path) in enumerate(history_images):
            if img_data:
                position = idx + 1
                if idx == 0:
                    label = f"At the beginning (image 1 of {total_images}), captured at {ts}"
                else:
                    ordinal = ordinal_labels[idx] if idx < len(ordinal_labels) else f"{idx+1}th"
                    label = f"In the {ordinal} capture (image {position} of {total_images}), at {ts}"
                
                parts.append(types.Part(text=f"\n[{label}]:"))
                parts.append(types.Part(inline_data=types.Blob(
                    mime_type="image/png", data=img_data
                )))
        
        # Add current image as the final/most recent
        current_position = len(history_images) + 1
        parts.append(types.Part(text=f"\n[NOW - Current view (image {current_position} of {total_images}), captured at {timestamp_str}]:"))
        parts.append(types.Part(inline_data=types.Blob(
            mime_type="image/png", data=img_bytes
        )))
        
        # Add conversation history summary if exists
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
            
            # Parse structured response
            try:
                parsed = response.parsed
                if not parsed:
                    parsed = WhiteCaneResponse.model_validate_json(response.text)
                
                result = {
                    "thought": parsed.thought,
                    "description": parsed.description,
                    "action": parsed.action,
                    "goal_achieved": parsed.goal_achieved,
                    "new_goal": parsed.new_goal
                }
            except Exception as parse_error:
                logger.warning(f"Structured parse failed, using fallback: {parse_error}")
                # Fallback to raw text
                result = {
                    "thought": "Processing...",
                    "description": response.text,
                    "action": "Continue as you were",
                    "goal_achieved": False,
                    "new_goal": None
                }
            
            # Cache this interaction
            self.conversation_history.append({
                "timestamp": timestamp_str,
                "summary": result["description"][:80] + "..." if len(result["description"]) > 80 else result["description"]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"White cane description failed: {e}")
            return {
                "thought": f"Error: {e}",
                "description": f"I encountered an error while analyzing the scene.",
                "action": "Please wait while I try again",
                "goal_achieved": False,
                "new_goal": None
            }

    def format_for_speech(self, result: Dict) -> str:
        """Format the structured result for spoken output to the user."""
        speech = result["description"]
        if result["action"]:
            speech += f" I recommend: {result['action']}"
        if result["goal_achieved"]:
            speech += " It looks like you've reached your goal!"
        return speech

    def activate(self, goal: str = None):
        """Activate white cane mode with an optional goal."""
        self.active = True
        self.current_goal = goal
        self.stop_event.clear()
        self.cached_images = []
        self.conversation_history = []
        
        logger = get_logger()
        logger.info(f"White cane activated. Goal: {goal}")
        
        response = f"White cane mode activated. {'Goal: ' + goal if goal else 'No specific goal set. I will describe what I see.'}\nSay 'disable white cane' to stop, or tell me your new goal anytime."
        self.audio.speak(response)
        return response

    def set_goal(self, goal: str):
        """Update the current goal."""
        old_goal = self.current_goal
        self.current_goal = goal
        logger = get_logger()
        logger.info(f"White cane goal changed: '{old_goal}' -> '{goal}'")
        return f"Goal updated from '{old_goal}' to '{goal}'"

    def deactivate(self):
        """Deactivate white cane mode."""
        self.active = False
        self.stop_event.set()
        if self.loop_thread and self.loop_thread.is_alive():
            if threading.current_thread() != self.loop_thread:
                self.loop_thread.join(timeout=2.0)
        
        logger = get_logger()
        logger.info("White cane deactivated.")
        
        return "White cane mode deactivated."

    def get_immediate_help(self, user_input: str = None) -> str:
        """Capture and describe immediately (when user says 'help')."""
        timestamp_str, img_bytes, file_path = self.capture_with_timestamp()
        
        if img_bytes:
            self.cached_images.append((timestamp_str, img_bytes, file_path))
        
        result = self.describe_and_recommend(timestamp_str, img_bytes, user_input)
        
        # Handle goal change if detected
        if result.get("new_goal"):
            self.set_goal(result["new_goal"])
            print(f"[Goal Changed]: {result['new_goal']}")
        
        return self.format_for_speech(result)

    def listen_command(self) -> Optional[str]:
        """Listen for a voice command (manual start/stop)."""
        return self.audio.listen_manual_stop()

    def run_loop(self, interval: float = 2.0, status_interval: float = 30.0):
        """
        Run the white cane loop in a separate thread.
        Captures every 'interval' seconds to build history.
        Announces status every 'status_interval' seconds.
        """
        logger = get_logger()
        import time
        last_announcement_time = time.time()
        
        while self.active and not self.stop_event.is_set():
            # Status Announcement
            if time.time() - last_announcement_time > status_interval:
                goal_msg = f"Goal: {self.current_goal}." if self.current_goal else "No specific goal."
                self.audio.speak(f"White cane active. {goal_msg} Say 'help' for details.")
                last_announcement_time = time.time()

            # Capture (Silent Monitoring)
            timestamp_str, img_bytes, file_path = self.capture_with_timestamp()
            
            if img_bytes:
                self.cached_images.append((timestamp_str, img_bytes, file_path))
                # Prune cache if it gets too large (optional, but good practice)
                if len(self.cached_images) > 20: 
                    self.cached_images.pop(0)

                # Log only, no speech/description
                logger.info(f"[White Cane] Silent Monitor: Captured {timestamp_str}")
                print(f"[White Cane] Monitoring... ({timestamp_str})", end='\r')
            
            # Wait for interval or until stopped
            self.stop_event.wait(timeout=interval)
        
        logger.info("White cane loop ended.")

    def start_background_loop(self, interval: float = 2.0):
        """Start the capture loop in a background thread."""
        if self.loop_thread and self.loop_thread.is_alive():
            return "White cane loop is already running."
        
        self.stop_event.clear()
        self.loop_thread = threading.Thread(
            target=self.run_loop,
            args=(interval,),
            daemon=True
        )
        self.loop_thread.start()
        return "White cane background loop started (every 2 seconds)."


# ============================================================================
# VERIFICATION (Gemini 2.5 Pro)
# ============================================================================

class Verifier:
    """Uses Gemini 2.5 Pro to verify actions."""
    def __init__(self, client):
        self.client = client
        self.model_name = MODEL_VERIFICATION

    def verify(self, image_data: bytes, action_description: str) -> str:
        prompt = f"""
        Verify if the following action was successful based on the image:
        Action: "{action_description}"
        
        Be critical. If you see failure, explain why. If success, confirm it.
        Use one sentence maximum as this will be read back to the user as a TTS feedback.
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(role="user", parts=[
                types.Part(text=prompt),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_data))
            ])]
        )
        return response.text

# ============================================================================
# DESCRIPTION (Gemini 2.5 Flash Lite)
# ============================================================================

class Describer:
    """Uses Gemini 2.5 Flash Lite for scene description."""
    def __init__(self, client):
        self.client = client
        self.model_name = MODEL_DESCRIPTION
        
    def describe(self, image_data: bytes, question: str) -> str:
        # Add accessibility context for blind users
        accessibility_prompt = (
            "IMPORTANT: This response is for a blind user and will be read aloud. "
            "Please provide a very detailed but succinct description. "
            "Specifically mention important items users might need, especially buttons or texts on the screen and their locations. "
            "CRITICAL: Do NOT use bullet points, lists, or multi-paragraph structured text. "
            "Output a single continuous paragraph. "
            "Ensure the description is fully detailed regardless of length so the user receives all necessary information."
        )
        
        full_query = f"{accessibility_prompt}\n\nQuestion: {question}"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(role="user", parts=[
                types.Part(text=full_query),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_data))
            ])]
        )
        return response.text
    
    def describe_multi(self, labeled_images: list, question: str) -> str:
        """
        Describe multiple images with labels (e.g., front/left/right views).
        
        Args:
            labeled_images: List of tuples (label, image_bytes)
            question: The prompt/question for the model
        """
        parts = [types.Part(text=question)]
        
        for label, img_data in labeled_images:
            # Add label for each image
            parts.append(types.Part(text=f"\n[{label.upper()} VIEW]:"))
            parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_data)))
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(role="user", parts=parts)]
        )
        return response.text

# ============================================================================
# EXECUTOR
# ============================================================================

class DirectMCPExecutor:
    """Loads mcp_server.py dynamically."""
    def __init__(self):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("mcp_server", "mcp_server.py")
        if not spec: raise ImportError("Could not find mcp_server.py")
        self.module = importlib.util.module_from_spec(spec)
        sys.modules["mcp_server"] = self.module
        spec.loader.exec_module(self.module)
        
    def call(self, tool: str, **kwargs):
        func = getattr(self.module, tool)
        return func(**kwargs)

# ============================================================================
# TOOL FUNCTIONS (Wrapped for Gemini)
# ============================================================================

# Global instances
_executor = None
_grounder = None
_tracker = None
_white_cane = None
_describer = None
_agent_ref = None # To access history

def _get_tools(executor, grounder, tracker, white_cane, describer, agent_ref):
    """Define tools as a list of callables."""
    global _executor, _grounder, _tracker, _white_cane, _describer, _agent_ref
    _executor = executor
    _grounder = grounder
    _tracker = tracker
    _white_cane = white_cane
    _describer = describer
    _agent_ref = agent_ref
    
    def _log_action(tool_name, **kwargs):
        get_logger().info(f"[TOOL] {tool_name}({kwargs})")
        print(f"Action: {tool_name} {kwargs}")

    def start_bridge():
        """Start the VR bridge connection."""
        _log_action("start_bridge")
        return _executor.call("start_vr_bridge")
        
    def move_relative(device: str, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move a device relative to current position. dz: -ve is forward, +ve is back."""
        _log_action("move_relative", device=device, dx=dx, dy=dy, dz=dz)
        return _executor.call("move_relative", device=device, dx=dx, dy=dy, dz=dz)

    def move_absolute(device: str, x: float, y: float, z: float):
        """Move a device to absolute coordinates."""
        _log_action("move_absolute", device=device, x=x, y=y, z=z)
        return _executor.call("teleport", device=device, x=x, y=y, z=z)
        
    def teleport(device: str, x: float, y: float, z: float):
        """Teleport to exact coordinates."""
        _log_action("teleport", device=device, x=x, y=y, z=z)
        return _executor.call("teleport", device=device, x=x, y=y, z=z)
        
    def rotate_device(device: str, pitch: float, yaw: float, roll: float):
        """Rotate device (degrees)."""
        _log_action("rotate_device", device=device, pitch=pitch, yaw=yaw, roll=roll)
        return _executor.call("rotate_device", device=device, pitch=pitch, yaw=yaw, roll=roll)
        
    def inspect_surroundings():
        """Take a picture."""
        _log_action("inspect_surroundings")
        return _executor.call("inspect_surroundings")
        
    def locate_object(object_description: str):
        """Find an object and return its center coordinates."""
        _log_action("locate_object", description=object_description)
        # 1. Capture
        res = _executor.call("inspect_surroundings")
        data = json.loads(res).get("data")
        if not data: return "Failed to capture image"
        
        # 2. Ground
        img_bytes = base64.b64decode(data)
        boxes = _grounder.ground_object(img_bytes, object_description)
        
        if not boxes: return f"Could not find '{object_description}'"
        
        # 3. Format
        results = []
        for box in boxes:
            y1, x1, y2, x2 = box['box_2d']
            cx, cy = (x1+x2)/2, (y1+y2)/2
            results.append(f"Found at Center(x={cx:.2f}, y={cy:.2f})")
            
        return "; ".join(results)
    def track_multiple_items(object_names: List[str]):
        """
        Track multiple objects simultaneously in a video.
        Example input: ["red cup", "keyboard", "blue pen"]
        """
        _log_action("track_multiple_items", objects=object_names)
        
        if not _tracker or not _tracker.available:
            return "Error: Object Tracking (SAM 3) is not available."

        # 1. Capture Video
        print(f"Capturing video to track: {object_names}...")
        res_str = _executor.call("capture_video", duration=3.0)
        
        # Parse video response
        try:
            res = json.loads(res_str)
            frames = res.get("frames", [])
            if not frames: return "Error: No frames captured."
        except:
            return "Error parsing video data."

        # Save Frames
        timestamp = datetime.now().strftime("%H%M%S")
        temp_dir = LOG_DIR / "tracking" / f"multi_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        for i, b64 in enumerate(frames):
            path = temp_dir / f"frame_{i:04d}.jpg"
            img_data = base64.b64decode(b64)
            # Robust Save
            try:
                with Image.open(io.BytesIO(img_data)) as img:
                    img.save(path, quality=95)  # High quality JPEG
            except:
                with open(path, "wb") as f: f.write(img_data)
            saved_frames.append(path)

        # Grounding
        print("Locating objects in first frame...")
        with open(saved_frames[0], "rb") as f:
            first_frame_data = f.read()

        # Call the new multi-grounder
        # Returns Dict: {'cup': [y,x,y,x], 'keyboard': [y,x,y,x]}
        initial_data = _grounder.ground_multiple_objects(first_frame_data, object_names)
        if not initial_data:
            print(f"FAILED: Grounding found 0 objects. Expected: {object_names}")
            # This return string tells the agent to try fallback
            return f"Could not locate any of the requested objects: {object_names}"
            
        print(f"Found {len(initial_data)} objects: {list(initial_data.keys())}")

        # Tracking
        print("Running SAM 3 Multi-Tracking...")
        result = _tracker.track_multi_objects(str(temp_dir), initial_data)
        
        if "error" in result:
            return f"Tracking failed: {result['error']}"
            
        video_path = result.get("video_path")
        print(f"\n[SUCCESS] Multi-Tracking Video: {video_path}\n")
        get_logger().info(f"Multi-Tracking Video: {video_path}")
        
        return f"Tracking Complete. Video saved to {video_path}"
    def finish_task(summary: str):
        """Call this when the user's request is fully completed."""
        _log_action("finish_task", summary=summary)
        return f"Task Completed: {summary}"

    def get_connection_status():
        """Check VR driver connection."""
        _log_action("get_connection_status")
        return _executor.call("get_connection_status")

    def kill_address():
        """
        Kills the process using the configured TCP port (default 5555).
        Use this when the server fails to start with "Address already in use".
        """
        _log_action("kill_address")
        return _executor.call("kill_address")

    def capture_video(duration: float = 3.0):
        """Capture a short video clip (sequence of frames) and save it."""
        _log_action("capture_video", duration=duration)
        res_str = _executor.call("capture_video", duration=duration)
        
        # Auto-save logic
        try:
            res = json.loads(res_str)
            if res.get("type") == "video":
                timestamp = datetime.now().strftime("%H%M%S")
                # Create a video directory
                vid_dir = LOG_DIR / "videos" / f"video_{timestamp}"
                vid_dir.mkdir(parents=True, exist_ok=True)
                
                # Save frames
                frames = res.get("frames", [])
                for i, b64 in enumerate(frames):
                    with open(vid_dir / f"frame_{i:04d}.jpg", "wb") as f:
                        f.write(base64.b64decode(b64))
                
                print(f"Video saved to: {vid_dir}")
                get_logger().info(f"Video saved to {vid_dir}")
                
                # Return a summary to the agent, not the huge base64 blob to save tokens
                return f"Video captured and saved to {vid_dir} ({len(frames)} frames)."
        except Exception as e:
            get_logger().error(f"Failed to auto-save video: {e}")
            
        return res_str

    def track_object(object_description: str):
        """
        Track an object in a video.
        1. Captures a 3-second video.
        2. Finds the object in the first frame.
        3. Tracks it using SAM 2.
        """
        _log_action("track_object", description=object_description)
        logger = get_logger()
        
        if not _tracker or not _tracker.available:
            return "Error: Object Tracking (SAM 2) is not available."

        # 1. Capture Video
        print("Capturing video...")
        res_str = _executor.call("capture_video", duration=3.0)
        if "Error" in res_str and not res_str.startswith("{"): return res_str
        
        try:
            res = json.loads(res_str)
            if res.get("type") != "video": return f"Error capturing video: {res_str[:100]}"
            frames = res.get("frames", [])
            if not frames: return "Error: No frames in video."
        except json.JSONDecodeError:
            return f"Error parsing video response: {res_str[:100]}"

        # 2. Save Frames to Temp Dir
        timestamp = datetime.now().strftime("%H%M%S")
        temp_dir = LOG_DIR / "tracking" / f"temp_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        for i, b64 in enumerate(frames):
            path = temp_dir / f"frame_{i:04d}.jpg"
            # Decode the base64 string to get the JPEG byte data. 
            # This is required because the server sends images as base64 strings in JSON.
            img_data = base64.b64decode(b64)
            
            # Use PIL to save, which handles/fixes truncated JPEGs gracefully.
            # This "cleans" the file structure so OpenCV won't complain about corruption.
            try:
                with Image.open(io.BytesIO(img_data)) as img:
                    img.save(path, quality=95)  # High quality JPEG
            except Exception as e:
                # Fallback if image is too broken even for PIL
                print(f"Warning: PIL failed to clean frame {i}: {e}. Saving raw.")
                with open(path, "wb") as f:
                    f.write(img_data)
            
            saved_frames.append(path)
            
        # 3. Ground in First Frame
        print("Locating object in first frame...")
        # Re-read the first frame from disk to ensure we use the 'fixed' version
        with open(saved_frames[0], "rb") as f:
            first_frame_data = f.read()

        boxes = _grounder.ground_object(first_frame_data, object_description)
        
        if not boxes:
            return f"Could not find '{object_description}' in the first frame to start tracking."
            
        # Select best box (first one)
        init_box = boxes[0]['box_2d']
        
        # 4. Track
        print("Running SAM 2 Tracking...")
        video_output = _tracker.track(str(temp_dir), init_box, object_description)
        
        # Explicitly print the location for the user
        print(f"\n[SUCCESS] Tracking Video Saved to: {video_output}\n")
        get_logger().info(f"Tracking Video Saved to: {video_output}")
        
        return f"Tracking completed. Output video: {video_output}"

    def create_tracking_video(object_description: str):
        """
        Creates a segmented video of the specified object.
        This function captures a video, locates the object, and then generates
        a video with the object segmented and tracked.
        """
        return track_object(object_description)

    def visual_servo_to_object(object_description: str, controller: str, ray_description: str = "blue VR controller ray"):
        f"""
        Rotates the controller (modifies yaw`/pitch) so that the 'blue ray of the {"left controller" if controller == "controller1" else "right controller"}' aligns with the center of the specified target object.
        Use this when the user asks to 'align', 'point', or 'aim' the controller or its ray at something.
        
        STRICT ALIGNMENT PIPELINE:
        1. Grounding: Use Gemini to find the object and controller ray in the scene.
        2. Tracking: Use SAM (Segment Anything Model) to track the object/ray in real-time.
        3. Control: Use PID control to align the controller based on tracking feedback.
        NO OTHER SYSTEM SHOULD BE IMPLEMENTED FOR ALIGNMENT.
        
        Args:
            object_description: The object to align with.
            controller: The controller to move. MUST be "controller1" (left) or "controller2" (right).
            ray_description: The description of the ray to align. Defaults to "blue VR controller ray of the {"left controller" if controller == "controller1" else "right controller"}".
        """
        # Auto-adjust ray description if it's the default generic one
        if ray_description == "blue VR controller ray":
            if controller == "controller1":
                ray_description = "blue VR controller ray of the left controller"
            else:
                ray_description = "blue VR controller ray of the right controller"

        _log_action("visual_servo_to_object", description=object_description, controller=controller, ray=ray_description)
        logger = get_logger()

        if not _tracker or not _tracker.available:
            return "Error: Object Tracking (SAM 3) is not available."

        # Config
        Kp_YAW = 0.01   # Gain for Horizontal (Yaw)
        Kp_PITCH = 0.01 # Gain for Vertical (Pitch)
        MAX_ITER = 100
        TOLERANCE_PX = 15  # Stop if within this many pixels (Relaxed from 5)
        
        # 1. Get Initial Pose
        print(f"Getting initial pose of {controller}...")
        curr_pitch = 0.0
        curr_yaw = 0.0
        curr_roll = 0.0
        
        status = _executor.call("get_current_pose", device=controller)
        try:
            if "Rotation: [" in status:
                rot_str = status.split("Rotation: [")[1].split("]")[0]
                # Fix: Handle numpy float string format if present
                rot_str = rot_str.replace("np.float64(", "").replace(")", "")
                curr_pitch, curr_yaw, curr_roll = map(float, rot_str.split(","))
                print(f"Initial Rotation: Pitch={curr_pitch}, Yaw={curr_yaw}, Roll={curr_roll}")
        except Exception as e:
            msg = f"Failed to parse initial pose: {e}"
            logger.error(msg)
            return msg

        # 2. Capture Initial Image for Grounding
        print("Capturing initial image for grounding...")
        res = _executor.call("inspect_surroundings")
        if isinstance(res, str) and res.startswith("Error"): return f"Capture failed: {res}"
        
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            # Use PIL to clean/load
            pil_img_init = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil_img_init.size
        except Exception as e:
            return f"Initial image parsing failed: {e}"

        # 3. Ground Targets (Gemini - Once)
        # STRICT REQUIREMENT: Only uses Gemini for initial grounding.
        # Do NOT use simple centroid logic without SAM.
        targets = {
            "ray": ray_description,
            "logo": object_description
        }
        current_boxes = {}
        
        print(f"Grounding '{targets['ray']}' and '{targets['logo']}'...")
        # We can use the multi-object grounder
        grounding_results = _grounder.ground_multiple_objects(img_bytes, list(targets.values()))
        
        all_found = True
        for key, desc in targets.items():
            if desc in grounding_results:
                # Normalized [ymin, xmin, ymax, xmax] -> Pixel [x, y, w, h]
                ymin, xmin, ymax, xmax = grounding_results[desc]
                box_x, box_y = xmin * w, ymin * h
                box_w, box_h = (xmax - xmin) * w, (ymax - ymin) * h
                current_boxes[key] = [box_x, box_y, box_w, box_h]
            else:
                print(f"[{key}] '{desc}' NOT found by Gemini.")
                all_found = False
        
        if not all_found:
             return f"Failed to find both the controller ray and '{object_description}'. CAUTION: The ray might be occluding the object if they are already aligned. Verify alignment manually or try a different viewing angle."
             
        print("Initial grounding successful. Starting control loop.")

        # 4. Control Loop
        prev_dist = float('inf')
        divergence_count = 0
        
        for i in range(MAX_ITER):
            print(f"\n--- Servo Iteration {i+1}/{MAX_ITER} ---")
            
            # Check for stop signal
            if _agent_ref and _agent_ref.stop_execution.is_set():
                print(f"Visual Servoing Stopped by User.")
                return "Visual Servoing Stopped by User."

            # A. Capture New Image
            res = _executor.call("inspect_surroundings")
            if isinstance(res, str) and res.startswith("Error"):
                print("Capture failed in loop.")
                break
                
            try:
                data = json.loads(res).get("data")
                img_bytes_loop = base64.b64decode(data)
                pil_img_loop = Image.open(io.BytesIO(img_bytes_loop)).convert("RGB")
                img_cv = cv2.cvtColor(np.array(pil_img_loop), cv2.COLOR_RGB2BGR) # For Vis
            except Exception as e:
                print(f"Image parse error in loop: {e}")
                break

            # B. SAM Tracking (Update state with new image)
            # STRICT REQUIREMENT: Must use SAM tracker for robust feature matching.
            inference_state = _tracker.processor.set_image(pil_img_loop)
            
            points = {}
            masks_for_viz = {}  # Store masks for visualization
            
            for key, desc in targets.items():
                if key not in current_boxes: continue # Should not happen unless lost
                
                box_x, box_y, box_w, box_h = current_boxes[key]
                
                # Format box for SAM
                box_input_xywh = _tracker.torch.tensor([box_x, box_y, box_w, box_h]).view(-1, 4)
                box_input_cxcywh = _tracker.box_xywh_to_cxcywh(box_input_xywh)
                norm_box_cxcywh = _tracker.normalize_bbox(box_input_cxcywh, w, h).flatten().tolist()
                
                _tracker.processor.reset_all_prompts(inference_state)
                # Set text prompt to help SAM differentiate between similar objects
                inference_state = _tracker.processor.set_text_prompt(desc, inference_state)
                inference_state = _tracker.processor.add_geometric_prompt(
                    state=inference_state, box=norm_box_cxcywh, label=True
                )
                
                # Get Mask
                mask = None
                if "masks" in inference_state and inference_state["masks"] is not None:
                    m = inference_state["masks"].detach().cpu().numpy() > 0.5
                    if m.ndim == 4: mask = m[0, 0]
                    elif m.ndim == 3: mask = m[0]
                
                if mask is None:
                    print(f"[{key}] Lost tracking (no mask). Attempting to reground...")
                    try:
                        # Retry logic: Reground using Gemini on the current frame
                        reground_res = _grounder.ground_multiple_objects(img_bytes_loop, [desc])
                        if desc in reground_res:
                            ymin, xmin, ymax, xmax = reground_res[desc]
                            # Update current_boxes
                            box_x, box_y = xmin * w, ymin * h
                            box_w, box_h = (xmax - xmin) * w, (ymax - ymin) * h
                            current_boxes[key] = [box_x, box_y, box_w, box_h]
                            print(f"[{key}] Reground Successful: {current_boxes[key]}")
                            # Skip this key for this frame (no points generated yet)
                            continue 
                        else:
                            print(f"[{key}] Reground Failed (Gemini could not find it).")
                            del current_boxes[key]
                            continue
                    except Exception as e:
                        print(f"[{key}] Reground Error: {e}")
                        del current_boxes[key]
                        continue
                
                # Store mask for visualization
                masks_for_viz[key] = mask
                    
                # Update Box & Extract Point
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    current_boxes[key] = [cmin, rmin, cmax - cmin, rmax - rmin]
                    
                    # Points logic
                    if key == "ray":
                        # Ray Tip = Min Y (Highest point visually)
                        ys, xs = np.where(mask)
                        if len(ys) > 0:
                            idx = np.argmin(ys) 
                            points[key] = (xs[idx], ys[idx])
                            cv2.circle(img_cv, (xs[idx], ys[idx]), 5, (0, 0, 255), -1) # Red Tip
                    elif key == "logo":
                        # Centroid
                        M = cv2.moments(mask.astype(np.uint8))
                        if M["m00"] != 0:
                            cx = int(M["m10"]/M["m00"])
                            cy = int(M["m01"]/M["m00"])
                            points[key] = (cx, cy)
                            cv2.circle(img_cv, (cx, cy), 5, (0, 255, 0), -1) # Green center
                else:
                    print(f"[{key}] Mask empty. Attempting to reground...")
                    try:
                        # Retry logic: Reground using Gemini on the current frame
                        reground_res = _grounder.ground_multiple_objects(img_bytes_loop, [desc])
                        if desc in reground_res:
                            ymin, xmin, ymax, xmax = reground_res[desc]
                            # Update current_boxes
                            box_x, box_y = xmin * w, ymin * h
                            box_w, box_h = (xmax - xmin) * w, (ymax - ymin) * h
                            current_boxes[key] = [box_x, box_y, box_w, box_h]
                            print(f"[{key}] Reground Successful: {current_boxes[key]}")
                            # Skip this key for this frame (no points generated yet)
                            continue 
                        else:
                            print(f"[{key}] Reground Failed (Gemini could not find it).")
                            del current_boxes[key]
                            continue
                    except Exception as e:
                        print(f"[{key}] Reground Error: {e}")
                        del current_boxes[key]
                        continue
            
            # Apply colored mask overlays for visualization
            for key, mask in masks_for_viz.items():
                if key == "ray":
                    # Blue overlay for ray (BGR: 255, 100, 0)
                    color = np.array([255, 100, 0], dtype=np.uint8)
                elif key == "logo":
                    # Green overlay for target object (BGR: 0, 255, 100)
                    color = np.array([0, 255, 100], dtype=np.uint8)
                else:
                    color = np.array([128, 128, 128], dtype=np.uint8)
                
                # Apply semi-transparent overlay
                overlay = img_cv.copy()
                overlay[mask] = color
                cv2.addWeighted(overlay, 0.35, img_cv, 0.65, 0, img_cv)
            
            # C. PID & Control
            if "ray" in points and "logo" in points:
                rx, ry = points["ray"]
                lx, ly = points["logo"]
                
                dx = lx - rx
                dy = ly - ry
                dist = math.sqrt(dx*dx + dy*dy)
                
                print(f"Error: dx={dx}, dy={dy}, dist={dist:.2f}")
                
                # Check Divergence
                if dist > prev_dist + 50.0: # Significant increase
                    divergence_count += 1
                    print(f"Warning: Divergence detected ({divergence_count}). Dist increased from {prev_dist:.1f} to {dist:.1f}")
                else:
                    divergence_count = 0
                prev_dist = dist
                
                if divergence_count >= 3:
                     return f"Visual Servoing Aborted: Divergence detected (error increasing). Ray might be tracking the wrong object."

                # Save debug image
                cv2.line(img_cv, (rx, ry), (lx, ly), (0, 255, 255), 2)
                
                # Prepare info text
                # We calculate PID terms here just for display before applying them
                d_yaw_disp = Kp_YAW * dx
                d_pitch_disp = Kp_PITCH * dy
                info_text = f"Err: {dist:.1f}px | dYaw: {d_yaw_disp:.2f} | dPitch: {d_pitch_disp:.2f}"
                
                # Draw text background
                (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_cv, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 0), -1)
                cv2.putText(img_cv, info_text, (15, 15 + th), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                timestamp = datetime.now().strftime("%H%M%S")
                debug_path = LOG_DIR / "tracking" / f"servo_{timestamp}_iter_{i}.jpg"
                if CV2_AVAILABLE:
                    cv2.imwrite(str(debug_path), img_cv)
                logger.info(f"Saved servo debug image: {debug_path}")

                if dist < TOLERANCE_PX:
                    msg = f"Visual Servoing Complete. Aligned with {object_description} (Error: {dist:.2f}px)."
                    logger.info(msg)
                    return msg
                
                # PID
                d_yaw = -Kp_YAW * dx
                d_pitch = -Kp_PITCH * dy
                
                curr_yaw += d_yaw
                curr_pitch += d_pitch
                
                print(f"Adjusting: dYaw={d_yaw:.2f}, dPitch={d_pitch:.2f} -> New Yaw={curr_yaw:.2f}, Pitch={curr_pitch:.2f}")
                
                _executor.call("rotate_device", device=controller, 
                               pitch=curr_pitch, yaw=curr_yaw, roll=curr_roll)
                               
                # Short wait for physical movement                
            else:
                if "ray" in current_boxes and "logo" in current_boxes:
                     print("Tracking recovering... skipping control updates this frame.")
                     continue
                return f"Lost tracking of one or both objects during loop. Stopping."

        final_dist = dist if 'dist' in locals() else float('inf')
        if final_dist < 50.0:
            msg = f"Visual Servoing finished max iterations. Aligned within {final_dist:.1f}px (acceptable)."
        else:
            msg = f"Visual servoing finished max iterations ({MAX_ITER}). Final error: {final_dist:.1f}."
        
        logger.info(msg)
        return msg

    def type_text(text: str, controller: str = "controller2"):
        """
        Types text on a virtual keyboard by visually aligning the controller ray 
        to each key and pressing the trigger.
        
        This function makes ONE Gemini API call to ground all unique characters,
        then uses PID control to align and select each key sequentially.
        
        Args:
            text: The text to type (e.g., "temp", "hello world")
            controller: Which controller to use (default: controller2)
        """
        _log_action("type_text", text=text, controller=controller)
        logger = get_logger()
        
        if not _tracker or not _tracker.available:
            return "Error: Object Tracking (SAM 3) is not available."
        
        if not text:
            return "Error: No text provided to type."
        
        # Config
        Kp_YAW = 0.02
        Kp_PITCH = 0.02
        MAX_ITER_PER_CHAR = 50
        TOLERANCE_PX = 8  # Slightly larger tolerance for keyboard keys
        
        # 1. Get Initial Pose
        print(f"Getting initial pose of {controller}...")
        curr_pitch = 0.0
        curr_yaw = 0.0
        curr_roll = 0.0
        
        status = _executor.call("get_current_pose", device=controller)
        try:
            if "Rotation: [" in status:
                rot_str = status.split("Rotation: [")[1].split("]")[0]
                # Fix: Handle numpy float string format if present
                rot_str = rot_str.replace("np.float64(", "").replace(")", "")
                curr_pitch, curr_yaw, curr_roll = map(float, rot_str.split(","))
                print(f"Initial Rotation: Pitch={curr_pitch}, Yaw={curr_yaw}, Roll={curr_roll}")
        except Exception as e:
            msg = f"Failed to parse initial pose: {e}"
            logger.error(msg)
            return msg
        
        # 2. Capture Initial Image for Grounding
        print("Capturing initial image for keyboard grounding...")
        res = _executor.call("inspect_surroundings")
        if isinstance(res, str) and res.startswith("Error"): 
            return f"Capture failed: {res}"
        
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            pil_img_init = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil_img_init.size
        except Exception as e:
            return f"Initial image parsing failed: {e}"
        
        # 3. Ground All Unique Characters + Controller Ray (Single API Call)
        unique_chars = list(set(text.lower()))  # Get unique characters
        print(f"Grounding keyboard keys for characters: {unique_chars}")
        
        # Build the prompt for Gemini to return JSON with key-value pairs
        chars_list = ", ".join([f'"{c}"' for c in unique_chars])
        prompt = f"""
Find the following keyboard keys in the image: {chars_list}
Also find the "blue VR controller ray of the {"left controller" if controller == "controller1" else "right controller"}".

You MUST return the answer in the following JSON format:
{{
    "thinking": "Describe what you see and where the keys are located...",
    "keys": {{
        "a": [ymin, xmin, ymax, xmax],
        "b": [ymin, xmin, ymax, xmax]
    }},
    "controller_ray": [ymin, xmin, ymax, xmax]
}}

Rules:
1. ymin, xmin, ymax, xmax must be normalized coordinates (0 to 1).
2. Include ONLY the keys you can confidently locate.
3. The "keys" object maps each character to its bounding box.
4. "controller_ray" is the bounding box of the blue VR controller ray.
"""
        
        try:
            # Clean image for API
            out_buffer = io.BytesIO()
            pil_img_init.save(out_buffer, format="JPEG")
            clean_image_data = out_buffer.getvalue()
            
            response = _grounder.client.models.generate_content(
                model=_grounder.model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=clean_image_data))
                    ])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            
            # Parse response
            if not response.candidates or not response.candidates[0].content.parts:
                return "Error: Gemini returned no grounding results."
            
            resp_text = response.candidates[0].content.parts[0].text
            if resp_text.startswith("```"):
                resp_text = resp_text.split("\n", 1)[-1].rsplit("\n", 1)[0]
            if resp_text.startswith("json"):
                resp_text = resp_text[4:]
            
            grounding_data = json.loads(resp_text.strip())
            
        except Exception as e:
            logger.error(f"Keyboard grounding failed: {e}")
            return f"Error grounding keyboard keys: {e}"
        
        # Extract grounded keys and ray
        grounded_keys = grounding_data.get("keys", {})
        ray_box_norm = grounding_data.get("controller_ray")
        
        if not ray_box_norm:
            return "Error: Could not find the controller ray in the image."
        
        # Log what was found
        print(f"Grounded {len(grounded_keys)} keys: {list(grounded_keys.keys())}")
        print(f"Controller ray found: {ray_box_norm}")
        
        # Check which characters we're missing
        missing_chars = [c for c in unique_chars if c not in grounded_keys]
        if missing_chars:
            print(f"Warning: Could not find keys for: {missing_chars}")
        
        # Convert normalized boxes to pixel coordinates [x, y, w, h]
        def norm_to_pixel_box(norm_box):
            ymin, xmin, ymax, xmax = norm_box
            # Handle 0-1000 scale
            if any(c > 1.0 for c in norm_box):
                ymin, xmin, ymax, xmax = [c / 1000.0 for c in norm_box]
            box_x = xmin * w
            box_y = ymin * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h
            return [box_x, box_y, box_w, box_h]
        
        key_boxes = {k: norm_to_pixel_box(v) for k, v in grounded_keys.items()}
        ray_box = norm_to_pixel_box(ray_box_norm)
        
        # 4. Type Each Character
        typed_chars = []
        failed_chars = []
        
        for char_idx, char in enumerate(text.lower()):
            print(f"\n=== Typing character '{char}' ({char_idx + 1}/{len(text)}) ===")
            
            # Check if we have this key grounded
            if char not in key_boxes:
                print(f"Skipping '{char}' - key not found on keyboard.")
                failed_chars.append(char)
                continue
            
            target_box = key_boxes[char]
            current_ray_box = ray_box.copy()
            
            # PID control loop to align to this key
            converged = False
            final_dist = 0
            
            for i in range(MAX_ITER_PER_CHAR):
                # Check for stop signal
                if _agent_ref and _agent_ref.stop_execution.is_set():
                    print(f"Typing Stopped by User.")
                    return "Typing Stopped by User."

                # A. Capture New Image
                res = _executor.call("inspect_surroundings")
                if isinstance(res, str) and res.startswith("Error"):
                    print("Capture failed in typing loop.")
                    break
                
                try:
                    data = json.loads(res).get("data")
                    img_bytes_loop = base64.b64decode(data)
                    pil_img_loop = Image.open(io.BytesIO(img_bytes_loop)).convert("RGB")
                    img_cv = cv2.cvtColor(np.array(pil_img_loop), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Image parse error: {e}")
                    break
                
                # B. SAM Tracking for ray and target key
                inference_state = _tracker.processor.set_image(pil_img_loop)
                
                points = {}
                boxes_to_track = {"ray": current_ray_box, "key": target_box}
                updated_boxes = {}
                masks_for_viz = {}  # Store masks for visualization
                
                for key, box in boxes_to_track.items():
                    box_x, box_y, box_w, box_h = box
                    
                    # Format box for SAM
                    box_input_xywh = _tracker.torch.tensor([box_x, box_y, box_w, box_h]).view(-1, 4)
                    box_input_cxcywh = _tracker.box_xywh_to_cxcywh(box_input_xywh)
                    norm_box_cxcywh = _tracker.normalize_bbox(box_input_cxcywh, w, h).flatten().tolist()
                    
                    _tracker.processor.reset_all_prompts(inference_state)
                    # Set text prompt to help SAM differentiate between similar objects
                    text_desc = "VR controller ray" if key == "ray" else f"keyboard key {char}"
                    inference_state = _tracker.processor.set_text_prompt(text_desc, inference_state)
                    inference_state = _tracker.processor.add_geometric_prompt(
                        state=inference_state, box=norm_box_cxcywh, label=True
                    )
                    
                    # Get Mask
                    mask = None
                    if "masks" in inference_state and inference_state["masks"] is not None:
                        m = inference_state["masks"].detach().cpu().numpy() > 0.5
                        if m.ndim == 4: mask = m[0, 0]
                        elif m.ndim == 3: mask = m[0]
                    
                    if mask is None:
                        print(f"[{key}] Lost tracking (no mask).")
                        continue
                    
                    # Store mask for visualization
                    masks_for_viz[key] = mask
                    
                    # Update Box & Extract Point
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    
                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        updated_boxes[key] = [cmin, rmin, cmax - cmin, rmax - rmin]
                        
                        if key == "ray":
                            # Ray Tip = Min Y (topmost point)
                            ys, xs = np.where(mask)
                            if len(ys) > 0:
                                idx = np.argmin(ys)
                                points[key] = (xs[idx], ys[idx])
                                cv2.circle(img_cv, (xs[idx], ys[idx]), 5, (0, 0, 255), -1)
                        elif key == "key":
                            # Key Centroid
                            M = cv2.moments(mask.astype(np.uint8))
                            if M["m00"] != 0:
                                cx = int(M["m10"]/M["m00"])
                                cy = int(M["m01"]/M["m00"])
                                points[key] = (cx, cy)
                                cv2.circle(img_cv, (cx, cy), 5, (0, 255, 0), -1)
                
                # Apply colored mask overlays for visualization
                for key, mask in masks_for_viz.items():
                    if key == "ray":
                        # Blue overlay for ray (BGR: 255, 100, 0)
                        color = np.array([255, 100, 0], dtype=np.uint8)
                    elif key == "key":
                        # Green overlay for key (BGR: 0, 255, 100)
                        color = np.array([0, 255, 100], dtype=np.uint8)
                    else:
                        color = np.array([128, 128, 128], dtype=np.uint8)
                    
                    # Apply semi-transparent overlay
                    overlay = img_cv.copy()
                    overlay[mask] = color
                    cv2.addWeighted(overlay, 0.35, img_cv, 0.65, 0, img_cv)
                
                # Update boxes for next iteration
                if "ray" in updated_boxes:
                    current_ray_box = updated_boxes["ray"]
                if "key" in updated_boxes:
                    target_box = updated_boxes["key"]
                
                # C. PID Control
                if "ray" in points and "key" in points:
                    rx, ry = points["ray"]
                    kx, ky = points["key"]
                    
                    dx = kx - rx
                    dy = ky - ry
                    dist = math.sqrt(dx*dx + dy*dy)
                    final_dist = dist
                    
                    # Draw debug visualization
                    cv2.line(img_cv, (rx, ry), (kx, ky), (0, 255, 255), 2)
                    cv2.putText(img_cv, f"'{char}' err:{dist:.1f}px", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save debug image
                    timestamp = datetime.now().strftime("%H%M%S")
                    debug_dir = LOG_DIR / "typing"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    debug_path = debug_dir / f"type_{char}_{timestamp}_iter_{i}.jpg"
                    cv2.imwrite(str(debug_path), img_cv)
                    
                    if dist < TOLERANCE_PX:
                        print(f"Aligned to '{char}' (Error: {dist:.2f}px). Pressing trigger...")
                        converged = True
                        break
                    
                    # Apply PID correction
                    d_yaw = -Kp_YAW * dx
                    d_pitch = -Kp_PITCH * dy
                    
                    curr_yaw += d_yaw
                    curr_pitch += d_pitch
                    
                    _executor.call("rotate_device", device=controller,
                                   pitch=curr_pitch, yaw=curr_yaw, roll=curr_roll)
                    time.sleep(0.05)
                else:
                    print(f"Lost tracking during alignment to '{char}'")
                    break
            
            # Press trigger if converged
            if converged:
                _executor.call("click_button", controller=controller, button="trigger", duration=0.1)
                time.sleep(0.15)  # Wait for key press to register
                typed_chars.append(char)
                print(f"Typed '{char}' successfully!")
            else:
                print(f"Failed to align to '{char}' (final error: {final_dist:.2f}px)")
                failed_chars.append(char)
        
        # 5. Summary
        result = f"Typed {len(typed_chars)}/{len(text)} characters: '{''.join(typed_chars)}'"
        if failed_chars:
            result += f". Failed: {failed_chars}"
        
        logger.info(result)
        return result

    def white_cane_describe():
        """
        Immediately captures an image and provides a description for a blind user.
        This is used when white cane mode is active and user says 'help'.
        """
        _log_action("white_cane_describe")
        if not _white_cane:
            return "Error: White cane assistant not available."
        
        if not _white_cane.active:
            return "White cane mode is not active. Say 'white cane' to activate."
        
        return _white_cane.get_immediate_help()

    def white_cane_set_goal(goal: str):
        """
        Set or update the goal for white cane navigation.
        
        Args:
            goal: Description of what the user is trying to find/reach.
        """
        _log_action("white_cane_set_goal", goal=goal)
        if not _white_cane:
            return "Error: White cane assistant not available."
        
        _white_cane.current_goal = goal
        return f"White cane goal updated: {goal}"
    
    def press_button(controller: str, button: str):
        """
        Press a button on a controller. The button stays pressed until release_button is called.
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
            button: One of: "trigger", "grip", "menu", "system", "trackpad", "a", "b"
        
        Common uses:
        - "trigger": Primary action (select, shoot, grab)
        - "grip": Secondary grab, hold objects
        - "menu": Open menu
        - "a"/"b": Context-dependent actions
        - "trackpad": Trackpad/joystick click
        """
        _log_action("press_button", controller=controller, button=button)
        return _executor.call("press_button", controller=controller, button=button)

    def release_button(controller: str, button: str):
        """
        Release a previously pressed button on a controller.
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
            button: One of: "trigger", "grip", "menu", "system", "trackpad", "a", "b"
        """
        _log_action("release_button", controller=controller, button=button)
        return _executor.call("release_button", controller=controller, button=button)

    def click_button(controller: str, button: str, duration: float = 0.1):
        """
        Click a button (press and release) on a controller.
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
            button: One of: "trigger", "grip", "menu", "system", "trackpad", "a", "b"
            duration: How long to hold the button in seconds (default 0.1)
        """
        _log_action("click_button", controller=controller, button=button, duration=duration)
        return _executor.call("click_button", controller=controller, button=button, duration=duration)

    def set_trigger(controller: str, value: float):
        """
        Set the analog trigger value (for partial pulls).
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
            value: Trigger value from 0.0 (released) to 1.0 (fully pressed)
        """
        _log_action("set_trigger", controller=controller, value=value)
        return _executor.call("set_trigger", controller=controller, value=value)

    def set_joystick(controller: str, x: float, y: float):
        """
        Set the joystick/trackpad position.
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
            x: Horizontal position from -1.0 (left) to 1.0 (right)
            y: Vertical position from -1.0 (down) to 1.0 (up)
        
        Common uses:
        - Movement: Use left controller joystick for locomotion
        - Camera: Use right controller joystick for turning
        - Menu navigation: Move through UI elements
        """
        _log_action("set_joystick", controller=controller, x=x, y=y)
        return _executor.call("set_joystick", controller=controller, x=x, y=y)

    def move_joystick_direction(controller: str, direction: str, magnitude: float = 1.0):
        """
        Move the joystick in a cardinal direction.
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
            direction: "up", "down", "left", "right", "center", "forward", "backward"
            magnitude: How far to push (0.0 to 1.0, default 1.0)
        """
        _log_action("move_joystick_direction", controller=controller, direction=direction, magnitude=magnitude)
        return _executor.call("move_joystick_direction", controller=controller, direction=direction, magnitude=magnitude)

    def perform_grab(controller: str):
        """
        Perform a grab action - press grip and trigger together.
        Common interaction pattern in VR for picking up objects.
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
        """
        _log_action("perform_grab", controller=controller)
        return _executor.call("perform_grab", controller=controller)

    def perform_release(controller: str):
        """
        Release a grabbed object - release grip and trigger.
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
        """
        _log_action("perform_release", controller=controller)
        return _executor.call("perform_release", controller=controller)

    def release_all_inputs(controller: str = "both"):
        """
        Release all buttons and reset joystick to center.
        
        Args:
            controller: "controller1", "controller2", or "both" (default)
        """
        _log_action("release_all_inputs", controller=controller)
        return _executor.call("release_all_inputs", controller=controller)

    def get_controller_state(controller: str):
        """
        Get the current input state of a controller (buttons, joystick, trigger).
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
        """
        _log_action("get_controller_state", controller=controller)
        return _executor.call("get_controller_state", controller=controller)

    def get_current_pose(device: str = "headset"):
        """
        Get the current position and rotation of a device.
        
        Args:
            device: "headset", "controller1", "controller2", or "all"
        """
        _log_action("get_current_pose", device=device)
        
        if device.lower() in ["all", "everything"]:
            results = []
            for d in ["headset", "controller1", "controller2"]:
                res = _executor.call("get_current_pose", device=d)
                results.append(f"{d}: {res}")
            return "\n".join(results)

        return _executor.call("get_current_pose", device=device)

    def reset_controller_positions():
        """
        Reset both controllers to natural positions relative to the headset.
        Controller1 (left): slightly left and in front
        Controller2 (right): slightly right and in front
        """
        _log_action("reset_controller_positions")
        return _executor.call("reset_controller_positions")

    def reset_controller_orientation():
        """
        Put controllers in front of user: Left pointing DOWN, Right pointing UP.
        """
        _log_action("reset_controller_orientation")
        
        # 1. Try Keyboard Controller (Best for locking offsets)
        if _agent_ref and hasattr(_agent_ref, 'keyboard_ctrl') and _agent_ref.keyboard_ctrl:
             _agent_ref.keyboard_ctrl.apply_reset_pose()
             return "Controllers reset (Left DOWN, Right UP) using keyboard controller."
        
        # 2. Fallback: Manual positioning
        # Left (Controller 1) -> Down
        _executor.call("position_controller_relative_to_headset", 
                       controller="controller1", forward=0.3, right=-0.2, up=-0.3)
        _executor.call("rotate_device", device="controller1", pitch=90, yaw=0, roll=0)
        
        # Right (Controller 2) -> Up
        _executor.call("position_controller_relative_to_headset", 
                       controller="controller2", forward=0.3, right=0.2, up=-0.3)
        _executor.call("rotate_device", device="controller2", pitch=45, yaw=0, roll=0)
        
        return "Controllers reset (Left DOWN, Right UP) via direct commands."

    def open_menu_sequence():
        """
        Sets rigid positions for headset and controllers and opens menu.
        """
        _log_action("open_menu_sequence")
        # Headset
        move_absolute("headset", 0.0, 1.5, 0.0)
        rotate_device("headset", -20, 0, 0)
        
        # Controller 1
        move_absolute("controller1", -0.18, 1.2, -0.4)
        rotate_device("controller1", 90, 0, -15)
        
        # Controller 2
        move_absolute("controller2", 0.16491913500649544, 1.2251347749891743, -0.23297393336220185)
        rotate_device("controller2", 48.7, 38.9, 0.0)
        
        return click_button("controller1", "menu")

    def position_controller_relative_to_headset(
        controller: str, 
        forward: float = -0.3, 
        right: float = 0.0, 
        up: float = -0.5
    ):
        """
        Position a controller relative to the headset position.
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
            forward: Distance in front of headset (negative = in front, positive = behind)
            right: Distance to the right (negative = left, positive = right)
            up: Distance up/down from headset (negative = below, positive = above)
        """
        _log_action("position_controller_relative_to_headset", 
                   controller=controller, forward=forward, right=right, up=up)
        return _executor.call("position_controller_relative_to_headset", 
                             controller=controller, forward=forward, right=right, up=up)

    def click_trackpad_direction(controller: str, direction: str, duration: float = 1.0):
        """
        Moves the joystick/trackpad to a specific direction and then clicks it.
        Useful for navigating menus or specific app interactions.
        
        Args:
            controller: "controller1" (left) or "controller2" (right)
            direction: "up", "down", "left", "right", "center"
            duration: Time to hold the click (default 1.0s)
        """
        _log_action("click_trackpad_direction", controller=controller, direction=direction, duration=duration)
        
        # 1. Move to direction
        move_joystick_direction(controller, direction, magnitude=1.0)
        time.sleep(0.1) # Wait for move
        
        # 2. Click trackpad
        return click_button(controller, "trackpad", duration=duration)

    return [
        # Movement & Orientation
        start_bridge, move_relative, move_absolute, teleport, rotate_device, get_current_pose,
        # Vision
        inspect_surroundings, locate_object, capture_video, 
        # Tracking
        track_object, track_multiple_items, visual_servo_to_object, create_tracking_video, type_text, 
        # White Cane Accessibility
        white_cane_describe, white_cane_set_goal,
        # Controller Positioning
        reset_controller_positions, reset_controller_orientation, position_controller_relative_to_headset, open_menu_sequence,
        # Button/Input Controls
        press_button, release_button, click_button, set_trigger,
        set_joystick, move_joystick_direction, click_trackpad_direction,
        perform_grab, perform_release, release_all_inputs,
        get_controller_state,
        # Utility
        finish_task, get_connection_status, kill_address
    ]
# ============================================================================
# AGENT (Multi-Model Orchestrator)
# ============================================================================

class GeminiAgent:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key: raise ValueError("GEMINI_API_KEY not set")
        
        # Clients for different models
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
        self.logger = get_logger()
        
        # Initialize Core Components
        self.executor = DirectMCPExecutor()
        self.grounder = VisualGrounder(self.client, LOG_DIR)
        self.planner = ActionPlanner(self.client)
        self.verifier = Verifier(self.client)
        self.describer = Describer(self.client)
        self.white_cane = WhiteCaneAssistant(self.client, self.executor, LOG_DIR)
        
        self.chat_history = [] # Store conversation
        
        if ObjectTracker:
            self.tracker = ObjectTracker(LOG_DIR)
        else:
            self.tracker = None
        
        # Threading control
        self.stop_execution = threading.Event()
        
        # Tools (for execution phase)
        self.tools = _get_tools(self.executor, self.grounder, self.tracker, self.white_cane, self.describer, self)
        # Create a mapping for manual execution from plan
        self.tool_map = {t.__name__: t for t in self.tools}
        self.tool_map["describe_view"] = self._describe_view_tool
        self.tool_map["verify_action"] = self._verify_action_tool
        
        # Start bridge immediately
        self.executor.call("start_vr_bridge")

        # Initialize keyboard controller (uses termios — works in WSL/Linux)
        try:
            from keyboard_controller import KeyboardVRController
            self.keyboard_ctrl = KeyboardVRController(self.executor.module)
            print("Keyboard VR control available (Default: Trackpad Mode). Type ` (backtick) at the prompt to toggle.")
        except ImportError:
            self.keyboard_ctrl = None
            
        # Voice Menu State
        self.menu_state = VoiceMenuState.IDLE
        self.pending_action = None # { "tool": str, "args": dict, "description": str }

    def trigger_main_menu(self):
        """Activates the main menu."""
        self.menu_state = VoiceMenuState.MAIN_MENU
        self.white_cane.audio.speak("Main menu. Say: Navigate, Describe, Identify, Repeat, or Help.")

    def handle_voice_input(self, user_input: str):
        """
        Central handler for voice commands.
        Routes based on current menu state.
        """
        cmd = user_input.lower().strip()
        
        # 1. Global Commands
        if cmd in ["repeat", "say that again", "what did you say"]:
            self.white_cane.audio.repeat_last()
            return
        
        if cmd in ["stop", "exit", "quit", "cancel"]:
            if self.menu_state != VoiceMenuState.IDLE:
                self.menu_state = VoiceMenuState.IDLE
                self.white_cane.audio.speak("Menu closed.")
                return
            # If idle, let it fall through to normal stop logic or do nothing
        
        if cmd == "menu":
            self.trigger_main_menu()
            return

        if cmd == "help":
            # Context-aware help
            if self.menu_state == VoiceMenuState.MAIN_MENU:
                self.white_cane.audio.speak("You are in the main menu. You can ask me to navigate, describe surroundings, or identify objects.")
            elif self.menu_state == VoiceMenuState.WHITE_CANE_MENU:
                self.white_cane.audio.speak("White cane mode. You can update your goal, ask for a description, or say stop to exit.")
            elif self.menu_state == VoiceMenuState.CONFIRMATION:
                self.white_cane.audio.speak(f"I need you to confirm if you want to {self.pending_action['description']}. Say confirm or cancel.")
            else:
                 self.white_cane.audio.speak("I am ready. Say menu for options, or just tell me what to do.")
            return

        if cmd == "options":
            # List available commands
            if self.menu_state == VoiceMenuState.MAIN_MENU:
                self.white_cane.audio.speak("Options: Navigate, Describe, Identify, Repeat, Help.")
            elif self.menu_state == VoiceMenuState.WHITE_CANE_MENU:
                self.white_cane.audio.speak("Options: Goal, Help, Stop, Disable.")
            elif self.menu_state == VoiceMenuState.CONFIRMATION:
                self.white_cane.audio.speak("Options: Confirm, Cancel.")
            else:
                self.white_cane.audio.speak("Options: Menu, White Cane, Stop, Help.")
            return
            
        if cmd == "tutorial":
             self.white_cane.audio.speak("I am your VR assistant. You can give me commands like 'find the keys' or 'describe the room'. Say 'menu' to see structured options. If you get lost, say 'help'.")
             return

        # 2. State-Based Routing
        if self.menu_state == VoiceMenuState.MAIN_MENU:
            self.handle_main_menu(cmd)
        elif self.menu_state == VoiceMenuState.WHITE_CANE_MENU:
            self.handle_white_cane_menu(cmd)
        elif self.menu_state == VoiceMenuState.CONFIRMATION:
            self.handle_confirmation(cmd)
        else:
            # IDLE state - Normal Agent Execution
            # Check for White Cane activation specifically here or in main loop?
            # The main loop currently handles "white cane" checks. 
            # We can leave IDLE to return False/None and let main loop handle it, 
            # OR we can return the command to be executed.
            return cmd # Return to main loop for execution processing

    def handle_main_menu(self, cmd: str):
        """Handle commands within the Main Menu."""
        if "navigate" in cmd:
            self.menu_state = VoiceMenuState.IDLE # Exit menu to execute
            self.white_cane.audio.speak("Navigation. Where do you want to go?")
            # We could enter a sub-state, but simple prompt is often enough. 
            # For now, let's just prompt and rely on next input.
            # But wait, if we return, the main loop process it. 
            # If we want to capture the NEXT input as the destination, we might need a state.
            # Simpler: Just guide them.
            return 
            
        elif "describe" in cmd:
            self.menu_state = VoiceMenuState.IDLE
            self.white_cane.audio.speak("Describing surroundings...")
            self._describe_view_tool("What do you see?")
            return

        elif "identify" in cmd:
            self.menu_state = VoiceMenuState.IDLE
            self.white_cane.audio.speak("What object should I look for?")
            return

        else:
            # Fallback to normal planner if it sounds like a command
            self.menu_state = VoiceMenuState.IDLE
            return cmd

    def handle_white_cane_menu(self, cmd: str):
        """Handle commands within White Cane Menu."""
        # This might be redundant if White Cane logic is separate, 
        # but good for unification.
        if "goal" in cmd:
            new_goal = cmd.replace("goal", "").strip()
            if new_goal:
                self.white_cane.current_goal = new_goal
                self.white_cane.audio.speak(f"Goal updated to: {new_goal}")
            else:
                self.white_cane.audio.speak("What is your new goal?")
        elif "stop" in cmd or "disable" in cmd:
            self.white_cane.deactivate()
            self.menu_state = VoiceMenuState.IDLE
            self.white_cane.audio.speak("White cane mode deactivated.")

    def handle_confirmation(self, cmd: str):
        """Handle confirmation for dangerous/long actions."""
        if "confirm" in cmd or "yes" in cmd or "do it" in cmd:
            if self.pending_action:
                action = self.pending_action
                self.pending_action = None
                self.menu_state = VoiceMenuState.IDLE
                self.white_cane.audio.speak("Confirmed. Executing.")
                # Execute the pending tool
                # We need to find the specific tool function and call it
                # This logic depends on how we stored it.
                # If we stored tool name and args:
                tool_name = action["tool"]
                args = action["args"]
                func = self.tool_map.get(tool_name)
                if func:
                     try:
                        res = func(**args)
                        # We should log/speak result here or return it?
                        # Since this is async/inside a handler, we should probably output directly.
                        if isinstance(res, str) and len(res) < 100:
                             self.white_cane.audio.speak(res)
                        else:
                             self.white_cane.audio.speak("Action completed.")
                     except Exception as e:
                         self.white_cane.audio.speak("Error executing action.")
                         print(e)
            else:
                 self.menu_state = VoiceMenuState.IDLE
                 self.white_cane.audio.speak("No pending action.")
        
        elif "cancel" in cmd or "no" in cmd:
            self.pending_action = None
            self.menu_state = VoiceMenuState.IDLE
            self.white_cane.audio.speak("Action cancelled.")
        else:
            self.white_cane.audio.speak("Please say confirm or cancel.")
        
    def _describe_view_tool(self, question: str):
        """Tool wrapper for description model."""
        print("Capturing image for description...")
        res = self.executor.call("inspect_surroundings")
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            return self.describer.describe(img_bytes, question)
        except Exception as e:
            return f"Description failed: {e}"

    def _verify_action_tool(self, action_description: str):
        """Tool wrapper for verification model."""
        print("Capturing image for verification...")
        res = self.executor.call("inspect_surroundings")
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            return self.verifier.verify(img_bytes, action_description)
        except Exception as e:
            return f"Verification failed: {e}"

    def _get_spoken_action_name(self, tool: str, args: dict) -> str:
        """Helper to get a natural language description of the action."""
        if tool == "locate_object":
            return f"Locating {args.get('object_description', 'object')}"
        elif tool == "visual_servo_to_object":
            return f"Aligning with {args.get('object_description', 'target')}"
        elif tool == "type_text":
            return f"Typing {args.get('text', 'text')}"
        elif tool == "click_button":
            return f"Clicking {args.get('button', 'button')}"
        elif tool == "press_button":
            return f"Pressing {args.get('button', 'button')}"
        elif tool == "release_button":
            return f"Releasing {args.get('button', 'button')}"
        elif tool == "move_joystick_direction":
            return f"Moving {args.get('direction', 'direction')}"
        elif tool == "inspect_surroundings":
            return "Inspecting surroundings"
        elif tool == "describe_view":
            return "Describing view"
        elif tool == "verify_action":
            return "Verifying action"
        elif tool == "track_object":
            return f"Tracking {args.get('object_description', 'object')}"
        elif tool == "track_multiple_items":
            return f"Tracking multiple items"
        elif tool == "white_cane_describe":
            return "Describing scene"
        elif tool == "white_cane_set_goal":
            return f"Setting goal to {args.get('goal', 'unknown')}"
        else:
            return f"Running {tool.replace('_', ' ')}"


    def run_task(self, user_input: str):
        """Runs the agent task in a separate thread (target for threading)."""
        self.stop_execution.clear()
        
        self.logger.info(f"User: {user_input}")
        self.chat_history.append({"role": "user", "content": user_input})
        print(f"\nAgent (Planner) is thinking about: '{user_input}'...")
        
        # 1. PLANNING PHASE (Gemini 3 Flash)
        if self.stop_execution.is_set(): return
        
        plan = self.planner.create_plan(user_input)
        
        if not plan:
            print("Failed to generate a plan.")
            return

        print(f"\nGenerated Plan ({len(plan)} steps):")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step.tool}: {step.description}")

        # 2. EXECUTION PHASE
        print("\nExecuting Plan...")
        for step in plan:
            # Check stop before each step
            if self.stop_execution.is_set():
                print("\n[Execution Stopped by User]")
                self.chat_history.append({"role": "agent", "content": "Execution stopped by user."})
                return

            print(f"\n>> Step: {step.tool}({step.args})")
            func = self.tool_map.get(step.tool)
            
            if func:
                try:
                    # Speak Action
                    action_desc = self._get_spoken_action_name(step.tool, step.args)
                    print(f"Speaking: {action_desc}...")
                    self.white_cane.audio.speak(action_desc)
                    
                    # Execute tool
                    result = func(**step.args)
                    print(f"Result: {str(result)[:200]}...")
                    self.chat_history.append({"role": "agent", "content": f"Executed {step.tool}: {result}"})
                    self.logger.info(f"Step '{step.tool}' Result: {result}")
                    
                    # Speak Result
                    if isinstance(result, str):
                        if result.lower().startswith("error") or "fail" in result.lower():
                             self.white_cane.audio.speak("Action failed.")
                        else:
                             # For very short results, maybe speak them? For now, just "Done" or "Success" to be succinct.
                             # If it's a description/verification, we might want to speak the result if it's short.
                             if step.tool in ["describe_view", "verify_action", "locate_object"]:
                                 # Speak the actual result for these information tools if short enough
                                 if len(result) < 100:
                                     self.white_cane.audio.speak(result)
                                 else:
                                     self.white_cane.audio.speak("Done. Content is long.")
                             else:
                                 self.white_cane.audio.speak("Success.")
                    else:
                        self.white_cane.audio.speak("Success.")

                except Exception as e:
                    print(f"Execution Error: {e}")
                    self.logger.error(f"Execution Error in {step.tool}: {e}")
                    self.white_cane.audio.speak("Error executing action.")
                    break
            else:
                print(f"Error: Unknown tool '{step.tool}'")
                self.white_cane.audio.speak(f"Unknown tool {step.tool}")



    def print_status(self):
        """Print current agent status."""
        print(f"\n--- Status (v2 Multi-Model) ---")
        print(f"Planner: {MODEL_PLANNER}")
        print(f"Grounding: {MODEL_GROUNDING}")
        print(f"Verification: {MODEL_VERIFICATION}")
        print(f"Description: {MODEL_DESCRIPTION}")
        print(f"Log Dir: {LOG_DIR}")        
        try:
            status = self.executor.call("get_connection_status")
            print(f"VR Bridge: {status}")
        except Exception as e:
            print(f"VR Bridge: Error getting status ({e})")
        print("--------------")



    def handle_direct_command(self, user_input: str):
        """
        Parses and executes a direct command in the format ((function arg1 arg2 ...))
        Arguments are automatically converted to int/float/bool/None if possible.
        """
        try:
            # Strip (( and ))
            content = user_input[2:-2].strip()
            if not content:
                print("Empty direct command.")
                return

            # Parse with shlex (handles quoted strings)
            parts = shlex.split(content)
            func_name = parts[0]
            raw_args = parts[1:]
            
            # Convert args
            args = []
            for arg in raw_args:
                if arg.lower() == 'true':
                    args.append(True)
                elif arg.lower() == 'false':
                    args.append(False)
                elif arg.lower() == 'none':
                    args.append(None)
                else:
                    try:
                        if '.' in arg:
                            args.append(float(arg))
                        else:
                            args.append(int(arg))
                    except ValueError:
                        args.append(arg) # Keep as string
            
            print(f"Direct Execution: {func_name}({args})")
            self.logger.info(f"Direct Execution: {func_name}({args})")

            # 1. Check Agent Tools (Wrapped functions)
            # self.tools is a list of callables
            tool_func = next((t for t in self.tools if t.__name__ == func_name), None)
            
            if tool_func:
                # Introspect to map args if needed, or just pass *args
                # Since our tools are simple python functions, *args usually works 
                # if the user provided them in order.
                import inspect
                sig = inspect.signature(tool_func)
                
                # Simple binding attempt
                try:
                    bound_args = sig.bind(*args)
                    bound_args.apply_defaults()
                    res = tool_func(*bound_args.args, **bound_args.kwargs)
                    print(f"Result: {res}")
                    self.logger.info(f"Result: {res}")
                    return
                except TypeError as e:
                    print(f"Argument mismatch for tool '{func_name}': {e}")
                    return

            # 2. Check MCP Server directly (underlying functions)
            # This allows calling functions not exposed as agent tools
                if hasattr(self.executor.module, func_name):
                    func = getattr(self.executor.module, func_name)
                    try:
                        res = func(*args)
                        print(f"Result: {res}")
                        self.logger.info(f"Result: {res}")
                        return
                    except Exception as e:
                         print(f"Error executing MCP function '{func_name}': {e}")
                         return

            print(f"Error: Function '{func_name}' not found in Agent Tools or MCP Server.")

        except Exception as e:
            print(f"Failed to execute direct command: {e}")
            traceback.print_exc()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    agent = GeminiAgent()
    print("VR Agent v4 (Multi-Model) Ready.")
    print("Commands: 'white cane' to activate accessibility mode, 'quit' to exit.")
    
    # Execution Thread
    execution_thread = None

    while True:
        try:
            # If keyboard control is active, skip input() — stdin is in cbreak mode.
            # The background thread handles keys. Sleep briefly and loop back.
            if hasattr(agent, 'keyboard_ctrl') and agent.keyboard_ctrl and agent.keyboard_ctrl.active:
                time.sleep(0.1)
                continue
            
            # Non-blocking check for thread status
            if execution_thread and execution_thread.is_alive():
                 # We need to allow user to interrupt. 
                 # Since input() is blocking, we rely on the user seeing the prompt.
                 # But if the thread is printing a lot, the prompt might get buried.
                 # Python's input() might fight with print() from thread. 
                 # For a simple console app, we just accept that input() holds the prompt.
                 pass

            user_input = input("\nYou (Type 'stop' to abort): ").strip()
            if not user_input:
                cmd = ""
            else:
                cmd = user_input.lower()

            # --- KEYBOARD VR TOGGLE ---
            if cmd == '`' and hasattr(agent, 'keyboard_ctrl') and agent.keyboard_ctrl:
                agent.keyboard_ctrl.activate()
                continue
            
            # --- STOP COMMAND ---
            if cmd == 'stop':
                if execution_thread and execution_thread.is_alive():
                    print("\n[Stopping Agent Execution...]")
                    agent.stop_execution.set()
                    execution_thread.join(timeout=2.0)
                    if execution_thread.is_alive():
                        print("Warning: Agent did not stop immediately. It may finish the current step first.")
                    else:
                        print("Agent stopped.")
                else: 
                    print("Agent is not running.")
                continue

            # --- QUIT/EXIT ---
            if cmd in ['quit', 'exit']:
                # Deactivate white cane if active
                if agent.white_cane.active:
                    agent.white_cane.deactivate()
                if hasattr(agent, 'keyboard_ctrl') and agent.keyboard_ctrl:
                    agent.keyboard_ctrl.stop()
                
                # Stop any running agent task
                if execution_thread and execution_thread.is_alive():
                    agent.stop_execution.set()
                    execution_thread.join(timeout=1.0)
                break

            elif cmd == 'status':
                agent.print_status()
                continue
            
            # --- WHITE CANE COMMANDS ---
            elif cmd in ['white cane', 'whitecane', 'enable white cane']:
                print("\nWhite Cane mode activating...")
                goal = input("What would you like to find or where do you want to go? (or press Enter to explore): ").strip()
                result = agent.white_cane.activate(goal if goal else None)
                print(result)
                
                # Start the automatic loop
                agent.white_cane.start_background_loop(interval=10.0)
                
                # Auto-activate keyboard controller
                if hasattr(agent, 'keyboard_ctrl') and agent.keyboard_ctrl:
                    print("Auto-activating Keyboard Controller for navigation...")
                    
                    # Define callback for Enter key
                    def on_enter_callback():
                        print("\n[Paused Keyboard] Listening for command...")
                        # Reuse the existing voice command logic
                        voice_cmd = agent.white_cane.listen_command()
                        if voice_cmd:
                            if any(x in voice_cmd.lower() for x in ['stop', 'exit', 'disable', 'quit']):
                                print(f"Voice Command: {voice_cmd} -> Stopping White Cane")
                                agent.white_cane.deactivate()
                                # Also stop keyboard if white cane stops? Maybe not.
                                # But we need to return to break the loop or update state
                            else:
                                # Always use get_immediate_help to engage LLM and potentially update goal
                                description = agent.white_cane.get_immediate_help(voice_cmd)
                                print(f"\n[White Cane]:\n{description}\n")
                                agent.white_cane.audio.speak(description)
                        print("[Resuming Keyboard]...")

                    agent.keyboard_ctrl.on_trigger_callback = on_enter_callback
                    agent.keyboard_ctrl.activate()
                
                continue
 
            # Normal Voice Input Trigger (White Cane INACTIVE)
            elif cmd == '' and not agent.white_cane.active:
                if execution_thread and execution_thread.is_alive():
                     print("Agent is busy. Type 'stop' to interrupt.")
                     continue
                     
                voice_cmd = agent.white_cane.listen_command()
                if voice_cmd:
                    print(f"Voice Command: {voice_cmd}")
                    # Use central handler
                    res = agent.handle_voice_input(voice_cmd)
                    if res: # If it returns a string, it's a command for the planner
                        execution_thread = threading.Thread(target=agent.run_task, args=(res,), daemon=True)
                        execution_thread.start()
                continue

            # White Cane Voice Input Trigger
            elif cmd == '' and agent.white_cane.active:
                # If user presses Enter while white cane is active, trigger listening
                voice_cmd = agent.white_cane.listen_command()
                if voice_cmd:
                    print(f"Voice Command: {voice_cmd}") 
                    # Use central handler first to check for menus/globals
                    # If handle_voice_input returns None, it handled it (e.g. menu, stop)
                    # If it returns string, it means "execute this" -> which for white cane might mean "update goal" or "describe"
                    
                    # We need to temporarily set state to WHITE_CANE_MENU if it's not already?
                    # agent.handle_voice_input checks menu_state. 
                    # If state is IDLE, it returns cmd.
                    # If state is WHITE_CANE_MENU, it handles it.
                    
                    # Problem: We want to support "menu" command even here.
                    # Agent state is likely IDLE unless we are in a specific menu.
                    # But White Cane is a "mode" not just a menu state.
                    
                    # Let's route through handle_voice_input
                    res = agent.handle_voice_input(voice_cmd)
                    
                    if res:
                        # If we got a result back, it wasn't a menu command.
                        # It's likely a goal update or help request.
                        if any(x in res.lower() for x in ['stop', 'exit', 'disable', 'quit']):
                             # Redundant check? handle_voice_input might have handled 'stop' if it was global.
                             # But 'stop' in IDLE returns 'stop'.
                             print(f"Voice Command: {res} -> Stopping White Cane")
                             agent.white_cane.deactivate()
                        else:
                             # Default White Cane behavior: Treat as help/goal
                             print(f"Processing White Cane Context: {res}")
                             description = agent.white_cane.get_immediate_help(res) # Pass user input as context
                             print(f"\n[White Cane]:\n{description}\n")
                             agent.white_cane.audio.speak(description)
                continue

            # White Cane Deactivation
            elif cmd in ['disable white cane', 'stop white cane', 'exit white cane']:
                result = agent.white_cane.deactivate()
                print(result)
                continue
            
            # White Cane Help
            elif agent.white_cane.active and cmd in ['help', 'what do you see', 'describe', "what's next", 'whats next']:
                print("\nGetting immediate description...")
                description = agent.white_cane.get_immediate_help()
                print(f"\n[White Cane]:\n{description}\n")
                agent.white_cane.audio.speak(description)
                continue
            
            # White Cane Goal Update
            elif agent.white_cane.active and cmd.startswith('goal '):
                new_goal = user_input[5:].strip()
                agent.white_cane.current_goal = new_goal
                print(f"Goal updated: {new_goal}")
                continue
                
            if user_input.startswith("((") and user_input.endswith("))"):
                agent.handle_direct_command(user_input)
                continue

            # --- RUN AGENT TASK (Threaded) ---
            if execution_thread and execution_thread.is_alive():
                print("Agent is busy! Type 'stop' to interrupt current task.")
            else:
                execution_thread = threading.Thread(target=agent.run_task, args=(user_input,), daemon=True)
                execution_thread.start()

        except KeyboardInterrupt:
            # Deactivate white cane if active
            if agent.white_cane.active:
                agent.white_cane.deactivate()
            if hasattr(agent, 'keyboard_ctrl') and agent.keyboard_ctrl:
                agent.keyboard_ctrl.stop()
            # Stop any running task
            if execution_thread and execution_thread.is_alive():
                agent.stop_execution.set()
            break
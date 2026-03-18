"""
vr_agent/agent.py
-----------------
QwenAgent: the main orchestrator class.
Initialises all components, manages chat history, plans and executes tasks.
"""

import json
import re
import shlex
import base64
import inspect
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from .config import (
    LOG_DIR, MODEL_PLANNER, MODEL_GROUNDING, MODEL_VERIFICATION, MODEL_DESCRIPTION,
    VoiceMenuState,
)
from .logger import get_logger
from .executor import DirectMCPExecutor
from .grounding import VisualGrounder
from .planning import ActionPlanner
from .verification import Verifier, Describer
from .white_cane import WhiteCaneAssistant
from .tools import _get_tools

try:
    from object_tracker import ObjectTracker
except ImportError:
    ObjectTracker = None


class QwenAgent:
    """
    Top-level VR agent.
    - Initialises all sub-components.
    - Manages the plan → execute loop.
    - Handles voice menus and direct commands.
    """

    def __init__(self):
        import os

        # Gemini client
        self.client = OpenAI(
            base_url="https://zippy-sarita-flabbier.ngrok-free.dev/v1", # Replace with your URL
            api_key="sk-no-key-required",
            default_headers={"ngrok-skip-browser-warning": "true"} # Bypasses the HTML warning
        )
        self.logger = get_logger()

        # Config
        self.config_file = LOG_DIR / "agent_config.json"
        self.config = self.load_config()

        # Core components
        self.executor = DirectMCPExecutor()
        self.grounder = VisualGrounder(self.client, LOG_DIR)
        self.planner = ActionPlanner(self.client)
        self.verifier = Verifier(self.client)
        self.describer = Describer(self.client)
        self.white_cane = WhiteCaneAssistant(self.client, self.executor, LOG_DIR)

        self.chat_history: list = []

        # Object tracker (optional — requires SAM 3)
        if ObjectTracker:
            self.tracker = ObjectTracker(LOG_DIR)
        else:
            self.tracker = None

        # Threading control
        self.stop_execution = threading.Event()

        # Tools
        self.tools = _get_tools(
            self.executor, self.grounder, self.tracker,
            self.white_cane, self.describer, self
        )
        self.tool_map = {t.__name__: t for t in self.tools}
        self.tool_map["describe_view"] = self._describe_view_tool
        self.tool_map["verify_action"] = self._verify_action_tool

        # Start VR bridge
        self.executor.call("start_vr_bridge")

        # Keyboard controller (Linux / WSL)
        try:
            from keyboard_controller import KeyboardVRController
            self.keyboard_ctrl = KeyboardVRController(self.executor.module)
            print("Keyboard VR control available (Default: Trackpad Mode). "
                  "Type ` (backtick) at the prompt to toggle.")
        except ImportError:
            self.keyboard_ctrl = None

        # Camera Walking controller
        try:
            from camera_walking_controller import CameraWalkingController
            self.camera_ctrl = CameraWalkingController(self.executor.module)
            print("Camera Walking control available.")
        except ImportError:
            self.camera_ctrl = None

        # Voice menu state
        self.menu_state = VoiceMenuState.IDLE
        self.pending_action: Optional[Dict] = None

    # ── Config ────────────────────────────────────────────────────────────────

    def load_config(self) -> Dict[str, Any]:
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
        return {"startup_message": True}

    def save_config(self):
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    # ── Status ────────────────────────────────────────────────────────────────

    def print_status(self):
        print("\n--- Status (v2 Multi-Model) ---")
        print(f"Planner:      {MODEL_PLANNER}")
        print(f"Grounding:    {MODEL_GROUNDING}")
        print(f"Verification: {MODEL_VERIFICATION}")
        print(f"Description:  {MODEL_DESCRIPTION}")
        print(f"Log Dir:      {LOG_DIR}")
        try:
            status = self.executor.call("get_connection_status")
            print(f"VR Bridge:    {status}")
        except Exception as e:
            print(f"VR Bridge:    Error ({e})")
        print("--------------")

    # ── Voice menu ────────────────────────────────────────────────────────────

    def trigger_main_menu(self):
        self.menu_state = VoiceMenuState.MAIN_MENU
        self.white_cane.audio.speak(
            "Main menu. Say: Navigate, Describe, Identify, Repeat, or Help."
        )

    def handle_voice_input(self, user_input: str):
        """
        Central handler for voice commands.
        Routes based on current menu state.
        Returns the command string if it should be passed to the planner, else None.
        """
        cmd = user_input.lower().strip()

        # Global commands
        if cmd in ["repeat", "say that again", "what did you say"]:
            self.white_cane.audio.repeat_last()
            return None

        if "enable startup message" in cmd:
            self.config["startup_message"] = True
            self.save_config()
            self.white_cane.audio.speak("Startup message enabled.")
            return None

        if "disable startup message" in cmd:
            self.config["startup_message"] = False
            self.save_config()
            self.white_cane.audio.speak("Startup message disabled.")
            return None

        if cmd in ["stop", "exit", "quit", "cancel"]:
            if self.menu_state != VoiceMenuState.IDLE:
                self.menu_state = VoiceMenuState.IDLE
                self.white_cane.audio.speak("Menu closed.")
                return None

        if cmd == "menu":
            self.trigger_main_menu()
            return None

        if cmd == "help":
            if self.menu_state == VoiceMenuState.MAIN_MENU:
                self.white_cane.audio.speak(
                    "You are in the main menu. You can ask me to navigate, "
                    "describe surroundings, identify objects, or click on different objects in the scene."
                )
            elif self.menu_state == VoiceMenuState.WHITE_CANE_MENU:
                self.white_cane.audio.speak(
                    "White cane mode. You can update your goal, ask for a description, or say stop to exit."
                )
            elif self.menu_state == VoiceMenuState.CONFIRMATION:
                self.white_cane.audio.speak(
                    f"I need you to confirm if you want to {self.pending_action['description']}. "
                    "Say confirm or cancel."
                )
            else:
                self.white_cane.audio.speak("I am ready. Say menu for options, or just tell me what to do.")
            return None

        if cmd == "options":
            if self.menu_state == VoiceMenuState.MAIN_MENU:
                self.white_cane.audio.speak("Options: Navigate, Describe, Identify, Repeat, Help.")
            elif self.menu_state == VoiceMenuState.WHITE_CANE_MENU:
                self.white_cane.audio.speak("Options: Goal, Help, Stop, Disable.")
            elif self.menu_state == VoiceMenuState.CONFIRMATION:
                self.white_cane.audio.speak("Options: Confirm, Cancel.")
            else:
                self.white_cane.audio.speak("Options: Menu, White Cane, Stop, Help.")
            return None

        if cmd == "tutorial":
            self.white_cane.audio.speak(
                "I am your VR assistant. You can give me commands like 'find the keys' or "
                "'describe the room'. Say 'menu' to see structured options. If you get lost, say 'help'."
            )
            return None

        # State-based routing
        if self.menu_state == VoiceMenuState.MAIN_MENU:
            return self.handle_main_menu(cmd)
        elif self.menu_state == VoiceMenuState.WHITE_CANE_MENU:
            self.handle_white_cane_menu(cmd)
            return None
        elif self.menu_state == VoiceMenuState.CONFIRMATION:
            self.handle_confirmation(cmd)
            return None
        else:
            # IDLE — return to main loop for planner execution
            return cmd

    def handle_main_menu(self, cmd: str):
        if "navigate" in cmd:
            self.menu_state = VoiceMenuState.IDLE
            self.white_cane.audio.speak("Navigation. Where do you want to go?")
            return None
        elif "describe" in cmd:
            self.menu_state = VoiceMenuState.IDLE
            self.white_cane.audio.speak("Describing surroundings...")
            self._describe_view_tool("What do you see?")
            return None
        elif "identify" in cmd:
            self.menu_state = VoiceMenuState.IDLE
            self.white_cane.audio.speak("What object should I look for?")
            return None
        else:
            self.menu_state = VoiceMenuState.IDLE
            return cmd

    def handle_white_cane_menu(self, cmd: str):
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
        if "confirm" in cmd or "yes" in cmd or "do it" in cmd:
            if self.pending_action:
                action = self.pending_action
                self.pending_action = None
                self.menu_state = VoiceMenuState.IDLE
                self.white_cane.audio.speak("Confirmed. Executing.")
                tool_name = action["tool"]
                args = action["args"]
                func = self.tool_map.get(tool_name)
                if func:
                    try:
                        res = func(**args)
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

    # ── Tool wrappers ─────────────────────────────────────────────────────────

    def _describe_view_tool(self, question: str):
        """Tool wrapper for the description model."""
        print("Capturing image for description...")
        res = self.executor.call("inspect_surroundings")
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            return self.describer.describe(img_bytes, question)
        except Exception as e:
            return f"Description failed: {e}"

    def _verify_action_tool(self, action_description: str):
        """Tool wrapper for the verification model."""
        print("Capturing image for verification...")
        res = self.executor.call("inspect_surroundings")
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            return self.verifier.verify(img_bytes, action_description)
        except Exception as e:
            return f"Verification failed: {e}"

    def _get_spoken_action_name(self, tool: str, args: dict) -> str:
        """Convert a tool name + args into a natural-language phrase for TTS."""
        mapping = {
            "locate_object":         lambda a: f"Locating {a.get('object_description', 'object')}",
            "visual_servo_to_object":lambda a: f"Aligning with {a.get('object_description', 'target')}",
            "type_text":             lambda a: f"Typing {a.get('text', 'text')}",
            "click_button":          lambda a: f"Clicking {a.get('button', 'button')}",
            "press_button":          lambda a: f"Pressing {a.get('button', 'button')}",
            "release_button":        lambda a: f"Releasing {a.get('button', 'button')}",
            "move_joystick_direction":lambda a: f"Moving {a.get('direction', 'direction')}",
            "inspect_surroundings":  lambda a: "Inspecting surroundings",
            "describe_view":         lambda a: "Describing view",
            "verify_action":         lambda a: "Verifying action",
            "track_object":          lambda a: f"Tracking {a.get('object_description', 'object')}",
            "track_multiple_items":  lambda a: "Tracking multiple items",
            "white_cane_describe":   lambda a: "Describing scene",
            "white_cane_set_goal":   lambda a: f"Setting goal to {a.get('goal', 'unknown')}",
            "provide_help":          lambda a: "Providing help",
            "provide_tutorial":      lambda a: "Tutorial",
            "provide_options":       lambda a: "Listing options",
        }
        fn = mapping.get(tool)
        if fn:
            return fn(args)
        return f"Running {tool.replace('_', ' ')}"

    # ── Main task execution ───────────────────────────────────────────────────

    def run_task(self, user_input: str):
        """Plan → execute loop. Intended to run in a daemon thread."""
        self.stop_execution.clear()

        self.logger.info(f"User: {user_input}")
        self.chat_history.append({"role": "user", "content": user_input})
        print(f"\nAgent (Planner) is thinking about: '{user_input}'...")

        # 1. PLANNING PHASE
        if self.stop_execution.is_set():
            return
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
            if self.stop_execution.is_set():
                print("\n[Execution Stopped by User]")
                self.chat_history.append({"role": "agent", "content": "Execution stopped by user."})
                return

            print(f"\n>> Step: {step.tool}({step.args})")
            func = self.tool_map.get(step.tool)

            if func:
                try:
                    action_desc = self._get_spoken_action_name(step.tool, step.args)
                    print(f"Speaking: {action_desc}...")
                    self.white_cane.audio.speak(action_desc)

                    result = func(**step.args)
                    print(f"Result: {str(result)[:200]}...")
                    self.chat_history.append({"role": "agent", "content": f"Executed {step.tool}: {result}"})
                    self.logger.info(f"Step '{step.tool}' Result: {result}")

                    # Speak result
                    if isinstance(result, str):
                        if result.lower().startswith("error") or "fail" in result.lower():
                            self.white_cane.audio.speak("Action failed.")
                        elif step.tool in ["describe_view", "verify_action", "locate_object"]:
                            if len(result) < 500:
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

    # ── Direct command execution ──────────────────────────────────────────────

    def handle_direct_command(self, user_input: str):
        """
        Parse and execute a direct command in the format ((function arg1 arg2 ...)).
        Arguments are auto-converted to int/float/bool/None where possible.
        """
        try:
            content = user_input[2:-2].strip()
            if not content:
                print("Empty direct command.")
                return

            parts = shlex.split(content)
            func_name = parts[0]
            raw_args = parts[1:]

            args = []
            for arg in raw_args:
                if arg.lower() == "true":
                    args.append(True)
                elif arg.lower() == "false":
                    args.append(False)
                elif arg.lower() == "none":
                    args.append(None)
                else:
                    try:
                        args.append(float(arg) if "." in arg else int(arg))
                    except ValueError:
                        args.append(arg)

            print(f"Direct Execution: {func_name}({args})")
            self.logger.info(f"Direct Execution: {func_name}({args})")

            # 1. Check agent tools
            tool_func = next((t for t in self.tools if t.__name__ == func_name), None)
            if tool_func:
                sig = inspect.signature(tool_func)
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

            # 2. Check MCP server directly
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

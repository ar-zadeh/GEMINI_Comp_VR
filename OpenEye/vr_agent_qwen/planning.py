"""
vr_agent/planning.py
--------------------
ActionPlanner: uses Gemini to produce a structured action plan from a user request.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .config import MODEL_PLANNER
from .logger import get_logger


@dataclass
class ActionPlanItem:
    tool: str
    args: Dict[str, Any]
    description: str


class ActionPlanner:
    """Uses Gemini 3 Flash to create a sequential list of tool calls."""

    def __init__(self, client):
        self.client = client
        self.model_name = MODEL_PLANNER

    @staticmethod
    def _coerce_response_text(content: Any) -> str:
        """Normalize chat content into plain text for JSON parsing."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text_val = part.get("text")
                    if isinstance(text_val, str):
                        parts.append(text_val)
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _extract_json_text(raw_text: str) -> str:
        """Extract JSON from fenced markdown or return trimmed raw text."""
        text = raw_text.strip()
        if not text:
            return text

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        if text.startswith("json\n"):
            return text[5:].strip()

        return text

    def create_plan(self, user_request: str) -> List[ActionPlanItem]:
        logger = get_logger()
        logger.info(f"Planning for request: {user_request}")

        prompt = f"""
        User Request: "{user_request}"

        You are a VR Agent Planner. Create a sequential list of tools to execute. In this environment (which is called VRChat), the movements are done using press of the track pad. You need to push it all the way on the direction you want to move.

        AVAILABLE TOOLS:
        1. GROUNDING & TRACKING:
           - locate_object(object_description) -> Use for single items.
           - visual_servo_to_object(object_description, controller, ray_description) -> Use to ALIGN or POINT. Note: BE AS DESCRIPTIVE AS POSSIBLE WITH THE OBJECT DESCRIPTION. Make sure to give the side of the controller as well. For example, if the user says "align the left controller blue ray with the button that says ASMR", you should use the following arguments: object_description="button that says ASMR", controller="controller1", ray_description="left controller blue ray".
           - track_multiple_items(object_names) -> Use when specific multiple items are requested.
           - type_text(text, controller) -> Use to type on virtual keyboard.
           - inspect_surroundings() -> Take a picture.
          - explore_environment(max_stations, min_moves_before_mapping, forward_move_m, rotate_probability, rotate_step_degrees, forward_depth_safety_margin_m, forward_depth_corridor_width_ratio, forward_depth_min_close_fraction, forward_depth_relative_close_ratio, obstacle_min_height_ratio, obstacle_max_height_ratio, debug_output, export_ply, save_ply_each_move, depth_engine, foundationstereo_repo, foundationstereo_checkpoint, foundationstereo_scale, foundationstereo_valid_iters, foundationstereo_max_disp, rotate_to_frontier, avoid_visited_forward) -> Use when the user asks to explore/map the environment. It uses current-view depth plus known movement/yaw to build a 2D occupancy map; set depth_engine to "fast_foundationstereo" for side-by-side stereo captures. Keep min_moves_before_mapping at 0 for immediate mapping, keep rotate_to_frontier and avoid_visited_forward true to prefer new frontier cells instead of revisiting mapped areas, and keep rotate_probability at 0.0 unless random scanning is explicitly requested.
           - describe_view(question) -> Describe what is seen. AT MOST 3 sentences. Only the essential details. Do not describe the background unless it is relevant to the user request.
           - verify_action(action_description) -> Check if action succeeded.
           - provide_help() -> Provide context-aware help (Use this for 'help', 'help me', 'I need help', etc.).
           - provide_tutorial() -> Provide a tutorial/introduction.
           - provide_options() -> List available options/commands.

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
            args: Dict[str, Any] = Field(
                description='The arguments for the tool as a dictionary (e.g. \'{"arg1": "value"}\').'
            )
            description: str = Field(description="Description of why this step is being taken.")

        class PlanResponse(BaseModel):
            plan: List[PlanItem] = Field(description="Sequential list of actions.")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                extra_body={
                   "chat_template_kwargs": {"enable_thinking": False}  
                },
                response_format={"type": "json_object"}
            )

            response_text = self._coerce_response_text(response.choices[0].message.content)
            response_text = self._extract_json_text(response_text)
            
            try:
                parsed = PlanResponse.model_validate_json(response_text)
            except Exception as e:
                logger.error(f"Plan validation failed: {e}. Text: {response_text}")
                return []

            plan = []
            for item in parsed.plan:
                args_dict = item.args if isinstance(item.args, dict) else {}
                plan.append(ActionPlanItem(
                    tool=item.tool,
                    args=args_dict,
                    description=item.description
                ))
            return plan

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return []

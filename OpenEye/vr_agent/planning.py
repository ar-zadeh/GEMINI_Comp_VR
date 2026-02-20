"""
vr_agent/planning.py
--------------------
ActionPlanner: uses Gemini to produce a structured action plan from a user request.
"""

import json
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

    def create_plan(self, user_request: str, chat_history: List[Dict[str, Any]] = None) -> List[ActionPlanItem]:
        logger = get_logger()
        logger.info(f"Planning for request: {user_request}")

        history_text = ""
        if chat_history:
            history_lines = []
            for msg in chat_history:
                role = str(msg.get("role", "unknown")).strip()
                content = str(msg.get("content", "")).strip()
                if content:
                    history_lines.append(f"- {role}: {content}")
            if history_lines:
                history_text = "\nConversation History (full):\n" + "\n".join(history_lines)

        prompt = f"""
        {history_text}

        User Request: "{user_request}"

        You are a VR Agent Planner. Create a sequential list of tools to execute. In this environment (which is called VRChat), the movements are done using press of the track pad. You need to push it all the way on the direction you want to move.
        Maintain continuity with earlier conversation context and previously stated goals.

        AVAILABLE TOOLS:
        1. GROUNDING & TRACKING:
           - locate_object(object_description) -> Use for single items.
           - visual_servo_to_object(object_description, controller, ray_description) -> Use to ALIGN or POINT. Note: BE AS DESCRIPTIVE AS POSSIBLE WITH THE OBJECT DESCRIPTION. Make sure to give the side of the controller as well. For example, if the user says "align the left controller blue ray with the button that says ASMR", you should use the following arguments: object_description="button that says ASMR", controller="controller1", ray_description="left controller blue ray".
           - track_multiple_items(object_names) -> Use when specific multiple items are requested.
           - type_text(text, controller) -> Use to type on virtual keyboard.
           - inspect_surroundings() -> Take a picture.
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
            args: str = Field(
                description='The arguments for the tool as a valid JSON object string (e.g. \'{"arg1": "value"}\').'
            )
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

            try:
                parsed = response.parsed
                if not parsed:
                    parsed = PlanResponse.model_validate_json(response.text)
            except Exception as e:
                logger.error(f"Plan validation failed: {e}. Text: {response.text}")
                return []

            plan = []
            for item in parsed.plan:
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

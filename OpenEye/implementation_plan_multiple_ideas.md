# VR Navigation Prompts Implementation Plan

This plan outlines the creation of 5 distinct Python scripts to evaluate different navigation strategies using the `Cosmos-Reason2-8b` model. These scripts aim to assist blind users in navigating a VR environment by providing movement direction and angles. Each script replaces the core logic of the baseline [teleport_to_point.py](file:///home/bourn23/projects/cosmos_competition/cosmos-reason2/scripts/teleport_to_point.py) with a unique prompting strategy and response parsing mechanism.

## Proposed Changes

### Configuration and Reusable Code
All scripts will share a similar underlying structure to [teleport_to_point.py](file:///home/bourn23/projects/cosmos_competition/cosmos-reason2/scripts/teleport_to_point.py), pointing to the local VLLM server endpoint (`http://localhost:8000/v1/chat/completions`) and utilizing the `nvidia/Cosmos-Reason2-8b` model.

---

### Implementation of Navigation Strategies (In Order of Simplicity)

#### [NEW] [midlevel_nav.py](file:///home/bourn23/projects/cosmos_competition/cosmos-reason2/scripts/midlevel_nav.py)
*   **Prompt**: Asks the model to act as a navigation brain and output "TURN LEFT/RIGHT X DEGREES" or "GO STRAIGHT" (turns between 5-45 degrees).
*   **Parser**: A regular expression parser to extract the direction (`LEFT`/`RIGHT`/`STRAIGHT`) and the rotation `X` in degrees.
*   **Output**: The calculated rotation angle to apply to the avatar.

#### [NEW] [safety_gate_nav.py](file:///home/bourn23/projects/cosmos_competition/cosmos-reason2/scripts/safety_gate_nav.py)
*   **Prompt 1**: "Is there any obstacle blocking the walking path in the center of this first-person view within roughly 2 meters? Answer YES or NO."
*   **Logic Flow**: If NO, output a clear straight path. If YES, proceed to Prompt 2.
*   **Prompt 2**: "The forward path is blocked. Looking at this image, estimate what percentage of open floor is on the left vs right side. Respond as: LEFT X% RIGHT Y%."
*   **Parser**: Extracts the quantitative ratio of open floor (X and Y percentage).
*   **Math Component**: Steering angle = [(left_pct - 50) / 50 * 45](file:///home/bourn23/projects/cosmos_competition/cosmos-reason2/scripts/teleport_to_point.py#12-15) degrees (capped at ±45°).

#### [NEW] [clockface_nav.py](file:///home/bourn23/projects/cosmos_competition/cosmos-reason2/scripts/clockface_nav.py)
*   **Prompt**: "Imagine a clock face overlay on this first-person view. 12 o'clock is straight ahead, 10 o'clock is left, 2 o'clock is right. Which clock direction (10, 11, 12, 1, or 2) has the clearest and longest obstacle-free walking path? Respond with only the clock number."
*   **Parser**: Extracts the discrete output numbers 10, 11, 12, 1, or 2.
*   **Mapping**: Map the clock output to rotation angles: `10 → -30°`, `11 → -15°`, `12 → 0°`, `1 → +15°`, `2 → +30°`.

#### [NEW] [freespace_nav.py](file:///home/bourn23/projects/cosmos_competition/cosmos-reason2/scripts/freespace_nav.py)
*   **Prompt**: "You are looking through the eyes of a person walking forward. Identify the walkable floor area in this image. Is the floor clear ahead for at least 3 steps? If not, is there more walkable floor space to the LEFT or RIGHT? Respond with: DIRECTION (LEFT, STRAIGHT, RIGHT) and a brief reason."
*   **Parser**: Simple string matching rules to extract the chosen direction and textual reason to guide the avatar's rotation.

#### [NEW] [three_point_nav.py](file:///home/bourn23/projects/cosmos_competition/cosmos-reason2/scripts/three_point_nav.py)
*   This is the most complex Strategy involving image manipulation and multiple queries.
*   **Image Processing 1**: Use `PIL.Image` to split the egocentric frame into 3 equal vertical slices (left, center, right).
*   **Stage 1 Prompting**: For each slice independently, query the model to output a walkable 2D floor point near the bottom-center of that slice.
*   **Image Processing 2**: Crop a region around the returned point for each slice.
*   **Stage 2 Prompting**: For each crop, ask: "Is there a clear walking path from the bottom of this image to the marked point? Rate confidence 0-10."
*   **Logic Loop**: Find the slice with the highest confidence. If the best score is `< 5`, widen the crop area and repeat Stage 2. 
*   **Output**: Return the final chosen direction (left/center/right) and its corresponding confidence score.

---

## Verification Plan

### Automated Tests
*   We will test each script locally using a sample test image (e.g., `assets/image_15_35_34.png` typically used in [teleport_to_point.py](file:///home/bourn23/projects/cosmos_competition/cosmos-reason2/scripts/teleport_to_point.py)).
*   The tests will be executed manually via the terminal after starting the local vLLM instance:
    *   Command: `mamba activate gemini_vr && cd /home/bourn23/projects/cosmos_competition/cosmos-reason2 && python scripts/midlevel_nav.py`
    *   Repeat for the other 4 scripts.
*   **Validation**: Verify that the scripts successfully query the Local VLM, correctly parse the reasoning/JSON response according to the defined rules, and print the desired rotation angles, directions, or confidence scores to the terminal.

### Manual Verification
*   The user can integrate these scripts as drop-in prompting replacements into their existing VR pipeline and observe the resulting avatar rotations.
*   The user can provide custom images containing various hallway configurations, obstacles, and open spaces to evaluate the robustness of each prompting approach directly.

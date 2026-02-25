# VR Navigation Scripts - Evaluation Suite

This directory contains 5 distinct navigation strategy implementations for helping blind users navigate VR environments using the `Cosmos-Reason2-8b` model.

## Overview

All scripts:
- Connect to local VLLM endpoint: `http://localhost:8000/v1/chat/completions`
- Use model: `nvidia/Cosmos-Reason2-8b`
- Process test image: `assets/image_15_35_34.png`
- Save visualizations to: `scripts/claude/outputs/`

## Scripts (Ordered by Complexity)

### 1. `midlevel_nav.py` - Direct Command Strategy ⭐ SIMPLEST
**Approach:** Ask the model to output a direct navigation command
**Prompt:** "Output TURN LEFT/RIGHT X DEGREES or GO STRAIGHT (5-45°)"
**Parser:** Regex extraction of direction and angle
**Output:** `output_midlevel.png` with command overlay

**Example Output:**
```
TURN LEFT 20°
```

---

### 2. `safety_gate_nav.py` - Two-Stage Gating Strategy
**Approach:** First check for obstacles, then analyze left/right open space
**Stage 1:** "Is there an obstacle within 2m? YES/NO"
**Stage 2 (if YES):** "Estimate open floor percentage - LEFT X% RIGHT Y%"
**Math:** `angle = ((left_pct - 50) / 50) * 45` capped at ±45°
**Output:** `output_safety_gate.png` with two-stage decision tree

**Example Output:**
```
Stage 1: Obstacle? YES
Stage 2: L=70% R=30%
TURN LEFT 18°
```

---

### 3. `clockface_nav.py` - Discrete Clock Mapping Strategy
**Approach:** Use clock face metaphor for spatial reasoning
**Prompt:** "Which clock direction (10, 11, 12, 1, 2) has the clearest path?"
**Mapping:**
- 10 o'clock → -30° (left)
- 11 o'clock → -15° (slight left)
- 12 o'clock → 0° (straight)
- 1 o'clock → +15° (slight right)
- 2 o'clock → +30° (right)

**Output:** `output_clockface.png` with clock visualization

**Example Output:**
```
CLOCK: 11 o'clock
Rotation: -15°
```

---

### 4. `freespace_nav.py` - Qualitative Reasoning Strategy
**Approach:** Natural language reasoning about walkable floor space
**Prompt:** "Is floor clear for 3 steps? If not, more space LEFT or RIGHT? Explain."
**Parser:** String matching for direction + reasoning extraction
**Mapping:** LEFT→-25°, STRAIGHT→0°, RIGHT→+25°
**Output:** `output_freespace.png` with direction and reasoning overlay

**Example Output:**
```
GO LEFT (-25°)
Reasoning: The forward path has furniture blocking it. The left
side shows more open carpet area suitable for walking.
```

---

### 5. `three_point_nav.py` - Multi-Stage Image Analysis ⭐ MOST COMPLEX
**Approach:** Split image into 3 slices, analyze each independently
**Stage 1:** Split image into LEFT, CENTER, RIGHT vertical slices
**Stage 2:** For each slice, query: "Rate confidence (0-10) this path is clear"
**Stage 3:** Select slice with highest confidence score
**Mapping:** LEFT→-30°, CENTER→0°, RIGHT→+30°
**Output:** `output_three_point.png` with multi-panel visualization

**Example Output:**
```
LEFT: Score: 8/10
CENTER: Score: 4/10
RIGHT: Score: 6/10
BEST PATH: LEFT (Confidence: 8/10, Angle: -30°)
```

---

## Running the Scripts

### Prerequisites
1. Start the local VLLM server with Cosmos-Reason2-8b model
2. Activate the conda environment: `mamba activate gemini_vr`
3. Ensure test image exists: `assets/image_15_35_34.png`

### Execution
```bash
# Navigate to project root
cd /home/bourn23/projects/cosmos_competition/cosmos-reason2

# Run individual scripts
python scripts/claude/midlevel_nav.py
python scripts/claude/safety_gate_nav.py
python scripts/claude/clockface_nav.py
python scripts/claude/freespace_nav.py
python scripts/claude/three_point_nav.py
```

### Run All Scripts at Once
```bash
for script in scripts/claude/*.py; do
    echo "Running $script..."
    python "$script"
    echo "---"
done
```

---

## Outputs

All visualizations are saved to `scripts/claude/outputs/`:
- `output_midlevel.png` - Command text overlay
- `output_safety_gate.png` - Two-stage decision display
- `output_clockface.png` - Clock face with direction indicator
- `output_freespace.png` - Direction with reasoning text
- `output_three_point.png` - Multi-panel slice analysis

---

## Evaluation Metrics

To compare approaches, consider:

1. **Accuracy**: Does the suggested direction avoid obstacles?
2. **Consistency**: Do repeated runs give similar results?
3. **Reasoning Quality**: Is the logic sound and explainable?
4. **Robustness**: How well does it handle edge cases?
5. **Latency**: How fast is the response (single vs multiple queries)?
6. **Granularity**: How precise are the rotation angles?

---

## Strategy Comparison

| Strategy | Queries | Complexity | Angle Range | Reasoning |
|----------|---------|------------|-------------|-----------|
| Midlevel | 1 | Low | 5-45° | Direct |
| Safety Gate | 1-2 | Medium | ±45° | Conditional |
| Clockface | 1 | Low | {-30,-15,0,+15,+30}° | Discrete |
| Freespace | 1 | Low | {-25,0,+25}° | Qualitative |
| Three-Point | 3 | High | {-30,0,+30}° | Quantitative |

---

## Customization

To test with different images, modify the `image_path` variable in each script's `main()` function:

```python
image_path = os.path.join(project_root, "assets", "your_test_image.png")
```

---

## Next Steps

1. Run all 5 scripts on multiple test images
2. Compare output visualizations
3. Integrate the best-performing strategy into your VR pipeline
4. Fine-tune prompts and parameters based on real-world performance


---
After running the following tests:
``` bash
python benchmark_nav.py --strategy freespace --mode 1 --interval 1
```

I found that the best strategy is to use the freespace and clockface strategies.

However, to better test these strategies I developed a test suite that we can use to evaluate how well we are doing in two main tasks:

1. Unobstructed exploration
2. Goal-oriented walking

``` bash
python -m agent_navigation_tests.runner --config agent_navigation_tests/configs/experiment_config.yaml
```
read more on how to setup your test suites in the `agent_navigation_tests` directory.


# Part 2
Here we'll examine how we can better incorporate VLM (cosmos reason 2) with the SLAM pipeline.
---

**Created:** 2026-02-22
**Model:** nvidia/Cosmos-Reason2-8b
**Purpose:** Evaluate navigation prompting strategies for blind VR users



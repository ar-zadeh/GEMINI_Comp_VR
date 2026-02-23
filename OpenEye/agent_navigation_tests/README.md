# Agent Navigation Testing Suite

Automated testing framework for evaluating VLM-based navigation agents.

## Quick Start

```bash
mamba activate gemini_vr
cd /path/to/OpenEye
python -m agent_navigation_tests.runner --config agent_navigation_tests/configs/experiment_config.yaml
```

### Run only one test

```bash
python -m agent_navigation_tests.runner --test exploration   # Test 1 only
python -m agent_navigation_tests.runner --test goal          # Test 2 only
```

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/experiment_config.yaml` | Global settings: VLM endpoint, strategies, max steps, stuck thresholds, goals |
| `configs/task1_prompts.yaml` | Prompt variants for **Unobstructed Exploration** |
| `configs/task2_prompts.yaml` | Prompt variants for **Goal-Oriented Walking** (uses `{goal_description}` template) |

### Key YAML Fields

- **`history_length`** — Number of past user+assistant message pairs kept in context (0 = stateless)
- **`stuck_position_threshold`** — Metres; if agent moves less than this → stuck tick
- **`goals[].target_position`** — `[x, y, z]` from `get_current_pose` at goal location
- **`goals[].goal_description`** — Text description passed to the LLM prompt

## Metrics

| Metric | What it measures |
|--------|-----------------|
| Distance traveled (XZ) | Exploration ability |
| Tortuosity | Path efficiency (path length / displacement, 1.0 = straight) |
| Stuck count | Position-based reliability (position delta < threshold) |
| Steering reversals | Smoothness (L→R or R→L direction changes) |
| Reversal rate | Reversals / total commands |
| Angular velocity variance | Lower = smoother steering |
| Goal distance | Final / min distance to target |
| Distance reduction rate | Δ(distance) per tick (negative = approaching) |
| VLM latency | P50, P95, max inference time |
| Consistency | Mean ± std across multiple runs |

## Output

Results are saved as JSON files in `results/`:
- Per-run files: `test1_<strategy>_<prompt>_run<N>_<timestamp>.json`
- Summary: `summary_<timestamp>.json`

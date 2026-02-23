#!/usr/bin/env python3
"""
agent_navigation_tests/runner.py
---------------------------------
Main test runner for the Agent Navigation Testing Suite.

Usage:
    mamba activate gemini_vr
    cd /home/bourn23/projects/GEMINI_Comp_VR/OpenEye
    python -m agent_navigation_tests.runner --config agent_navigation_tests/configs/experiment_config.yaml
"""

import os
import sys
import re
import json
import time
import base64
import argparse
import requests
import yaml
import threading
from datetime import datetime
from typing import Dict, List, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vr_agent.executor import DirectMCPExecutor
from keyboard_controller import KeyboardVRController
from agent_navigation_tests.metrics import MetricTracker, Pose, compute_consistency


# ------------------------------------------------------------------ #
#  Direction parsers (reused from benchmark_nav.py)
# ------------------------------------------------------------------ #
def parse_freespace_direction(content: str) -> Optional[float]:
    c = content.upper()
    if "STRAIGHT" in c:
        return 0.0
    elif "LEFT" in c:
        return -25.0
    elif "RIGHT" in c:
        return 25.0
    return None


def parse_clock_direction(content: str) -> Optional[float]:
    numbers = re.findall(r"\d+", content)
    mapping = {9: -45.0, 10: -30.0, 11: -15.0, 12: 0.0, 1: 15.0, 2: 30.0, 3: 45.0}
    for num_str in numbers:
        num = int(num_str)
        if num in mapping:
            return mapping[num]
    return None


PARSERS = {
    "freespace": parse_freespace_direction,
    "clockface": parse_clock_direction,
}


# ------------------------------------------------------------------ #
#  AutoWalker (Continuous walking thread)
# ------------------------------------------------------------------ #
class AutoWalker(threading.Thread):
    def __init__(self, ctrl: KeyboardVRController, walk_tick: float = 1.0):
        super().__init__(daemon=True)
        self.ctrl = ctrl
        self.walk_tick = walk_tick
        self.running = False

    def start_walking(self):
        self.running = True
        self.start()

    def stop_walking(self):
        self.running = False

    def run(self):
        while self.running:
            # We call the controller's internal _handle_char(ch) which
            # performs the movement or rotation-sync logic.
            # In 'headset' mode, 'w' moves the headset forward.
            self.ctrl._handle_char("w")
            time.sleep(self.walk_tick)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def capture_frame(executor: DirectMCPExecutor) -> Optional[bytes]:
    """Grab one frame via MCP."""
    res = executor.call("inspect_surroundings")
    try:
        data = json.loads(res).get("data")
        if data:
            return base64.b64decode(data)
    except Exception:
        pass
    return None


def get_pose(executor: DirectMCPExecutor) -> Pose:
    """Get the current headset pose via MCP."""
    raw = executor.call("get_current_pose", device="headset")
    return Pose.from_mcp_string(raw)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  Single run executor
# ------------------------------------------------------------------ #
def execute_run(
    executor: DirectMCPExecutor,
    ctrl: KeyboardVRController,
    vlm_cfg: dict,
    prompt_text: str,
    system_text: str,
    strategy: str,
    max_steps: int,
    query_interval: float,
    walk_tick: float,
    history_length: int,
    stuck_threshold: float,
    stuck_limit: int,
    goal_position: Optional[List[float]] = None,
    goal_reached_threshold: float = 0.5,
) -> Dict:
    """
    Run a single navigation test episode.
    Returns the tracker summary dict + trajectory.
    """
    tracker = MetricTracker(goal_position=goal_position)
    parse_func = PARSERS[strategy]

    # Conversation history: list of {"role": ..., "content": ...} dicts
    conversation_history: List[dict] = []

    print(f"    [Run] Starting — max_steps={max_steps}, interval={query_interval}s, "
          f"history={history_length}")

    # Start the AutoWalker background thread
    walker = AutoWalker(ctrl, walk_tick=walk_tick)
    walker.start_walking()

    try:
        for step in range(max_steps):
            # 1. (Continuous walking is now handled by walker thread)
            # We just wait for the interval between VLM queries.
            # (In the first step, we might want to wait a bit before capturing)
            time.sleep(query_interval)

            # 2. Get pose
            pose = get_pose(executor)

            # 3. Capture frame + query VLM
            frame = capture_frame(executor)
            rotation_cmd = None
            vlm_latency = None
            vlm_response = None

            if frame:
                b64 = base64.b64encode(frame).decode("utf-8")

                # Build messages with conversation history
                messages: List[dict] = []

                # System message (if provided)
                if system_text.strip():
                    messages.append({"role": "system", "content": system_text})

                # Add prior conversation history
                if history_length > 0 and conversation_history:
                    window = conversation_history[-history_length * 2:]  # pairs of user+assistant
                    messages.extend(window)

                # Current user message with image
                current_user_msg = {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
                messages.append(current_user_msg)

                payload = {
                    "model": vlm_cfg["model"],
                    "messages": messages,
                    "max_tokens": vlm_cfg.get("max_tokens", 150),
                    "temperature": vlm_cfg.get("temperature", 0.1),
                }

                t0 = time.time()
                try:
                    resp = requests.post(vlm_cfg["url"], json=payload, timeout=60)
                    resp.raise_for_status()
                    result = resp.json()
                    vlm_response = result["choices"][0]["message"]["content"].strip()
                    vlm_latency = time.time() - t0

                    # Parse and apply rotation
                    rotation_cmd = parse_func(vlm_response)
                    if rotation_cmd is not None and rotation_cmd != 0.0:
                        ctrl._rotate_headset(dyaw=rotation_cmd)

                    # Update conversation history
                    # Store a simplified user message (text only) for history
                    conversation_history.append({"role": "user", "content": prompt_text})
                    conversation_history.append({"role": "assistant", "content": vlm_response})

                except Exception as e:
                    vlm_latency = time.time() - t0
                    print(f"      [VLM Error @ step {step}]: {e}")

            # 4. Record tick
            rec = tracker.record_tick(
                tick=step,
                pose=pose,
                rotation_command=rotation_cmd,
                vlm_latency=vlm_latency,
                vlm_response=vlm_response,
            )

            # 5. Print progress
            stuck_flag = ""
            if step > 0:
                from agent_navigation_tests.metrics import xz_distance
                prev_pose = tracker.ticks[-2].pose
                if xz_distance(prev_pose, pose) < stuck_threshold:
                    stuck_flag = " [STUCK]"
            goal_info = f" | d_goal={rec.distance_to_goal:.2f}" if rec.distance_to_goal is not None else ""
            lat_info = f" | lat={vlm_latency:.2f}s" if vlm_latency else ""
            print(f"      step {step:3d}: pos=({pose.x:.2f}, {pose.z:.2f}) "
                  f"rot_cmd={rotation_cmd}{goal_info}{lat_info}{stuck_flag}")

            # 6. Early stop if goal reached
            if goal_position is not None and rec.distance_to_goal is not None:
                if rec.distance_to_goal < goal_reached_threshold:
                    print(f"      *** Goal reached at step {step}! ***")
                    break

            # 7. (VLM query interval is now at the top of the loop)
    finally:
        # Stop background walking
        walker.stop_walking()
        walker.join(timeout=1.0)

    # Build result
    summary = tracker.summary(stuck_threshold=stuck_threshold, stuck_limit=stuck_limit)
    return {
        "summary": summary,
        "trajectory": tracker.trajectory_as_list(),
    }


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Agent Navigation Test Runner")
    parser.add_argument(
        "--config", type=str,
        default="agent_navigation_tests/configs/experiment_config.yaml",
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--test", type=str, choices=["exploration", "goal", "all"], default="all",
        help="Which test to run (default: all)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    vlm_cfg = cfg["vlm"]
    settings = cfg["settings"]
    output_cfg = cfg["output"]
    results_dir = output_cfg["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Init MCP + Keyboard controller
    print("=" * 70)
    print("  Agent Navigation Testing Suite")
    print("=" * 70)
    executor = DirectMCPExecutor()
    print("[Init] Starting VR Bridge...")
    executor.call("start_vr_bridge")
    time.sleep(2)

    ctrl = KeyboardVRController(executor.module)
    if ctrl.mode != "headset":
        ctrl.mode = "headset"

    input("\n[Ready] VR Bridge initialised. Press ENTER to start tests...\n")

    # Capture initial pose for automatic resetting
    print("[Init] Capturing starting pose for automatic teleportation...")
    initial_pose = get_pose(executor)
    print(f"       Start pose: ({initial_pose.x:.2f}, {initial_pose.z:.2f}) Yaw: {initial_pose.ry:.1f}")

    all_results: Dict = {}

    # ============================================================== #
    #  TEST 1 — Unobstructed Exploration
    # ============================================================== #
    if args.test in ("exploration", "all") and cfg.get("test1_exploration", {}).get("enabled"):
        print("\n" + "=" * 70)
        print("  TEST 1: Unobstructed Exploration")
        print("=" * 70)

        prompt_cfg = load_yaml(cfg["test1_exploration"]["prompt_file"])
        prompts = prompt_cfg["prompts"]

        test1_results: Dict = {}

        for prompt_name, prompt_entry in prompts.items():
            strategy = prompt_entry["strategy"]
            if strategy not in settings["strategies"]:
                continue

            user_text = prompt_entry["user"]
            sys_text = prompt_entry.get("system", "")

            print(f"\n  ── Strategy: {strategy} | Prompt: {prompt_name} ──")
            run_summaries = []

            for run_idx in range(settings["num_runs"]):
                print(f"\n  ▸ Run {run_idx + 1}/{settings['num_runs']}")

                # Teleport to start
                print(f"    [Run] Teleporting agent to start position...")
                executor.call("teleport", device="headset", x=initial_pose.x, y=initial_pose.y, z=initial_pose.z)
                executor.call("rotate_device", device="headset",
                              pitch=initial_pose.rx, yaw=initial_pose.ry, roll=initial_pose.rz)
                time.sleep(1.0)  # Give time for bridge to sync

                result = execute_run(
                    executor=executor,
                    ctrl=ctrl,
                    vlm_cfg=vlm_cfg,
                    prompt_text=user_text,
                    system_text=sys_text,
                    strategy=strategy,
                    max_steps=settings["max_steps"],
                    query_interval=settings["query_interval"],
                    walk_tick=settings["walk_tick"],
                    history_length=settings["history_length"],
                    stuck_threshold=settings["stuck_position_threshold"],
                    stuck_limit=settings["stuck_consecutive_limit"],
                )
                run_summaries.append(result["summary"])

                # Save per-run result
                run_file = os.path.join(
                    results_dir,
                    f"test1_{strategy}_{prompt_name}_run{run_idx}_{timestamp}.json",
                )
                with open(run_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"    Saved → {run_file}")

            # Consistency across runs
            consistency = compute_consistency(run_summaries)
            test1_results[f"{strategy}_{prompt_name}"] = {
                "runs": run_summaries,
                "consistency": consistency,
            }

        all_results["test1_exploration"] = test1_results

    # ============================================================== #
    #  TEST 2 — Goal-Oriented Walking
    # ============================================================== #
    if args.test in ("goal", "all") and cfg.get("test2_goal_walking", {}).get("enabled"):
        print("\n" + "=" * 70)
        print("  TEST 2: Goal-Oriented Walking")
        print("=" * 70)

        prompt_cfg = load_yaml(cfg["test2_goal_walking"]["prompt_file"])
        prompts = prompt_cfg["prompts"]
        goals = cfg["test2_goal_walking"]["goals"]

        test2_results: Dict = {}

        for goal_entry in goals:
            goal_name = goal_entry["name"]
            target_pos = goal_entry["target_position"]
            goal_desc = goal_entry["goal_description"]
            goal_thresh = goal_entry.get("goal_reached_threshold", 0.5)

            print(f"\n  ── Goal: {goal_name} ──")
            print(f"    Target: {target_pos}")
            print(f"    Description: {goal_desc}")

            for prompt_name, prompt_entry in prompts.items():
                strategy = prompt_entry["strategy"]
                if strategy not in settings["strategies"]:
                    continue

                # Substitute {goal_description} into the prompt
                user_text = prompt_entry["user"].replace("{goal_description}", goal_desc)
                sys_text = prompt_entry.get("system", "")

                print(f"\n    ── Strategy: {strategy} | Prompt: {prompt_name} ──")
                run_summaries = []

                for run_idx in range(settings["num_runs"]):
                    print(f"\n    ▸ Run {run_idx + 1}/{settings['num_runs']}")

                    # Teleport to start
                    print(f"      [Run] Teleporting agent to start position for goal '{goal_name}'...")
                    executor.call("teleport", device="headset", x=initial_pose.x, y=initial_pose.y, z=initial_pose.z)
                    executor.call("rotate_device", device="headset",
                                  pitch=initial_pose.rx, yaw=initial_pose.ry, roll=initial_pose.rz)
                    time.sleep(1.0)  # Give time for bridge to sync

                    result = execute_run(
                        executor=executor,
                        ctrl=ctrl,
                        vlm_cfg=vlm_cfg,
                        prompt_text=user_text,
                        system_text=sys_text,
                        strategy=strategy,
                        max_steps=settings["max_steps"],
                        query_interval=settings["query_interval"],
                        walk_tick=settings["walk_tick"],
                        history_length=settings["history_length"],
                        stuck_threshold=settings["stuck_position_threshold"],
                        stuck_limit=settings["stuck_consecutive_limit"],
                        goal_position=target_pos,
                        goal_reached_threshold=goal_thresh,
                    )
                    run_summaries.append(result["summary"])

                    run_file = os.path.join(
                        results_dir,
                        f"test2_{goal_name}_{strategy}_{prompt_name}_run{run_idx}_{timestamp}.json",
                    )
                    with open(run_file, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"      Saved → {run_file}")

                consistency = compute_consistency(run_summaries)
                key = f"{goal_name}_{strategy}_{prompt_name}"
                test2_results[key] = {
                    "runs": run_summaries,
                    "consistency": consistency,
                }

        all_results["test2_goal_walking"] = test2_results

    # ============================================================== #
    #  Final Summary
    # ============================================================== #
    summary_file = os.path.join(results_dir, f"summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"  All tests complete. Summary → {summary_file}")
    print(f"{'=' * 70}")

    # Print a comparison table to stdout
    print("\n──── Strategy / Prompt Comparison ────")
    for test_name, test_data in all_results.items():
        print(f"\n  {test_name}")
        print(f"  {'Variant':<40s} {'Dist':>8s} {'Tort':>8s} {'Stuck':>6s} {'Rev%':>6s} {'LatP50':>8s}")
        print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*8}")
        for variant, vdata in test_data.items():
            c = vdata["consistency"]
            dist = c.get("distance_traveled_xz", {}).get("mean", 0)
            tort = c.get("tortuosity", {}).get("mean", 0)
            stuck = c.get("stuck_count", {}).get("mean", 0)
            rev = c.get("reversal_rate", {}).get("mean", 0)
            lat = c.get("latency_p50", {}).get("mean", 0)
            print(f"  {variant:<40s} {dist:>8.2f} {tort:>8.2f} {stuck:>6.1f} {rev:>6.3f} {lat:>8.2f}s")


if __name__ == "__main__":
    main()

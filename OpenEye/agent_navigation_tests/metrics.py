"""
agent_navigation_tests/metrics.py
---------------------------------
MetricTracker: collects per-tick data (pose, commands, latency) and
computes all required navigation metrics.
"""

import math
import time
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


# ------------------------------------------------------------------ #
#  Pose helper
# ------------------------------------------------------------------ #
@dataclass
class Pose:
    """Parsed pose from get_current_pose output."""
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    timestamp: float = 0.0

    @staticmethod
    def from_mcp_string(raw: str) -> "Pose":
        """
        Parse a string like:
        'headset - Position: [0.26, 1.5, -0.69], Rotation: [0.0, -14.0, 0.0]'
        """
        pos_match = re.search(r"Position:\s*\[([^\]]+)\]", raw)
        rot_match = re.search(r"Rotation:\s*\[([^\]]+)\]", raw)
        if not pos_match or not rot_match:
            raise ValueError(f"Cannot parse pose from: {raw}")
        px, py, pz = [float(v.strip()) for v in pos_match.group(1).split(",")]
        rx, ry, rz = [float(v.strip()) for v in rot_match.group(1).split(",")]
        return Pose(x=px, y=py, z=pz, rx=rx, ry=ry, rz=rz, timestamp=time.time())


def xz_distance(a: Pose, b: Pose) -> float:
    """Euclidean distance on the XZ plane (ignoring height)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.z - b.z) ** 2)


# ------------------------------------------------------------------ #
#  Tick record
# ------------------------------------------------------------------ #
@dataclass
class TickRecord:
    """One step of the benchmark loop."""
    tick: int
    pose: Pose
    rotation_command: Optional[float] = None     # degrees applied this tick
    vlm_latency: Optional[float] = None          # seconds for this VLM call
    vlm_response: Optional[str] = None           # raw LLM text
    distance_to_goal: Optional[float] = None     # only for goal-oriented test


# ------------------------------------------------------------------ #
#  MetricTracker
# ------------------------------------------------------------------ #
class MetricTracker:
    """Accumulates tick data and computes summary metrics."""

    def __init__(self, goal_position: Optional[List[float]] = None):
        self.ticks: List[TickRecord] = []
        self.goal_position = goal_position   # [x, y, z] or None

    # ---- recording ------------------------------------------------ #
    def record_tick(
        self,
        tick: int,
        pose: Pose,
        rotation_command: Optional[float] = None,
        vlm_latency: Optional[float] = None,
        vlm_response: Optional[str] = None,
    ) -> TickRecord:
        dist_to_goal = None
        if self.goal_position is not None:
            gx, gz = self.goal_position[0], self.goal_position[2]
            dist_to_goal = math.sqrt((pose.x - gx) ** 2 + (pose.z - gz) ** 2)

        rec = TickRecord(
            tick=tick,
            pose=pose,
            rotation_command=rotation_command,
            vlm_latency=vlm_latency,
            vlm_response=vlm_response,
            distance_to_goal=dist_to_goal,
        )
        self.ticks.append(rec)
        return rec

    # ---- trajectory metrics --------------------------------------- #
    def total_distance_xz(self) -> float:
        """Sum of XZ segment lengths along the trajectory."""
        dist = 0.0
        for i in range(1, len(self.ticks)):
            dist += xz_distance(self.ticks[i - 1].pose, self.ticks[i].pose)
        return dist

    def displacement_xz(self) -> float:
        """Straight-line XZ distance from first to last pose."""
        if len(self.ticks) < 2:
            return 0.0
        return xz_distance(self.ticks[0].pose, self.ticks[-1].pose)

    def tortuosity(self) -> float:
        """Path length / displacement.  1.0 = perfectly straight."""
        d = self.displacement_xz()
        if d < 1e-6:
            return float("inf")
        return self.total_distance_xz() / d

    # ---- stuck detection (position-based) ------------------------- #
    def stuck_ticks(self, threshold: float = 0.05) -> List[int]:
        """Return tick indices where the agent barely moved since the previous tick."""
        stuck = []
        for i in range(1, len(self.ticks)):
            if xz_distance(self.ticks[i - 1].pose, self.ticks[i].pose) < threshold:
                stuck.append(self.ticks[i].tick)
        return stuck

    def stuck_count(self, threshold: float = 0.05) -> int:
        return len(self.stuck_ticks(threshold))

    def consecutive_stuck_episodes(self, threshold: float = 0.05, limit: int = 3) -> int:
        """Count how many times the agent was stuck for >= `limit` consecutive ticks."""
        sticks = self.stuck_ticks(threshold)
        if not sticks:
            return 0
        episodes = 0
        run = 1
        for i in range(1, len(sticks)):
            if sticks[i] == sticks[i - 1] + 1:
                run += 1
            else:
                if run >= limit:
                    episodes += 1
                run = 1
        if run >= limit:
            episodes += 1
        return episodes

    # ---- steering metrics ----------------------------------------- #
    def _rotation_commands(self) -> List[float]:
        return [t.rotation_command for t in self.ticks if t.rotation_command is not None]

    def steering_reversals(self) -> int:
        """Count direction reversals (positive→negative or vice-versa)."""
        cmds = self._rotation_commands()
        reversals = 0
        for i in range(1, len(cmds)):
            if cmds[i] * cmds[i - 1] < 0:       # sign change
                reversals += 1
        return reversals

    def reversal_rate(self) -> float:
        """Direction changes / total commands."""
        cmds = self._rotation_commands()
        if len(cmds) <= 1:
            return 0.0
        return self.steering_reversals() / len(cmds)

    def angular_velocity_variance(self) -> float:
        """Variance of issued rotation commands (lower = smoother)."""
        cmds = self._rotation_commands()
        if len(cmds) < 2:
            return 0.0
        return float(np.var(cmds))

    # ---- latency -------------------------------------------------- #
    def _latencies(self) -> List[float]:
        return [t.vlm_latency for t in self.ticks if t.vlm_latency is not None]

    def latency_stats(self) -> Dict[str, float]:
        lats = self._latencies()
        if not lats:
            return {"p50": 0, "p95": 0, "max": 0, "mean": 0}
        arr = np.array(lats)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    # ---- goal-oriented metrics ------------------------------------ #
    def final_distance_to_goal(self) -> Optional[float]:
        if self.goal_position is None or not self.ticks:
            return None
        return self.ticks[-1].distance_to_goal

    def min_distance_to_goal(self) -> Optional[float]:
        dists = [t.distance_to_goal for t in self.ticks if t.distance_to_goal is not None]
        return min(dists) if dists else None

    def distance_reduction_rates(self) -> List[float]:
        """Δ(distance to goal) per tick.  Negative = getting closer (good)."""
        rates = []
        for i in range(1, len(self.ticks)):
            d0 = self.ticks[i - 1].distance_to_goal
            d1 = self.ticks[i].distance_to_goal
            if d0 is not None and d1 is not None:
                rates.append(d1 - d0)
        return rates

    def mean_distance_reduction_rate(self) -> Optional[float]:
        rates = self.distance_reduction_rates()
        if not rates:
            return None
        return float(np.mean(rates))

    # ---- full trajectory dump ------------------------------------- #
    def trajectory_as_list(self) -> List[Dict]:
        return [
            {
                "tick": t.tick,
                "x": t.pose.x,
                "y": t.pose.y,
                "z": t.pose.z,
                "ry": t.pose.ry,
                "rotation_cmd": t.rotation_command,
                "vlm_latency": t.vlm_latency,
                "vlm_response": t.vlm_response,
                "distance_to_goal": t.distance_to_goal,
            }
            for t in self.ticks
        ]

    # ---- summary dict --------------------------------------------- #
    def summary(self, stuck_threshold: float = 0.05, stuck_limit: int = 3) -> Dict:
        s: Dict = {
            "total_ticks": len(self.ticks),
            "distance_traveled_xz": round(self.total_distance_xz(), 4),
            "displacement_xz": round(self.displacement_xz(), 4),
            "tortuosity": round(self.tortuosity(), 4),
            "stuck_count": self.stuck_count(stuck_threshold),
            "consecutive_stuck_episodes": self.consecutive_stuck_episodes(stuck_threshold, stuck_limit),
            "steering_reversals": self.steering_reversals(),
            "reversal_rate": round(self.reversal_rate(), 4),
            "angular_velocity_variance": round(self.angular_velocity_variance(), 4),
            "latency": self.latency_stats(),
        }
        if self.goal_position is not None:
            s["final_distance_to_goal"] = round(self.final_distance_to_goal() or 0, 4)
            s["min_distance_to_goal"] = round(self.min_distance_to_goal() or 0, 4)
            s["mean_distance_reduction_rate"] = round(self.mean_distance_reduction_rate() or 0, 4)
        return s


# ------------------------------------------------------------------ #
#  Consistency helper (across multiple runs)
# ------------------------------------------------------------------ #
def compute_consistency(summaries: List[Dict]) -> Dict:
    """Given a list of per-run summary dicts, compute mean ± std for scalars."""
    if not summaries:
        return {}
    keys = [
        "distance_traveled_xz", "displacement_xz", "tortuosity",
        "stuck_count", "steering_reversals", "reversal_rate",
        "angular_velocity_variance",
    ]
    # Include goal keys if present
    if "final_distance_to_goal" in summaries[0]:
        keys += ["final_distance_to_goal", "min_distance_to_goal", "mean_distance_reduction_rate"]

    result: Dict = {}
    for k in keys:
        vals = [s[k] for s in summaries if k in s]
        if vals:
            result[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

    # Latency aggregation
    lat_keys = ["p50", "p95", "max", "mean"]
    for lk in lat_keys:
        vals = [s["latency"][lk] for s in summaries if "latency" in s and lk in s["latency"]]
        if vals:
            result[f"latency_{lk}"] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
    return result

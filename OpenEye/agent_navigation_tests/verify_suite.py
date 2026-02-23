#!/usr/bin/env python3
"""Quick verification of the metrics module and YAML configs."""

import yaml
from agent_navigation_tests.metrics import MetricTracker, Pose, compute_consistency

# Test pose parsing
raw = 'headset - Position: [0.26156315222784843, 1.5, -0.6929768648226491], Rotation: [0.0, -14.0, 0.0]'
p = Pose.from_mcp_string(raw)
print(f"Parsed pose: x={p.x:.4f}, y={p.y:.4f}, z={p.z:.4f}, ry={p.ry:.1f}")
assert abs(p.x - 0.2616) < 0.001
assert abs(p.z - (-0.6930)) < 0.001
print("✓ Pose parsing OK")

# Test metric tracker
tracker = MetricTracker(goal_position=[2.0, 1.5, -3.0])
poses = [
    Pose(0.0, 1.5, 0.0, 0, 0, 0),
    Pose(0.1, 1.5, 0.2, 0, 10, 0),
    Pose(0.1, 1.5, 0.21, 0, -5, 0),  # stuck tick
    Pose(0.5, 1.5, 1.0, 0, 15, 0),
]
for i, p2 in enumerate(poses):
    cmd = [None, 10.0, -5.0, 15.0][i]
    tracker.record_tick(i, p2, rotation_command=cmd, vlm_latency=0.5 + i * 0.1)

s = tracker.summary()
print(f"Distance: {s['distance_traveled_xz']:.4f}")
print(f"Tortuosity: {s['tortuosity']:.4f}")
print(f"Stuck count: {s['stuck_count']}")
print(f"Reversals: {s['steering_reversals']}")
print(f"Reversal rate: {s['reversal_rate']:.4f}")
print(f"Final dist to goal: {s['final_distance_to_goal']:.4f}")
print(f"Latency P50: {s['latency']['p50']:.2f}s")
assert s['stuck_count'] >= 1, 'Expected at least 1 stuck tick'
assert s['steering_reversals'] >= 1, 'Expected at least 1 reversal'
assert s['tortuosity'] >= 1.0, 'Tortuosity must be >= 1'
print("✓ All metric calculations OK")

# Test consistency
c = compute_consistency([s, s])
print(f"Consistency std for distance: {c['distance_traveled_xz']['std']}")
print("✓ Consistency computation OK")

# Test YAML loading
with open('agent_navigation_tests/configs/experiment_config.yaml') as f:
    cfg = yaml.safe_load(f)
assert 'vlm' in cfg
assert 'test2_goal_walking' in cfg
assert len(cfg['test2_goal_walking']['goals']) == 3
print(f"✓ YAML config loaded: {len(cfg['test2_goal_walking']['goals'])} goals")

# Test prompt YAML loading
with open('agent_navigation_tests/configs/task1_prompts.yaml') as f:
    t1 = yaml.safe_load(f)
assert len(t1['prompts']) >= 2
print(f"✓ Task1 prompts loaded: {len(t1['prompts'])} variants")

with open('agent_navigation_tests/configs/task2_prompts.yaml') as f:
    t2 = yaml.safe_load(f)
assert len(t2['prompts']) >= 2
# Test template substitution
sample = list(t2['prompts'].values())[0]['user']
assert '{goal_description}' in sample
filled = sample.replace('{goal_description}', 'the kitchen table')
assert 'the kitchen table' in filled
print(f"✓ Task2 prompts loaded: {len(t2['prompts'])} variants, template substitution works")

print()
print("=" * 50)
print("  ALL VERIFICATION CHECKS PASSED")
print("=" * 50)

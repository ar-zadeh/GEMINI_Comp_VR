
import math

def calculate_control_action(telemetry_data, effector_label, target_label):
    """
    Calculates the movement command based on the last frame of telemetry.
    
    Args:
        telemetry_data: List of frame dicts [{'eff': [x,y], 'tgt': [x,y]}, ...]
        effector_label: Key for the pointer/hand
        target_label: Key for the button/object
        
    Returns:
        Dict with keys: action_needed (bool), dx, dy, reason
    """
    if not telemetry_data:
        return {"action_needed": False, "reason": "No telemetry data"}
        
    # Get last frame (most recent state)
    last_frame = telemetry_data[-1]
    
    if effector_label not in last_frame or target_label not in last_frame:
        return {"action_needed": False, "reason": f"Labels not found in telemetry: {last_frame.keys()}"}
        
    eff_pos = last_frame[effector_label] # [x, y] normalized
    tgt_pos = last_frame[target_label]   # [x, y] normalized
    
    # Calculate vector Difference (Target - Effector)
    # Coordinate system:
    # X: 0 (Left) -> 1 (Right)
    # Y: 0 (Top) -> 1 (Bottom)
    
    # But VR Move Relative usually mapping depends on headset.
    # Typically:
    # Screen X+ (Right) -> VR Right (+X or +Right)
    # Screen Y+ (Down)  -> VR Down (-Y or -Up) 
    
    diff_x = tgt_pos[0] - eff_pos[0]
    diff_y = tgt_pos[1] - eff_pos[1]
    
    distance = math.sqrt(diff_x**2 + diff_y**2)
    
    # Threshold for "Close Enough" (e.g. 2% of screen width)
    if distance < 0.02:
        return {
            "action_needed": False, 
            "reason": "Aligned",
            "distance": distance
        }
        
    # Scale factor (Gain)
    # If objects are far, move fast. If close, move slow.
    # VR units are meters. Screen is 0-1.
    # Heuristic: Full screen width ~ 1 meter? Let's be conservative.
    GAIN = 0.5 
    
    move_x = diff_x * GAIN
    move_y = -(diff_y * GAIN) # Invert Y because Screen Down is VR Down/Negative
    
    return {
        "action_needed": True,
        "dx": move_x,
        "dy": move_y,
        "distance": distance,
        "reason": f"Aligning dist {distance:.3f}"
    }

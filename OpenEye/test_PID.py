import numpy as np
import time

class VirtualVREnvironment:
    """
    Simulates the VR Null Driver and the optical physics of a laser pointer.
    """
    def __init__(self, depth_meters=5.0):
        # State
        self.depth = depth_meters
        self.current_pitch = 0.0 # Vertical angle in radians
        self.current_yaw = 0.0   # Horizontal angle in radians
        
        # Target location (randomly placed on the virtual wall)
        self.target_x = np.random.uniform(-2.0, 2.0)
        self.target_y = np.random.uniform(-1.0, 1.0)
    
    def get_vision_data(self):
        """
        Simulates SAM2 returning 2D screen coordinates.
        In reality, you'd get pixels. Here we use meters on wall for simplicity.
        """
        # Projection math: pos = tan(angle) * depth
        laser_x = np.tan(self.current_yaw) * self.depth
        laser_y = np.tan(self.current_pitch) * self.depth
        
        return (laser_x, laser_y), (self.target_x, self.target_y)

    def move_driver(self, d_pitch, d_yaw):
        """
        Simulates sending a command to the Null Driver.
        """
        self.current_pitch += d_pitch
        self.current_yaw += d_yaw

class AdaptiveVisualServo:
    def __init__(self):
        # Initial guess for gain. 
        # "How much angle should I change per meter of error?"
        self.gain_x = 0.01 
        self.gain_y = 0.01
        
        # Memory to detect overshooting
        self.last_error_x = 0
        self.last_error_y = 0

    def compute_move(self, laser_pos, target_pos):
        lx, ly = laser_pos
        tx, ty = target_pos
        
        # 1. Calculate Error
        error_x = tx - lx
        error_y = ty - ly
        
        # 2. ADAPTIVE LOGIC (The "Magic" part)
        # Check X Axis Overshoot: If error sign flipped, we went too far!
        if np.sign(error_x) != np.sign(self.last_error_x) and self.last_error_x != 0:
            self.gain_x *= 0.5  # Dampen the gain aggressively
            print(f"  [!] X Overshot! Reducing gain to {self.gain_x:.4f}")
        else:
            self.gain_x *= 1.05 # Slowly be more aggressive if we haven't hit yet
            
        # Check Y Axis Overshoot
        if np.sign(error_y) != np.sign(self.last_error_y) and self.last_error_y != 0:
            self.gain_y *= 0.5
            print(f"  [!] Y Overshot! Reducing gain to {self.gain_y:.4f}")
        else:
            self.gain_y *= 1.05

        # 3. Calculate Command
        d_yaw = error_x * self.gain_x
        d_pitch = error_y * self.gain_y
        
        # Store errors for next frame
        self.last_error_x = error_x
        self.last_error_y = error_y
        
        return d_pitch, d_yaw

# --- MAIN SIMULATION LOOP ---

def run_test(scenario_name, depth):
    print(f"\n--- Starting Scenario: {scenario_name} (Depth: {depth}m) ---")
    vr = VirtualVREnvironment(depth_meters=depth)
    agent = AdaptiveVisualServo()
    
    for step in range(20): # Max 20 steps to converge
        laser, target = vr.get_vision_data()
        
        dist = np.sqrt((laser[0]-target[0])**2 + (laser[1]-target[1])**2)
        print(f"Step {step}: Dist to Target: {dist:.3f}m | Gain: {agent.gain_x:.4f}")
        
        if dist < 0.05: # Converged (within 5cm)
            print("✅ TARGET HIT!")
            break
            
        # Agent decides move
        d_pitch, d_yaw = agent.compute_move(laser, target)
        
        # Move VR Driver
        vr.move_driver(d_pitch, d_yaw)

# Run 1: Close Range (Requires HIGH angular change)
# run_test("Close Range Interaction", depth=1.0)

# Run 2: Far Range (Requires TINY angular change - High Overshoot Risk)
run_test("Far Distance Sniping", depth=50.0)
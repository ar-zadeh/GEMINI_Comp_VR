
import sys
import os

# Add the directory containing mcp_server.py to the python path
sys.path.append("/home/amir/gemini_project_VR/GEMINI_Comp_VR/OpenEye")

import mcp_server
import json

def test_trackpad_press_top():
    print("Testing 'press top of trackpad' capability...")
    
    controller = "controller1"
    
    # 1. Reset state
    mcp_server.release_all_inputs(controller)
    
    # 2. Move Joystick/Trackpad to Top (Up)
    # y=1.0 means UP
    print("Step 1: Moving joystick/trackpad to UP position (y=1.0)...")
    mcp_server.set_joystick(controller, 0.0, 1.0)
    
    # Check intermediate state
    state = mcp_server.controller_inputs[controller]
    print(f"State after move: joystickY={state['joystickY']}, trackpadTouch={state['trackpadTouch']}")
    
    assert state['joystickY'] == 1.0, "Joystick Y should be 1.0"
    assert state['trackpadTouch'] == True, "Trackpad should be touched"
    
    # 3. Press Trackpad Button
    print("Step 2: Pressing trackpad button...")
    mcp_server.press_button(controller, "trackpad")
    
    # Check final state
    state = mcp_server.controller_inputs[controller]
    print(f"State after press: trackpadClick={state['trackpadClick']}")
    
    assert state['trackpadClick'] == True, "Trackpad should be clicked"
    assert state['joystickY'] == 1.0, "Joystick Y should remain 1.0"
    
    print("\nSUCCESS: Agent is capable of pressing top direction of trackpad.")
    print("Implementation: Combine 'set_joystick(c, 0, 1)' and 'press_button(c, 'trackpad')'")

if __name__ == "__main__":
    test_trackpad_press_top()

import time
import os
import sys

# Ensure we can import from vr_agent_qwen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vr_agent_qwen.executor import DirectMCPExecutor

def test_driver_capture():
    print("========================================")
    print("Testing VR Driver Frame Capture directly")
    print("========================================")
    print("Starting DirectMCPExecutor...")
    
    executor = DirectMCPExecutor()
    
    print("\nAttempting to call 'inspect_surroundings' on the VR driver...")
    print("Make sure SteamVR is running and the OpenEye driver is active.\n")
    
    try:
        # Start VR bridge to connect to the driver
        executor.call("start_vr_bridge")
        time.sleep(1.0) # wait a moment for the connection
        
        print("Calling inspect_surroundings...")
        res = executor.call("inspect_surroundings")
        
        if isinstance(res, str):
            if res.startswith("Error"):
                print(f"\n[FAIL] Driver returned an error:\n{res}")
                print("\nPossible reasons:")
                print("1. 'Headset Window' or 'VR View' is minimized, closed, or obstructed.")
                print("2. The C++ driver BitBlt screen capture failed.")
                print("3. Check the SteamVR 'vrserver.txt' logs to see the exact CFrameCapture error.")
            else:
                try:
                    # Let's see if we got JSON bytes
                    import json
                    data = json.loads(res)
                    if data.get("type") == "frame" and "data" in data:
                        img_bytes = data["data"]
                        print(f"\n[SUCCESS] Captured frame successfully! Base64 payload length: {len(img_bytes)}")
                    else:
                        print(f"\n[WARNING] Unexpected JSON response format:\n{str(data)[:200]}")
                except Exception as e:
                     print(f"\n[WARNING] Could not parse JSON response (not an error, but malformed):\n{e}\nResponse excerpt: {res[:200]}")
        else:
             print(f"\n[WARNING] Response was not a string: {type(res)}")
             
    except Exception as e:
        print(f"\n[EXCEPTION] An unexpected error occurred in Python:\n{e}")

if __name__ == "__main__":
    test_driver_capture()

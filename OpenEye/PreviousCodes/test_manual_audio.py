
import sys
import os
from pathlib import Path

# Add the directory to sys.path
sys.path.append("/home/amir/gemini_project_VR/GEMINI_Comp_VR/OpenEye")

from gemini_vr_agent_v6 import AudioAssistant

def test_manual_recording():
    print("Initializing AudioAssistant...")
    log_dir = Path("test_logs")
    log_dir.mkdir(exist_ok=True)
    
    audio = AudioAssistant(log_dir)
    
    print("\n--- Testing Manual Recording ---")
    print("When prompted, press Enter to start, speak something, then press Enter to stop.")
    
    text = audio.listen_manual_stop()
    
    print(f"\nResult: '{text}'")

if __name__ == "__main__":
    test_manual_recording()

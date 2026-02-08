import sys
import os
from pathlib import Path
import time

# Add current directory to path
sys.path.append(os.getcwd())

print("Importing AudioAssistant from gemini_vr_agent_v6...")
try:
    from gemini_vr_agent_v6 import AudioAssistant, AUDIO_AVAILABLE
except ImportError as e:
    print(f"Failed to import: {e}")
    print("Ensure you are running this from the same directory as gemini_vr_agent_v6.py")
    sys.exit(1)

if not AUDIO_AVAILABLE:
    print("Audio is NOT available (imports failed). Check logs above.")
    sys.exit(1)

def test():
    log_dir = Path("test_logs")
    audio = AudioAssistant(log_dir)
    
    print("\n1. Testing TTS...")
    audio.speak("Testing Text to Speech. If you can hear this, it is working.")
    time.sleep(5) # Wait for speech to finish
    
    print("\n2. Testing STT (Recording for 5 seconds)...")
    print("Please say something after the countdown.")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("GO!")
    
    text = audio.listen(duration=5)
    
    print("\n--- Result ---")
    print(f"Transcription: {text}")
    
    if text:
        print("\n3. Speaking back what you said...")
        audio.speak(f"You said: {text}")
        time.sleep(5)

if __name__ == "__main__":
    test()

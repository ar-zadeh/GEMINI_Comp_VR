#!/usr/bin/env python3
import subprocess
import sys

def run_command(cmd):
    print(f"\n--- Output of '{cmd}' ---")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
    except FileNotFoundError:
        print(f"Command not found: {cmd.split()[0]}")

def main():
    print("Checking Microphone Status...")
    
    # 1. List Capture Devices
    run_command("arecord -l")
    
    # 2. Check ALSA Capture Settings (Mute/Volume)
    run_command("amixer sget Capture")
    
    # 3. Check PulseAudio Sources (if available)
    print("\n--- PulseAudio Source Status (Mute) ---")
    try:
        # grep for Mute status of sources
        result = subprocess.run("pactl list sources | grep -E 'Source|Mute|Volume'", shell=True, capture_output=True, text=True)
        print(result.stdout)
    except Exception:
        print("pactl not found or failed.")

    print("\nTo test recording and playback, run:")
    print("arecord -f cd -d 5 test_mic.wav && aplay test_mic.wav")

if __name__ == "__main__":
    main()

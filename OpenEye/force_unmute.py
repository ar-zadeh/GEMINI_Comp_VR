#!/usr/bin/env python3
"""
Force unmute the microphone.
Tries to:
1. Connect to the VR driver (requires SteamVR running) and send 'unmute' command.
2. Unmute local Linux system via amixer.
"""

import sys
import time
import os
import mcp_server

def wait_for_driver(timeout=15):
    """Start the bridge and wait for the driver to connect."""
    print("Starting VR bridge...")
    try:
        msg = mcp_server.start_vr_bridge()
        print(msg)
    except Exception as e:
        print(f"Failed to start bridge: {e}")
        # If server already running (e.g. hung process), we might still try? 
        # But usually start_vr_bridge handles that check.

    print(f"Waiting for driver connection (up to {timeout}s)...")
    print("Please ensure SteamVR is running and the driver is active.")
    for i in range(timeout):
        status = mcp_server.get_connection_status()
        if "Connected" in status:
            print(f"  {status}")
            return True
        time.sleep(1)
        if i % 5 == 0:
            print(f"  ...waiting {i}s")

    print("Warning: Driver did not connect in time. Cannot unmute via VR Driver.")
    return False

def force_unmute():
    # 1. System Level (Linux/ALSA)
    print("\n[1/2] Attempting to unmute local Linux system (ALSA)...")
    try:
        # Check if amixer exists
        if os.system("which amixer > /dev/null 2>&1") == 0:
            os.system("amixer set Capture cap > /dev/null 2>&1")
            os.system("amixer set Capture unmute > /dev/null 2>&1")
            os.system("amixer set Master unmute > /dev/null 2>&1")
            print("Executed amixer unmute commands.")
        else:
            print("'amixer' not found. Skipping ALSA unmute.")
    except Exception as e:
        print(f"Linux unmute failed: {e}")

    # 2. VR Driver Level
    print("\n[2/2] Attempting to unmute via VR Driver...")
    if wait_for_driver(timeout=10):
        try:
            print("Sending Unmute command...")
            result = mcp_server.unmute_microphone()
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error calling unmute_microphone: {e}")
    else:
        print("Skipping VR Driver unmute (no connection).")

if __name__ == "__main__":
    force_unmute()
    print("\nDone. Please check your system microphone settings manually if issues persist.")

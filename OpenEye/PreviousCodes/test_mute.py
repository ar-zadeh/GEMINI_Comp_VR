#!/usr/bin/env python3
"""
Test mute/unmute functionality via the MCP server.

Requires the VR driver to be connected (SteamVR running with the sample driver).

Usage:
    python test_mute.py
"""

import sys
import time
import os

# Add parent dir so we can import mcp_server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcp_server


def wait_for_driver(timeout=15):
    """Start the bridge and wait for the driver to connect."""
    print("Starting VR bridge...")
    mcp_server.start_vr_bridge()

    print(f"Waiting for driver connection (up to {timeout}s)...")
    for i in range(timeout):
        status = mcp_server.get_connection_status()
        if "Connected" in status:
            print(f"  {status}")
            return True
        time.sleep(1)
        print(f"  {i+1}s - {status}")

    print("ERROR: Driver did not connect in time.")
    return False


def test_get_state():
    """Test querying the current mute state."""
    print("\n--- get_mute_state ---")
    result = mcp_server.get_mute_state("microphone")
    print(f"  Microphone: {result}")
    result = mcp_server.get_mute_state("system")
    print(f"  System:     {result}")


def test_mute_unmute():
    """Test mute then unmute the microphone."""
    print("\n--- mute_microphone ---")
    result = mcp_server.mute_microphone()
    print(f"  {result}")

    time.sleep(1)

    print("\n--- get_mute_state (should be muted) ---")
    result = mcp_server.get_mute_state("microphone")
    print(f"  {result}")

    time.sleep(2)

    print("\n--- unmute_microphone ---")
    result = mcp_server.unmute_microphone()
    print(f"  {result}")

    time.sleep(1)

    print("\n--- get_mute_state (should be unmuted) ---")
    result = mcp_server.get_mute_state("microphone")
    print(f"  {result}")


def test_toggle():
    """Test toggling mute on and off."""
    print("\n--- toggle_mute (1st - should mute) ---")
    result = mcp_server.toggle_mute("microphone")
    print(f"  {result}")

    time.sleep(2)

    print("\n--- toggle_mute (2nd - should unmute) ---")
    result = mcp_server.toggle_mute("microphone")
    print(f"  {result}")


def main():
    if not wait_for_driver():
        sys.exit(1)

    test_get_state()
    test_mute_unmute()
    test_toggle()

    print("\nAll mute tests completed.")


if __name__ == "__main__":
    main()

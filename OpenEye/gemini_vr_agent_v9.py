#!/usr/bin/env python3
"""
gemini_vr_agent_v9.py
---------------------
Entry point for the VR Agent (v9 — modular package edition).
All logic lives in the vr_agent/ package; this file is the main loop only.

Usage:
    conda activate myenv
    python gemini_vr_agent_v9.py
"""

import time
import threading
import argparse
from dotenv import load_dotenv

load_dotenv()

from vr_agent.agent import GeminiAgent

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VR Agent v9")
    parser.add_argument(
        "--tracking-model",
        default="SAM3",
        choices=["SAM3", "VNS-SAM"],
        help="Select tracking backend (default: SAM3).",
    )
    args = parser.parse_args()

    agent = GeminiAgent(tracking_model=args.tracking_model)
    print("VR Agent v9 (Modular) Ready.")
    print("Commands: 'white cane' to activate accessibility mode, 'quit' to exit.")

    if agent.config.get("startup_message", True):
        agent.white_cane.audio.speak(
            "VR Agent initialized. I am your VR assistant. "
            "Press enter, talk to me and press enter again. "
            "You can give me commands like 'find the keys' or 'describe the room'. "
            "Say 'menu' to see structured options. If you get lost, say 'help'."
        )

    execution_thread = None

    while True:
        try:
            # Keyboard controller active — skip stdin
            if hasattr(agent, "keyboard_ctrl") and agent.keyboard_ctrl and agent.keyboard_ctrl.active:
                time.sleep(0.1)
                continue

            user_input = input("\nYou (Type 'stop' to abort): ").strip()
            cmd = user_input.lower() if user_input else ""

            # ── Keyboard VR toggle ────────────────────────────────────────────
            if cmd == "`" and hasattr(agent, "keyboard_ctrl") and agent.keyboard_ctrl:
                agent.keyboard_ctrl.activate()
                continue

            # ── Stop ──────────────────────────────────────────────────────────
            if cmd == "stop":
                if execution_thread and execution_thread.is_alive():
                    print("\n[Stopping Agent Execution...]")
                    agent.stop_execution.set()
                    execution_thread.join(timeout=2.0)
                    if execution_thread.is_alive():
                        print("Warning: Agent did not stop immediately.")
                    else:
                        print("Agent stopped.")
                else:
                    print("Agent is not running.")
                continue

            # ── Quit ──────────────────────────────────────────────────────────
            if cmd in ["quit", "exit"]:
                if agent.white_cane.active:
                    agent.white_cane.deactivate()
                if hasattr(agent, "keyboard_ctrl") and agent.keyboard_ctrl:
                    agent.keyboard_ctrl.stop()
                if execution_thread and execution_thread.is_alive():
                    agent.stop_execution.set()
                    execution_thread.join(timeout=1.0)
                break

            # ── Status ────────────────────────────────────────────────────────
            elif cmd == "status":
                agent.print_status()
                continue

            # ── White Cane activation ─────────────────────────────────────────
            elif cmd in ["white cane", "whitecane", "enable white cane"]:
                print("\nWhite Cane mode activating...")
                result = agent.white_cane.activate()
                print(result)

                agent.white_cane.start_background_loop(interval=10.0)

                if hasattr(agent, "keyboard_ctrl") and agent.keyboard_ctrl:
                    print("Auto-activating Keyboard Controller for navigation...")

                    def on_enter_callback():
                        print("\n[Paused Keyboard] Listening for command...")
                        voice_cmd = agent.white_cane.listen_command()
                        if voice_cmd:
                            if any(x in voice_cmd.lower() for x in ["stop", "exit", "disable", "quit"]):
                                print(f"Voice Command: {voice_cmd} -> Stopping White Cane")
                                agent.white_cane.deactivate()
                            else:
                                description = agent.white_cane.perform_360_scan(voice_cmd)
                                print(f"\n[White Cane]:\n{description}\n")
                                agent.white_cane.audio.speak(description)
                        print("[Resuming Keyboard]...")

                    agent.keyboard_ctrl.on_trigger_callback = on_enter_callback
                    agent.keyboard_ctrl.activate()
                continue

            # ── Normal voice input (White Cane INACTIVE) ──────────────────────
            elif cmd == "" and not agent.white_cane.active:
                if execution_thread and execution_thread.is_alive():
                    print("Agent is busy. Type 'stop' to interrupt.")
                    continue
                voice_cmd = agent.white_cane.listen_command()
                if voice_cmd:
                    print(f"Voice Command: {voice_cmd}")
                    res = agent.handle_voice_input(voice_cmd)
                    if res:
                        execution_thread = threading.Thread(
                            target=agent.run_task, args=(res,), daemon=True
                        )
                        execution_thread.start()
                continue

            # ── White Cane voice input ────────────────────────────────────────
            elif cmd == "" and agent.white_cane.active:
                voice_cmd = agent.white_cane.listen_command()
                if voice_cmd:
                    print(f"Voice Command: {voice_cmd}")
                    res = agent.handle_voice_input(voice_cmd)
                    if res:
                        if any(x in res.lower() for x in ["stop", "exit", "disable", "quit"]):
                            print(f"Voice Command: {res} -> Stopping White Cane")
                            agent.white_cane.deactivate()
                        else:
                            print(f"Processing White Cane Context: {res}")
                            description = agent.white_cane.perform_360_scan(res)
                            print(f"\n[White Cane]:\n{description}\n")
                            agent.white_cane.audio.speak(description)
                continue

            # ── White Cane deactivation ───────────────────────────────────────
            elif cmd in ["disable white cane", "stop white cane", "exit white cane"]:
                result = agent.white_cane.deactivate()
                print(result)
                continue

            # ── White Cane help / describe ────────────────────────────────────
            elif agent.white_cane.active and cmd in [
                "help", "what do you see", "describe", "what's next", "whats next"
            ]:
                print("\nGetting immediate description...")
                description = agent.white_cane.get_immediate_help()
                print(f"\n[White Cane]:\n{description}\n")
                agent.white_cane.audio.speak(description)
                continue

            # ── Push-to-talk (hold grip) ──────────────────────────────────────
            elif cmd == "f":
                def hold_grip_loop(stop_event):
                    while not stop_event.is_set():
                        agent.executor.call("press_button", controller="controller1", button="grip")
                        if stop_event.wait(timeout=1.0):
                            break
                    agent.executor.call("release_button", controller="controller1", button="grip")
                    print("\n[Push to Talk] Grip released.")

                ptt_stop_event = threading.Event()
                ptt_thread = threading.Thread(target=hold_grip_loop, args=(ptt_stop_event,), daemon=True)
                ptt_thread.start()
                print("\n[Push to Talk] Holding Grip (Controller 1)... Press ENTER to release.")
                input()
                ptt_stop_event.set()
                ptt_thread.join()
                continue

            # ── Direct command ((function arg1 arg2)) ─────────────────────────
            if user_input.startswith("((") and user_input.endswith("))"):
                agent.handle_direct_command(user_input)
                continue

            # ── Normal text → planner ─────────────────────────────────────────
            if execution_thread and execution_thread.is_alive():
                print("Agent is busy! Type 'stop' to interrupt current task.")
            else:
                execution_thread = threading.Thread(
                    target=agent.run_task, args=(user_input,), daemon=True
                )
                execution_thread.start()

        except KeyboardInterrupt:
            if agent.white_cane.active:
                agent.white_cane.deactivate()
            if hasattr(agent, "keyboard_ctrl") and agent.keyboard_ctrl:
                agent.keyboard_ctrl.stop()
            if execution_thread and execution_thread.is_alive():
                agent.stop_execution.set()
            break

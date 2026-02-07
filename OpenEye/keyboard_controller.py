#!/usr/bin/env python3
"""
Keyboard-based VR movement controller (WSL/Linux compatible, non-blocking).

Uses termios + select for raw terminal input — no X11/Wayland required.
Works in WSL, SSH, and any Linux terminal.

Non-blocking: activate() returns immediately, key reading happens in a
background thread. The main agent loop keeps running (white cane, etc.).

Controls (when active):
    WASD    = headset position (yaw-relative)
    Q/E     = headset up/down
    Arrows  = headset rotation (pitch/yaw)
    `       = toggle keyboard control off (return to normal input)

Controllers maintain fixed relative offsets to the headset, even when
the agent (not keyboard) moves/rotates the headset via mcp_server.

Usage:
    from keyboard_controller import KeyboardVRController
    ctrl = KeyboardVRController(mcp_module)
    ctrl.activate()   # non-blocking — returns immediately
    # ... later ...
    ctrl.deactivate()  # or press backtick
"""

import sys
import os
import math
import time
import select
import threading
from typing import Dict

# Terminal control — Unix only (WSL/Linux/macOS)
try:
    import termios
    import tty
    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False


class KeyboardVRController:
    """
    Keyboard-based VR movement controller.

    WASD = headset position (yaw-relative)
    Q/E  = headset up/down
    Arrow keys = headset rotation (pitch/yaw)
    Backtick (`) = toggle off, return to normal input.

    Controllers maintain fixed relative offsets to headset.
    Uses termios cbreak mode — works in WSL, SSH, any terminal.
    Non-blocking: activate() spawns a background thread.
    """

    def __init__(self, mcp_module, move_step: float = 0.05, rotate_step: float = 2.0):
        """
        Args:
            mcp_module: The live mcp_server module (provides current_poses, state_lock, broadcast_state).
            move_step: Meters per keypress for WASD/QE (default 0.05m = 5cm).
            rotate_step: Degrees per keypress for arrow keys (default 2.0 degrees).
        """
        self.mcp = mcp_module
        self.move_step = move_step
        self.rotate_step = rotate_step

        self.active = False
        self._thread = None
        self._old_term_settings = None

        # Modes: 'trackpad' (default) or 'headset'
        self.mode = 'trackpad' 
        self.target_controller = 'controller1' # Right controller for trackpad

        # Controller offsets relative to headset in headset-local coordinates.
        # forward: +ve = in front of headset (toward -Z at yaw=0)
        # right:   +ve = to the right of headset (+X at yaw=0)
        # up:      +ve = above headset, -ve = below
        self._controller_offsets: Dict[str, Dict[str, float]] = {
            'controller1': {
                'forward': 0.3, 'right': -0.3, 'up': -0.5,
                'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0
            },
            'controller2': {
                'forward': 0.3, 'right': 0.3, 'up': -0.5,
                'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def activate(self):
        """
        Enter keyboard VR control mode (non-blocking).

        Switches terminal to cbreak mode, spawns a background thread to
        read keys, and returns immediately. The main thread keeps running.
        Press backtick (`) to deactivate.
        """
        if not TERMIOS_AVAILABLE:
            print("[KeyboardVRController] termios not available (Windows?). Use WSL or Linux.")
            return

        if self.active:
            return

        self.active = True
        self.capture_current_offsets()

        # Suppress noisy mcp_server logging during keyboard mode
        self.mcp.suppress_logging = True

        # Register callback so controllers follow agent-initiated headset changes
        self.mcp._headset_changed_callback = self._on_headset_changed

        # Save terminal settings and enter cbreak mode
        fd = sys.stdin.fileno()
        self._old_term_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        # Spawn background thread for key reading
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()

        print("\n[Keyboard VR Control] ENABLED")
        print(f"  Current Mode: {self.mode.upper()}")
        print("  Controls:")
        print("    WASD    = Move Headset OR Trackpad (Top/Left/Down/Right)")
        print("    Q/E     = Up/Down (Headset)")
        print("    Arrows  = Rotate (Headset)")
        print("    m       = Toggle Mode (Trackpad <-> Headset)")
        print("    `       = Exit Keyboard Mode")

    def deactivate(self):
        """
        Exit keyboard VR control mode.
        Restores terminal, unregisters callback, re-enables logging.
        """
        if not self.active:
            return

        self.active = False

        # Wait for background thread to finish
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Restore terminal
        if self._old_term_settings is not None:
            try:
                fd = sys.stdin.fileno()
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_term_settings)
            except Exception:
                pass
            self._old_term_settings = None

        # Unregister callback and re-enable logging
        self.mcp._headset_changed_callback = None
        self.mcp.suppress_logging = False

        print("\n[Keyboard VR Control] DISABLED")

    def stop(self):
        """Force-stop keyboard control (e.g., on application exit)."""
        self.deactivate()

    def set_controller_offset(
        self,
        controller: str,
        forward: float = 0.3,
        right: float = 0.0,
        up: float = -0.5,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
    ):
        """
        Manually set a controller's position/rotation offset relative to the headset.

        Args:
            controller: "controller1" or "controller2"
            forward: Distance in front of headset (+ve = in front).
            right: Distance to the right (+ve = right, -ve = left).
            up: Distance above headset (+ve = above, -ve = below).
            pitch/yaw/roll: Rotation offset in degrees added to headset rotation.
        """
        if controller not in self._controller_offsets:
            print(f"[KeyboardVRController] Invalid controller: {controller}")
            return

        self._controller_offsets[controller] = {
            'forward': forward, 'right': right, 'up': up,
            'pitch': pitch, 'yaw': yaw, 'roll': roll,
        }

        self._update_controllers()
        self.mcp.broadcast_state()
        print(f"[KeyboardVRController] {controller} offset set: "
              f"fwd={forward}, right={right}, up={up}, "
              f"pitch={pitch}, yaw={yaw}, roll={roll}")

    def capture_current_offsets(self):
        """
        Compute controller offsets from current world poses.
        Call this to snapshot whatever positions the controllers are currently at
        so they stay locked there relative to the headset.
        """
        with self.mcp.state_lock:
            headset_pos = self.mcp.current_poses['headset']['pos']
            headset_rot = self.mcp.current_poses['headset']['rot']
            headset_yaw = math.radians(-float(headset_rot[1]))

            for ctrl_name in ['controller1', 'controller2']:
                ctrl_pos = self.mcp.current_poses[ctrl_name]['pos']
                ctrl_rot = self.mcp.current_poses[ctrl_name]['rot']

                # World-space delta from headset to controller
                wx = float(ctrl_pos[0]) - float(headset_pos[0])
                wy = float(ctrl_pos[1]) - float(headset_pos[1])
                wz = float(ctrl_pos[2]) - float(headset_pos[2])

                # Inverse yaw rotation: world -> headset-local
                forward = wx * math.sin(headset_yaw) - wz * math.cos(headset_yaw)
                right = wx * math.cos(headset_yaw) + wz * math.sin(headset_yaw)
                up = wy

                # Rotation offset = controller rotation - headset rotation
                pitch_off = float(ctrl_rot[0]) - float(headset_rot[0])
                yaw_off = float(ctrl_rot[1]) - float(headset_rot[1])
                roll_off = float(ctrl_rot[2]) - float(headset_rot[2])

                self._controller_offsets[ctrl_name] = {
                    'forward': forward, 'right': right, 'up': up,
                    'pitch': pitch_off, 'yaw': yaw_off, 'roll': roll_off,
                }

        print("[KeyboardVRController] Controller offsets captured from current poses.")
        for name, off in self._controller_offsets.items():
            print(f"  {name}: fwd={off['forward']:.3f}, right={off['right']:.3f}, "
                  f"up={off['up']:.3f}, pitch={off['pitch']:.1f}, "
                  f"yaw={off['yaw']:.1f}, roll={off['roll']:.1f}")

    def apply_reset_pose(self):
        """
        Reset controllers to a fixed 'holding' pose relative to the headset.
        Left (controller1): Pointing DOWN (pitch=90), slightly left/front.
        Right (controller2): Pointing UP (pitch=-90), slightly right/front.
        """
        # Define the desired offsets
        reset_offsets = {
            'controller1': { # Left: Pointing DOWN
                'forward': 0.3, 'right': -0.2, 'up': -0.3,
                'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0
            },
            'controller2': { # Right: Pointing UP
                'forward': 0.3, 'right': 0.2, 'up': -0.3,
                'pitch': 45.0, 'yaw': 0.0, 'roll': 0.0
            }
        }

        # Apply to local state
        self._controller_offsets = reset_offsets
        
        # Apply to world immediately
        self._update_controllers()
        self.mcp.broadcast_state()
        
        print("[KeyboardVRController] Controllers RESET: Left DOWN, Right UP.")

    # ------------------------------------------------------------------
    # Background key reading thread
    # ------------------------------------------------------------------

    def _input_loop(self):
        """Background thread: poll stdin in cbreak mode, dispatch keys."""
        try:
            while self.active:
                # select() with 50ms timeout — efficient, no CPU spinning
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue

                ch = sys.stdin.read(1)

                if ch == '`':
                    # Toggle off — deactivate from this thread
                    self.active = False
                    # Restore terminal and cleanup (in main thread context via deactivate)
                    # We set active=False first so the main loop can call deactivate
                    self._restore_terminal()
                    self.mcp._headset_changed_callback = None
                    self.mcp.suppress_logging = False
                    print("\n[Keyboard VR Control] DISABLED")
                    break
                elif ch == '\x1b':
                    # ESC — could be start of arrow key escape sequence
                    self._handle_escape_sequence()
                elif ch == '\x03':
                    # Ctrl+C — exit keyboard mode gracefully
                    self.active = False
                    self._restore_terminal()
                    self.mcp._headset_changed_callback = None
                    self.mcp.suppress_logging = False
                    print("\n[Keyboard VR Control] DISABLED (Ctrl+C)")
                    break
                else:
                    self._handle_char(ch)
        except Exception:
            # If anything goes wrong, make sure terminal is restored
            self.active = False
            self._restore_terminal()
            self.mcp._headset_changed_callback = None
            self.mcp.suppress_logging = False

    def _restore_terminal(self):
        """Restore original terminal settings."""
        if self._old_term_settings is not None:
            try:
                fd = sys.stdin.fileno()
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_term_settings)
            except Exception:
                pass
            self._old_term_settings = None

    # ------------------------------------------------------------------
    # Key dispatch
    # ------------------------------------------------------------------

    def _handle_char(self, ch: str):
        """Handle a single character keypress (WASD, QE, m)."""
        ch = ch.lower()

        # Mode Toggle
        if ch == 'm':
            self.mode = 'headset' if self.mode == 'trackpad' else 'trackpad'
            print(f"\n[Keyboard] Switched to {self.mode.upper()} mode.")
            return

        # TRACKPAD MODE (Default)
        if self.mode == 'trackpad':
            # WASD maps to trackpad directions on right controller
            direction = None
            if ch == 'w': direction = 'up'     # Top
            elif ch == 's': direction = 'down' # Bottom
            elif ch == 'a': direction = 'left' # Left
            elif ch == 'd': direction = 'right'# Right
            
            if direction:
                # Use primitives from mcp_server directly since click_trackpad_direction is not available there
                try:
                    # 1. Move Joystick/Trackpad to direction
                    if hasattr(self.mcp, 'move_joystick_direction'):
                        self.mcp.move_joystick_direction(self.target_controller, direction, magnitude=1.0)
                    
                    # Wait briefly for move to register
                    # time.sleep(0.001)
                    
                    # 2. Click Trackpad
                    if hasattr(self.mcp, 'click_button'):
                        self.mcp.click_button(self.target_controller, 'trackpad', duration=0.02)
                        
                except Exception as e:
                    print(f"[Keyboard] Error triggering trackpad: {e}")
            
            # Allow Q/E/Arrows to still work for headset even in trackpad mode?
            # User said "create two modes with the default being this".
            # Usually users want to move their head too. 
            # I will allow Q/E and Arrows for headset movement in BOTH modes, 
            # as they don't conflict with WASD trackpad usage.
            if ch == 'q':
                self._move_headset(up=self.move_step)
            elif ch == 'e':
                self._move_headset(up=-self.move_step)

        # HEADSET MODE (Legacy)
        elif self.mode == 'headset':
            if ch == 'w':
                self._move_headset(forward=self.move_step)
            elif ch == 's':
                self._move_headset(forward=-self.move_step)
            elif ch == 'a':
                self._move_headset(right=-self.move_step)
            elif ch == 'd':
                self._move_headset(right=self.move_step)
            elif ch == 'q':
                self._move_headset(up=self.move_step)
            elif ch == 'e':
                self._move_headset(up=-self.move_step)

    def _handle_escape_sequence(self):
        """Parse an escape sequence (arrow keys: ESC [ A/B/C/D)."""
        # Wait briefly for the rest of the sequence
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not ready:
            return  # Bare ESC press — ignore

        ch2 = sys.stdin.read(1)
        if ch2 != '[':
            return  # Not a CSI sequence — ignore

        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not ready:
            return

        ch3 = sys.stdin.read(1)
        if ch3 == 'A':       # Up arrow
            self._rotate_headset(dpitch=self.rotate_step)
        elif ch3 == 'B':     # Down arrow
            self._rotate_headset(dpitch=-self.rotate_step)
        elif ch3 == 'C':     # Right arrow
            self._rotate_headset(dyaw=-self.rotate_step)
        elif ch3 == 'D':     # Left arrow
            self._rotate_headset(dyaw=self.rotate_step)

    # ------------------------------------------------------------------
    # Headset change callback (called by mcp_server when agent moves headset)
    # ------------------------------------------------------------------

    def _on_headset_changed(self):
        """
        Called by mcp_server._notify_headset_changed() when the agent
        (not keyboard) moves or rotates the headset. Repositions controllers
        to maintain their relative offsets.
        """
        with self.mcp.state_lock:
            self._update_controllers_locked()
        self.mcp.broadcast_state()

    # ------------------------------------------------------------------
    # Internal movement logic
    # ------------------------------------------------------------------

    def _move_headset(self, forward: float = 0.0, right: float = 0.0, up: float = 0.0):
        """Move headset in its local forward/right/up directions, then reposition controllers."""
        with self.mcp.state_lock:
            pos = self.mcp.current_poses['headset']['pos']
            yaw_rad = math.radians(-float(self.mcp.current_poses['headset']['rot'][1]))

            # Transform local movement to world space
            # forward=+ve moves toward -Z at yaw=0 (VR forward)
            dx = forward * math.sin(yaw_rad) + right * math.cos(yaw_rad)
            dz = forward * (-math.cos(yaw_rad)) + right * math.sin(yaw_rad)
            dy = up

            self.mcp.current_poses['headset']['pos'] = [
                float(pos[0]) + dx,
                float(pos[1]) + dy,
                float(pos[2]) + dz,
            ]

            # Update controllers while we still hold the lock
            self._update_controllers_locked()

        # broadcast_state acquires its own lock
        self.mcp.broadcast_state()

    def _rotate_headset(self, dpitch: float = 0.0, dyaw: float = 0.0, droll: float = 0.0):
        """Incrementally rotate headset, then reposition controllers."""
        with self.mcp.state_lock:
            rot = self.mcp.current_poses['headset']['rot']
            self.mcp.current_poses['headset']['rot'] = [
                float(rot[0]) + dpitch,
                float(rot[1]) + dyaw,
                float(rot[2]) + droll,
            ]

            # Update controllers while we still hold the lock
            self._update_controllers_locked()

        self.mcp.broadcast_state()

    def _update_controllers(self):
        """Recalculate controller world poses from headset pose + stored offsets (acquires lock)."""
        with self.mcp.state_lock:
            self._update_controllers_locked()

    def _update_controllers_locked(self):
        """Recalculate controller world poses. Caller MUST hold state_lock."""
        headset_pos = self.mcp.current_poses['headset']['pos']
        headset_rot = self.mcp.current_poses['headset']['rot']
        headset_yaw = math.radians(-float(headset_rot[1]))

        for ctrl_name, offsets in self._controller_offsets.items():
            fwd = offsets['forward']
            rgt = offsets['right']
            up = offsets['up']

            # World position from headset-local offsets (same formula as mcp_server)
            world_x = float(headset_pos[0]) + fwd * math.sin(headset_yaw) + rgt * math.cos(headset_yaw)
            world_y = float(headset_pos[1]) + up
            world_z = float(headset_pos[2]) + fwd * (-math.cos(headset_yaw)) + rgt * math.sin(headset_yaw)

            self.mcp.current_poses[ctrl_name]['pos'] = [world_x, world_y, world_z]

            # Controller rotation = headset rotation + offset
            self.mcp.current_poses[ctrl_name]['rot'] = [
                float(headset_rot[0]) + offsets['pitch'],
                float(headset_rot[1]) + offsets['yaw'],
                float(headset_rot[2]) + offsets['roll'],
            ]

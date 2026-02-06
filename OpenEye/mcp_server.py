#!/usr/bin/env python3
"""
MCP Server for OpenVR driver pose control with Vision capabilities.
Allows LLMs to control virtual VR headset and controllers via SteamVR,
and capture what the headset "sees" for visual reasoning.

Install:
    pip install mcp pillow

Run:
    python mcp_server.py
"""

import socket
import sys
import json
import threading
import queue
import time
import math
import base64
import io
import asyncio
import websockets
try:
    import openvr
except ImportError:
    openvr = None
    print("Warning: openvr module not found. Real-time tracking will be disabled.")

from typing import Optional, List, Dict, Any, Set
from mcp.server.fastmcp import FastMCP
import logging

# Configure logging
logger = logging.getLogger("mcp_server")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Configuration
HOST = '127.0.0.1'
PORT = 5555
VISION_TIMEOUT = 30  # seconds to wait for vision response
MAX_ARM_REACH = 0.8  # Maximum distance controllers can be from headset (meters)

WS_PORT = 8765
# Thread-safe state
state_lock = threading.Lock()

# Share state across re-imports if they exist in sys.modules
_existing_mcp = sys.modules.get("mcp_server")
if _existing_mcp and hasattr(_existing_mcp, "clients"):
    clients = _existing_mcp.clients
    ws_clients = _existing_mcp.ws_clients
    server_running = getattr(_existing_mcp, "server_running", False)
    vision_broadcast_running = getattr(_existing_mcp, "vision_broadcast_running", False)
    ws_loop = getattr(_existing_mcp, "ws_loop", None)
    command_queue = getattr(_existing_mcp, "command_queue", queue.Queue())
else:
    clients: List[socket.socket] = []
    ws_clients: Set[websockets.WebSocketServerProtocol] = set()
    server_running = False
    vision_broadcast_running = False
    ws_loop = None
    command_queue = queue.Queue()

vision_response: Optional[Dict] = None
vision_event = threading.Event()

# Default Poses
default_poses = {
    'headset': {'pos': [0.0, 1.5, 0.0], 'rot': [0.0, 0.0, 0.0]},
    'controller1': {'pos': [-0.3, 1.0, -0.3], 'rot': [0.0, 0.0, 0.0]},
    'controller2': {'pos': [0.3, 1.0, -0.3], 'rot': [0.0, 0.0, 0.0]},
}
current_poses = json.loads(json.dumps(default_poses))

# Controller input state
default_input = {
    'system': False,
    'menu': False,
    'grip': False,
    'triggerClick': False,
    'trackpadClick': False,
    'trackpadTouch': False,
    'buttonA': False,
    'buttonB': False,
    'triggerValue': 0.0,
    'joystickX': 0.0,
    'joystickY': 0.0,
}
controller_inputs = {
    'controller1': json.loads(json.dumps(default_input)),
    'controller2': json.loads(json.dumps(default_input)),
}

mcp = FastMCP("openvr-controller")

def distance_3d(pos1: List[float], pos2: List[float]) -> float:
    """Calculate 3D distance between two positions."""
    return math.sqrt(
        (float(pos1[0]) - float(pos2[0]))**2 + 
        (float(pos1[1]) - float(pos2[1]))**2 + 
        (float(pos1[2]) - float(pos2[2]))**2
    )

def enforce_controller_tether():
    """
    Ensure controllers stay within arm's reach of the headset.
    If a controller is too far, move it to be within MAX_ARM_REACH.
    Controllers maintain their relative offset direction but are pulled closer.
    """
    headset_pos = current_poses['headset']['pos']
    
    for ctrl in ['controller1', 'controller2']:
        ctrl_pos = current_poses[ctrl]['pos']
        dist = distance_3d(headset_pos, ctrl_pos)
        
        if dist > MAX_ARM_REACH:
            # Calculate direction from headset to controller
            dx = float(ctrl_pos[0]) - float(headset_pos[0])
            dy = float(ctrl_pos[1]) - float(headset_pos[1])
            dz = float(ctrl_pos[2]) - float(headset_pos[2])
            
            # Normalize and scale to max reach
            scale = MAX_ARM_REACH / dist
            new_pos = [
                float(headset_pos[0]) + dx * scale,
                float(headset_pos[1]) + dy * scale,
                float(headset_pos[2]) + dz * scale
            ]
            
            current_poses[ctrl]['pos'] = new_pos

# State tracking for noise reduction
last_ws_state = None

def broadcast_state():
    """Thread-safe broadcast of state to all connected drivers and UI."""
    global last_ws_state
    with state_lock:
        # Enforce controller tethering before broadcasting
        enforce_controller_tether()
        
        active_clients = []
        
        # Build individual device updates for TCP clients
        messages = []
        for device, pose in current_poses.items():
            msg = {
                "device": device,
                "pos": pose['pos'],
                "rot": pose['rot']
            }
            if device in controller_inputs:
                msg["input"] = controller_inputs[device]
            messages.append(json.dumps(msg, separators=(',', ':')) + '\n')
        
        for client in clients:
            try:
                for msg in messages:
                    client.sendall(msg.encode('utf-8'))
                active_clients.append(client)
            except Exception:
                try: client.close()
                except: pass
        
        clients[:] = active_clients
        
        # Broadcast to WebSockets only if state changed
        if ws_clients:
            # Create a compact state for comparison
            current_state_summary = {
                "poses": current_poses,
                "inputs": controller_inputs
            }
            
            # Use string representation for simple equality check (robust enough for this scale)
            state_str = json.dumps(current_state_summary, separators=(',', ':'))
            
            if state_str != last_ws_state:
                last_ws_state = state_str
                msg_data = {
                    "type": "state_update",
                    "poses": current_poses,
                    "inputs": controller_inputs
                }
                asyncio.run_coroutine_threadsafe(
                    broadcast_ws(json.dumps(msg_data)), 
                    ws_loop
                )

async def broadcast_ws(message: str):
    """Broadcast a message to all connected WebSocket clients."""
    if not ws_clients:
        return
    websockets_to_remove = set()
    for ws in ws_clients:
        try:
            # Only print for state updates or if explicitly requested to avoid log spam
            # if '"state_update"' in message:
            #     print(f"[WS] Broadcasting state to {ws.remote_address}", flush=True)
            await ws.send(message)
        except Exception as e:
            print(f"[WS] Send failed to {ws.remote_address}: {e}")
            websockets_to_remove.add(ws)
    
    for ws in websockets_to_remove:
        if ws in ws_clients:
            ws_clients.remove(ws)

async def ws_handler(websocket):
    """Handle individual WebSocket connections."""
    print(f"[WS] Client connected: {websocket.remote_address}")
    ws_clients.add(websocket)
    
    # Send initial state immediately upon connection
    try:
        msg_data = {
            "type": "state_update",
            "poses": current_poses,
            "inputs": controller_inputs
        }
        await websocket.send(json.dumps(msg_data))
    except Exception as e:
        print(f"[WS] Error sending initial state: {e}")
        
    try:
        async for message in websocket:
            # Handle incoming messages from UI (decisions/requests)
            try:
                data = json.loads(message)
                # print(f"[WS] Received: {data}")
                
                msg_type = data.get('type')
                
                if msg_type == 'user_request':
                    # Put request in queue for Agent to process
                    content = data.get('content')
                    if content:
                        command_queue.put(content)
                        
                elif msg_type == 'trigger_action':
                    device = data.get('device')
                    action = data.get('action')
                    if device in ['controller1', 'controller2'] and action == 'click':
                        click_button(device, 'trigger')

                elif msg_type == 'move_relative':
                    # Handle direct movement control from UI
                    device = data.get('device')
                    dx = float(data.get('dx', 0))
                    dy = float(data.get('dy', 0))
                    dz = float(data.get('dz', 0))
                    
                    if device in current_poses:
                        move_relative(device, dx, dy, dz)
                        
                elif msg_type == 'rotate_relative':
                    device = data.get('device')
                    dp = float(data.get('dp', 0))
                    dy = float(data.get('dy', 0))
                    dr = float(data.get('dr', 0))
                    
                    if device in current_poses:
                        rotate_relative(device, dp, dy, dr)

                elif msg_type == 'read_pose':
                    # Returns current poses for all devices in a unified response
                    with state_lock:
                        msg_data = {
                            "type": "pose_data",
                            "poses": current_poses,
                            "timestamp": time.time()
                        }
                    await websocket.send(json.dumps(msg_data))
                        
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"[WS] Error processing message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"[WS] Client disconnected: {websocket.remote_address}")
        if websocket in ws_clients:
            ws_clients.remove(websocket)

ws_loop = None
def run_ws_server():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    
    # Use a try-except block to handle cases where the event loop might be closed
    async def _start():
        try:
            async with websockets.serve(ws_handler, HOST, WS_PORT):
                logger.info(f"WebSocket Server listening on {HOST}:{WS_PORT}")
                await asyncio.Future()  # run forever
        except Exception as e:
            logger.error(f"WebSocket Server error: {e}")

    try:
        ws_loop.run_until_complete(_start())
    except Exception as e:
        print(f"WebSocket Server Error: {e}")

def run_tracking_loop():
    """Periodically poll OpenVR for device poses and update current_poses."""
    if not openvr:
        return
        
    print("[TRACKING] Initializing OpenVR...")
    try:
        vr_system = openvr.init(openvr.VRApplication_Background)
    except Exception as e:
        print(f"[TRACKING] Failed to initialize OpenVR: {e}")
        return
        
    print("[TRACKING] Loop started.")

    last_broadcast = time.time()
    
    while True:
        try:
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )
            
            with state_lock:
                for i in range(openvr.k_unMaxTrackedDeviceCount):
                    device_class = vr_system.getTrackedDeviceClass(i)
                    
                    if device_class in [openvr.TrackedDeviceClass_HMD, openvr.TrackedDeviceClass_Controller]:
                        pose = poses[i]
                        if pose.bPoseIsValid:
                            matrix = pose.mDeviceToAbsoluteTracking
                            
                            # Extract Position
                            x = matrix[0][3]
                            y = matrix[1][3]
                            z = matrix[2][3]
                            
                            # Extract Rotation
                            # OpenVR -> Euler conversion
                            yaw = math.degrees(math.atan2(matrix[0][2], matrix[2][2]))
                            pitch = math.degrees(math.asin(-matrix[1][2]))
                            roll = math.degrees(math.atan2(matrix[1][0], matrix[1][1]))
                            
                            dev_name = None
                            if device_class == openvr.TrackedDeviceClass_HMD:
                                dev_name = 'headset'
                            else:
                                role = vr_system.getControllerRoleForTrackedDeviceIndex(i)
                                if role == openvr.TrackedControllerRole_LeftHand:
                                    dev_name = 'controller1'
                                elif role == openvr.TrackedControllerRole_RightHand:
                                    dev_name = 'controller2'
                                    
                            if dev_name:
                                current_poses[dev_name]['pos'] = [x, y, z]
                                current_poses[dev_name]['rot'] = [pitch, yaw, roll]
                                
            # Broadcast updates at ~30Hz
            now = time.time()
            if now - last_broadcast > 0.033:
                # Debug print occasionally
                if int(now) % 5 == 0 and int(last_broadcast) % 5 != 0:
                     print(f"[TRACKING] Valid poses. Headset: {current_poses['headset']['pos']}", flush=True)
                
                broadcast_state()
                last_broadcast = now
                
            time.sleep(0.005) # Sleep small amount to not hog CPU
            
        except Exception as e:
            print(f"[TRACKING] Error in loop: {e}")
            time.sleep(1)

def run_vision_broadcast_loop():
    """Periodically request vision frames if clients are connected."""
    global vision_broadcast_running
    # print("[VISION] Starting periodic broadcast loop...")
    while vision_broadcast_running:
        if ws_clients and clients:
            # Send a vision request
            # logger.info(f"[VISION] Periodic request. WS: {len(ws_clients)}, TCP: {len(clients)}")
            request = {
                "type": "vision_request",
                "action": "capture_frame"
            }
            send_vision_request(request)
        time.sleep(5.0)
    print("[VISION] Periodic broadcast loop stopped.")

def send_device_update(device: str):
    """Send update for a single device instead of all devices. Lower latency for targeted updates."""
    import datetime
    with state_lock:
        if device not in current_poses:
            return
        
        # Build message for just this device
        pose = current_poses[device]
        msg = {
            "device": device,
            "pos": pose['pos'],
            "rot": pose['rot']
        }
        # Add input state for controllers
        if device in controller_inputs:
            msg["input"] = controller_inputs[device]
        
        # Add timestamp for latency debugging
        send_ts = datetime.datetime.now()
        msg["send_ts"] = send_ts.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
        print(f"[SEND] {device} at {msg['send_ts']}")
        
        payload = json.dumps(msg, separators=(',', ':')) + '\n'
        
        active_clients = []
        for client in clients:
            try:
                client.sendall(payload.encode('utf-8'))
                active_clients.append(client)
            except Exception:
                try: client.close()
                except: pass
        
        clients[:] = active_clients


def send_vision_request(request: Dict) -> bool:
    """Send a vision request to the driver."""
    with state_lock:
        if not clients:
            return False
        payload = json.dumps(request) + '\n'
        for client in clients:
            try:
                client.sendall(payload.encode('utf-8'))
                return True
            except Exception:
                pass
    return False

def handle_client(client_socket: socket.socket):
    """Handle incoming data from driver (poses and vision responses)."""
    global vision_response
    
    with state_lock:
        clients.append(client_socket)
    
    broadcast_state()
    buffer = ""
    
    while True:
        try:
            client_socket.settimeout(2.0)
            data = client_socket.recv(5242880)  # 5MB buffer for large vision data
            if not data: break
            
            buffer += data.decode('utf-8')
            
            # Process complete JSON messages (newline-delimited)
            # For large vision responses, we need to ensure we have complete JSON
            while '\n' in buffer:
                newline_pos = buffer.find('\n')
                line = buffer[:newline_pos]
                
                if line.strip():
                    # Try to parse as JSON - if it fails, the message might be incomplete
                    # This can happen if the base64 data contains characters that look like newlines
                    try:
                        msg = json.loads(line)
                        # Successfully parsed - remove from buffer and process
                        buffer = buffer[newline_pos + 1:]
                        
                        # Check if this is a vision response
                        if msg.get('type') in ['frame', 'video', 'status', 'error']:
                            vision_response = msg
                            vision_event.set()
                            
                            # Also broadcast frame to WebSockets for UI
                            if msg.get('type') == 'frame' and ws_clients:
                                frames = msg.get('frames', [])
                                if frames:
                                    msg_data = {
                                        "type": "vision_update",
                                        "frame": frames[0],
                                        "width": msg.get('width'),
                                        "height": msg.get('height')
                                    }
                                    if ws_loop and ws_loop.is_running():
                                        # logger.info(f"[VISION] Broadcasting frame ({len(frames[0])} bytes) to {len(ws_clients)} UI clients")
                                        asyncio.run_coroutine_threadsafe(
                                            broadcast_ws(json.dumps(msg_data)), 
                                            ws_loop
                                        )
                                    else:
                                        print("[VISION] Skip broadcast - ws_loop not ready", flush=True)
                    except json.JSONDecodeError:
                        # JSON incomplete - might be a partial vision response
                        # Check if this looks like a vision response (has "frames" field)
                        if '"frames"' in line or '"type":"frame"' in line or '"type":"video"' in line:
                            # This is likely a truncated vision response - wait for more data
                            break
                        else:
                            # Not a vision response, just bad JSON - skip it
                            buffer = buffer[newline_pos + 1:]
                else:
                    # Empty line, skip it
                    buffer = buffer[newline_pos + 1:]
                        
        except socket.timeout:
            continue
        except Exception:
            break
            
    with state_lock:
        if client_socket in clients:
            clients.remove(client_socket)
    client_socket.close()

def run_tcp_server():
    global server_running
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        # Enable address reuse to fix reconnection issues
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Disable Nagle's algorithm for lower latency
        server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.bind((HOST, PORT))
        server.listen(5)
        logger.info(f"TCP Server listening on {HOST}:{PORT}")
        
        while server_running:
            try:
                server.settimeout(1.0)
                client, _ = server.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                t = threading.Thread(target=handle_client, args=(client,), daemon=True)
                t.start()
            except socket.timeout:
                continue
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.close()

@mcp.tool()
def kill_address() -> str:
    """
    Kills the process using the configured TCP port (default 5555).
    Useful if the server cannot start because 'Address already in use'.
    """
    import subprocess
    try:
        # Try fuser first (standard on many Linux distros)
        cmd = f"fuser -k -n tcp {PORT}"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"Successfully killed process on port {PORT}"
    except subprocess.CalledProcessError:
        try:
            # Fallback to lsof + kill
            # This complex command finds PIDs using the port and kills them
            cmd = f"lsof -t -i:{PORT} | xargs -r kill"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return f"Successfully killed process on port {PORT} (using lsof)"
        except subprocess.CalledProcessError:
             # Fallback to netstat if lsof is missing? Or just return error.
             return f"Could not kill process on port {PORT}. Please do it manually (e.g. 'fuser -k {PORT}/tcp')."
    except Exception as e:
        return f"Error executing kill command: {e}"

# --- MCP TOOLS: Connection Management ---

@mcp.tool()
def start_vr_bridge() -> str:
    """Start the TCP server to listen for the OpenVR driver."""
    global server_running
    if server_running: return "Server already running."
    
    server_running = True
    
    t = threading.Thread(target=run_tcp_server, daemon=True)
    t.start()
    
    # Start WebSocket server
    ws_t = threading.Thread(target=run_ws_server, daemon=True)
    ws_t.start()
    
    # Start Vision Broadcast loop
    global vision_broadcast_running
    vision_broadcast_running = True
    vision_t = threading.Thread(target=run_vision_broadcast_loop, daemon=True)
    vision_t.start()
    
    # Start Tracking Loop (OpenVR)
    tracking_t = threading.Thread(target=run_tracking_loop, daemon=True)
    tracking_t.start()
    
    return "VR Bridge, WebSocket Server, and Tracking Loop started. Please launch SteamVR."

@mcp.tool()
def get_connection_status() -> str:
    """Check if the VR driver is connected."""
    with state_lock:
        count = len(clients)
    if count == 0:
        return "No VR driver connected. Make sure SteamVR is running with the sample driver."
    return f"Connected to {count} VR driver(s)."

# --- MCP TOOLS: Movement & Orientation ---

@mcp.tool()
def look_at(target_x: float, target_y: float, target_z: float) -> str:
    """
    Make the headset look at a specific coordinate in 3D space.
    Calculates the necessary Yaw and Pitch.
    """
    with state_lock:
        hx, hy, hz = current_poses['headset']['pos']
    
    # Cast inputs to float
    target_x, target_y, target_z = float(target_x), float(target_y), float(target_z)
    
    dx = target_x - float(hx)
    dy = target_y - float(hy)
    dz = target_z - float(hz)
    
    yaw = math.degrees(math.atan2(dx, -dz))
    dist = math.sqrt(dx*dx + dz*dz)
    pitch = math.degrees(math.atan2(dy, dist))
    
    with state_lock:
        current_poses['headset']['rot'] = [-pitch, yaw, 0.0]
    
    broadcast_state()
    return f"Looking at [{target_x}, {target_y}, {target_z}] (Yaw: {yaw:.1f}, Pitch: {-pitch:.1f})"

@mcp.tool()
def teleport(device: str, x: float, y: float, z: float) -> str:
    """Teleport a device (headset/controller1/controller2) to exact coordinates."""
    if device not in current_poses:
        return f"Invalid device. Options: {list(current_poses.keys())}"
    
    with state_lock:
        current_poses[device]['pos'] = [float(x), float(y), float(z)]
    
    broadcast_state()
    return f"{device} teleported to [{x}, {y}, {z}]"

@mcp.tool()
def rotate_device(device: str, pitch: float, yaw: float, roll: float) -> str:
    """Set the rotation of a device in degrees (pitch=up/down, yaw=left/right, roll=tilt)."""
    import time
    if device not in current_poses:
        return f"Invalid device. Options: {list(current_poses.keys())}"
    
    with state_lock:
        current_poses[device]['rot'] = [pitch, yaw, roll]
    
    # Use targeted update for lower latency (only sends this device, not all 3)
    send_device_update(device)
    
    # Verification: wait 0.1s and check if values match
    time.sleep(0.1)
    with state_lock:
        stored_rot = current_poses[device]['rot']
    expected = [pitch, yaw, roll]
    if stored_rot == expected:
        print(f"[VERIFY] good - {device} rot={stored_rot}")
    else:
        print(f"[VERIFY] not good - {device} expected={expected} got={stored_rot}")
    
    return f"{device} rotated to pitch={pitch}, yaw={yaw}, roll={roll}"

@mcp.tool()
def walk_path(x: float, z: float, steps: int = 10) -> str:
    """
    Simulate walking to a destination over time. 
    Prevents 'teleporting' sickness in visuals and looks more natural.
    """
    with state_lock:
        start_x, start_y, start_z = current_poses['headset']['pos']
        
    # Cast inputs to float
    x, z = float(x), float(z)
    
    for i in range(1, steps + 1):
        t = i / steps
        curr_x = float(start_x) + (x - float(start_x)) * t
        curr_z = float(start_z) + (z - float(start_z)) * t
        
        with state_lock:
            current_poses['headset']['pos'] = [curr_x, start_y, curr_z]
        broadcast_state()
        time.sleep(0.1)
        
    return f"Walked to [{x}, {start_y}, {z}]"

@mcp.tool()
def get_current_pose(device: str = "headset") -> str:
    """Get the current position and rotation of a device."""
    if device not in current_poses:
        return f"Invalid device. Options: {list(current_poses.keys())}"
    
    with state_lock:
        pose = current_poses[device]
    
    return f"{device} - Position: {pose['pos']}, Rotation: {pose['rot']}"

@mcp.tool()
def move_relative(device: str, dx: float = 0, dy: float = 0, dz: float = 0) -> str:
    """Move a device relative to its current position."""
    if device not in current_poses:
        return f"Invalid device. Options: {list(current_poses.keys())}"
    
    with state_lock:
        pos = current_poses[device]['pos']
        dx, dy, dz = float(dx), float(dy), float(dz)
        current_poses[device]['pos'] = [float(pos[0]) + dx, float(pos[1]) + dy, float(pos[2]) + dz]
        new_pos = current_poses[device]['pos']
    
    broadcast_state()
    return f"{device} moved to {new_pos}"

def rotate_relative(device: str, dp: float = 0, dy: float = 0, dr: float = 0) -> str:
    """Rotate a device relative to its current orientation."""
    if device not in current_poses:
        return f"Invalid device. Options: {list(current_poses.keys())}"
    
    with state_lock:
        rot = current_poses[device]['rot']
        dp, dy, dr = float(dp), float(dy), float(dr)
        current_poses[device]['rot'] = [float(rot[0]) + dp, float(rot[1]) + dy, float(rot[2]) + dr]
        new_rot = current_poses[device]['rot']
    
    broadcast_state()
    return f"{device} rotated to {new_rot}"

@mcp.tool()
def position_controller_relative_to_headset(
    controller: str, 
    forward: float = -0.3, 
    right: float = 0.0, 
    up: float = -0.5
) -> str:
    """
    Position a controller relative to the headset position.
    This is useful for keeping controllers in natural positions as you move.
    
    Args:
        controller: "controller1" (left) or "controller2" (right)
        forward: Distance in front of headset (negative = in front, positive = behind)
        right: Distance to the right (negative = left, positive = right)
        up: Distance up/down from headset (negative = below, positive = above)
    
    Default positions are natural "hands at sides" pose.
    Controllers are automatically kept within arm's reach (0.8m).
    """
    if controller not in ['controller1', 'controller2']:
        return "Invalid controller. Use 'controller1' (left) or 'controller2' (right)"
    
    with state_lock:
        headset_pos = current_poses['headset']['pos']
        headset_yaw = math.radians(float(current_poses['headset']['rot'][1]))
        
        # Calculate world-space offset based on headset orientation
        # Forward is -Z in VR, Right is +X
        world_x = float(headset_pos[0]) + float(forward) * math.sin(headset_yaw) + float(right) * math.cos(headset_yaw)
        world_y = float(headset_pos[1]) + float(up)
        world_z = float(headset_pos[2]) + float(forward) * (-math.cos(headset_yaw)) + float(right) * math.sin(headset_yaw)
        
        current_poses[controller]['pos'] = [world_x, world_y, world_z]
    
    broadcast_state()
    
    with state_lock:
        final_pos = current_poses[controller]['pos']
    
    return f"{controller} positioned at {final_pos} (relative to headset: forward={forward}, right={right}, up={up})"

@mcp.tool()
def reset_controller_positions() -> str:
    """
    Reset both controllers to natural positions relative to the headset.
    Controller1 (left): slightly left and in front
    Controller2 (right): slightly right and in front
    """
    with state_lock:
        headset_pos = current_poses['headset']['pos']
        
        # Natural resting positions
        current_poses['controller1']['pos'] = [
            headset_pos[0] - 0.3,  # Left
            headset_pos[1] - 0.5,  # Below head (waist level)
            headset_pos[2] - 0.3   # In front
        ]
        current_poses['controller2']['pos'] = [
            headset_pos[0] + 0.3,  # Right
            headset_pos[1] - 0.5,  # Below head (waist level)
            headset_pos[2] - 0.3   # In front
        ]
    
    broadcast_state()
    return f"Controllers reset to natural positions near headset at {headset_pos}"

# --- MCP TOOLS: Controller Input (Buttons & Joystick) ---

@mcp.tool()
def press_button(controller: str, button: str) -> str:
    """
    Press a button on a controller. The button stays pressed until release_button is called.
    
    Args:
        controller: "controller1" (left) or "controller2" (right)
        button: One of: "trigger", "grip", "menu", "system", "trackpad", "a", "b"
    
    Common uses:
    - "trigger": Primary action (select, shoot, grab)
    - "grip": Secondary grab, hold objects
    - "menu": Open menu
    - "a"/"b": Context-dependent actions
    - "trackpad": Trackpad/joystick click
    """
    if controller not in controller_inputs:
        return f"Invalid controller. Use 'controller1' or 'controller2'"
    
    button_map = {
        'trigger': 'triggerClick',
        'grip': 'grip',
        'menu': 'menu',
        'system': 'system',
        'trackpad': 'trackpadClick',
        'a': 'buttonA',
        'b': 'buttonB',
    }
    
    if button.lower() not in button_map:
        return f"Invalid button. Options: {list(button_map.keys())}"
    
    with state_lock:
        controller_inputs[controller][button_map[button.lower()]] = True
        # Also set trigger value if pressing trigger
        if button.lower() == 'trigger':
            controller_inputs[controller]['triggerValue'] = 1.0
        # Debug: show current input state
        print(f"[DEBUG] {controller} input state: {controller_inputs[controller]}", flush=True)
    
    broadcast_state()
    return f"{controller}: {button} pressed"

@mcp.tool()
def release_button(controller: str, button: str) -> str:
    """
    Release a previously pressed button on a controller.
    
    Args:
        controller: "controller1" (left) or "controller2" (right)
        button: One of: "trigger", "grip", "menu", "system", "trackpad", "a", "b"
    """
    if controller not in controller_inputs:
        return f"Invalid controller. Use 'controller1' or 'controller2'"
    
    button_map = {
        'trigger': 'triggerClick',
        'grip': 'grip',
        'menu': 'menu',
        'system': 'system',
        'trackpad': 'trackpadClick',
        'a': 'buttonA',
        'b': 'buttonB',
    }
    
    if button.lower() not in button_map:
        return f"Invalid button. Options: {list(button_map.keys())}"
    
    with state_lock:
        controller_inputs[controller][button_map[button.lower()]] = False
        # Also reset trigger value if releasing trigger
        if button.lower() == 'trigger':
            controller_inputs[controller]['triggerValue'] = 0.0
    
    broadcast_state()
    return f"{controller}: {button} released"

@mcp.tool()
def click_button(controller: str, button: str, duration: float = 0.1) -> str:
    """
    Click a button (press and release) on a controller.
    
    Args:
        controller: "controller1" (left) or "controller2" (right)
        button: One of: "trigger", "grip", "menu", "system", "trackpad", "a", "b"
        duration: How long to hold the button in seconds (default 0.1)
    """
    result = press_button(controller, button)
    if "Invalid" in result:
        return result
    
    time.sleep(duration)
    release_button(controller, button)
    return f"{controller}: {button} clicked"

@mcp.tool()
def set_trigger(controller: str, value: float) -> str:
    """
    Set the analog trigger value (for partial pulls).
    
    Args:
        controller: "controller1" (left) or "controller2" (right)
        value: Trigger value from 0.0 (released) to 1.0 (fully pressed)
    """
    if controller not in controller_inputs:
        return f"Invalid controller. Use 'controller1' or 'controller2'"
    
    value = max(0.0, min(1.0, value))
    
    with state_lock:
        controller_inputs[controller]['triggerValue'] = value
        controller_inputs[controller]['triggerClick'] = value > 0.9
    
    broadcast_state()
    return f"{controller}: trigger set to {value:.2f}"

@mcp.tool()
def set_joystick(controller: str, x: float, y: float) -> str:
    """
    Set the joystick/trackpad position.
    
    Args:
        controller: "controller1" (left) or "controller2" (right)
        x: Horizontal position from -1.0 (left) to 1.0 (right)
        y: Vertical position from -1.0 (down) to 1.0 (up)
    
    Common uses:
    - Movement: Use left controller joystick for locomotion
    - Camera: Use right controller joystick for turning
    - Menu navigation: Move through UI elements
    """
    if controller not in controller_inputs:
        return f"Invalid controller. Use 'controller1' or 'controller2'"
    
    x = max(-1.0, min(1.0, x))
    y = max(-1.0, min(1.0, y))
    
    with state_lock:
        controller_inputs[controller]['joystickX'] = x
        controller_inputs[controller]['joystickY'] = y
        # Set touch state if joystick is being used
        controller_inputs[controller]['trackpadTouch'] = (x != 0 or y != 0)
    
    broadcast_state()
    return f"{controller}: joystick set to ({x:.2f}, {y:.2f})"

@mcp.tool()
def move_joystick_direction(controller: str, direction: str, magnitude: float = 1.0) -> str:
    """
    Move the joystick in a cardinal direction.
    
    Args:
        controller: "controller1" (left) or "controller2" (right)
        direction: "up", "down", "left", "right", "center"
        magnitude: How far to push (0.0 to 1.0, default 1.0)
    """
    directions = {
        'up': (0, 1),
        'down': (0, -1),
        'left': (-1, 0),
        'right': (1, 0),
        'center': (0, 0),
        'forward': (0, 1),
        'backward': (0, -1),
    }
    
    if direction.lower() not in directions:
        return f"Invalid direction. Options: {list(directions.keys())}"
    
    dx, dy = directions[direction.lower()]
    magnitude = max(0.0, min(1.0, magnitude))
    
    return set_joystick(controller, dx * magnitude, dy * magnitude)

@mcp.tool()
def release_all_inputs(controller: str = "both") -> str:
    """
    Release all buttons and reset joystick to center.
    
    Args:
        controller: "controller1", "controller2", or "both" (default)
    """
    controllers = []
    if controller == "both":
        controllers = ['controller1', 'controller2']
    elif controller in controller_inputs:
        controllers = [controller]
    else:
        return f"Invalid controller. Use 'controller1', 'controller2', or 'both'"
    
    with state_lock:
        for ctrl in controllers:
            controller_inputs[ctrl] = json.loads(json.dumps(default_input))
    
    broadcast_state()
    return f"All inputs released for {controller}"

@mcp.tool()
def get_controller_state(controller: str) -> str:
    """
    Get the current input state of a controller.
    
    Args:
        controller: "controller1" (left) or "controller2" (right)
    """
    if controller not in controller_inputs:
        return f"Invalid controller. Use 'controller1' or 'controller2'"
    
    with state_lock:
        inputs = controller_inputs[controller].copy()
        pose = current_poses[controller].copy()
    
    # Format nicely
    pressed = [k for k, v in inputs.items() if v is True]
    
    return f"""{controller} state:
  Position: {pose['pos']}
  Rotation: {pose['rot']}
  Joystick: ({inputs['joystickX']:.2f}, {inputs['joystickY']:.2f})
  Trigger: {inputs['triggerValue']:.2f}
  Pressed buttons: {pressed if pressed else 'none'}"""

@mcp.tool()
def perform_grab(controller: str) -> str:
    """
    Perform a grab action - press grip and trigger together.
    Common interaction pattern in VR for picking up objects.
    
    Args:
        controller: "controller1" (left) or "controller2" (right)
    """
    if controller not in controller_inputs:
        return f"Invalid controller. Use 'controller1' or 'controller2'"
    
    with state_lock:
        controller_inputs[controller]['grip'] = True
        controller_inputs[controller]['triggerClick'] = True
        controller_inputs[controller]['triggerValue'] = 1.0
    
    broadcast_state()
    return f"{controller}: grab initiated (grip + trigger)"

@mcp.tool()
def perform_release(controller: str) -> str:
    """
    Release a grabbed object - release grip and trigger.
    
    Args:
        controller: "controller1" (left) or "controller2" (right)
    """
    if controller not in controller_inputs:
        return f"Invalid controller. Use 'controller1' or 'controller2'"
    
    with state_lock:
        controller_inputs[controller]['grip'] = False
        controller_inputs[controller]['triggerClick'] = False
        controller_inputs[controller]['triggerValue'] = 0.0
    
    broadcast_state()
    return f"{controller}: object released"

# --- MCP TOOLS: Vision (The "Eye" of the Agent) ---

@mcp.tool()
def inspect_surroundings() -> str:
    """
    Capture a single frame of what the VR headset currently sees.
    Returns a base64-encoded JPEG image that can be analyzed.
    
    Use this to:
    - Look for objects in the VR environment
    - Verify navigation was successful
    - Understand the current scene before taking action
    """
    global vision_response
    vision_response = None
    vision_event.clear()
    
    request = {
        "type": "vision_request",
        "action": "capture_frame"
    }
    
    if not send_vision_request(request):
        return "Error: No VR driver connected. Cannot capture frame."
    
    # Wait for response
    if not vision_event.wait(timeout=VISION_TIMEOUT):
        return "Error: Timeout waiting for frame capture."
    
    if vision_response is None:
        return "Error: No response received from driver."
    
    if vision_response.get('type') == 'error':
        return f"Error: {vision_response.get('message', 'Unknown error')}"
    
    frames = vision_response.get('frames', [])
    if not frames:
        return "Error: No frame data received."
    
    width = vision_response.get('width', 0)
    height = vision_response.get('height', 0)
    
    # Return the base64 image data for the LLM to analyze
    return json.dumps({
        "type": "image",
        "format": "jpeg",
        "width": width,
        "height": height,
        "data": frames[0]  # Base64 encoded JPEG
    })

@mcp.tool()
def capture_video(duration: float = 3.0, fps: int = 10) -> str:
    """
    Capture a video (sequence of frames) of what the VR headset sees.
    
    Args:
        duration: Length of video in seconds (default 3.0, max 10.0)
        fps: Frames per second (default 10, max 30)
    
    Returns a list of base64-encoded JPEG frames that can be analyzed
    to understand motion, track objects, or observe changes over time.
    
    Use this to:
    - Observe how objects move in the scene
    - Track your own movement through the environment
    - Capture a sequence for temporal reasoning
    """
    global vision_response
    vision_response = None
    vision_event.clear()
    
    # Clamp values to reasonable limits
    duration = min(max(duration, 0.5), 10.0)
    fps = min(max(fps, 1), 30)
    
    request = {
        "type": "vision_request",
        "action": "capture_video",
        "duration": duration,
        "fps": fps
    }
    
    if not send_vision_request(request):
        return "Error: No VR driver connected. Cannot capture video."
    
    # Wait for response (longer timeout for video)
    timeout = duration + 10  # Extra time for encoding
    if not vision_event.wait(timeout=timeout):
        return "Error: Timeout waiting for video capture."
    
    if vision_response is None:
        return "Error: No response received from driver."
    
    if vision_response.get('type') == 'error':
        return f"Error: {vision_response.get('message', 'Unknown error')}"
    
    frames = vision_response.get('frames', [])
    if not frames:
        return "Error: No frame data received."
    
    width = vision_response.get('width', 0)
    height = vision_response.get('height', 0)
    frame_count = vision_response.get('frameCount', len(frames))
    
    return json.dumps({
        "type": "video",
        "format": "jpeg_sequence",
        "width": width,
        "height": height,
        "frameCount": frame_count,
        "fps": fps,
        "duration": duration,
        "frames": frames  # List of base64 encoded JPEGs
    })

@mcp.tool()
def look_around_and_observe() -> str:
    """
    Perform a 360-degree scan of the environment by rotating the headset
    and capturing frames at different angles. Returns multiple images
    for comprehensive scene understanding.
    
    This is useful for:
    - Getting a complete picture of the surroundings
    - Finding objects that might be behind you
    - Building a mental map of the VR space
    """
    global vision_response
    
    frames_data = []
    angles = [0, 90, 180, 270]  # Look in 4 directions
    
    with state_lock:
        original_rot = current_poses['headset']['rot'].copy()
    
    for angle in angles:
        # Rotate headset
        with state_lock:
            current_poses['headset']['rot'] = [original_rot[0], angle, original_rot[2]]
        broadcast_state()
        time.sleep(0.2)  # Let the view settle
        
        # Capture frame
        vision_response = None
        vision_event.clear()
        
        request = {"type": "vision_request", "action": "capture_frame"}
        if send_vision_request(request):
            if vision_event.wait(timeout=5):
                if vision_response and vision_response.get('type') == 'frame':
                    frames = vision_response.get('frames', [])
                    if frames:
                        frames_data.append({
                            "angle": angle,
                            "data": frames[0]
                        })
    
    # Restore original rotation
    with state_lock:
        current_poses['headset']['rot'] = original_rot
    broadcast_state()
    
    if not frames_data:
        return "Error: Could not capture any frames during scan."
    
    return json.dumps({
        "type": "panorama_scan",
        "format": "jpeg",
        "directions": frames_data,
        "description": "4 frames captured at 0°, 90°, 180°, 270° yaw angles"
    })

@mcp.tool()
def navigate_to_object(description: str) -> str:
    """
    High-level navigation helper. Describe what you're looking for,
    and this tool will help you plan the search.
    
    Args:
        description: What you're looking for (e.g., "red cube", "door", "table")
    
    Returns guidance on how to find the object using the vision tools.
    
    Note: This doesn't actually find the object - you need to use
    inspect_surroundings() or look_around_and_observe() and analyze
    the images yourself to locate objects.
    """
    return f"""To find "{description}", follow these steps:

1. First, call look_around_and_observe() to get a 360° view
2. Analyze the returned images to locate the {description}
3. If found, note which angle (0°, 90°, 180°, 270°) it was visible at
4. Use look_at(x, y, z) to face that direction, or rotate_device("headset", 0, angle, 0)
5. Call inspect_surroundings() to verify you're facing the right way
6. Use walk_path(x, z) to move toward it, checking with inspect_surroundings() periodically
7. Repeat until you reach the {description}

Current headset position: {current_poses['headset']['pos']}
Current headset rotation: {current_poses['headset']['rot']}"""

# --- Entry Point ---

if __name__ == '__main__':
    print("OpenVR MCP Server with Vision & Input Support")
    print("==============================================")
    print("Tools available:")
    print("  Connection: start_vr_bridge, get_connection_status")
    print("  Movement: teleport, walk_path, move_relative, look_at, rotate_device")
    print("  Controllers: position_controller_relative_to_headset, reset_controller_positions")
    print("  Buttons: press_button, release_button, click_button, set_trigger")
    print("  Joystick: set_joystick, move_joystick_direction")
    print("  Actions: perform_grab, perform_release, release_all_inputs")
    print("  Vision: inspect_surroundings, capture_video, look_around_and_observe")
    print("")
    mcp.run()

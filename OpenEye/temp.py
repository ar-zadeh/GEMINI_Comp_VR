import dxcam
import subprocess
import json

def get_vr_coords():
    # We still use PowerShell once to find the window location
    cmd = 'powershell.exe -NoProfile -Command "Get-Process -Name vrcompositor | Select-Object -ExpandProperty MainWindowHandle"'
    hwnd = subprocess.check_output(cmd, shell=True).decode().strip()
    
    # Get the rect via PowerShell (or hardcode if the window doesn't move)
    # Based on your previous output: X=-1920, Y=360, W=1280, H=720
    # Let's define the region: (left, top, right, bottom)
    region = (-1920, 360, -1920 + 1280, 360 + 720)
    return region

region = get_vr_coords()
camera = dxcam.create(device_idx=0, output_idx=1) # output_idx depends on which monitor it is

# To capture a single frame efficiently:
frame = camera.grab(region=region) 

# If you want a continuous high-speed loop:
camera.start(region=region, target_fps=30)
while True:
    frame = camera.get_latest_frame() # This is extremely low-latency
    if frame is not None:
        # 'frame' is a numpy array (RGB). 
        # You can pass this directly to Gemini or process it.
        pass
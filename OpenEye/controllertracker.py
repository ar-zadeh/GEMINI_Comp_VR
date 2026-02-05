import openvr
import time
import math

# Initialize OpenVR as a background application
vr = openvr.init(openvr.VRApplication_Background)

def get_pose_data(matrix):
    """Extracts position (x, y, z) and rotation (pitch, yaw, roll)"""
    # 1. Extract Position
    x = matrix[0][3]
    y = matrix[1][3]
    z = matrix[2][3]

    # 2. Extract Rotation (Euler Angles from 3x4 Matrix)
    # Using the standard conversion for a row-major extract from OpenVR
    # Note: OpenVR uses a right-handed coordinate system
    try:
        yaw = math.atan2(matrix[0][2], matrix[2][2])
        pitch = math.asin(-matrix[1][2])
        roll = math.atan2(matrix[1][0], matrix[1][1])
        
        # Convert radians to degrees
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        roll = math.degrees(roll)
    except Exception:
        pitch, yaw, roll = 0.0, 0.0, 0.0

    return (x, y, z), (pitch, yaw, roll)

print("Searching for devices... (Press Ctrl+C to stop)")

try:
    while True:
        poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = vr.getTrackedDeviceClass(i)
            
            if device_class in [openvr.TrackedDeviceClass_HMD, openvr.TrackedDeviceClass_Controller]:
                pose = poses[i]
                if pose.bPoseIsValid:
                    role = vr.getControllerRoleForTrackedDeviceIndex(i)
                    
                    if device_class == openvr.TrackedDeviceClass_HMD:
                        label = "HMD"
                    elif role == openvr.TrackedControllerRole_LeftHand:
                        label = "Left Controller"
                    elif role == openvr.TrackedControllerRole_RightHand:
                        label = "Right Controller"
                    else:
                        label = f"Generic ({i})"

                    pos, rot = get_pose_data(pose.mDeviceToAbsoluteTracking)
                    
                    # Clean output formatting
                    print(f"{label:18} | POS: X:{pos[0]:.2f} Y:{pos[1]:.2f} Z:{pos[2]:.2f} | ROT: P:{rot[0]:.1f}° Y:{rot[1]:.1f}° R:{rot[2]:.1f}°")

        print("-" * 80)
        time.sleep(5) 

except KeyboardInterrupt:
    print("\nShutting down...")
    openvr.shutdown()
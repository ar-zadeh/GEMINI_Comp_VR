import openvr
import time

# Initialize OpenVR as a background application
vr = openvr.init(openvr.VRApplication_Background)



def get_pose_position(matrix):
    """Extracts x, y, z from the 3x4 OpenVR matrix"""
    x = matrix[0][3]
    y = matrix[1][3]
    z = matrix[2][3]
    return x, y, z

print("Searching for devices... (Make sure they are turned on)")

try:
    while True:
        # Get poses for all possible 64 devices
        poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = vr.getTrackedDeviceClass(i)
            
            # Check if the device is a HMD or a Controller
            if device_class == openvr.TrackedDeviceClass_HMD or \
               device_class == openvr.TrackedDeviceClass_Controller:
                
                pose = poses[i]
                if pose.bPoseIsValid:
                    # Get the device role (Left Hand, Right Hand, or HMD)
                    role = vr.getControllerRoleForTrackedDeviceIndex(i)
                    
                    if device_class == openvr.TrackedDeviceClass_HMD:
                        label = "HMD"
                    elif role == openvr.TrackedControllerRole_LeftHand:
                        label = "Left Controller"
                    elif role == openvr.TrackedControllerRole_RightHand:
                        label = "Right Controller"
                    else:
                        label = f"Generic Controller ({i})"

                    x, y, z = get_pose_position(pose.mDeviceToAbsoluteTracking)
                    print(f"{label:18} | X: {x:7.3f} | Y: {y:7.3f} | Z: {z:7.3f}")

        print("-" * 50) # Divider for readability
        time.sleep(0.5) # Slowed down slightly so you can read the output

except KeyboardInterrupt:
    print("\nShutting down...")
    openvr.shutdown()
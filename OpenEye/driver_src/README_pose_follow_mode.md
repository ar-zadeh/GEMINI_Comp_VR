# Physical HMD Follow Mode (Pose Modifier)

This driver can now operate in a mode where it **follows a real tracked headset pose** and then applies your incoming TCP pose as a transform.

## Important limitation

OpenVR server drivers cannot directly rewrite another vendor driver's tracked pose in-place.
This implementation works by publishing this driver's HMD pose as:

- Base pose from a physical HMD (`GetRawTrackedDevicePoses`)
- Plus optional TCP offset (`pos`, `rot`)

So it is a pose-modifier pipeline for this driver's device, not an in-place mutation of another driver's internals.

## New settings (`default.vrsettings`, section `driver_openeye_pose`)

- `followPhysicalHmd` (`bool`)
  - `false` (default): legacy behavior, TCP pose is treated as absolute pose.
  - `true`: driver tries to find a connected physical HMD and use it as base.

- `physicalHmdSerial` (`string`)
  - Empty string (default): first connected HMD (excluding this driver) is used.
  - Non-empty: only the HMD with this serial is used as the physical source.

- `tcpPoseIsOffset` (`bool`)
  - `true` (default): TCP pose is applied as offset from the physical HMD base pose.
  - `false`: TCP pose is treated as absolute pose even when `followPhysicalHmd=true`.

## Typical configuration

```json
{
  "driver_openeye_pose": {
    "enable": true,
    "followPhysicalHmd": true,
    "physicalHmdSerial": "", 
    "tcpPoseIsOffset": true,
    "tcpEnabled": true,
    "tcpHost": "127.0.0.1",
    "tcpPort": 5555
  }
}
```

## Runtime notes

- Position offset is currently applied in tracking/world axes: `finalPos = physicalPos + tcpPos`.
- Rotation offset is composed as quaternion multiply: `finalRot = physicalRot * tcpRot`.
- If no physical HMD is found while `followPhysicalHmd=true`, driver logs a fallback message and continues using TCP/last values.

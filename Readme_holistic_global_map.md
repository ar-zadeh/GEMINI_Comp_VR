# Holistic/Global Map from `template.mp4` (VGGT + SfM-style VO)

This script combines:
- **VGGT** for per-frame depth + intrinsics
- **SfM-style visual odometry** (feature matching + PnP) for camera motion
- **Depth fusion** into one global occupancy map as the camera moves


## Installation

``` bash
git clone https://github.com/ByteDance-Seed/depth-anything-3
cd depth-anything-3
pip install -e . # Basic
export TORCH_CUDA_ARCH_LIST="10.0" # this is needed if the next step fails
pip install  --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
```

if your build still failed (especially i'm testing on WSL):
```bash
export TORCH_CUDA_ARCH_LIST="10.0"
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
pip install git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 --no-build-isolation
```


## Script

- `holistic_obstacle_mapper.py`

## Quick run (uses `template.mp4` by default)

```bash
python holistic_obstacle_mapper.py --output_prefix holistic_obstacle_map_hd
```

## Explicit input

```bash
python holistic_obstacle_mapper.py \
  --input template.mp4 \
  --output_prefix holistic_obstacle_map_hd
```

## Useful tuning

```bash
python holistic_obstacle_mapper.py \
  --input template.mp4 \
  --output_prefix holistic_obstacle_map_hd \
  --frame_stride 2 \
  --max_frames 160 \
  --map_resolution_m 0.03 \
  --depth_conf_threshold 5.0 \
  --depth_min_m 0.2 \
  --depth_max_m 15.0
```

## Outputs

- `holistic_obstacle_map_hd.npz`
  - `occupancy` (top-down normalized occupancy grid)
  - `origin_xz` (world origin of grid in meters)
  - `resolution_m` (cell size in meters)
  - `trajectory_xyz` (camera centers in world frame)
  - `poses_c2w` (camera-to-world transforms)
  - `world_points_sample` (downsampled fused point cloud)

- `holistic_obstacle_map_hd.png`
  - Occupancy heatmap + start/end + camera trajectory overlay

## Notes

- First run downloads VGGT weights from Hugging Face.
- A CUDA GPU is strongly recommended.
- If too sparse, lower `--depth_conf_threshold` (for example, `3.0`).
- If map looks noisy, increase `--point_stride` (for example, `6`) or increase `--frame_stride`.

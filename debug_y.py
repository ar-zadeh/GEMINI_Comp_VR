import numpy as np
import trimesh
import sys

pc = trimesh.load("occupancy_grid_vggt_pointcloud.ply")
y = pc.vertices[:, 1]
print("Min Y:", np.min(y))
print("Max Y:", np.max(y))
print("5th perc Y:", np.percentile(y, 5))
print("95th perc Y:", np.percentile(y, 95))
print("Mean Y:", np.mean(y))

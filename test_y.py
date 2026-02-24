import trimesh
import numpy as np
pc = trimesh.load('occupancy_grid_vggt_pointcloud.ply')
pts = pc.vertices
cam_pts = pts[-1680:] # last 40*42 points
scene_pts = pts[:-1680]
print("cam mean Y:", np.mean(cam_pts[:, 1]))
print("cam min Y:", np.min(cam_pts[:, 1]), "max Y:", np.max(cam_pts[:, 1]))
print("scene 5% Y:", np.percentile(scene_pts[:, 1], 5))
print("scene 95% Y:", np.percentile(scene_pts[:, 1], 95))

import numpy as np
import open3d as o3d
import trimesh

def test_on_ply():
    pc = trimesh.load("occupancy_grid_vggt_pointcloud.ply")
    points = np.array(pc.vertices)
    
    # 1. basic RANSAC
    import time
    t0 = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # downsample
    pcd = pcd.voxel_down_sample(voxel_size=0.05) if len(points) > 50000 else pcd
    pcd = pcd.random_down_sample(50000 / len(pcd.points)) if len(pcd.points) > 50000 else pcd
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=1000)
    print(f"RANSAC found plane: {plane_model} in {time.time()-t0:.3f}s with {len(inliers)} inliers")
    
test_on_ply()

import numpy as np
import open3d as o3d
import trimesh

def classify_points_by_height(points):
    try:
        import open3d as o3d
        USE_RANSAC = True
    except ImportError:
        USE_RANSAC = False

    y_range = np.percentile(points[:, 1], 95) - np.percentile(points[:, 1], 5)
    
    if USE_RANSAC and len(points) > 100:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if len(points) > 50000:
            pcd_down = pcd.random_down_sample(50000 / len(points))
        else:
            pcd_down = pcd
            
        plane_model, _ = pcd_down.segment_plane(distance_threshold=max(0.02 * y_range, 0.05),
                                                ransac_n=3,
                                                num_iterations=1000)
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        
        if normal[1] < 0:
            normal = -normal
            d = -d
            
        if normal[1] >= 0.5:
            distances = np.dot(points, normal) + d
            # Shift plane down slightly if it fitted the middle of the floor noise
            # Instead of median, let's just say floor is dist in [-0.05*y_range, 0.05*y_range]
            
            # actually let's re-anchor the floor to the ~5th percentile of distances
            dist_5th = np.percentile(distances, 5)
            distances = distances - dist_5th
            
            floor_mask = (distances >= -0.05 * y_range) & (distances < 0.05 * y_range)
            wall_mask = (distances >= 0.05 * y_range) & (distances < 0.75 * y_range)
            print("RANSAC successful.")
            return points[floor_mask], points[wall_mask]
        else:
            print("RANSAC found a non-horizontal plane. Falling back to simple percentiles.")

    print("Using simple percentiles.")
    y = points[:, 1]
    floor_y = np.percentile(y, 5)
    floor_band_top = floor_y + 0.05 * y_range
    wall_band_top = floor_y + 0.75 * y_range
    floor_mask = (y >= floor_y) & (y < floor_band_top)
    wall_mask = (y >= floor_band_top) & (y < wall_band_top)
    return points[floor_mask], points[wall_mask]

pc = trimesh.load("occupancy_grid_vggt_pointcloud.ply")
pts = np.array(pc.vertices)
floor, wall = classify_points_by_height(pts)
print(f"Floor points: {len(floor)}, Wall points: {len(wall)}")

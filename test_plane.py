import numpy as np
import open3d as o3d

def find_ground_plane(points, dist_thresh=0.05, max_tilt_deg=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Downsample
    n_pts = len(points)
    if n_pts > 50000:
        pcd = pcd.random_down_sample(50000 / n_pts)

    best_plane = None
    best_inliers = None
    max_plane_size = 0

    # Iteratively find the largest roughly horizontal plane
    temp_pcd = pcd
    for i in range(5):
        if len(temp_pcd.points) < 1000:
            break
        plane_model, inliers = temp_pcd.segment_plane(distance_threshold=dist_thresh,
                                                      ransac_n=3,
                                                      num_iterations=1000)
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        # Check angle with Y-axis [0, 1, 0]
        # Since it can be up or down
        angle = np.arccos(np.abs(np.dot(normal, [0, 1, 0])))
        deg = np.degrees(angle)
        
        if deg < max_tilt_deg:
            if len(inliers) > max_plane_size:
                max_plane_size = len(inliers)
                best_plane = plane_model
                best_inliers = inliers
        
        # remove inliers and find next plane
        temp_pcd = temp_pcd.select_by_index(inliers, invert=True)

    return best_plane

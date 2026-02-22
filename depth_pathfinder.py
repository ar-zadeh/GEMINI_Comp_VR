import argparse
import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import heapq

# Add Depth-Anything-3 to path
sys.path.append(os.path.join(os.path.dirname(__file__), "Depth-Anything-3", "src"))
from depth_anything_3.api import DepthAnything3

def load_depth_model(device="cuda", model_id="depth-anything/DA3MONO-LARGE"):
    print(f"Loading Depth-Anything-3 model on {device}...")
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device).eval()
    print(f"Loaded pretrained model: {model_id}")
    return model

def get_depth_map(model, image_path, device="cuda"):
    print(f"Processing image: {image_path}")
    
    # Get depth map using the correct API
    with torch.no_grad():
        prediction = model.inference(
            [image_path],
            process_res=1024,
            process_res_method="lower_bound_resize",
        )
        depth = prediction.depth[0].astype(np.float32)

    print(
        f"Depth stats -> min: {float(np.min(depth)):.4f}, "
        f"max: {float(np.max(depth)):.4f}, mean: {float(np.mean(depth)):.4f}"
    )
        
    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image_rgb, depth

def create_occupancy_grid(depth_map, threshold_percentile=70, grid_size=(50, 50), crop_top_ratio=0.25):
    """
    Converts a depth map into a 2D occupancy grid (bird's eye view approximation).
    This is a simplified projection assuming the camera is looking forward.
    """
    h, w = depth_map.shape
    
    # Robust normalization to suppress outliers/noise.
    p2, p98 = np.percentile(depth_map, [2, 98])
    depth_norm = np.clip((depth_map - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)

    # Crop the bottom portion of the image (floor / path ahead).
    bottom = depth_norm[int(h * crop_top_ratio):, :]

    # --- Obstacle detection via floor-plane deviation ---
    # Estimate the floor depth at each row using the median across columns.
    # The floor is the dominant surface, so its depth ≈ the row median.
    row_medians = np.median(bottom, axis=1, keepdims=True)

    # Pixels significantly CLOSER than the floor median are obstacles
    # (furniture, walls, etc. that protrude above the floor plane).
    # depth_norm: low = close, high = far  →  (median - pixel) > 0 means closer.
    obstacle_score = np.clip(row_medians - bottom, 0, None)

    # Also add depth-gradient magnitude to capture obstacle edges.
    grad_y = cv2.Sobel(bottom, cv2.CV_64F, 0, 1, ksize=5)
    grad_x = cv2.Sobel(bottom, cv2.CV_64F, 1, 0, ksize=5)
    gradient_mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
    g_max = gradient_mag.max()
    if g_max > 0:
        gradient_mag /= g_max

    # Combine: mainly floor-deviation, boosted by edge signal.
    combined = obstacle_score + 0.3 * gradient_mag

    # Resize to our grid size
    grid_raw = cv2.resize(combined, grid_size, interpolation=cv2.INTER_AREA)
    
    # Adaptive threshold on obstacle likelihood.
    threshold = np.percentile(grid_raw, threshold_percentile)
    occupancy_grid = (grid_raw >= threshold).astype(np.uint8)

    # Denoise small speckles.
    kernel = np.ones((3, 3), np.uint8)
    occupancy_grid = cv2.morphologyEx(occupancy_grid, cv2.MORPH_OPEN, kernel)
    occupancy_grid = cv2.morphologyEx(occupancy_grid, cv2.MORPH_CLOSE, kernel)
    
    # Ensure the bottom center (where the user is) is always free space
    user_pos = (grid_size[0] - 1, grid_size[1] // 2)
    occupancy_grid[user_pos[0]-3:user_pos[0]+1, user_pos[1]-3:user_pos[1]+4] = 0
    
    return occupancy_grid, grid_raw

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_pathfinding(grid, start, goal):
    """
    A* pathfinding on a 2D grid.
    grid: 2D numpy array (0 = free, 1 = obstacle)
    start: (y, x) tuple
    goal: (y, x) tuple
    """
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1] # Reverse to get path from start to goal
            
        close_set.add(current)
        
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            
            # Check bounds
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    # Check if obstacle
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue
                
            # Diagonal movement cost is slightly higher
            cost = 1.414 if i != 0 and j != 0 else 1.0
            tentative_g_score = gscore[current] + cost
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue
                
            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    return None # No path found

def visualize_results(image, depth_map, occupancy_grid, path, output_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. Original Image
    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')
    
    # 2. Depth Map
    im = axs[0, 1].imshow(depth_map, cmap='inferno')
    axs[0, 1].set_title("Depth Map (Depth-Anything-3)")
    axs[0, 1].axis('off')
    plt.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Occupancy Grid
    axs[1, 0].imshow(occupancy_grid, cmap='Greys')
    axs[1, 0].set_title("2D Occupancy Grid (Top-Down)")
    
    # 4. Path on Grid
    axs[1, 1].imshow(occupancy_grid, cmap='Greys')
    axs[1, 1].set_title("A* Path to Goal")
    
    if path:
        path_y = [p[0] for p in path]
        path_x = [p[1] for p in path]
        axs[1, 1].plot(path_x, path_y, 'r-', linewidth=2)
        axs[1, 1].plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')
        axs[1, 1].plot(path_x[-1], path_y[-1], 'bo', markersize=8, label='Goal')
        axs[1, 1].legend()
    else:
        axs[1, 1].text(occupancy_grid.shape[1]//2, occupancy_grid.shape[0]//2, 
                      "NO PATH FOUND", color='red', ha='center', va='center', 
                      fontsize=14, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Depth-based Pathfinding")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="pathfinding_result.png", help="Path to save visualization")
    parser.add_argument("--goal_x", type=float, default=0.5, help="Goal X position (0.0 to 1.0, left to right)")
    parser.add_argument("--goal_y", type=float, default=0.1, help="Goal Y position (0.0 to 1.0, top to bottom of grid)")
    parser.add_argument("--model_id", type=str, default="depth-anything/DA3MONO-LARGE", help="DepthAnything3 model id")
    parser.add_argument("--crop_top_ratio", type=float, default=0.25, help="Top crop ratio for occupancy projection")
    parser.add_argument(
        "--goal_bbox",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Goal 2D bounding box in image pixels; path target becomes bbox center",
    )
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 1. Load Model
        model = load_depth_model(device, model_id=args.model_id)
        
        # 2. Get Depth Map
        image_rgb, depth_map = get_depth_map(model, args.image, device)
        
        # 3. Create Occupancy Grid
        grid_size = (50, 50)
        occupancy_grid, _ = create_occupancy_grid(
            depth_map,
            threshold_percentile=82,
            grid_size=grid_size,
            crop_top_ratio=args.crop_top_ratio,
        )
        
        # 4. Pathfinding
        # Start is bottom center (user position)
        start = (grid_size[0] - 1, grid_size[1] // 2)

        # Goal can come from bbox center in image-space or normalized grid coordinates.
        if args.goal_bbox is not None:
            x1, y1, x2, y2 = args.goal_bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            src_h, src_w = image_rgb.shape[:2]
            depth_h, depth_w = depth_map.shape

            depth_cx = cx * (depth_w / max(1, src_w))
            depth_cy = cy * (depth_h / max(1, src_h))

            crop_top = int(depth_h * args.crop_top_ratio)

            goal_x_idx = int(np.clip(depth_cx / max(1, depth_w - 1) * (grid_size[1] - 1), 0, grid_size[1] - 1))
            goal_y_ratio = (depth_cy - crop_top) / max(1, (depth_h - crop_top - 1))
            goal_y_idx = int(np.clip(goal_y_ratio * (grid_size[0] - 1), 0, grid_size[0] - 1))

            print(
                f"Using bbox goal center ({cx:.1f}, {cy:.1f}) -> depth ({depth_cx:.1f}, {depth_cy:.1f}) -> grid goal ({goal_y_idx}, {goal_x_idx})"
            )
        else:
            # Goal is based on normalized arguments (default is top center)
            goal_x_idx = int(args.goal_x * (grid_size[1] - 1))
            goal_y_idx = int(args.goal_y * (grid_size[0] - 1))

        goal = (goal_y_idx, goal_x_idx)
        
        threshold_trials = [70, 74, 78, 82, 86, 90, 94, 98]
        path = None
        chosen_threshold = None

        for th in threshold_trials:
            occupancy_grid, _ = create_occupancy_grid(
                depth_map,
                threshold_percentile=th,
                grid_size=grid_size,
                crop_top_ratio=args.crop_top_ratio,
            )

            current_goal = goal
            if occupancy_grid[current_goal[0], current_goal[1]] == 1:
                min_dist = float('inf')
                best_goal = current_goal
                for y in range(grid_size[0]):
                    for x in range(grid_size[1]):
                        if occupancy_grid[y, x] == 0:
                            dist = heuristic((y, x), current_goal)
                            if dist < min_dist:
                                min_dist = dist
                                best_goal = (y, x)
                current_goal = best_goal

            path = astar_pathfinding(occupancy_grid, start, current_goal)
            if path is not None:
                goal = current_goal
                chosen_threshold = th
                print(f"Path found with occupancy threshold percentile {th}.")
                break

        if path is None:
            print("No path after adaptive threshold search.")
        
        if path:
            print(f"Found path with {len(path)} steps.")
            if chosen_threshold is not None:
                print(f"Using occupancy threshold percentile: {chosen_threshold}")
            # Generate a simple text instruction based on the first few steps
            if len(path) > 5:
                dx = path[5][1] - start[1]
                if dx < -2:
                    instruction = "Turn left and move forward."
                elif dx > 2:
                    instruction = "Turn right and move forward."
                else:
                    instruction = "Move straight forward."
                print(f"Navigation Instruction: {instruction}")
        else:
            print("No safe path found to the goal!")
            
        # 5. Visualize
        visualize_results(image_rgb, depth_map, occupancy_grid, path, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

import numpy as np
from PIL import Image
import random
import math
import os

def generate_map(filename, width=100, height=100, num_obstacles=12, obstacle_max_size=20):
    # Initialize a white map (navigable space)
    # 255 is white
    grid = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add random rectangular obstacles (black)
    for _ in range(num_obstacles):
        ow = random.randint(5, obstacle_max_size)
        oh = random.randint(5, obstacle_max_size)
        ox = random.randint(0, width - ow - 1)
        oy = random.randint(0, height - oh - 1)
        
        # Black color
        grid[oy:oy+oh, ox:ox+ow] = [0, 0, 0]
        
    # Helper to find valid empty spot (white space)
    def find_empty():
        while True:
            x = random.randint(3, width-4)
            y = random.randint(3, height-4)
            # Check if an area around it is white (to ensure it doesn't overlap completely with walls)
            if np.all(grid[y-2:y+3, x-2:x+3] == [255, 255, 255]):
                return x, y
                
    # Initial position (green)
    start_x, start_y = find_empty()
    # Draw a 5x5 square for better visibility
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            grid[start_y + dy, start_x + dx] = [0, 255, 0] # Green
    
    # Destination (red)
    goal_x, goal_y = find_empty()
    # Ensure some distance between start and goal
    while math.hypot(goal_x - start_x, goal_y - start_y) < min(width, height) / 3:
        goal_x, goal_y = find_empty()
        
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            grid[goal_y + dy, goal_x + dx] = [255, 0, 0] # Red
                
    img = Image.fromarray(grid)
    img.save(filename)
    print(f"Generated and saved: {filename}")

if __name__ == '__main__':
    output_dir = "sample_maps"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating 3 random maps...")
    for i in range(3):
        output_file = os.path.join(output_dir, f"map_{i+1}.png")
        generate_map(output_file)
    print("Map generation complete.")

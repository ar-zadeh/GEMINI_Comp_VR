import cv2
import os
import re
import glob

def create_videos_from_images(image_folder, output_folder=None, fps=10):
    if output_folder is None:
        output_folder = image_folder

    # Get all .jpg files
    files = glob.glob(os.path.join(image_folder, "*.jpg"))
    
    # Parse filenames: servo_<timestamp>_iter_<iteration>.jpg
    # We want to group them into sessions. 
    # Assumption: A new session usually starts with iter_0, or we can just group by sorting and cutting when iter resets.
    
    parsed_files = []
    pattern = re.compile(r"servo_(\d+)_iter_(\d+)\.jpg")
    
    for f in files:
        basename = os.path.basename(f)
        match = pattern.match(basename)
        if match:
            timestamp = int(match.group(1))
            iteration = int(match.group(2))
            parsed_files.append({
                'path': f,
                'filename': basename,
                'timestamp': timestamp,
                'iteration': iteration
            })
    
    # Sort by timestamp
    parsed_files.sort(key=lambda x: x['timestamp'])
    
    # Group into sessions
    sessions = []
    current_session = []
    
    for img in parsed_files:
        if img['iteration'] == 0:
            if current_session:
                sessions.append(current_session)
            current_session = [img]
        else:
            # If we have a current session, append. 
            # If for some reason we start with non-zero (files deleted?), we create a session or append if it makes sense.
            # But relying on timestamp continuity is header.
            # Simple heuristic: if current_session is empty, start a new one (even if not iter 0, though unlikely with this logic if we want strictly iter 0 start)
            if not current_session:
                 current_session = [img]
            else:
                 current_session.append(img)
                 
    if current_session:
        sessions.append(current_session)
        
    print(f"Found {len(sessions)} sessions.")
    
    for i, session in enumerate(sessions):
        if not session:
            continue
            
        first_ts = session[0]['timestamp']
        first_img_path = session[0]['path']
        
        # Read first image to get dimensions
        frame = cv2.imread(first_img_path)
        if frame is None:
            print(f"Failed to read {first_img_path}, skipping session.")
            continue
            
        height, width, layers = frame.shape
        
        # Output filename
        video_name = os.path.join(output_folder, f"session_{first_ts}.mp4")
        print(f"Creating {video_name} with {len(session)} frames...")
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1' or 'XVID'
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        
        for img_info in session:
            frame = cv2.imread(img_info['path'])
            if frame is not None:
                video.write(frame)
            else:
                print(f"Warning: Could not read frame {img_info['path']}")
                
        video.release()
        print(f"Finished {video_name}")

if __name__ == "__main__":
    # Target folder relative to this script or absolute
    target_dir = os.path.join(os.path.dirname(__file__), "agent_logs_v2", "tracking")
    
    # If the script is run from OpenEye/, the path should be correct. 
    # Just in case, checking absolute vs relative.
    if not os.path.exists(target_dir):
        # bold assumption of the path, let's try the one provided by user context if it fails?
        # User path: f:\GEMINI_Comp_VR\OpenEye\agent_logs_v2\tracking
        # Script location: f:\GEMINI_Comp_VR\OpenEye\create_tracking_videos.py
        # So os.path.join(os.path.dirname(__file__), "agent_logs_v2", "tracking") is correct.
        print(f"Directory not found: {target_dir}")
    else:
        create_videos_from_images(target_dir, fps=10)

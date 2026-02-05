
import sys
import os

# Add directory to path
sys.path.append("/home/amir/gemini_project_VR/GEMINI_Comp_VR/OpenEye")

try:
    print("Importing gemini_vr_agent_v2...")
    import gemini_vr_agent_v2
    print("Import successful.")

    # Check VisualGrounder internal class
    # Since they are defined inside methods, we can't easily access them without running the method.
    # But if the file imported, syntax is valid.
    
    print("Syntax check passed.")

except ImportError as e:
    print(f"ImportError: {e}")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
except Exception as e:
    print(f"Error: {e}")

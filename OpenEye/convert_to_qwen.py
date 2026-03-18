import os
import glob
import re

package_dir = 'OpenEye/vr_agent_qwen'
if not os.path.exists(package_dir):
    os.system('cp -r OpenEye/vr_agent OpenEye/vr_agent_qwen')
    
os.system('sed -e "s/gemini_vr_agent_v9\\.py/qwen_vr_agent.py/g" -e "s/GeminiAgent/QwenAgent/g" -e "s/vr_agent/vr_agent_qwen/g" OpenEye/gemini_vr_agent_v9.py > OpenEye/qwen_vr_agent.py')
os.system(f'sed -i "s/GeminiAgent/QwenAgent/g" {package_dir}/*.py')
os.system(f'sed -i "s/gemini_api_key/qwen_api_key/g" {package_dir}/*.py')
os.system(f'sed -i "s/GEMINI_API_KEY/QWEN_API_KEY/g" {package_dir}/*.py')

def replace_agent():
    file_path = f'{package_dir}/agent.py'
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace genai import with openai
    content = content.replace('from google import genai', 'from openai import OpenAI')
    
    # Replace client initialization
    old_client = """        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1alpha"}
        )"""
    new_client = """        self.client = OpenAI(
            base_url="https://zippy-sarita-flabbier.ngrok-free.dev/v1", # Replace with your URL
            api_key="sk-no-key-required",
            default_headers={"ngrok-skip-browser-warning": "true"} # Bypasses the HTML warning
        )"""
    content = content.replace(old_client, new_client)
    
    with open(file_path, 'w') as f:
        f.write(content)

replace_agent()

print("Initial copy and agent.py setup done. Ready for generation call replacements.")

#!/bin/bash

# Configuration
MODELS=("Firworks/Cosmos-Reason2-8B-nvfp4") # "nvidia/Cosmos-Reason2-2B" "nvidia/Cosmos-Reason2-8b" "Firworks/Cosmos-Reason2-8B-nvfp4"
PROMPTS_FILE="prompts.yaml"
PLANNING_SCRIPT="001.basic_planning.py"

# Activate environment
eval "$(mamba shell hook --shell bash)"
mamba activate gemini_vr

echo "Starting experiments..."

# Use python to parse YAML and run the trials
# This avoids needing yq installed
python3 <<EOF
import yaml
import subprocess
import os

with open("$PROMPTS_FILE", 'r') as f:
    config = yaml.safe_load(f)

models = ["${MODELS[@]}"]
prompts = config.get('prompts', [])

for model in models:
    for p in prompts:
        p_id = p.get('id')
        p_text = p.get('prompt')
        
        print(f"Executing: Model={model}, PromptID={p_id}")
        
        cmd = [
            "python3", "$PLANNING_SCRIPT",
            "--model", model,
            "--prompt", p_text,
            "--prompt_id", p_id,
            "--img_path", "../occupancy_grid.png"
        ]
        
        # Run the command and wait
        subprocess.run(cmd)

print("All experiments completed.")
EOF

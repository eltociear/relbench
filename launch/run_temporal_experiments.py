import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default=None, required=True)
parser.add_argument("--script", type=str, default="examples/gnn_node_temporal_feat.py")
args = parser.parse_args()


# Define your configurations


configs = [
    {#"dataset": "rel-f1", "task": "driver-position", "batch-size": 256, "lr": 0.005},
    "dataset": "rel-f1", "task": "driver-dnf", "batch-size": 256, "lr": 0.005},
    #{"dataset": "rel-f1", "task": "driver-top3", "batch-size": 256},
    # Add more configurations as needed
]
"""

configs = [
    {"dataset": "rel-f1", "task": "driver-top3", "use-ar-label": True},
    {"dataset": "rel-f1", "task": "driver-dnf", "use-ar-label": True},
    {"dataset": "rel-f1", "task": "driver-position", "use-ar-label": True},
    # {"dataset": "rel-f1", "task": "driver-top3", "batch_size": 256},
    # Add more configurations as needed
]
"""
num_seeds = 5
# add seeds to configs
configs = [
    {**config, "seed": seed}
    for config in configs
    for seed in range(num_seeds)
]

def run_script(config, log_dir, script):
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Construct the command
    command = ["python", script]
    for arg, value in config.items():
        command.append(f"--{arg}")
        if isinstance(value, bool):
            pass
        else:
            command.append(str(value))
    
    # Create a log file path with all config values
    log_file = os.path.join(log_dir, "_".join([f"{k}_{v}" for k, v in config.items()]) + ".log")

    # Run the script and save output to log file
    with open(log_file, "w") as log:
        subprocess.run(command, stdout=log, stderr=log, text=True)
    
    print(f"Run completed: {config} (log saved to {log_file})")

# Directory to save logs
log_directory = os.path.join("results", "temporal-gnn", args.experiment_name)
# make directory if it doesn't exist
os.makedirs(log_directory, exist_ok=True)

for config in configs:
    run_script(config, log_directory, args.script)

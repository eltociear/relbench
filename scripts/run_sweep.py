import subprocess
import argparse
import time

def run_experiment(dataset, task, seed):
    # Get current Unix timestamp
    timestamp = int(time.time())

    # Construct filename with dataset, task, run number, and timestamp
    filename = f"results/sweep/{dataset}_{task}_run{seed}_{timestamp}.log"

    # Build the command list
    command = [
        "python",
        "examples/lightgbm_gnn_features_node.py",
        "--dataset",
        dataset,
        "--task",
        task,
        f"--seed={seed}",
    ]

    # Run the command and capture output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()

    # Decode output and error messages
    output = output.decode("utf-8")
    err = err.decode("utf-8")

    # Write output to the file
    with open(filename, "w") as f:
        f.write(output)

    # Handle potential errors
    if err:
        print(f"Error running experiment: {dataset}, {task}, {seed}")
        print(f"Error message: {err}")
    else:
        print(f"Results saved to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=1, help="Number of times to run each experiment")
    args = parser.parse_args()

    data_task_pairs = [#('rel-trial', 'study-adverse'),
                       #('rel-trial', 'site-success')]
                        #('rel-amazon', 'user-churn'),    
                        ##('rel-amazon', 'user-ltv'),
                        #('rel-amazon', 'item-churn'),    
                        #('rel-amazon', 'item-ltv')]
                        ('rel-hm', 'item-sales')]
    
                       

    seeds = [42 + i for i in range(args.num_runs)]

    for dataset, task in data_task_pairs:
        for seed in seeds:
            run_experiment(dataset, task, seed)

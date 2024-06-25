import os
import pandas as pd
import argparse
import ast

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if line.startswith(args.line):
            best_metrics = line.split(args.line)[1].strip()
            return best_metrics
    return None

def parse_filename(filename):
    parts = filename.replace(".log", "").split('_')
    params = {}
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            params[parts[i]] = parts[i + 1]
    return params

def main(args):
    results_dir = args.results_dir
    data = []

    for filename in os.listdir(results_dir):
        if filename.endswith(".log"):
            file_path = os.path.join(results_dir, filename)
            best_metrics = parse_log_file(file_path)
            if best_metrics is not None:
                params = parse_filename(filename)
                params["Best test metrics"] = best_metrics
                # Convert string to dictionary
                best_metrics = ast.literal_eval(best_metrics)
                
                params["mae"] = best_metrics.get("mae", None)
                params["roc_auc"] = best_metrics.get("roc_auc", None)
                data.append(params)
            else:
                params = parse_filename(filename)
                params["Best test metrics"] = None
                params["mae"] = None
                params["roc_auc"] = None
                data.append(params)

    df = pd.DataFrame(data)
    print("Summary of results:")
    print(df)

    if args.save_csv:
        output_file = os.path.join(results_dir, "results_summary.csv")
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    if args.average:
        # keep only dataset, task and numeric columns
        df = df[["dataset", "task", "mae", "roc_auc"]]
        df_grouped = df.groupby(["dataset", "task"]).mean()

        # do std dev and count also
        df_grouped["mae_std"] = df.groupby(["dataset", "task"])["mae"].std()
        df_grouped["roc_auc_std"] = df.groupby(["dataset", "task"])["roc_auc"].std()

        df_grouped["mae_count"] = df.groupby(["dataset", "task"])["mae"].count()
        df_grouped["roc_auc_count"] = df.groupby(["dataset", "task"])["roc_auc"].count()

        print("\nAverage metrics over dataset and task pairs:")
        print(df_grouped)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metrics from log files and summarize them in a DataFrame.")
    parser.add_argument("--results_dir", type=str, help="Directory containing the log files.")
    parser.add_argument("--save_csv", action="store_true", help="Save the results to a CSV file.")
    parser.add_argument("--line", type=str, default="Best test metrics: ")
    parser.add_argument("--average", action="store_true", help="Average metrics over dataset and task pairs.")
    args = parser.parse_args()

    main(args)

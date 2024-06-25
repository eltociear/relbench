import pandas as pd
import os
import re


def collect_metrics(log_dir):
    data = []
    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):  # Adjust based on log file extension
            match = re.search(r"(.*?)_(.*?)_run(\d+)_(\d+)\.log", filename)
            if match:
                dataset, task, run, timestamp = match.groups()
                # Convert timestamp to datetime (optional)
                # datetime_str = datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
                with open(os.path.join(log_dir, filename), "r") as f:
                    for line in f:
                        if line.startswith("Test:") or line.startswith("Val:"):
                            if line.startswith("Test:"):
                                metrics_str = line.split("Test: ")[-1].strip()
                                split = 'Test'
                            else:
                                metrics_str = line.split("Val: ")[-1].strip()
                                split = 'Val'
                            metrics = eval(metrics_str)
                            data.append({
                                "dataset": dataset,
                                "task": task,
                                "run": int(run),
                                "timestamp": int(timestamp),
                                "split": split,
                                'metrics': metrics
                            })
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Replace with the directory containing your experiment logs
    log_dir = "results/sweep"
    df = collect_metrics(log_dir)

    # Filter only the rows where split is 'Test'
    df_test = df[df['split'] == 'Test']

    # Extract mae and auroc (if available) from the 'results' dictionary
    df_test['mae'] = df_test['metrics'].apply(lambda x: x.get('mae'))
    df_test['auroc'] = df_test['metrics'].apply(lambda x: x.get('roc_auc', None))

    # Group by 'task' and calculate mean and std dev for mae and auroc
    result = df_test.groupby('task').agg(
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        auroc_mean=('auroc', 'mean'),
        auroc_std=('auroc', 'std')
    ).reset_index()

    # Drop columns with all NaN values (e.g., auroc if it's not applicable)
    result = result.dropna(axis=1, how='all')

    print(result)
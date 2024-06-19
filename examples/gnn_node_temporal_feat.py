import argparse
import copy
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from inferred_stypes import dataset2inferred_stypes
from model_temporal import TemporalModel as Model
from text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.data import HeteroData
from temporal_loader.neighbor_loader import TemporalNeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.data import NodeTask, RelBenchDataset
from relbench.data.task_base import TaskType
from relbench.datasets import get_dataset
from relbench.external.graph import get_temporal_node_train_table_input, make_pkey_fkey_graph

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stackex")
parser.add_argument("--task", type=str, default="rel-stackex-engage")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_ar", type=int, default=3)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--process", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

root_dir = "./data"

# TODO: remove process=True once correct data/task is uploaded.
dataset: RelBenchDataset = get_dataset(name=args.dataset, process=args.process)
task: NodeTask = dataset.get_task(args.task, process=True)

col_to_stype_dict = dataset2inferred_stypes[args.dataset]

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=os.path.join(root_dir, f"{args.dataset}_materialized_cache"),
)

train_table = task.train_table
val_table = task.val_table
task.test_table # justto compute ground truths
test_table = task._full_test_table

ar_label_cols = []
### Adding AR labels into train/val/test_table
whole_df = pd.concat([train_table.df, val_table.df, test_table.df], axis=0)
num_ar_labels = args.num_ar #max(train_table.df[train_table.time_col].nunique() - 2, 1)

sorted_unique_times = np.sort(whole_df[train_table.time_col].unique())
timedelta = sorted_unique_times[1:] - sorted_unique_times[:-1]
TIME_IDX_COL = "time_idx"
time_df = pd.DataFrame(
    {
        task.time_col: sorted_unique_times,
        "time_idx": np.arange(len(sorted_unique_times)),
    }
)

whole_df = whole_df.merge(time_df, how="left", on=task.time_col)
whole_df.drop(task.time_col, axis=1, inplace=True)
# Shift timestamp of whole_df iteratively and join it with train/val/test_table
for i in range(1, num_ar_labels + 1):
    whole_df_shifted = whole_df.copy(deep=True)
    # Shift time index by i
    whole_df_shifted[TIME_IDX_COL] += i
    # Map time index back to datetime timestamp
    whole_df_shifted = whole_df_shifted.merge(time_df, how="inner", on=TIME_IDX_COL)
    whole_df_shifted.drop(TIME_IDX_COL, axis=1, inplace=True)
    ar_label = f"AR_{i}"
    ar_label_cols.append(ar_label)
    whole_df_shifted.rename(columns={task.target_col: ar_label}, inplace=True)

    for table in [train_table, val_table, test_table]:
        table.df = table.df.merge(
            whole_df_shifted, how="left", on=(task.entity_col, task.time_col)
        )


loader_dict: Dict[str, TemporalNeighborLoader] = {}
for split, table in [
    ("train", train_table),
    ("val", val_table),
    ("test", test_table),
]:
    table_input = get_temporal_node_train_table_input(table=table, task=task, num_ar=args.num_ar)
    entity_table = table_input.nodes[0]
    loader_dict[split] = TemporalNeighborLoader(
        data,
        num_neighbors=[
            int(args.num_neighbors / 2**i) for i in range(args.num_layers)
        ],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        previous_times=table_input.previous_times,
        transform=table_input.transform,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    # Get the clamp value at inference time
    clamp_min, clamp_max = np.percentile(
        task.train_table.df[task.target_col].to_numpy(), [2, 98]
    )

model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",
    task_type=task.task_type,
    num_ar=args.num_ar,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )


        """
        loss = 0.
        # stack pred and y
        for key in pred.keys():
            pred[key] = pred[key].view(-1) if pred[key].size(1) == 1 else pred[key]
            target = batch[key][entity_table].y

            # get mask of nans in target
            mask = torch.isnan(target)

            loss += loss_fn(pred[key][~mask], target[~mask])
        """
        pred = pred['root'].view(-1) if pred['root'].size(1) == 1 else pred['root']
        loss = loss_fn(pred, batch['root'][entity_table].y)

        #pred = pred.view(-1) if pred.size(1) == 1 else pred
        #loss = loss_fn(pred, batch['root'][entity_table].y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: TemporalNeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(
            batch,
            task.entity_table,
        )

        pred = pred['root']

        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


state_dict = None
best_val_metric = 0 if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.val_table)
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

    if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
        not higher_is_better and val_metrics[tune_metric] < best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())

model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.val_table)
print(f"Best Val metrics: {val_metrics}")


test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")

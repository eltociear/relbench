import argparse
import copy
import math
import time
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from inferred_stypes import dataset2inferred_stypes
from model_structural import Model
from text_embedder import GloveTextEmbedding
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.data import LinkTask, RelBenchDataset
from relbench.data.task_base import TaskType
from relbench.datasets import get_dataset
from relbench.external.graph import get_link_train_table_input, make_pkey_fkey_graph, merge_batch
from relbench.external.loader import LinkNeighborLoader

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--task", type=str, default="rel-hm-rec")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--num_neighbors", type=int, default=160)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
# Use the same seed time across the mini-batch and share the negatives
parser.add_argument("--share_same_time", action="store_true")
# Whether to use shallow embedding on dst nodes or not.
parser.add_argument("--use_shallow", action="store_true")
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--log_dir", type=str, default="link_results")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--freeze_base", action="store_true")


# method specific arguments
parser.add_argument("--num_breakings", type=int, default=10)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

root_dir = "./data"

assert args.share_same_time # Unified model assumes this is True

# TODO: remove process=True once correct data/task is uploaded.
dataset: RelBenchDataset = get_dataset(name=args.dataset)
task: LinkTask = dataset.get_task(args.task, process=True)
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

col_to_stype_dict = dataset2inferred_stypes[args.dataset]


data, col_stats_dict = make_pkey_fkey_graph(
    dataset.db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=os.path.join(root_dir, f"{args.dataset}_materialized_cache"),
)

num_neighbors = [int(args.num_neighbors // 2**i) for i in range(args.num_layers)]

train_table_input = get_link_train_table_input(task.train_table, task)
train_loader = LinkNeighborLoader(
    data=data,
    num_neighbors=num_neighbors,
    time_attr="time",
    src_nodes=train_table_input.src_nodes,
    dst_nodes=train_table_input.dst_nodes,
    num_dst_nodes=train_table_input.num_dst_nodes,
    src_time=train_table_input.src_time,
    share_same_time=args.share_same_time,
    batch_size=args.batch_size,
    temporal_strategy=args.temporal_strategy,
    # if share_same_time is True, we use sampler, so shuffle must be set False
    shuffle=not args.share_same_time,
    num_workers=args.num_workers,
)

eval_loaders_dict: Dict[str, tuple[NeighborLoader, NeighborLoader]] = {}
for split in ["val", "test"]:
    seed_time = task.val_seed_time if split == "val" else task.test_seed_time
    target_table = task.val_table if split == "val" else task.test_table
    src_node_indices = torch.from_numpy(target_table.df[task.src_entity_col].values)
    src_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=(task.src_entity_table, src_node_indices),
        input_time=torch.full(
            size=(len(src_node_indices),), fill_value=seed_time, dtype=torch.long
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    dst_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=task.dst_entity_table,
        input_time=torch.full(
            size=(task.num_dst_nodes,), fill_value=seed_time, dtype=torch.long
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    eval_loaders_dict[split] = (src_loader, dst_loader)

# load model from checkpoint 
if args.checkpoint is not None:
    print(f"Loading weights from {args.checkpoint}")
    unified_state_dict = torch.load(os.path.join(args.checkpoint, "model.pth"))
    model_args =  torch.load(os.path.join(args.checkpoint, "args.pth"))
else:
    model_args = args
    model_args.out_channels = args.channels
    


unified_model = Model(
                data=data,
                col_stats_dict=col_stats_dict,
                num_layers=model_args.num_layers,
                channels=model_args.channels,
                out_channels=model_args.channels,
                aggr=model_args.aggr,
                norm="layer_norm",
                shallow_list=[task.dst_entity_table] if args.use_shallow else [],
                conv_type="unified",
                num_breakings=args.num_breakings,
            ).to(device)

if args.checkpoint is not None:
    unified_model.load_state_dict(unified_state_dict, strict=False)
    # freeze all parameters except the head
    if args.freeze_base:
        for name, param in unified_model.named_parameters():
            unfrozen = ['head'] #, 'rho', 'breaking_projection']
            if not any([x in name for x in unfrozen]):   
                param.requires_grad = False

optimizer = torch.optim.Adam(unified_model.parameters(), lr=args.lr)


def train() -> Dict[str, float]:
    unified_model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(train_loader), args.max_steps_per_epoch)
    for batch in tqdm(train_loader, total=total_steps):
        breakpoint()
        """
        src_batch, batch_pos_dst, batch_neg_dst = batch
        src_batch, batch_pos_dst, batch_neg_dst = (
            src_batch.to(device),
            batch_pos_dst.to(device),
            batch_neg_dst.to(device),
        )


        x_src = unified_model(src_batch, task.src_entity_table)

        # merge dst to avoid model learning batch shortcut features
        dst_batch = merge_batch(batch_pos_dst, batch_neg_dst,  col_stats_dict.keys())
        x_dst = unified_model(dst_batch, task.dst_entity_table)
        
        # split back to pos and neg
        bs = x_src.size(0)
        x_pos_dst = x_dst[:bs]
        x_neg_dst = x_dst[bs:]

        # [batch_size, ]
        pos_score = torch.sum(x_src * x_pos_dst, dim=1)
        #pos_score = unified_model.scorer(x_src, x_pos_dst, all_to_all=False)
        if args.share_same_time:
            # [batch_size, batch_size]
            neg_score = x_src @ x_neg_dst.t()
            #neg_score = unified_model.scorer(x_src, x_neg_dst)

            # [batch_size, 1]
            pos_score = pos_score.view(-1, 1)
        else:
            # [batch_size, ]
            neg_score = torch.sum(x_src * x_neg_dst, dim=1)
            #neg_score = unified_model.scorer(x_src, x_neg_dst, all_to_all=False)
        optimizer.zero_grad()
        # BPR loss        
        diff_score = pos_score - neg_score
        loss = F.softplus(-diff_score).mean()
        loss.backward()

        optimizer.step()

        loss_accum += loss.item() * x_src.size(0)
        count_accum += x_src.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break
        """
    return 0. #loss_accum / count_accum


@torch.no_grad()
def test(src_loader: NeighborLoader, dst_loader: NeighborLoader) -> np.ndarray:
    unified_model.eval()

    dst_embs: list[Tensor] = []
    for batch in tqdm(dst_loader):
        batch = batch.to(device)
        emb = unified_model(batch, task.dst_entity_table).detach()
        dst_embs.append(emb)
    dst_emb = torch.cat(dst_embs, dim=0)
    del dst_embs

    pred_index_mat_list: list[Tensor] = []
    for batch in tqdm(src_loader):
        batch = batch.to(device)
        emb = unified_model(batch, task.src_entity_table).detach()
        score = emb @ dst_emb.t()
        #score = unified_model.scorer(emb, dst_emb)
        _, pred_index_mat = torch.topk(score, k=task.eval_k, dim=1)
        pred_index_mat_list.append(pred_index_mat.cpu())
    pred = torch.cat(pred_index_mat_list, dim=0).numpy()
    return pred


writer = SummaryWriter(log_dir=args.log_dir)

state_dict = None
best_val_metric = 0
if not args.eval_only:
    for epoch in range(1, args.epochs + 1):
        train_loss = train()
        if epoch % args.eval_epochs_interval == 0:
            val_pred = test(*eval_loaders_dict["val"])
            val_metrics = task.evaluate(val_pred, task.val_table)
            print(
                f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
                f"Val metrics: {val_metrics}"
            )

            if val_metrics[tune_metric] > best_val_metric:
                best_val_metric = val_metrics[tune_metric]
                state_dict = copy.deepcopy(unified_model.state_dict())

            writer.add_scalar("train/loss", train_loss, epoch)
            for name, metric in val_metrics.items():
                writer.add_scalar(f"val/{name}", metric, epoch)

    unified_model.load_state_dict(state_dict)

val_pred = test(*eval_loaders_dict["val"])
val_metrics = task.evaluate(val_pred, task.val_table)
print(f"Best Val metrics: {val_metrics}")

test_pred = test(*eval_loaders_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")

for name, metric in test_metrics.items():
    writer.add_scalar(f"test/{name}", metric, 0)

if args.checkpoint is None:
    model_dir = f"unified_v2_{args.dataset}_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(os.path.join(args.log_dir, model_dir), exist_ok=True)
    torch.save(state_dict, os.path.join(args.log_dir, model_dir, "model.pth"))
    torch.save(args, os.path.join(args.log_dir, model_dir, "args.pth"))


writer.flush()
writer.close()

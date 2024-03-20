import argparse
import logging
import sys
import os

import torch
import numpy as np
from tqdm import tqdm
from torch_geometric import seed_everything

sys.path.append("./")

from relbench.data.task_base import TaskType
from relgym.config import cfg, dump_cfg, load_cfg, set_out_dir, set_run_dir
from relgym.loader import create_loader, create_dataset_and_task, transform_dataset_to_graph
from relgym.models.model_builder import create_model
from relgym.loss import create_loss_fn
from relgym.utils.comp_budget import params_count
from relgym.utils.device import auto_select_device
from relgym.utils.checkpoint import load_ckpt


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description="RelGym")

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        type=str,
        required=True,
        help="The configuration file path.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="The number of repeated jobs."
    )
    parser.add_argument(
        "--mark_done",
        action="store_true",
        help="Mark yaml as done after a job has finished.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="See graphgym/config.py for remaining options.",
    )
    parser.add_argument(
        "--auto_select_device",
        action="store_true",
        help="Automatically select gpu for training.",
    )

    return parser.parse_args()


@torch.no_grad()
def eval_epoch_pred_node(
    loader_dict, model, task, loss_fn, loss_utils, split="val"
):
    model.eval()

    entity_table = task.entity_table

    pred_list = []
    for batch in tqdm(loader_dict[split]):
        batch = batch.to(cfg.device)

        pred = model(
            batch,
            entity_table,
        )

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = torch.sigmoid(pred)
        elif task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, loss_utils["clamp_min"], loss_utils["clamp_max"])

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    all_pred = torch.cat(pred_list, dim=0).numpy()
    return all_pred


if __name__ == "__main__":
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    # set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    # torch.set_num_threads(cfg.num_threads)
    # dump_cfg(cfg)
    if args.auto_select_device:
        auto_select_device()
    else:
        cfg.device = "cuda"
    # Load dataset
    dataset, task = create_dataset_and_task()
    data, col_stats_dict = transform_dataset_to_graph(dataset)

    # Set machine learning pipeline
    loader_dict = create_loader(data, task)
    model = create_model(
        data=data,
        col_stats_dict=col_stats_dict,
        task=task,
        to_device=cfg.device,
        shallow_list=[
            task.dst_entity_table
        ] if cfg.model.use_shallow and task.task_type == TaskType.LINK_PREDICTION else [],
    )
    loss_fn, loss_utils = create_loss_fn(task)

    # Repeat for different random seeds
    splits = ["val", "test"]
    pred_across_runs = {_: [] for _ in splits}
    for i in range(args.repeat):
        cfg.run_dir = os.path.join(cfg.out_dir, str(cfg.seed))
        load_ckpt(model, best=True)
        # setup_printing()
        # Set configurations for each run
        # seed_everything(cfg.seed)
        # Print model info
        # logging.info(model)
        # logging.info(cfg)
        cfg.params = params_count(model)
        # logging.info("Num parameters: %s", cfg.params)

        for split in ["val", "test"]:
            all_pred = eval_epoch_pred_node(loader_dict, model, task, loss_fn, loss_utils, split=split)
            if split == "val":
                metrics = task.evaluate(all_pred, task.val_table)
            elif split == "test":
                metrics = task.evaluate(all_pred)
            else:
                raise RuntimeError(f"split should be val or test, got {split}")
            print(f"seed {cfg.seed}, split {split}")
            print(metrics)
            pred_across_runs[split].append(all_pred)

        # logging.info(f"Complete trial {i}")
        print(f"Complete trial {i}")
        cfg.seed = cfg.seed + 1

    # Do ensemble here
    print('Ensemble:')
    for split in ["val", "test"]:
        pred_ensemble = np.stack(pred_across_runs[split], axis=0)  # [n_runs, ...]
        pred_ensemble = np.mean(pred_ensemble, axis=0)
        if split == "val":
            metrics = task.evaluate(pred_ensemble, task.val_table)
        elif split == "test":
            metrics = task.evaluate(pred_ensemble)
        else:
            raise RuntimeError(f"split should be val or test, got {split}")
        print(split)
        print(metrics)

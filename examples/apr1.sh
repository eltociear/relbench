python gnn_link.py --task rel-hm-rec --dataset rel-hm --log_dir runs/hm-rec --share_same_time
python gnn_node.py --task rel-hm-churn --dataset rel-hm --log_dir runs/hm-churn-vanilla --max_steps_per_epoch 10 --epochs 50
python gnn_node.py --task rel-hm-churn --dataset rel-hm --log_dir runs/hm-churn-pretrain --max_steps_per_epoch 10 --epochs 50 --load_ckpt

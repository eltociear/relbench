from typing import Any, Dict, List

import torch
from torch import Tensor
from torch import nn
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType


from relbench.external.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.data.task_base import TaskType

class TemporalModel(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        task_type: str, 
        num_ar: int,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )

        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )

        self.head = MLP(
            channels * 2 * (num_ar + 1),
            hidden_channels=channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=3,
        )

        self.past_mlp = MLP(
            channels * 2,
            hidden_channels=2*channels,
            out_channels=2*channels,
            norm=norm,
            num_layers=2,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None

        self.label_embedder = LabelEmbedder(task_type, channels)

        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

        self.num_ar = num_ar
        self.ar_key_to_idx = {}
    
        # Map the AR keys
        for i in range(num_ar, 0, -1):
            self.ar_key_to_idx[f'AR_{i}'] = num_ar - i
        
        # Map the root key to the last index
        self.ar_key_to_idx['root'] = num_ar

        # and the reverse
        self.idx_to_ar_key = {v: k for k, v in self.ar_key_to_idx.items()}

        self.y_nan_emb = torch.nn.Embedding(1, channels)


    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()



    def add_label_and_time_embeddings(self, outs, entity_table, batch_dict):
        x_dict = {k: out[entity_table] for k, out in outs.items()}

        y_dict = {k: batch[entity_table].y for k, batch in batch_dict.items() if k != "root"}

        for i, ar_key in  self.idx_to_ar_key.items():
            if i > 0:
                # w.p. 0.5 zero out the x during train
                #if torch.rand(1) < 0.25: # and self.training:
                #    x_dict[ar_key] = torch.zeros_like(x_dict[ar_key])
                # add y label from previous time step

                # mask to zero out the x if y is nan
                #mask = torch.isnan(y_dict[self.idx_to_ar_key[i-1]]).unsqueeze(-1)
                #x_dict[ar_key] = torch.where(mask, self.y_nan_emb.weight.repeat(mask.size(0), 1), x_dict[ar_key])


                x_dict[ar_key] = torch.cat([torch.zeros_like(x_dict[ar_key]), self.label_embedder(y_dict[self.idx_to_ar_key[i-1]])], dim=-1)
            else:
                #x_dict[ar_key] = x_dict[ar_key]
                x_dict[ar_key] = torch.cat([x_dict[ar_key], torch.zeros_like(x_dict[ar_key])], dim=-1) 

        return x_dict


    def fuse_past(self, x_dict, seed_times):
        # make seq_len x n x dim tensor
        for i, ar_key in  self.idx_to_ar_key.items():
            x_dict[ar_key] = x_dict[ar_key][: seed_times[ar_key].size(0)]

        #x = torch.cat(xs, dim=0) # seq_len x n x dim

        #for i, ar_key in  self.idx_to_ar_key.items():
        #    x_dict[ar_key] = x[i]

        for ar_key in x_dict.keys():
            if ar_key != 'root':
                x_dict[ar_key] = self.past_mlp(x_dict[ar_key])

        return x_dict

    def forward_batch(self, inputs, key):
        seed_time, tf_dict, time_dict, batch_dict, edge_index_dict, num_sampled_nodes_dict, num_sampled_edges_dict, node_type_dict = inputs

        x_dict = self.encoder(tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, time_dict, batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(node_type_dict[node_type])

        x_dict = self.gnn(
            x_dict,
            edge_index_dict,
            num_sampled_nodes_dict,
            num_sampled_edges_dict,
        )

        return x_dict
    

    def forward(
        self,
        batch_dict: Dict[str, HeteroData],
        entity_table: NodeType,
    ) -> Dict[str, Tensor]:
        # Prepare input tensors for parallel processing
        seed_times = {k: batch[entity_table].seed_time for k, batch in batch_dict.items()}
        tf_dicts = {k: batch.tf_dict for k, batch in batch_dict.items()}
        time_dicts = {k: batch.time_dict for k, batch in batch_dict.items()}
        batch_dicts = {k: batch.batch_dict for k, batch in batch_dict.items()}
        edge_index_dicts = {k: batch.edge_index_dict for k, batch in batch_dict.items()}
        num_sampled_nodes_dicts = {k: batch.num_sampled_nodes_dict for k, batch in batch_dict.items()}
        num_sampled_edges_dicts = {k: batch.num_sampled_edges_dict for k, batch in batch_dict.items()}
        node_type_dicts = {k: {node_type: batch[node_type].n_id for node_type in self.embedding_dict.keys()} for k, batch in batch_dict.items()}


        # Prepare the input tensors as a sequence of tuples
        inputs = {k:
            (
                seed_times[k],
                tf_dicts[k],
                time_dicts[k],
                batch_dicts[k],
                edge_index_dicts[k],
                num_sampled_nodes_dicts[k],
                num_sampled_edges_dicts[k],
                node_type_dicts[k]
            )
            for k in seed_times.keys()
        }

        # Run forward pass in parallel on the single GPU
        outs = {k: self.forward_batch(batch, k) for k, batch in inputs.items()}
        # Combine the outputs into a dictionary

        x_time_dict = self.add_label_and_time_embeddings(outs, entity_table, batch_dict)
        out = self.fuse_past(x_time_dict, seed_times)

        out = torch.cat([out[k] for k in out.keys()], dim=-1)
        return {"root": self.head(out)}


class LabelEmbedder(torch.nn.Module):
    """
    Class to embed labels for the model
    """
    def __init__(self, task_type, channels):
        super(LabelEmbedder, self).__init__()
        
        if task_type == TaskType.BINARY_CLASSIFICATION:
            self.label_embedder = torch.nn.Embedding(2, channels)
        elif task_type == TaskType.REGRESSION:
            self.label_embedder = torch.nn.Linear(1, channels)
        
        self.task_type = task_type
        self.nan_embedder = torch.nn.Parameter(torch.randn(channels))  # Learned embedding for NaN values

        self.channels = channels
    def forward(self, y):
        is_nan = torch.isnan(y)
        not_nan = ~is_nan
        
        embedded_labels = torch.zeros((y.size(0), self.channels), device=y.device)
        
        if self.task_type == TaskType.REGRESSION:
            y = y.float().unsqueeze(1)
        else:
            y = y.long()
        if not_nan.any():
            embedded_labels[not_nan] = self.label_embedder(y[not_nan])
        
        if is_nan.any():
           embedded_labels[is_nan] = self.nan_embedder
        
        return embedded_labels

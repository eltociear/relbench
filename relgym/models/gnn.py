from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATConv, HeteroConv, LayerNorm, SAGEConv, MLP
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import trim_to_layer
from torch_scatter import scatter

conv_name_to_func = {
    "sage": SAGEConv,
    "gat": partial(GATConv, add_self_loops=False),
}


class HybridConv(nn.Module):
    def __init__(self, conv_func, aggr, channels):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in aggr:
            _conv = conv_func((channels, channels), channels, aggr=_)
            self.convs.append(_conv)

    def forward(self, *args, **kwargs):
        return sum([conv(*args, **kwargs) for conv in self.convs])


def parse_conv_func(conv, aggr, channels):
    conv_func = conv_name_to_func[conv]
    if type(aggr) is str and aggr != "hybrid":
        return conv_func((channels, channels), channels, aggr=aggr)
    else:
        if aggr == "hybrid":
            aggr = ["sum", "mean"]
        return HybridConv(conv_func, aggr, channels)


class SelfJoinLayer(torch.nn.Module):
    def __init__(
        self,
        node_types,
        channels,
        node_type_considered=None,
        num_filtered=20,
        normalize_score=True,
        selfjoin_aggr="sum",
        retrieve_label=True,
    ):
        super().__init__()

        if node_type_considered is None:
            node_type_considered = []
        elif node_type_considered == "all":
            node_type_considered = node_types
        elif type(node_type_considered) is str:
            node_type_considered = [
                node_type_considered
            ]  # If only one type as str, treat it as a list
        else:
            assert type(node_type_considered) is list
        self.node_type_considered = node_type_considered
        self.num_filtered = num_filtered
        self.normalize_score = normalize_score
        self.selfjoin_aggr = selfjoin_aggr
        self.retrieve_label = retrieve_label

        # Initialize the message passing module
        self.msg_dict = torch.nn.ModuleDict()
        self.upd_dict = torch.nn.ModuleDict()
        for node_type in self.node_type_considered:
            self.msg_dict[node_type] = MLP(
                channel_list=[channels * 2 + 1, channels, channels]
            )
            self.upd_dict[node_type] = MLP(
                channel_list=[channels * 2, channels, channels]
            )

        # Initialize the similarity score computation module
        self.query_dict = torch.nn.ModuleDict()
        self.key_dict = torch.nn.ModuleDict()
        for node_type in self.node_type_considered:
            self.query_dict[node_type] = nn.Linear(channels, channels)
            self.key_dict[node_type] = nn.Linear(channels, channels)

    def forward(self, x_dict: Dict, bank_x_dict: Dict, bank_y: Tensor, seed_time: Tensor, bank_seed_time: Tensor):
        upd_x_dict = {}
        for node_type, feature in x_dict.items():
            if node_type not in self.node_type_considered:
                upd_x_dict[node_type] = feature
                continue

            feature_bank = bank_x_dict[node_type][: bank_seed_time.size(0)]  # [N', H]
            _feature = feature.clone()
            feature = feature[: seed_time.size(0)]

            # Compute similarity score
            q = self.query_dict[node_type](feature_bank)  # [N', H]
            k = self.key_dict[node_type](feature)  # [N, H]
            sim_score = torch.matmul(k, q.transpose(0, 1))  # [N, N']

            # Avoid time leakage
            valid_mask = (seed_time.unsqueeze(-1) < bank_seed_time.unsqueeze(0))  # [N, N']
            sim_score[~valid_mask] = -torch.inf
            # Select Top K
            sim_score, index_sampled = torch.topk(
                sim_score, k=min(self.num_filtered, sim_score.shape[1]), dim=1
            )  # [N, K], [N, K]

            # Normalize
            if self.normalize_score:
                sim_score = torch.softmax(sim_score, dim=-1)  # [N, K]
                # fix the extreme case where there is a row with all -inf, leading to nan sim_score
                sim_score = torch.nan_to_num(sim_score, nan=0)

            # Construct the graph over the retrieved entries
            edge_index_i = (
                torch.arange(index_sampled.size(0))
                .to(sim_score.device)
                .unsqueeze(-1)
                .repeat(1, index_sampled.size(1))
                .view(-1)
            )  # [NK]
            edge_index_j = index_sampled.view(-1)  # [NK]
            edge_index = torch.stack((edge_index_i, edge_index_j), dim=0)  # [2, NK]

            h_i, h_j = (
                feature[edge_index[0]],
                feature_bank[edge_index[1]],
            )  # [M, H], M = N * K
            y_j = bank_y[edge_index[1]].unsqueeze(-1).to(h_j)  # [M,]
            if not self.retrieve_label:  # mask the label by all zeros
                y_j = torch.zeros_like(y_j)
            score = self.msg_dict[node_type](
                torch.cat((h_i, h_j, y_j), dim=-1)
            ) * sim_score.view(
                -1, 1
            )  # [M, H]
            h_agg = scatter(
                score, edge_index[0], dim=0, reduce=self.selfjoin_aggr
            )  # [N, H]
            feature_out = feature + self.upd_dict[node_type](
                torch.cat((feature, h_agg), dim=-1)
            )  # [N, H]

            _feature[: seed_time.size(0)] = feature_out
            upd_x_dict[node_type] = _feature
            # upd_x_dict[node_type] = feature_out

        return upd_x_dict


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        conv: str,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        hetero_aggr: str = "sum",
        num_layers: int = 2,
        feature_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            _conv = HeteroConv(
                {
                    edge_type: parse_conv_func(conv, aggr, channels)
                    for edge_type in edge_types
                },
                aggr=hetero_aggr,
            )
            self.convs.append(_conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

        self.feature_dropout = feature_dropout

        # Self Join
        self.use_self_join = kwargs.pop("use_self_join")
        self.self_joins = torch.nn.ModuleList()
        if self.use_self_join:
            for _ in range(num_layers):
                self.self_joins.append(SelfJoinLayer(node_types, channels, **kwargs))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
        **kwargs,
    ) -> Dict[NodeType, Tensor]:

        # Apply dropout to the input features
        x_dict = {
            key: nn.functional.dropout(
                x, p=self.feature_dropout, training=self.training
            )
            for key, x in x_dict.items()
        }

        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # TODO: Re-introduce this.
            # Trim graph and features to only hold required data per layer:
            # if num_sampled_nodes_dict is not None:
            #     assert num_sampled_edges_dict is not None
            #     x_dict, edge_index_dict, _ = trim_to_layer(
            #         layer=i,
            #         num_sampled_nodes_per_hop=num_sampled_nodes_dict,
            #         num_sampled_edges_per_hop=num_sampled_edges_dict,
            #         x=x_dict,
            #         edge_index=edge_index_dict,
            #     )
            if self.use_self_join:
                x_dict = self.self_joins[i](x_dict, **kwargs)

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

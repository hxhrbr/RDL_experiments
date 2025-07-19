from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder

def partition_edges_by_time_dict(
    edge_index_dict,
    time_dict,
    start_time,
    end_time,
    period_duration,
):
    r"""
    :param edge_index_dict:
    :param time_dict:
    :param start_time:
    :param end_time:
    :param period_duration:
    :return:
    """

    number_of_periods = (end_time-start_time+period_duration)//period_duration
    intro_times = {}
    output = [{} for _ in range(number_of_periods)]

    # Compute the introduction time of each edge
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        src, dst = edge_index[0], edge_index[1]

        # Check that tensors are in the same device
        device = edge_index.device
        for node_type in {src_type, dst_type}:
            if node_type in time_dict:
                assert time_dict[node_type].device == device, (
                    f"Time tensor for node type '{node_type}' is on {time_dict[node_type].device}, "
                    f"but expected device is {device}"
                )

        if src_type not in time_dict and dst_type not in time_dict:
            continue
        elif src_type not in time_dict:
            intro_times[edge_type] = time_dict[dst_type][dst]
        elif dst_type not in time_dict:
            intro_times[edge_type] = time_dict[src_type][src]
        else:
            intro_times[edge_type] = torch.maximum(time_dict[src_type][src], time_dict[dst_type][dst])

    # Account for atemporal edges by adding them to each partition part
    for edge_type in edge_index_dict.keys():
        if edge_type not in intro_times:
            for p in range(number_of_periods):
                output[p][edge_type] = edge_index_dict[edge_type]

    # Assign temporal edges to the correct period
    for edge_type in intro_times.keys():

        periods = (intro_times[edge_type] - start_time) // period_duration
        assert ((periods >= 0) & (periods < number_of_periods)).all()

        for p in range(number_of_periods):
            mask = (periods == p)
            output[p][edge_type] = edge_index_dict[edge_type][:, mask]

    return output

class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
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
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        min_timestamp,
        max_timestamp,
        update_period,
        batch: HeteroData,
        entity_table: NodeType,
        hidden_dict: Dict = None,
    ) -> Tensor:
        x_dict = hidden_dict
        seed_time = batch[entity_table].seed_time

        if hidden_dict is None:
            x_dict = self.encoder(batch.tf_dict)

            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )

            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

            for node_type, embedding in self.embedding_dict.items():
                x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        parts = partition_edges_by_time_dict(
            batch.edge_index_dict,
            batch.time_dict,
            min_timestamp,
            max_timestamp,
            update_period
        )
        '''
        for edge_index_dict in parts:

            gnn_out = self.gnn(
                x_dict,
                edge_index_dict,
                batch.num_sampled_nodes_dict,
                batch.num_sampled_edges_dict,
            )

            for node_type in x_dict:
                x_dict[node_type] = x_dict[node_type] + gnn_out[node_type]

        return self.head(x_dict[entity_table][: seed_time.size(0)])
        '''
        for edge_index_dict in parts:
            involved_nodes = {}
            for (src_type, _, dst_type), edge_index in edge_index_dict.items():
                for node_type, idxs in zip([src_type, dst_type], [edge_index[0], edge_index[1]]):
                    if node_type not in involved_nodes:
                        involved_nodes[node_type] = []
                    involved_nodes[node_type].append(idxs)

            x_sub_dict = {}
            node_id_map = {}

            for node_type, list_of_node_indices in involved_nodes.items():
                all_nodes = torch.cat(list_of_node_indices)
                node_ids, inverse = torch.unique(all_nodes, sorted=True, return_inverse=True)

                x_sub_dict[node_type] = x_dict[node_type][node_ids].contiguous()
                node_id_map[node_type] = (node_ids, inverse)

                involved_nodes[node_type] = node_ids  # reuse for writing back

            edge_index_sub_dict = {}
            for (src_type, rel_type, dst_type), edge_index in edge_index_dict.items():
                src_node_ids, _ = node_id_map[src_type]
                dst_node_ids, _ = node_id_map[dst_type]

                remapped_src = torch.bucketize(edge_index[0], src_node_ids)
                remapped_dst = torch.bucketize(edge_index[1], dst_node_ids)

                edge_index_sub_dict[(src_type, rel_type, dst_type)] = torch.stack([remapped_src, remapped_dst], dim=0)

            gnn_out = self.gnn(
                x_sub_dict,
                edge_index_sub_dict,
                batch.num_sampled_nodes_dict,
                batch.num_sampled_edges_dict,
            )

            for node_type, out_feats in gnn_out.items():
                node_ids = involved_nodes[node_type]
                x_dict[node_type][node_ids] += out_feats

        out = self.head(x_dict[entity_table][: batch[entity_table].seed_time.size(0)])
        return out

    def forward_dst_readout(
        self,
        min_timestamp,
        max_timestamp,
        update_period,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
        hidden_dict: Dict = None,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )

        x_dict = hidden_dict
        seed_time = batch[entity_table].seed_time

        if hidden_dict is None:
            x_dict = self.encoder(batch.tf_dict)

            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )

            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

            for node_type, embedding in self.embedding_dict.items():
                x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        parts = partition_edges_by_time_dict(
            batch.edge_index_dict,
            batch.time_dict,
            min_timestamp,
            max_timestamp,
            update_period
        )

        for edge_index_dict in parts:

            gnn_out = self.gnn(
                x_dict,
                edge_index_dict,
            )

            for node_type in x_dict:
                x_dict[node_type] = x_dict[node_type] + gnn_out[node_type]

        return self.head(x_dict[dst_table])

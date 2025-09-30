from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from torch_geometric.data import Batch
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





def make_subgraph_for_part(
    x_dict,
    edge_index_dict,
    entity_table,
    num_seeds,
    ensure_seed_presence: bool = True,
    device='cuda',
    dtype=torch.long,
):
    # TODO: think of pruning useless parts of this subgraph (i.e. disconnected from the seed node)
    """
    Build per-part subgraph tensors (minimal changes from your original code).

    Returns:
        x_sub_dict:           {node_type: Tensor[num_nodes_of_type_node_type_in_part, embedding_dim]}
        edge_index_sub_dict:  {(src, rel, dst): LongTensor[2, num_edges_of_type__src_rel_dst__in_part]}
        involved_nodes:       {node_type: LongTensor[sorted global ids used in this part]}
                               (sorted global ids enable fast alignment later via searchsorted)
    """
    involved_nodes = {}

    # Keep seed nodes (as you had it) so they are never dropped
    if ensure_seed_presence:
        involved_nodes[entity_table] = [torch.arange(num_seeds, device=device, dtype=dtype)]

    # Collect node indices touched by edges (your original pattern)
    for (src_type, _, dst_type), edge_index in edge_index_dict.items():
        for node_type, idxs in zip(
            [src_type, dst_type],
            [edge_index[0], edge_index[1]],
        ):
            # NOTE: dictionary has keys only of node types for which at least one involved node is present!
            if node_type not in involved_nodes:
                involved_nodes[node_type] = []
            involved_nodes[node_type].append(idxs)

    # Slice features for the unique, sorted set of nodes per type
    x_sub_dict = {}
    for node_type, list_of_node_indices in involved_nodes.items():
        all_nodes = torch.cat(list_of_node_indices)
        node_ids = torch.unique(all_nodes, sorted=True)  # sorted -> enables searchsorted later
        x_sub_dict[node_type] = x_dict[node_type][node_ids].contiguous()
        involved_nodes[node_type] = node_ids  # reuse for writing back / alignment

    # Remap edges to the local (per-part) node id space
    edge_index_sub_dict = {}
    for (src_type, rel_type, dst_type), edge_index in edge_index_dict.items():
        src_node_ids = involved_nodes[src_type]  # sorted
        dst_node_ids = involved_nodes[dst_type]  # sorted

        remapped_src = torch.bucketize(edge_index[0], src_node_ids)
        remapped_dst = torch.bucketize(edge_index[1], dst_node_ids)

        edge_index_sub_dict[(src_type, rel_type, dst_type)] = torch.stack(
            [remapped_src, remapped_dst], dim=0
        )

    return x_sub_dict, edge_index_sub_dict, involved_nodes





def build_part_data(
    x_dict,
    edge_index_dict_for_part,
    entity_table,
    num_seeds
):
    """
    Expects `make_subgraph_for_part` to return:
        x_sub_dict, edge_index_sub_dict, involved_nodes
    where involved_nodes[ntype] is a sorted 1D LongTensor of GLOBAL node ids.
    """
    x_sub_dict, edge_index_sub_dict, involved_nodes = make_subgraph_for_part(
        x_dict,
        edge_index_dict_for_part,
        entity_table,
        num_seeds
    )

    data = HeteroData()
    for ntype, x in x_sub_dict.items():
        data[ntype].x = x
    for (src, rel, dst), eidx in edge_index_sub_dict.items():
        data[(src, rel, dst)].edge_index = eidx

    # Keep global id order used to slice features (needed to align seeds later)
    data.involved_nodes = involved_nodes  # dict: ntype -> 1D LongTensor (sorted)
    return data





def stack_seed_embeddings(
    out_dict,
    batched,
    entity_type,
    num_parts,
    num_seeds
):
    """
    Assumes:
      - For each part, the first `num_seeds` rows of `entity_type` are the seeds
        in identical order across parts.
    """
    feats_all = out_dict[entity_type]
    part_id_vec = batched[entity_type].batch
    P, S, D = num_parts, num_seeds, feats_all.size(-1)

    # Count how many entity nodes per part (so we know where each part starts)
    sizes = torch.bincount(part_id_vec, minlength=P)          # [P]
    offsets = F.pad(sizes.cumsum(0), (1, 0))[:-1]             # [P], start index of each part

    # Calculate the indices of all the seeds in the combined data
    row_in_part = torch.arange(S, device=feats_all.device)   # [S]
    seeds_all_idxs = offsets.unsqueeze(1) + row_in_part.unsqueeze(0) # [P, S]

    # Gather and reshape to [S, D, P]
    embs = feats_all[seeds_all_idxs.reshape(-1)].view(P, S, D).permute(1, 2, 0).contiguous()
    return embs  # [S, D, P]





class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        aggr: str,
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
        max_parts_in_parallel: int = None
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

        num_parts = len(parts)
        num_seeds = seed_time.size(0)

        def parallel_gnn_run(parts_slice):
            # build HeteroData list for these parts
            data_list = [build_part_data(x_dict, edge_idx_dict, entity_table, num_seeds) for edge_idx_dict in parts_slice]
            # pack into one disjoint union
            batched = Batch.from_data_list(data_list)
            gnn_out = self.gnn(
                batched.x_dict,
                batched.edge_index_dict,
            )

            return stack_seed_embeddings(
                gnn_out,
                batched,
                entity_type=entity_table,
                num_parts=len(parts_slice),
                num_seeds=num_seeds,
            )

        if max_parts_in_parallel is None:
            return parallel_gnn_run(parts)
        else:
            out = []
            for i in range(0, num_parts, max_parts_in_parallel):
                out.append(parallel_gnn_run(parts[i : i + max_parts_in_parallel]))
            return torch.cat(out, dim=1)




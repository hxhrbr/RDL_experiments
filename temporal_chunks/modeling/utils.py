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
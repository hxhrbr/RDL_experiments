import duckdb
import torch
import pandas as pd
import numpy as np
from relbench.base import Database, Table

def extract_unique_driverIds_form_tables(*tables):
    tables = [table['driverId'] for table in tables]
    return pd.concat(tables).drop_duplicates()



def mask_future_labels(
        min_time,
        max_time,
        period,
        delta,
        time_series,
        seed_time,
        filler_value = -2
):
    n_seeds = time_series.shape[0]
    if n_seeds != seed_time.shape[0]:
        raise ValueError("time_series and seed_time must have same number of elements in the first dimension")
    n_periods = (max_time - min_time + period) // period
    if n_periods != time_series.shape[1]:
        raise ValueError("time_series number of columns is incongruent with min_time, max_time and period")

    row_indices = torch.arange(n_periods).to(time_series.device)
    last_usable_indices = (seed_time-delta)//period
    mask = row_indices.unsqueeze(0) > last_usable_indices.unsqueeze(1)
    time_series[mask] = filler_value


def driverDNFlabels(db: Database, seed_ids, timestamps: "pd.Series[pd.Timestamp]", seed_timestamps: "pd.Series[pd.Timestamp]" = None, device = "cuda"):

    timedelta = pd.Timedelta(days=30)
    timestamp_df = pd.DataFrame({"timestamp": timestamps})
    #Note: We assume timestamps is sorted in ascending order
    results_df = db.table_dict["results"].df
    drivers_df = db.table_dict["drivers"].df
    if (isinstance(seed_ids, list)):
        # in this case the seed_ids are treated as positional indices
        driver_ids = drivers_df.iloc[seed_ids]["driverId"].tolist()
        relevant_drivers = drivers_df.iloc[seed_ids]
    else:
        # we assume then that seed_ids is a pd.Series of driverIds
        driver_ids = seed_ids.tolist()
        relevant_drivers = drivers_df[drivers_df["driverId"].isin(seed_ids)]

    relevant_results = duckdb.sql(
        f"""
            SELECT *
            FROM
                results_df re
            WHERE
                re.driverId IN (SELECT driverId FROM relevant_drivers)
        ;
        """
    ).df()


    out_cols = []

    for i in range(len(timestamp_df)):
        one_timestamp = timestamp_df.iloc[[i]]

        df = duckdb.sql(
            f"""
                SELECT
                    re.driverId as driverId,
                    MAX(CASE WHEN re.statusId != 1 THEN 1 ELSE 0 END) AS did_not_finish
                FROM
                    one_timestamp t
                JOIN
                    relevant_results re
                ON
                    re.date <= t.timestamp + INTERVAL '{timedelta}'
                    and re.date  > t.timestamp
                GROUP BY re.driverId
            ;
            """
        ).df()
        df = df.set_index("driverId").reindex(driver_ids, fill_value=-1).reset_index()
        col = torch.tensor(df["did_not_finish"].values).to(device)
        out_cols.append(col)

    out_cols = torch.stack(out_cols, dim=1)

    if seed_timestamps is None:
        return out_cols

    last_usable_timestamp_index = np.searchsorted(
        timestamps.to_numpy(),
        (seed_timestamps - timedelta).to_numpy(),
        side = "right"
    )
    last_usable_timestamp_index = torch.from_numpy(last_usable_timestamp_index).to(device)

    index_grid = torch.arange(len(timestamps), device = device).expand(len(seed_ids), -1)
    mask = last_usable_timestamp_index.unsqueeze(1) <= index_grid
    out_cols[mask] = -2

    """
        TODO: Take action on the following observation: Above we use -2 to indicate that the label can't be calculated for that timestamp,
        but for how we set up things, this is not necessarily true, becauese we are ignoring that, for a seed_node driver and a timestamp we can
        already know the labels if we have data just for the next day, assuming the next days sees a result where the driver finishes.
        So it would be possible to refine this thing to take that into account
    """

    return out_cols

def driverDFNlabels_from_batch(db, pd_period_beginnings, batch, device = "cuda"):
    x = batch["drivers"]
    seed_timestamps = x.seed_time.cpu()
    seed_timestamps = pd.to_datetime(seed_timestamps, unit="s")
    seed_timestamps = pd.Series(seed_timestamps)
    seed_ids = x.n_id[:len(seed_timestamps)].cpu()
    seed_ids = seed_ids.numpy()
    out = driverDNFlabels(db, seed_ids, pd_period_beginnings, seed_timestamps, device)
    return out




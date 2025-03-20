# %%
import argparse
import json
import multiprocessing as mp
import os
import pickle
from contextlib import nullcontext

import numpy as np
import pandas as pd
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pairs")
    parser.add_argument("--stations", type=str, default="test_data/ridgecrest/stations.csv")
    parser.add_argument("--events", type=str, default="test_data/ridgecrest/gamma_events.csv")
    parser.add_argument("--picks", type=str, default="test_data/ridgecrest/gamma_picks.csv")
    parser.add_argument("--result_path", type=str, default="results/ridgecrest")
    return parser.parse_args()


# %%
def convert_dd(
    pairs,
    picks_by_event,
    min_obs=8,
    max_obs=20,
    i=0,
):

    station_index = []
    event_index1 = []
    event_index2 = []
    phase_type = []
    phase_score = []
    phase_dtime = []
    for idx1, idx2 in tqdm(pairs, desc=f"CPU {i}", position=i):
        picks1 = picks_by_event.get_group(idx1)
        picks2 = picks_by_event.get_group(idx2)

        common = picks1.merge(picks2, on=["idx_sta", "phase_type"], how="inner")
        if len(common) < min_obs:
            continue
        common["phase_score"] = (common["phase_score_x"] + common["phase_score_y"]) / 2.0
        common.sort_values("phase_score", ascending=False, inplace=True)
        common = common.head(max_obs)
        event_index1.extend(common["idx_eve_x"].values)
        event_index2.extend(common["idx_eve_y"].values)
        station_index.extend(common["idx_sta"].values)
        phase_type.extend(common["phase_type"].values)
        phase_score.extend(common["phase_score"].values)
        phase_dtime.extend(np.round(common["travel_time_x"].values - common["travel_time_y"].values, 5))

    return {
        "event_index1": event_index1,
        "event_index2": event_index2,
        "station_index": station_index,
        "phase_type": phase_type,
        "phase_score": phase_score,
        "phase_dtime": phase_dtime,
    }


# %%
if __name__ == "__main__":

    args = parse_args()
    result_path = args.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # %%
    MAX_PAIR_DIST = 10  # km
    MAX_NEIGHBORS = 50
    MIN_NEIGHBORS = 8
    MIN_OBS = 8
    MAX_OBS = 100
    mapping_phase_type_int = {"P": 0, "S": 1}

    # %%
    stations = pd.read_csv(args.stations)
    picks = pd.read_csv(args.picks, parse_dates=["phase_time"])
    events = pd.read_csv(args.events, parse_dates=["time"])

    picks = picks[picks["event_index"] != -1]
    # check phase_type is P/S or 0/1
    if set(picks["phase_type"].unique()).issubset(set(mapping_phase_type_int.keys())):  # P/S
        picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)

    # %%
    if "idx_eve" in events.columns:
        events = events.drop("idx_eve", axis=1)
    if "idx_sta" in stations.columns:
        stations = stations.drop("idx_sta", axis=1)
    if "idx_eve" in picks.columns:
        picks = picks.drop("idx_eve", axis=1)
    if "idx_sta" in picks.columns:
        picks = picks.drop("idx_sta", axis=1)

    # %%
    # reindex in case the index does not start from 0 or is not continuous
    stations = stations[stations["station_id"].isin(picks["station_id"].unique())]
    events = events[events["event_index"].isin(picks["event_index"].unique())]
    stations["idx_sta"] = np.arange(len(stations))
    events["idx_eve"] = np.arange(len(events))

    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    # %%
    lon0 = stations["longitude"].median()
    lat0 = stations["latitude"].median()
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")

    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["depth_km"] = -stations["elevation_m"] / 1000
    stations["z_km"] = stations["depth_km"]

    events[["x_km", "y_km"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events["z_km"] = events["depth_km"]

    picks = picks.merge(events[["idx_eve", "time"]], on="idx_eve")
    picks["travel_time"] = (picks["phase_time"] - picks["time"]).dt.total_seconds()
    picks.drop("time", axis=1, inplace=True)

    # %%
    picks_by_event = picks.groupby("idx_eve")

    # Option 1:
    neigh = NearestNeighbors(radius=MAX_PAIR_DIST, n_jobs=-1)
    neigh.fit(events[["x_km", "y_km", "z_km"]].values)
    pairs = set()
    neigh_ind = neigh.radius_neighbors(sort_results=True)[1]
    for i, neighs in enumerate(tqdm(neigh_ind, desc="Generating pairs")):
        if len(neighs) < MIN_NEIGHBORS:
            continue
        for j in neighs[:MAX_NEIGHBORS]:
            if i > j:
                pairs.add((j, i))
            else:
                pairs.add((i, j))
    pairs = list(pairs)

    # Option 2:
    # neigh = NearestNeighbors(radius=MAX_PAIR_DIST, n_jobs=-1)
    # neigh.fit(events[["x_km", "y_km", "z_km"]].values)
    # pairs = set()
    # neigh_ind = neigh.radius_neighbors()[1]
    # for i, neighs in enumerate(tqdm(neigh_ind, desc="Generating pairs")):
    #     if len(neighs) < MIN_NEIGHBORS:
    #         continue
    #     neighs = neighs[np.argsort(events.loc[neighs, "num_picks"])]  ## TODO: check if useful
    #     for j in neighs[:MAX_NEIGHBORS]:
    #         if i > j:
    #             pairs.add((j, i))
    #         else:
    #             pairs.add((i, j))
    # pairs = list(pairs)

    # %%
    NCPU = min(32, mp.cpu_count())
    with mp.Manager() as manager:

        pool = mp.Pool(NCPU)
        results = pool.starmap(
            convert_dd,
            [
                (
                    pairs[i::NCPU],
                    picks_by_event,
                    MIN_OBS,
                    MAX_OBS,
                    i,
                )
                for i in range(NCPU)
            ],
        )
        pool.close()
        pool.join()

        print("Collecting results")
        event_index1 = np.concatenate([r["event_index1"] for r in results])
        event_index2 = np.concatenate([r["event_index2"] for r in results])
        station_index = np.concatenate([r["station_index"] for r in results])
        phase_type = np.concatenate([r["phase_type"] for r in results])
        phase_score = np.concatenate([r["phase_score"] for r in results])
        phase_dtime = np.concatenate([r["phase_dtime"] for r in results])

        # Filter large P and S time differences
        idx = ((phase_type == 0) & (np.abs(phase_dtime) < 1.0)) | ((phase_type == 1) & (np.abs(phase_dtime) < 1.5))
        event_index1 = event_index1[idx]
        event_index2 = event_index2[idx]
        station_index = station_index[idx]
        phase_type = phase_type[idx]
        phase_score = phase_score[idx]
        phase_dtime = phase_dtime[idx]
        print(f"Saving to disk: {len(event_index1)} pairs")
        # np.savez_compressed(
        #     os.path.join(catalog_path, "adloc_dt.npz"),
        #     event_index1=event_index1,
        #     event_index2=event_index2,
        #     station_index=station_index,
        #     phase_type=phase_type,
        #     phase_score=phase_score,
        #     phase_dtime=phase_dtime,
        # )

        dtypes = np.dtype(
            [
                ("event_index1", np.int32),
                ("event_index2", np.int32),
                ("station_index", np.int32),
                ("phase_type", np.int32),
                ("phase_score", np.float32),
                ("phase_dtime", np.float32),
            ]
        )
        pairs_array = np.memmap(
            os.path.join(result_path, "pair_dt.dat"),
            mode="w+",
            shape=(len(phase_dtime),),
            dtype=dtypes,
        )
        pairs_array["event_index1"] = event_index1
        pairs_array["event_index2"] = event_index2
        pairs_array["station_index"] = station_index
        pairs_array["phase_type"] = phase_type
        pairs_array["phase_score"] = phase_score
        pairs_array["phase_dtime"] = phase_dtime
        with open(os.path.join(result_path, "pair_dtypes.pkl"), "wb") as f:
            pickle.dump(dtypes, f)

        events.to_csv(os.path.join(result_path, "pair_events.csv"), index=False)
        stations.to_csv(os.path.join(result_path, "pair_stations.csv"), index=False)
        picks.to_csv(os.path.join(result_path, "pair_picks.csv"), index=False)

# %%

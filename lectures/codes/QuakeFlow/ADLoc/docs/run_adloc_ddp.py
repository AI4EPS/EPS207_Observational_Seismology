import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from adloc.adloc import TravelTime
from adloc.data import PhaseDataset
from adloc.eikonal2d import init_eikonal2d, traveltime
from utils import plotting

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":

    # %%
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend="gloo")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        print(f"DDP rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size}")
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        print("Non-DDP run")

    # %%
    EPOCHS = 100
    MAX_SST_ITER = 10

    # %%
    region = "synthetic"
    # region = "ridgecrest"
    data_path = f"test_data/{region}"
    result_path = f"results/{region}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    figure_path = f"figures/{region}/"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    picks_file = os.path.join(data_path, "picks.csv")
    events_file = os.path.join(data_path, "events.csv")
    stations_file = os.path.join(data_path, "stations.csv")
    config_file = os.path.join(data_path, "config.json")

    # %% JSON format
    # with open(args.stations, "r") as fp:
    #     stations = json.load(fp)
    # stations = pd.DataFrame.from_dict(stations, orient="index")
    # stations["station_id"] = stations.index
    # %% CSV format
    stations = pd.read_csv(stations_file)
    picks = pd.read_csv(picks_file, parse_dates=["phase_time"])
    events = pd.read_csv(events_file, parse_dates=["time"])
    config = json.load(open(config_file))

    # %%
    ## Automatic region; you can also specify a region
    # lon0 = stations["longitude"].median()
    # lat0 = stations["latitude"].median()
    # proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")
    lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
    lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0} +lat_ts={lat0} +units=km")

    if "depth_km" not in stations:
        stations["depth_km"] = -stations["elevation_m"] / 1000
    if "station_term" not in stations:
        stations["station_term"] = 0.0
    events["x_km"], events["y_km"] = proj(events["longitude"], events["latitude"])
    events["z_km"] = events["depth_km"]
    stations["x_km"], stations["y_km"] = proj(stations["longitude"], stations["latitude"])
    stations["z_km"] = stations["depth_km"]

    if ("xlim_km" not in config) or ("ylim_km" not in config) or ("zlim_km" not in config):
        xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
        xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
        zmin = stations["z_km"].min()
        zmax = 20
        config = {}
        config["xlim_km"] = (xmin, xmax)
        config["ylim_km"] = (ymin, ymax)
        config["zlim_km"] = (zmin, zmax)

    # %%
    mapping_phase_type_int = {"P": 0, "S": 1}
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    config["eikonal"] = None

    ## Eikonal for 1D velocity model
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 5.5, 6.7, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.3
    vel = {"Z": zz, "P": vp, "S": vs}
    config["eikonal"] = {
        "vel": vel,
        "h": h,
        "xlim_km": config["xlim_km"],
        "ylim_km": config["ylim_km"],
        "zlim_km": config["zlim_km"],
    }
    config["eikonal"] = init_eikonal2d(config["eikonal"])

    # %%
    config["bfgs_bounds"] = (
        (config["xlim_km"][0] - 1, config["xlim_km"][1] + 1),  # x
        (config["ylim_km"][0] - 1, config["ylim_km"][1] + 1),  # y
        (0, config["zlim_km"][1] + 1),
        (None, None),  # t
    )

    # %% reindex in case the index does not start from 0 or is not continuous
    stations["idx_sta"] = np.arange(len(stations))
    events["idx_eve"] = np.arange(len(events))

    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    # %%
    picks["travel_time"] = picks.apply(
        lambda x: (x["phase_time"] - events.loc[x["idx_eve"], "time"]).total_seconds(), axis=1
    )

    # %%
    num_event = len(events)
    num_station = len(stations)

    station_loc = stations[["x_km", "y_km", "z_km"]].values
    station_dt = None

    # %%
    events_init = events.copy()

    print(np.ceil(len(picks) / ddp_world_size))
    phase_dataset = PhaseDataset(
        picks,
        events,
        stations,
        config=config,
        rank=ddp_local_rank,
        world_size=ddp_world_size,
        batch_size=np.ceil(len(events) / ddp_world_size).astype(int),
    )
    data_loader = DataLoader(phase_dataset, batch_size=None, shuffle=False, num_workers=0)

    # %%
    event_loc_init = np.zeros((num_event, 3))
    event_loc_init[:, 2] = np.mean(config["zlim_km"])
    travel_time = TravelTime(
        num_event,
        num_station,
        station_loc,
        # event_loc=event_loc_init,  # Initial location
        event_loc=events[["x_km", "y_km", "z_km"]].values,
        # event_time=event_time,
        velocity={"P": vp, "S": vs},
        eikonal=config["eikonal"],
    )
    if ddp:
        travel_time = DDP(travel_time)
    raw_travel_time = travel_time.module if ddp else travel_time

    # %% Conventional location
    if ddp_local_rank == 0:
        print(
            f"Dataset: {len(picks)} picks, {len(events)} events, {len(stations)} stations, {len(phase_dataset)} batches"
        )
        print(f"============================ Basic location ============================")

    optimizer = optim.Adam(params=travel_time.parameters(), lr=1.0)
    for i in range(EPOCHS):
        optimizer.zero_grad()

        prev_loss = 0
        loss = 0
        for meta in data_loader:
            loss_ = travel_time(
                meta["idx_sta"],
                meta["idx_eve"],
                meta["phase_type"],
                meta["phase_time"],
                meta["phase_weight"],
            )["loss"]
            loss_.backward()
            if ddp:
                dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
            loss += loss_

        if ddp_local_rank == 0 and i % 10 == 0:
            print(f"Epoch = {i}, Loss = {loss:.3f}")
        if abs(loss - prev_loss) < 1e-3:
            break
        prev_loss = loss

        optimizer.step()

    # if ddp:
    #     dist.barrier()
    if ddp_local_rank == 0:
        print(f"Loss (Adam):  {loss}")

    # optimizer = optim.LBFGS(params=travel_time.parameters(), max_iter=1000, line_search_fn="strong_wolfe", lr=1.0)

    # def closure():
    #     optimizer.zero_grad()
    #     loss = 0
    #     for meta in data_loader:
    #         station_index = meta["station_index"]
    #         event_index = meta["event_index"]
    #         phase_time = meta["phase_time"]
    #         phase_type = meta["phase_type"]
    #         phase_weight = meta["phase_weight"]
    #         loss_ = travel_time(
    #             station_index,
    #             event_index,
    #             phase_type,
    #             phase_time,
    #             phase_weight,
    #         )["loss"]
    #         loss_.backward()
    #         if ddp:
    #             dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
    #         loss += loss_

    #     # print(f"Loss (LBFGS):  {loss}")
    #     return loss

    # optimizer.step(closure)

    # %%
    if ddp_local_rank == 0:
        invert_event_loc = raw_travel_time.event_loc.weight.clone().detach().numpy()
        invert_event_time = raw_travel_time.event_time.weight.clone().detach().numpy()

        events = events_init.copy()
        events["time"] = events["time"] + pd.to_timedelta(np.squeeze(invert_event_time), unit="s")
        events["x_km"] = invert_event_loc[:, 0]
        events["y_km"] = invert_event_loc[:, 1]
        events["z_km"] = invert_event_loc[:, 2]
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z_km"]
        events.to_csv(
            f"{result_path}/adloc_events.csv", index=False, float_format="%.5f", date_format="%Y-%m-%dT%H:%M:%S.%f"
        )

        plotting(stations, figure_path, config, picks, events, events, suffix="basic")

    # %% Location with SST (station static term)
    if ddp_local_rank == 0:
        print(f"============================ Location with SST ============================")

        idx_sta = torch.tensor(picks["idx_sta"].values, dtype=torch.long)
        idx_eve = torch.tensor(picks["idx_eve"].values, dtype=torch.long)
        phase_type = picks["phase_type"].values
        phase_time = torch.tensor(picks["travel_time"].values, dtype=torch.float32)
        phase_weight = torch.tensor(picks["phase_score"].values, dtype=torch.float32)
        weighted_mean = lambda x, w: np.sum(x * w) / np.sum(w)

    for i in range(MAX_SST_ITER):
        if ddp_local_rank == 0:
            with torch.inference_mode():
                picks["residual"] = (
                    travel_time(idx_sta, idx_eve, phase_type, phase_time, phase_weight)["residual"].detach().numpy()
                )
            # station_term = picks.groupby("idx_sta").agg({"residual": "mean"}).reset_index()
            station_term = (
                picks.groupby("idx_sta")
                .apply(lambda x: weighted_mean(x["residual"], x["phase_score"]))
                .reset_index(name="residual")
            )
            stations["station_term"] += (
                stations["idx_sta"].map(station_term.set_index("idx_sta")["residual"]).fillna(0)
            )
            raw_travel_time.station_dt.weight.data = torch.tensor(
                stations["station_term"].values, dtype=torch.float32
            ).view(-1, 1)

            # if ddp:
            #     dist.broadcast(raw_travel_time.station_dt.weight.data, src=0)

        optimizer = optim.Adam(params=travel_time.parameters(), lr=1.0)
        for j in range(EPOCHS):
            optimizer.zero_grad()

            prev_loss = 0
            loss = 0
            for meta in data_loader:

                loss_ = travel_time(
                    meta["idx_sta"],
                    meta["idx_eve"],
                    meta["phase_type"],
                    meta["phase_time"],
                    meta["phase_weight"],
                )["loss"]
                loss_.backward()
                if ddp:
                    dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
                loss += loss_

            if ddp_local_rank == 0 and j % 10 == 0:
                print(f"Epoch = {j}, Loss = {loss:.3f}")
            if abs(loss - prev_loss) < 1e-3:
                break
            prev_loss = loss

            optimizer.step()

        # if ddp:
        #     dist.barrier()
        if ddp_local_rank == 0:
            print(f"Loss (Adam):  {loss}")

            invert_event_loc = raw_travel_time.event_loc.weight.clone().detach().numpy()
            invert_event_time = raw_travel_time.event_time.weight.clone().detach().numpy()

            events = events_init.copy()
            events["time"] = events["time"] + pd.to_timedelta(np.squeeze(invert_event_time), unit="s")
            events["x_km"] = invert_event_loc[:, 0]
            events["y_km"] = invert_event_loc[:, 1]
            events["z_km"] = invert_event_loc[:, 2]
            events[["longitude", "latitude"]] = events.apply(
                lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
            )
            events["depth_km"] = events["z_km"]
            plotting(stations, figure_path, config, picks, events, events, suffix=f"sst_{i}")

    if ddp_local_rank == 0:
        plotting(stations, figure_path, config, picks, events, events, suffix="sst")

        # %% Location with grid search
        print(f"============================ Location with grid search ============================")
        event_loc = raw_travel_time.event_loc.weight.clone().detach().numpy()
        event_time = raw_travel_time.event_time.weight.clone().detach().numpy()
        nx, ny, nz = 11, 11, 21
        search_grid = np.stack(
            np.meshgrid(np.linspace(-5, 5, nx), np.linspace(-5, 5, ny), np.linspace(0, 20, nz), indexing="ij"), axis=-1
        ).reshape(-1, 3)
        num_grid = search_grid.shape[0]
        picks_ = picks.copy()
        event_loc_gs = []  # grid_search location
        event_time_gs = []  # grid_search time
        for i, (event_loc_, event_time_) in tqdm(
            enumerate(zip(event_loc, event_time)), desc="Grid search:", total=num_event
        ):
            picks_per_event = picks_[picks_["idx_eve"] == i]
            num_picks = len(picks_per_event)

            idx_sta = picks_per_event["idx_sta"].values
            phase_type = picks_per_event["phase_type"].values
            phase_weight = picks_per_event["phase_score"].values

            idx_eve = np.repeat(np.arange(num_grid), num_picks)
            idx_sta = np.tile(idx_sta, num_grid)
            phase_type = np.tile(phase_type, num_grid)
            station_dt_ = stations.iloc[idx_sta]["station_term"].values
            event_loc0 = event_loc_ - np.array([0, 0, np.round(event_loc_[2])])
            # event_loc0 = event_loc_ - np.array([0, 0, event_loc_[2]])
            event_time0 = event_time_
            tt = (
                event_time_
                + station_dt_
                + traveltime(
                    idx_eve,
                    idx_sta,
                    phase_type,
                    search_grid + event_loc0,
                    station_loc,
                    eikonal=config["eikonal"],
                )
            )
            tt = np.reshape(tt, (num_grid, num_picks))
            dt = tt - picks_per_event["travel_time"].values
            dt_mean = np.sum(dt * phase_weight, axis=-1) / np.sum(phase_weight)
            dt_std = np.sqrt(np.sum((dt - dt_mean[:, None]) ** 2 * phase_weight, axis=-1) / np.sum(phase_weight))
            idx = np.argmin(dt_std)
            event_loc_gs.append(search_grid[idx] + event_loc0)
            event_time_gs.append(np.mean(dt, axis=-1)[idx] + event_time0)

            # tt = np.reshape(tt, (nx, ny, nz, num_picks))
            # dt = np.reshape(dt, (nx, ny, nz, num_picks))
            # dt = np.sum(np.abs(dt) * phase_weight, axis=-1) / np.sum(phase_weight)
            # tmp_loc = search_grid[np.argmin(dt)]
            # tmp_x = np.linspace(-5, 5, nx)
            # tmp_y = np.linspace(-5, 5, ny)
            # tmp_z = np.linspace(0, 30, nz)
            # fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.8, 1, 1]})
            # im = ax[0].pcolormesh(tmp_x, tmp_y, dt[:, :, nz // 2].T, cmap="viridis")
            # ax[0].scatter(tmp_loc[0], tmp_loc[1], c="r", marker="x")
            # fig.colorbar(im, ax=ax[0])
            # ax[0].set_title("Depth")
            # im = ax[1].pcolormesh(tmp_x, tmp_z, dt[nx // 2, :, :].T, cmap="viridis")
            # ax[1].scatter(tmp_loc[1], tmp_loc[2], c="r", marker="x")
            # fig.colorbar(im, ax=ax[1])
            # ax[1].set_title("X")
            # im = ax[2].pcolormesh(tmp_y, tmp_z, dt[:, ny // 2, :].T, cmap="viridis")
            # ax[2].scatter(tmp_loc[0], tmp_loc[2], c="r", marker="x")
            # fig.colorbar(im, ax=ax[2])
            # ax[2].set_title("Y")
            # ax[1].invert_yaxis()
            # ax[2].invert_yaxis()
            # plt.savefig("debug.png", bbox_inches="tight", dpi=300)
            # plt.close(fig)

        events = events_init.copy()
        events["time"] = events["time"] + pd.to_timedelta(np.squeeze(event_time_gs), unit="s")
        events["x_km"] = np.array(event_loc_gs)[:, 0]
        events["y_km"] = np.array(event_loc_gs)[:, 1]
        events["z_km"] = np.array(event_loc_gs)[:, 2]
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z_km"]
        events.to_csv(
            f"{result_path}/adloc_events_grid_search.csv",
            index=False,
            float_format="%.5f",
            date_format="%Y-%m-%dT%H:%M:%S.%f",
        )

        plotting(stations, figure_path, config, picks, events, events, suffix="gridsearch")

import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pyproj import Proj
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
    EPOCHS = 100
    MAX_SST_ITER = 10

    # %%
    # ##################################### DEMO DATA #####################################
    # region = "synthetic"
    # # region = "ridgecrest"
    # data_path = f"test_data/{region}"
    # result_path = f"results/{region}"

    # picks_file = os.path.join(data_path, "picks.csv")
    # events_file = os.path.join(data_path, "events.csv")
    # stations_file = os.path.join(data_path, "stations.csv")
    # config_file = os.path.join(data_path, "config.json")

    # ## JSON format
    # # with open(args.stations, "r") as fp:
    # #     stations = json.load(fp)
    # # stations = pd.DataFrame.from_dict(stations, orient="index")
    # # stations["station_id"] = stations.index
    # ## CSV format
    # stations = pd.read_csv(stations_file)
    # picks = pd.read_csv(picks_file, parse_dates=["phase_time"])
    # events = pd.read_csv(events_file, parse_dates=["time"])
    # config = json.load(open(config_file))
    # config["mindepth"] = 0
    # config["maxdepth"] = 15

    # ## Eikonal for 1D velocity model
    # zz = [0.0, 5.5, 16.0, 32.0]
    # vp = [5.5, 5.5, 6.7, 7.8]
    # vp_vs_ratio = 1.73
    # vs = [v / vp_vs_ratio for v in vp]
    # h = 0.3

    # ##################################### DEMO DATA #####################################

    ##################################### Stanford DATA #####################################
    region = "stanford"
    data_path = f"./{region}/"
    result_path = f"results/{region}"
    figure_path = f"figures/{region}/"

    picks = pd.read_csv(f"{data_path}/phase.csv", parse_dates=["time"])
    events = None
    stations = pd.read_csv(f"{data_path}/station.csv")

    picks.rename({"time": "phase_time", "evid": "event_index", "phase": "phase_type"}, axis=1, inplace=True)
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    picks["station_id"] = picks["network"] + "." + picks["station"]
    picks["phase_score"] = 1.0

    stations.rename({"elevation": "elevation_m"}, axis=1, inplace=True)
    stations["station_id"] = stations["network"] + "." + stations["station"]

    config = {
        "maxlongitude": -117.10,
        "minlongitude": -118.2,
        "maxlatitude": 36.4,
        "minlatitude": 35.3,
        "mindepth": 0,
        "maxdepth": 15,
    }

    ## Eikonal for 1D velocity model
    zz = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0]
    vp = [4.746, 4.793, 4.799, 5.045, 5.721, 5.879, 6.504, 6.708, 6.725, 7.800]
    vs = [2.469, 2.470, 2.929, 2.930, 3.402, 3.403, 3.848, 3.907, 3.963, 4.500]
    h = 0.3

    ##################################### Stanford DATA #####################################

    # %%
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

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
    stations["x_km"], stations["y_km"] = proj(stations["longitude"], stations["latitude"])
    stations["z_km"] = stations["depth_km"]
    if events is not None:
        events["x_km"], events["y_km"] = proj(events["longitude"], events["latitude"])
        events["z_km"] = events["depth_km"]

    if ("xlim_km" not in config) or ("ylim_km" not in config) or ("zlim_km" not in config):
        xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
        xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
        zmin, zmax = config["mindepth"], config["maxdepth"]
        config["xlim_km"] = (xmin, xmax)
        config["ylim_km"] = (ymin, ymax)
        config["zlim_km"] = (zmin, zmax)

    # %%
    mapping_phase_type_int = {"P": 0, "S": 1}
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    config["eikonal"] = None

    ## Eikonal for 1D velocity model
    # zz = [0.0, 5.5, 16.0, 32.0]
    # vp = [5.5, 5.5, 6.7, 7.8]
    # vp_vs_ratio = 1.73
    # vs = [v / vp_vs_ratio for v in vp]
    # zz = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0]
    # vp = [4.746, 4.793, 4.799, 5.045, 5.721, 5.879, 6.504, 6.708, 6.725, 7.800]
    # vs = [2.469, 2.470, 2.929, 2.930, 3.402, 3.403, 3.848, 3.907, 3.963, 4.500]
    # h = 0.3
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
    if events is not None:
        events["idx_eve"] = np.arange(len(events))
    else:
        picks = picks.merge(stations[["station_id", "x_km", "y_km", "z_km"]], on="station_id")
        events = picks.groupby("event_index").agg({"x_km": "mean", "y_km": "mean", "z_km": "mean", "phase_time": "min"})
        events["z_km"] = 10.0  # km default depth
        events.rename({"phase_time": "time"}, axis=1, inplace=True)
        events["event_index"] = events.index
        events.reset_index(drop=True, inplace=True)
        events["idx_eve"] = np.arange(len(events))

    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    # %%
    picks = picks.merge(events[["idx_eve", "time"]], on="idx_eve")
    picks["travel_time"] = (picks["phase_time"] - picks["time"]).dt.total_seconds()

    # %%
    num_event = len(events)
    num_station = len(stations)

    station_loc = stations[["x_km", "y_km", "z_km"]].values
    station_dt = None

    # %%
    events_init = events.copy()

    phase_dataset = PhaseDataset(
        picks,
        events,
        stations,
        config=config,
    )
    data_loader = DataLoader(phase_dataset, batch_size=None, shuffle=False, num_workers=0)

    # %%
    travel_time = TravelTime(
        num_event,
        num_station,
        station_loc,
        event_loc=events[["x_km", "y_km", "z_km"]].values,
        velocity={"P": vp, "S": vs},
        eikonal=config["eikonal"],
    )

    # %% Conventional location
    print(f"Dataset: {len(picks)} picks, {len(events)} events, {len(stations)} stations, {len(phase_dataset)} batches")
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
            loss += loss_

        if i % 10 == 0:
            print(f"Epoch = {i}, Loss = {loss:.3f}")
        if abs(loss - prev_loss) < 1e-3:
            break
        prev_loss = loss

        optimizer.step()
        with torch.no_grad():
            travel_time.event_loc.weight.data[:, 2] = torch.clamp(
                travel_time.event_loc.weight.data[:, 2], min=config["zlim_km"][0] + 0.1, max=config["zlim_km"][1] - 0.1
            )

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
    invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    invert_event_time = travel_time.event_time.weight.clone().detach().numpy()

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
    print(f"============================ Location with SST ============================")

    idx_sta = torch.tensor(picks["idx_sta"].values, dtype=torch.long)
    idx_eve = torch.tensor(picks["idx_eve"].values, dtype=torch.long)
    phase_type = picks["phase_type"].values
    phase_time = torch.tensor(picks["travel_time"].values, dtype=torch.float32)
    phase_weight = torch.tensor(picks["phase_score"].values, dtype=torch.float32)
    weighted_mean = lambda x, w: np.sum(x * w) / np.sum(w)

    for i in range(MAX_SST_ITER):
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
        stations["station_term"] += stations["idx_sta"].map(station_term.set_index("idx_sta")["residual"]).fillna(0)
        travel_time.station_dt.weight.data = torch.tensor(stations["station_term"].values, dtype=torch.float32).view(
            -1, 1
        )

        optimizer = optim.Adam(params=travel_time.parameters(), lr=0.01)
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
                loss += loss_

            if j % 10 == 0:
                print(f"Epoch = {j}, Loss = {loss:.3f}")
            if abs(loss - prev_loss) < 1e-3:
                break
            prev_loss = loss

            optimizer.step()
            with torch.no_grad():
                travel_time.event_loc.weight.data[:, 2] = torch.clamp(
                    travel_time.event_loc.weight.data[:, 2],
                    min=config["zlim_km"][0] + 0.1,
                    max=config["zlim_km"][1] - 0.1,
                )

        print(f"Loss (Adam):  {loss}")

        # %%
        invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
        invert_event_time = travel_time.event_time.weight.clone().detach().numpy()

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

    # %%
    invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    invert_event_time = travel_time.event_time.weight.clone().detach().numpy()

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
        f"{result_path}/adloc_events_sst.csv", index=False, float_format="%.5f", date_format="%Y-%m-%dT%H:%M:%S.%f"
    )
    plotting(stations, figure_path, config, picks, events, events, suffix="sst")

    # %% Location with grid search
    print(f"============================ Location with grid search ============================")
    event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    event_time = travel_time.event_time.weight.clone().detach().numpy()
    nx, ny, nz = 11, 11, 21
    # nx * ny * nz, 3
    z = np.linspace(config["zlim_km"][0], config["zlim_km"][1], nz)
    search_grid = np.stack(
        np.meshgrid(np.linspace(-5, 5, nx), np.linspace(-5, 5, ny), z, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    num_grid = search_grid.shape[0]
    picks_ = picks.copy()
    event_loc_gs = []  # grid_search location
    event_time_gs = []  # grid_search time
    event_uncertainty = []
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
        # event_loc0 = event_loc_ - np.array([0, 0, np.round(event_loc_[2])])
        # make suare event_loc_[2] can be sampled
        if event_loc_[2] < z[0]:
            shiftz = event_loc_[2] - z[0]
        else:
            shiftz = event_loc_[2] - z[z <= event_loc_[2]][-1]
        event_loc0 = np.array([event_loc_[0], event_loc_[1], shiftz])
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
        dt_mean = np.sum(dt * phase_weight, axis=-1, keepdims=True) / np.sum(phase_weight)
        dt_std = np.sqrt(np.sum((dt - dt_mean) ** 2 * phase_weight, axis=-1) / np.sum(phase_weight))
        idx = np.argmin(dt_std)
        event_loc_gs.append(search_grid[idx] + event_loc0)
        event_time_gs.append(np.mean(dt, axis=-1)[idx] + event_time0)

        ## uncertainty
        T = 0.03  # temperature
        loss = np.sum(np.abs(dt) * phase_weight, axis=-1, keepdims=True) / np.sum(phase_weight)
        prob = np.exp(-loss / T)
        prob /= np.sum(prob)
        mean = np.sum(search_grid * prob, axis=0)
        variance = np.sum((search_grid - mean) ** 2 * prob, axis=0)
        uncertainty = np.sqrt(variance)
        event_uncertainty.append(uncertainty)

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
    events[["sigma_x_km", "sigma_y_km", "sigma_z_km"]] = np.array(event_uncertainty)
    events["x_km"] = np.array(event_loc_gs)[:, 0]
    events["y_km"] = np.array(event_loc_gs)[:, 1]
    events["z_km"] = np.array(event_loc_gs)[:, 2]
    events[["longitude", "latitude"]] = events.apply(
        lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
    )
    events["depth_km"] = events["z_km"]
    events.to_csv(
        f"{result_path}/adloc_events_gridsearch.csv",
        index=False,
        float_format="%.5f",
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )

    plotting(stations, figure_path, config, picks, events, events, suffix="gridsearch")

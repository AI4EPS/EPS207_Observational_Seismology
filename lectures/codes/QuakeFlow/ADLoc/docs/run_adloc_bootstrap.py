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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import utils
from adloc import PhaseDataset
from adloc.adloc import TravelTime
from adloc.eikonal2d import init_eikonal2d, traveltime

torch.manual_seed(0)
np.random.seed(0)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    parser.add_argument("-dd", "--double_difference", action="store_true", help="Use double difference")
    parser.add_argument("--eikonal", action="store_true", help="Use eikonal")
    parser.add_argument("--dd_weight", default=1.0, type=float, help="weight for double difference")
    parser.add_argument("--min_pair_dist", default=20.0, type=float, help="minimum distance between pairs")
    parser.add_argument("--max_time_res", default=0.5, type=float, help="maximum time residual")
    parser.add_argument("--config", default="test_data/synthetic/config.json", type=str, help="config file")
    parser.add_argument("--stations", type=str, default="test_data/synthetic/stations.csv", help="station json")
    parser.add_argument("--picks", type=str, default="test_data/synthetic/picks.csv", help="picks csv")
    parser.add_argument("--events", type=str, default="test_data/synthetic/events.csv", help="events csv")
    parser.add_argument("--result_path", type=str, default="results/synthetic", help="result path")
    parser.add_argument("--figure_path", type=str, default="figures/synthetic", help="result path")
    # parser.add_argument("--config", default="test_data/ridgecrest/config.json", type=str, help="config file")
    # parser.add_argument("--stations", type=str, default="test_data/ridgecrest/stations.csv", help="station json")
    # parser.add_argument("--picks", type=str, default="test_data/ridgecrest/gamma_picks.csv", help="picks csv")
    # parser.add_argument("--events", type=str, default="test_data/ridgecrest/gamma_events.csv", help="events csv")
    # parser.add_argument("--result_path", type=str, default="results/ridgecrest", help="result path")
    # parser.add_argument("--figure_path", type=str, default="figures/ridgecrest", help="result path")
    parser.add_argument("--bootstrap", default=0, type=int, help="bootstrap")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch_size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=1000, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="lbfgs", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument(
        "--wd",
        "--weight_decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser.parse_args()


def plotting(stations, figure_path, config, picks, events_old, locations, station_term=None, iter=0):

    vmin = min(locations["z_km"].min(), events_old["depth_km"].min())
    vmax = max(locations["z_km"].max(), events_old["depth_km"].max())
    # xmin, xmax = stations["x_km"].min(), stations["x_km"].max()
    # ymin, ymax = stations["y_km"].min(), stations["y_km"].max()
    xmin = min(stations["x_km"].min(), locations["x_km"].min())
    xmax = max(stations["x_km"].max(), locations["x_km"].max())
    ymin = min(stations["y_km"].min(), locations["y_km"].min())
    ymax = max(stations["y_km"].max(), locations["y_km"].max())
    zmin, zmax = config["zlim_km"]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
    # fig, ax = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={"height_ratios": [2, 1]})
    im = ax[0, 0].scatter(
        locations["x_km"],
        locations["y_km"],
        c=locations["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    cbar = fig.colorbar(im, ax=ax[0, 0])
    cbar.set_label("Depth (km)")
    ax[0, 0].set_title(f"ADLoc: {len(locations)} events")

    im = ax[0, 1].scatter(
        stations["x_km"],
        stations["y_km"],
        c=stations["station_term"],
        cmap="viridis_r",
        s=100,
        marker="^",
        alpha=0.5,
    )
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    cbar = fig.colorbar(im, ax=ax[0, 1])
    cbar.set_label("Residual (s)")
    ax[0, 1].set_title(f"Station term: {np.mean(np.abs(stations['station_term'].values)):.4f} s")

    im = ax[1, 0].scatter(
        locations["x_km"],
        locations["z_km"],
        c=locations["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
    )
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([zmax, zmin])
    cbar = fig.colorbar(im, ax=ax[1, 0])
    cbar.set_label("Depth (km)")

    im = ax[1, 1].scatter(
        locations["y_km"],
        locations["z_km"],
        c=locations["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
    )
    ax[1, 1].set_xlim([ymin, ymax])
    ax[1, 1].set_ylim([zmax, zmin])
    cbar = fig.colorbar(im, ax=ax[1, 1])
    cbar.set_label("Depth (km)")
    plt.savefig(os.path.join(figure_path, f"location_{iter}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def optimize(args, config, data_loader, travel_time):
    if (args.opt.lower() == "lbfgs") or (args.opt.lower() == "bfgs"):
        optimizer = optim.LBFGS(params=travel_time.parameters(), max_iter=1000, line_search_fn="strong_wolfe", lr=1.0)
    elif args.opt.lower() == "adam":
        optimizer = optim.Adam(params=travel_time.parameters(), lr=1.0)
    elif args.opt.lower() == "sgd":
        optimizer = optim.SGD(params=travel_time.parameters(), lr=10.0)
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    # init loss
    loss = 0
    for meta in data_loader:
        station_index = meta["station_index"]
        event_index = meta["event_index"]
        phase_time = meta["phase_time"]
        phase_type = meta["phase_type"]
        phase_weight = meta["phase_weight"]
        loss += travel_time(
            station_index,
            event_index,
            phase_type,
            phase_time,
            phase_weight,
        )["loss"]
        if args.distributed:
            dist.barrier()
            dist.all_reduce(loss)
    print(f"Init loss: {loss}")

    if (args.opt.lower() == "lbfgs") or (args.opt.lower() == "bfgs"):

        def closure():
            travel_time.event_loc.weight.data[:, 0].clamp_(min=config["xlim_km"][0], max=config["xlim_km"][1])
            travel_time.event_loc.weight.data[:, 1].clamp_(min=config["ylim_km"][0], max=config["ylim_km"][1])
            travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])
            optimizer.zero_grad()
            for meta in data_loader:
                station_index = meta["station_index"]
                event_index = meta["event_index"]
                phase_time = meta["phase_time"]
                phase_type = meta["phase_type"]
                phase_weight = meta["phase_weight"]
                loss = travel_time(
                    station_index,
                    event_index,
                    phase_type,
                    phase_time,
                    phase_weight,
                )["loss"]
                if args.distributed:
                    dist.barrier()
                    dist.all_reduce(loss)
                loss.backward()
            return loss

        optimizer.step(closure)
        travel_time.event_loc.weight.data[:, 0].clamp_(min=config["xlim_km"][0], max=config["xlim_km"][1])
        travel_time.event_loc.weight.data[:, 1].clamp_(min=config["ylim_km"][0], max=config["ylim_km"][1])
        travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])

    else:
        for i in range(args.epochs):
            optimizer.zero_grad()

            prev_loss = 0
            loss = 0
            for meta in data_loader:
                station_index = meta["station_index"]
                event_index = meta["event_index"]
                phase_time = meta["phase_time"]
                phase_type = meta["phase_type"]
                phase_weight = meta["phase_weight"]

                loss = travel_time(
                    station_index,
                    event_index,
                    phase_type,
                    phase_time,
                    phase_weight,
                )["loss"]
                loss.backward()

            optimizer.step()

            if abs(loss - prev_loss) < 1e-3:
                print(f"Loss:  {loss}")
                break
            prev_loss = loss

            if i % 10 == 0:
                print(f"Epoch = {i}, Loss = {loss:.3f}")

            # set variable range
            # travel_time.event_loc.weight.data[:, 2] += (
            #     torch.randn_like(travel_time.event_loc.weight.data[:, 2]) * (args.epochs - i) / args.epochs
            # )
            travel_time.event_loc.weight.data[:, 0].clamp_(min=config["xlim_km"][0], max=config["xlim_km"][1])
            travel_time.event_loc.weight.data[:, 1].clamp_(min=config["ylim_km"][0], max=config["ylim_km"][1])
            travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])

    loss = 0
    for meta in data_loader:
        station_index = meta["station_index"]
        event_index = meta["event_index"]
        phase_time = meta["phase_time"]
        phase_type = meta["phase_type"]
        phase_weight = meta["phase_weight"]
        loss += travel_time(
            station_index,
            event_index,
            phase_type,
            phase_time,
            phase_weight,
        )["loss"]

    print(f"Invert loss: {loss}")


def main(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)

    with open(args.config, "r") as fp:
        config = json.load(fp)

    if "latitude0" not in config:
        config["latitude0"] = (config["minlatitude"] + config["maxlatitude"]) / 2
    if "longitude0" not in config:
        config["longitude0"] = (config["minlongitude"] + config["maxlongitude"]) / 2
    if "mindepth" not in config:
        config["mindepth"] = 0.0
    if "maxdepth" not in config:
        config["maxdepth"] = 20.0

    # %%
    if ("xlim_km" not in config) or ("ylim_km" not in config) or ("zlim_km" not in config):
        proj = Proj(f"+proj=sterea +lon_0={config['longitude0']} +lat_0={config['latitude0']} +units=km")
        config["xlim_km"] = proj(
            longitude=[config["minlongitude"], config["maxlongitude"]], latitude=[config["latitude0"]] * 2
        )[0]
        config["ylim_km"] = proj(
            longitude=[config["longitude0"]] * 2, latitude=[config["minlatitude"], config["maxlatitude"]]
        )[1]
        config["zlim_km"] = [config["mindepth"], config["maxdepth"]]

    vp = 6.0
    vs = vp / 1.73
    eikonal = None

    if args.eikonal:
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
        eikonal = init_eikonal2d(config["eikonal"])

    # %% JSON format
    # with open(args.stations, "r") as fp:
    #     stations = json.load(fp)
    # stations = pd.DataFrame.from_dict(stations, orient="index")
    # stations["station_id"] = stations.index
    # %% CSV format
    stations = pd.read_csv(args.stations)
    picks = pd.read_csv(args.picks, parse_dates=["phase_time"])
    events = pd.read_csv(args.events, parse_dates=["time"])

    # %%
    lon0 = stations["longitude"].median()
    lat0 = stations["latitude"].median()
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")
    if "depth_km" not in stations:
        stations["depth_km"] = -stations["elevation_m"] / 1000
    if "station_term" not in stations:
        stations["station_term"] = 0.0
    mapping_int = {"P": 0, "S": 1}
    if ("P" in picks["phase_type"].values) or ("S" in picks["phase_type"].values):
        picks["phase_type"] = picks["phase_type"].apply(lambda x: mapping_int[x])

    # %%
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z_km"] = stations["depth_km"]
    events[["x_km", "y_km"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events["z_km"] = events["depth_km"]

    # %%
    stations["idx_sta"] = stations.index  # reindex in case the index does not start from 0 or is not continuous
    events["idx_eve"] = events.index  # reindex in case the index does not start from 0 or is not continuous

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
    events_origin = events.copy()
    # %%
    utils.init_distributed_mode(args)
    print(args)

    phase_dataset = PhaseDataset(picks, events, stations, config=args)
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(phase_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(phase_dataset)
    data_loader = DataLoader(phase_dataset, batch_size=None, sampler=sampler, num_workers=args.workers, collate_fn=None)

    # %%
    event_loc_init = np.zeros((num_event, 3))
    event_loc_init[:, 2] = np.mean(config["zlim_km"])
    travel_time = TravelTime(
        num_event,
        num_station,
        station_loc,
        event_loc=event_loc_init,  # Initial location
        # event_loc=events[["x_km", "y_km", "z_km"]].values,
        # event_time=event_time,
        velocity={"P": vp, "S": vs},
        eikonal=eikonal,
    )

    # %% Conventional location
    print(f"Dataset: {len(picks)} picks, {len(events)} events, {len(stations)} stations, {len(phase_dataset)} batches")
    print(f"============================ Basic location ============================")
    for i in range(10):
        optimize(args, config, data_loader, travel_time)

    # %%
    invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    invert_event_time = travel_time.event_time.weight.clone().detach().numpy()

    events = events_origin.copy()
    events["time"] = events["time"] + pd.to_timedelta(np.squeeze(invert_event_time), unit="s")
    events["x_km"] = invert_event_loc[:, 0]
    events["y_km"] = invert_event_loc[:, 1]
    events["z_km"] = invert_event_loc[:, 2]
    events[["longitude", "latitude"]] = events.apply(
        lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
    )
    events["depth_km"] = events["z_km"]
    events.to_csv(
        f"{args.result_path}/adloc_events.csv", index=False, float_format="%.5f", date_format="%Y-%m-%dT%H:%M:%S.%f"
    )

    plotting(stations, args.figure_path, config, picks, events, events, station_dt, suffix="basic")

    # %%
    events_boostrap = []
    for i in tqdm(range(args.bootstrap), desc="Bootstrapping:"):
        picks_by_event = picks.groupby("idx_eve")
        # picks_by_event_sample = picks_by_event.apply(lambda x: x.sample(frac=1.0, replace=True))  # Bootstrap
        picks_by_event_sample = picks_by_event.apply(lambda x: x.sample(n=len(x) - 1, replace=False))  # Jackknife
        picks_sample = picks_by_event_sample.reset_index(drop=True)
        phase_dataset = PhaseDataset(picks_sample, events, stations, config=args)
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(phase_dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(phase_dataset)
        data_loader = DataLoader(
            phase_dataset, batch_size=None, sampler=sampler, num_workers=args.workers, collate_fn=None
        )

        travel_time = TravelTime(
            num_event,
            num_station,
            station_loc,
            station_dt=station_dt,
            event_loc=event_loc_init,  # Initial location
            velocity={"P": vp, "S": vs},
            eikonal=eikonal,
        )
        optimize(args, config, data_loader, travel_time)

        invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
        invert_event_time = travel_time.event_time.weight.clone().detach().numpy()
        invert_station_dt = travel_time.station_dt.weight.clone().detach().numpy()
        events_invert = events[["idx_eve", "event_index"]].copy()
        events_invert["t_s"] = invert_event_time[:, 0]
        events_invert["x_km"] = invert_event_loc[:, 0]
        events_invert["y_km"] = invert_event_loc[:, 1]
        events_invert["z_km"] = invert_event_loc[:, 2]
        events_boostrap.append(events_invert)

    if len(events_boostrap) > 0:
        events_boostrap = pd.concat(events_boostrap)
        events_by_index = events_boostrap.groupby("idx_eve")
        events_boostrap = events_by_index.mean()
        events_boostrap_std = events_by_index.std()
        events_boostrap["time"] = events["time"] + pd.to_timedelta(np.squeeze(events_boostrap["t_s"]), unit="s")
        events_boostrap[["longitude", "latitude"]] = events_boostrap.apply(
            lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        )
        events_boostrap["depth_km"] = events_boostrap["z_km"]
        events_boostrap["std_x_km"] = events_boostrap_std["x_km"]
        events_boostrap["std_y_km"] = events_boostrap_std["y_km"]
        events_boostrap["std_z_km"] = events_boostrap_std["z_km"]
        events_boostrap["std_t_s"] = events_boostrap_std["t_s"]
        events_boostrap.reset_index(inplace=True)
        events_boostrap["event_index"] = events_boostrap["event_index"].astype(int)
        events_boostrap["idx_eve"] = events_boostrap["idx_eve"].astype(int)
        events_boostrap.to_csv(
            f"{args.result_path}/adloc_events_bootstrap.csv",
            index=False,
            float_format="%.5f",
            date_format="%Y-%m-%dT%H:%M:%S.%f",
        )

        plotting(stations, args.figure_path, config, picks, events, events, station_dt, suffix="bootstrap")


if __name__ == "__main__":
    args = get_args_parser()

    main(args)

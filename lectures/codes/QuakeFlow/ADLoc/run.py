import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import utils
from adloc import PhaseDataset, TravelTime, initialize_eikonal, optimize
from matplotlib import pyplot as plt
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    parser.add_argument("--config", default="config.json", type=str, help="config file")
    parser.add_argument("--stations", type=str, default="tests/stations.json", help="station json")
    parser.add_argument("--picks", type=str, default="tests/picks.csv", help="picks csv")
    parser.add_argument("--events", type=str, default="tests/events.csv", help="events csv")
    parser.add_argument("--result_path", type=str, default="results", help="result path")
    parser.add_argument("--bootstrap", default=0, type=int, help="bootstrap")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch_size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
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


def main(args):
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
    if "degree2km" not in config:
        config["degree2km"] = 111.19

    proj = Proj(f"+proj=sterea +lon_0={config['longitude0']} +lat_0={config['latitude0']} +units=km")

    # %%
    config["xlim_km"] = proj(
        longitude=[config["minlongitude"], config["maxlongitude"]], latitude=[config["latitude0"]] * 2
    )
    config["ylim_km"] = proj(
        longitude=[config["longitude0"]] * 2, latitude=[config["minlatitude"], config["maxlatitude"]]
    )
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
        h = 1.0
        vel = {"z": zz, "p": vp, "s": vs}
        config["eikonal"] = {
            "vel": vel,
            "h": h,
            "xlim": config["xlim_km"],
            "ylim": config["ylim_km"],
            "zlim": config["zlim_km"],
        }
        eikonal = initialize_eikonal(config["eikonal"])

    # %%
    with open(args.stations, "r") as fp:
        stations = json.load(fp)
    stations = pd.DataFrame.from_dict(stations, orient="index")
    stations["station_id"] = stations.index
    # stations = pd.read_csv(args.stations)
    picks = pd.read_csv(args.picks, parse_dates=["phase_time"])
    events = pd.read_csv(args.events, parse_dates=["time"])

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
    num_event = len(events)
    num_station = len(stations)

    stations.reset_index(inplace=True, drop=True)
    stations["index"] = stations.index.values  # reindex starts from 0 continuously
    stations.set_index("station_id", inplace=True)
    station_loc = stations[["x_km", "y_km", "z_km"]].values
    station_dt = None

    events.reset_index(inplace=True, drop=True)
    events["index"] = events.index.values  # reindex starts from 0 continuously
    event_loc = events[["x_km", "y_km", "z_km"]].values
    event_time = events["time"].values

    event_index_map = {x: i for i, x in enumerate(events["event_index"])}
    picks = picks[picks["event_index"] != -1]
    picks["index"] = picks["event_index"].apply(lambda x: event_index_map[x])  # map index starts from 0 continuously
    picks["phase_time"] = picks.apply(lambda x: (x["phase_time"] - event_time[x["index"]]).total_seconds(), axis=1)

    if args.double_difference:
        picks = picks.merge(events[["index", "x_km", "y_km", "z_km"]], left_on="index", right_on="index")

    # %%
    utils.init_distributed_mode(args)
    print(args)

    phase_dataset = PhaseDataset(picks, events, stations, double_difference=False, config=args)
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(phase_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(phase_dataset)
    data_loader = DataLoader(phase_dataset, batch_size=None, sampler=sampler, num_workers=args.workers, collate_fn=None)

    if args.double_difference:
        phase_dataset_dd = PhaseDataset(picks, events, stations, double_difference=True, config=args)
        if args.distributed:
            sampler_dd = torch.utils.data.distributed.DistributedSampler(phase_dataset_dd, shuffle=False)
        else:
            sampler_dd = torch.utils.data.SequentialSampler(phase_dataset_dd)

        data_loader_dd = DataLoader(
            phase_dataset_dd, batch_size=None, sampler=sampler_dd, num_workers=args.workers, collate_fn=None
        )
    else:
        data_loader_dd = None

    # %%
    event_loc_init = np.zeros((num_event, 3))
    event_loc_init[:, 2] = np.mean(config["zlim_km"])
    travel_time = TravelTime(
        num_event,
        num_station,
        station_loc,
        event_loc=event_loc_init,  # Initial location
        # event_loc=event_loc,  # Initial location
        # event_time=event_time,
        velocity={"P": vp, "S": vs},
        eikonal=eikonal,
    )

    print(f"Dataset: {len(picks)} picks, {len(events)} events, {len(stations)} stations, {len(phase_dataset)} batches")
    optimize(args, config, data_loader, data_loader_dd, travel_time)

    # %%
    station_dt = travel_time.station_dt.weight.clone().detach().numpy()
    print(
        f"station_dt: max = {np.max(station_dt)}, min = {np.min(station_dt)}, median = {np.median(station_dt)}, mean = {np.mean(station_dt)}, std = {np.std(station_dt)}"
    )
    invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    invert_event_time = travel_time.event_time.weight.clone().detach().numpy()
    invert_station_dt = travel_time.station_dt.weight.clone().detach().numpy()

    stations["dt_s"] = invert_station_dt[:, 0]
    with open(f"{args.result_path}/stations.json", "w") as fp:
        json.dump(stations.to_dict(orient="index"), fp, indent=4)
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

    events_boostrap = []
    for i in tqdm(range(args.bootstrap), desc="Bootstrapping:"):
        picks_by_event = picks.groupby("index")
        # picks_by_event_sample = picks_by_event.apply(lambda x: x.sample(frac=1.0, replace=True))  # Bootstrap
        picks_by_event_sample = picks_by_event.apply(lambda x: x.sample(n=len(x) - 1, replace=False))  # Jackknife
        picks_sample = picks_by_event_sample.reset_index(drop=True)
        phase_dataset = PhaseDataset(picks_sample, events, stations, double_difference=False, config=args)
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
            event_loc=event_loc_init,  # Initial location
            velocity={"P": vp, "S": vs},
            eikonal=eikonal,
        )
        optimize(args, config, data_loader, data_loader_dd, travel_time)

        invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
        invert_event_time = travel_time.event_time.weight.clone().detach().numpy()
        invert_station_dt = travel_time.station_dt.weight.clone().detach().numpy()
        events_invert = events[["index", "event_index"]].copy()
        events_invert["t_s"] = invert_event_time[:, 0]
        events_invert["x_km"] = invert_event_loc[:, 0]
        events_invert["y_km"] = invert_event_loc[:, 1]
        events_invert["z_km"] = invert_event_loc[:, 2]
        events_boostrap.append(events_invert)

    if len(events_boostrap) > 0:
        events_boostrap = pd.concat(events_boostrap)
        events_by_index = events_boostrap.groupby("index")
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
        events_boostrap["index"] = events_boostrap["index"].astype(int)
        events_boostrap.to_csv(
            f"{args.result_path}/adloc_events_bootstrap.csv",
            index=False,
            float_format="%.5f",
            date_format="%Y-%m-%dT%H:%M:%S.%f",
        )


if __name__ == "__main__":
    args = get_args_parser()

    main(args)

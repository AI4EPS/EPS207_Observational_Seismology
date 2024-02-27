# %%
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import utils
from adloc.seismic_ops import initialize_eikonal
from adloc.travel_time import CalcTravelTime

torch.manual_seed(0)
np.random.seed(0)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    parser.add_argument("-dd", "--double_difference", action="store_true", help="Use double difference")
    parser.add_argument("--eikonal", action="store_true", help="Use eikonal")
    parser.add_argument("--dd_weight", default=1.0, type=float, help="weight for double difference")
    parser.add_argument("--min_pair_dist", default=3.0, type=float, help="minimum distance between pairs")
    parser.add_argument("--max_time_res", default=0.5, type=float, help="maximum time residual")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
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
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser


class PhaseDataset(Dataset):
    def __init__(self, picks, events, stations, double_difference=False, config=None):
        self.picks = picks
        self.events = events
        self.stations = stations
        self.config = config
        self.double_difference = double_difference
        if double_difference:
            self.read_data_dd()
        else:
            self.read_data()

    def __len__(self):
        ## TODO: return batch
        return 1

    def read_data(self):
        # event_index = []
        # station_index = []
        # phase_score = []
        # phase_time = []
        # phase_type = []

        # for i in range(len(self.events)):
        #     phase_time.append(
        #         self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_time"].values
        #     )
        #     phase_score.append(
        #         self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_score"].values
        #     )
        #     phase_type.extend(
        #         self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_type"].values.tolist()
        #     )
        #     event_index.extend([i] * len(self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]))
        #     station_index.append(
        #         self.stations.loc[
        #             self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["station_id"],
        #             "index",
        #         ].values
        #     )

        # phase_time = np.concatenate(phase_time)
        # phase_score = np.concatenate(phase_score)
        # phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
        # event_index = np.array(event_index)
        # station_index = np.concatenate(station_index)

        # self.station_index = torch.tensor(station_index, dtype=torch.long)
        # self.event_index = torch.tensor(event_index, dtype=torch.long)
        # self.phase_weight = torch.tensor(phase_score, dtype=torch.float32)
        # self.phase_time = torch.tensor(phase_time[:, np.newaxis], dtype=torch.float32)
        # self.phase_type = torch.tensor(phase_type, dtype=torch.long)

        # print(
        #     f"{self.station_index.shape = }, {self.event_index.shape = }, {self.phase_weight.shape = }, {self.phase_time.shape = }, {self.phase_type.shape = }"
        # )

        event_index = []
        station_index = []
        phase_score = []
        phase_time = []
        phase_type = []

        picks_by_event = self.picks.groupby("index")
        for key, group in picks_by_event:
            if key == -1:
                continue
            phase_time.append(group["phase_time"].values)
            phase_score.append(group["phase_score"].values)
            phase_type.extend(group["phase_type"].values.tolist())
            event_index.extend([key] * len(group))
            station_index.append(self.stations.loc[group["station_id"], "index"].values)

        phase_time = np.concatenate(phase_time)
        phase_score = np.concatenate(phase_score)
        phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
        event_index = np.array(event_index)
        station_index = np.concatenate(station_index)

        self.station_index = torch.tensor(station_index, dtype=torch.long)
        self.event_index = torch.tensor(event_index, dtype=torch.long)
        self.phase_weight = torch.tensor(phase_score, dtype=torch.float32)
        self.phase_time = torch.tensor(phase_time, dtype=torch.float32)
        self.phase_type = torch.tensor(phase_type, dtype=torch.long)

        # print(
        #     f"{self.station_index.shape = }, {self.event_index.shape = }, {self.phase_weight.shape = }, {self.phase_time.shape = }, {self.phase_type.shape = }"
        # )

    def read_data_dd(self):
        event_index = []
        station_index = []
        phase_score = []
        phase_time = []
        phase_type = []

        event_loc = self.events[["x_km", "y_km", "z_km"]].values
        event_time = self.events["time"].values[:, np.newaxis]

        neigh = NearestNeighbors(radius=self.config.min_pair_dist, metric="euclidean")
        neigh.fit(event_loc)

        picks_by_event = self.picks.groupby("index")

        for key1, group1 in tqdm(picks_by_event, total=len(picks_by_event), desc="Generating pairs"):
            if key1 == -1:
                continue

            for key2 in neigh.radius_neighbors([event_loc[key1]], return_distance=False)[0]:
                if key1 >= key2:
                    continue

                common = group1.merge(picks_by_event.get_group(key2), on=["station_id", "phase_type"], how="inner")
                phase_time.append(common["phase_time_x"].values - common["phase_time_y"].values)
                phase_score.append(common["phase_score_x"].values * common["phase_score_y"].values)
                phase_type.extend(common["phase_type"].values.tolist())
                event_index.extend([[key1, key2]] * len(common))
                station_index.append(self.stations.loc[common["station_id"], "index"].values)

        phase_time = np.concatenate(phase_time)
        phase_score = np.concatenate(phase_score)
        phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
        event_index = np.array(event_index)
        station_index = np.concatenate(station_index)

        # %%
        self.station_index = torch.tensor(station_index, dtype=torch.long)
        self.event_index = torch.tensor(event_index, dtype=torch.long)
        self.phase_weight = torch.tensor(phase_score, dtype=torch.float32)
        self.phase_time = torch.tensor(phase_time, dtype=torch.float32)
        self.phase_type = torch.tensor(phase_type, dtype=torch.long)

    def __getitem__(self, i):
        # phase_time = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_time"].values
        # phase_score = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_score"].values
        # phase_type = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]][
        #     "phase_type"
        # ].values.tolist()
        # event_index = np.array([i] * len(self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]))
        # station_index = self.stations.loc[
        #     self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["station_id"], "index"
        # ].values

        return {
            "event_index": self.event_index,
            "station_index": self.station_index,
            "phase_time": self.phase_time,
            "phase_weight": self.phase_weight,
            "phase_type": self.phase_type,
        }


# %%
class TravelTime(nn.Module):
    def __init__(
        self,
        num_event,
        num_station,
        station_loc,
        station_dt=None,
        event_loc=None,
        event_time=None,
        reg=0.1,
        velocity={"P": 6.0, "S": 6.0 / 1.73},
        eikonal=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.num_event = num_event
        self.event_loc = nn.Embedding(num_event, 3)
        self.event_time = nn.Embedding(num_event, 1)
        self.station_loc = nn.Embedding(num_station, 3)
        self.station_dt = nn.Embedding(num_station, 2)  # vp, vs
        self.station_loc.weight = torch.nn.Parameter(torch.tensor(station_loc, dtype=dtype), requires_grad=False)
        if station_dt is not None:
            self.station_dt.weight = torch.nn.Parameter(torch.tensor(station_dt, dtype=dtype))  # , requires_grad=False)
        else:
            self.station_dt.weight = torch.nn.Parameter(
                torch.zeros(num_station, 2, dtype=dtype)
            )  # , requires_grad=False)
        self.velocity = [velocity["P"], velocity["S"]]

        self.reg = reg
        if event_loc is not None:
            self.event_loc.weight = torch.nn.Parameter(torch.tensor(event_loc, dtype=dtype).contiguous())
        if event_time is not None:
            self.event_time.weight = torch.nn.Parameter(torch.tensor(event_time, dtype=dtype).contiguous())

        self.eikonal = eikonal

    def calc_time(self, event_loc, station_loc, phase_type, double_difference=False):
        if self.eikonal is None:
            dist = torch.linalg.norm(event_loc - station_loc, axis=-1, keepdim=True)
            tt = dist / self.velocity[phase_type]
            tt = tt.float()
            # tt = tt.unsqueeze(-1)
            # print(f"{event_loc[:10] = }")
            # print(f"{station_loc[:10] = }")
            # print(f"{tt[:10] = }")
            # raise
        else:
            if double_difference:
                nb1, ne1, nc1 = event_loc.shape  # batch, event, xyz
                nb2, ne2, nc2 = station_loc.shape
                assert ne1 % ne2 == 0
                assert nb1 == nb2
                station_loc = torch.repeat_interleave(station_loc, ne1 // ne2, dim=1)
                event_loc = event_loc.view(nb1 * ne1, nc1)
                station_loc = station_loc.view(nb1 * ne1, nc2)

            r = torch.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)  ## nb, 2 (pair), 3
            z = event_loc[:, 2] - station_loc[:, 2]

            timetable = self.eikonal["up"] if phase_type == 0 else self.eikonal["us"]
            rgrid0 = self.eikonal["rgrid"][0]
            zgrid0 = self.eikonal["zgrid"][0]
            nr = self.eikonal["nr"]
            nz = self.eikonal["nz"]
            h = self.eikonal["h"]
            tt = CalcTravelTime.apply(r, z, timetable, rgrid0, zgrid0, nr, nz, h)

            tt = tt.float()
            if double_difference:
                tt = tt.view(nb1, ne1, 1)
            else:
                tt = tt.unsqueeze(-1)

        return tt

    def forward(
        self,
        station_index,
        event_index=None,
        phase_type=None,
        phase_time=None,
        phase_weight=None,
        double_difference=False,
    ):
        loss = 0.0
        pred_time = torch.zeros(len(phase_type), dtype=torch.float32)
        for type in [0, 1]:
            station_index_ = station_index[phase_type == type]  # (nb,)
            event_index_ = event_index[phase_type == type]  # (nb,)
            phase_weight_ = phase_weight[phase_type == type]  # (nb,)

            # print(f"{event_index_.shape = }")

            station_loc_ = self.station_loc(station_index_)  # (nb, 3)
            station_dt_ = self.station_dt(station_index_)[:, [type]]  # (nb, 1)

            event_loc_ = self.event_loc(event_index_)  # (nb, 3)
            event_time_ = self.event_time(event_index_)  # (nb, 1)

            # print(f"{station_loc_.shape = }, {event_loc_.shape = }, {station_dt_.shape = }, {event_time_.shape = }")

            if double_difference:
                station_loc_ = station_loc_.unsqueeze(1)  # (nb, 1, 3)
                station_dt_ = station_dt_.unsqueeze(1)  # (nb, 1, 1)

            tt_ = self.calc_time(
                event_loc_, station_loc_, type, double_difference=double_difference
            )  # (nb, 1) or (nb, 2) for double_difference

            # print(f"{tt_.shape = }")

            # print(f"{event_time_.shape = }, {tt_.shape = }, {station_dt_.shape = }")

            t_ = event_time_ + tt_ + station_dt_  # (nb, 1) or (nb, 2, 1) for double_difference

            # print(f"{event_time_[:10] = }")
            # print(f"{tt_[:10] = }")
            # print(f"{station_dt_[:10] = }")

            if double_difference:
                t_ = t_[:, 0] - t_[:, 1]  # (nb, 1)

            t_ = t_.squeeze(1)  # (nb, )

            pred_time[phase_type == type] = t_  # (nb, )

            if phase_time is not None:
                phase_time_ = phase_time[phase_type == type]

                if double_difference:
                    loss += torch.mean(F.huber_loss(t_, phase_time_, reduction="none") * phase_weight_)
                else:
                    # loss = torch.mean(phase_weight * (t - phase_time) ** 2)
                    # loss += torch.mean(
                    #     F.huber_loss(tt_ + station_dt_, phase_time_ - event_time_, reduction="none") * phase_weight_
                    # )
                    loss += torch.mean(F.huber_loss(t_, phase_time_, reduction="none") * phase_weight_)

                    # print(f"{t_.shape = }, {phase_time_.shape = }, {phase_weight_.shape = }")
                    # loss += self.reg * torch.mean(
                    #     torch.abs(station_dt_)
                    # )  ## prevent the trade-off between station_dt and event_time

                    # tmp = F.huber_loss(tt_ + station_dt_, phase_time_ - event_time_, reduction="none")

                    # print(f"{tmp[:10] = }")
                    # print(f"{tmp.shape = } {phase_weight_.shape = }")
                    # print(f"{phase_weight_[:10] = }")
                    # print(torch.mean(tmp * phase_weight_))
                    # print(phase_weight_.dtype)
                    # raise

                    # print(f"{(tt_ + station_dt_)[:10] = }")
                    # print(f"{(phase_time_ - event_time_)[:10] = }")
                    # print(f"{loss = }")
                    # raise

        return {"phase_time": pred_time, "loss": loss}


def main(args):
    # %%
    data_path = Path("test_data")
    figure_path = Path("figures")
    figure_path.mkdir(exist_ok=True)

    ##TODO: clean up config
    config = {
        "center": (-117.504, 35.705),
        "xlim_degree": [-118.004, -117.004],
        "ylim_degree": [35.205, 36.205],
        "degree2km": 111.19492474777779,
        "starttime": datetime(2019, 7, 4, 17, 0),
        "endtime": datetime(2019, 7, 5, 0, 0),
    }

    ## Eikonal for 1D velocity model
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 5.5, 6.7, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 1.0
    vel = {"z": zz, "p": vp, "s": vs}
    config["x(km)"] = (
        (np.array(config["xlim_degree"]) - np.array(config["center"][0]))
        * config["degree2km"]
        * np.cos(np.deg2rad(config["center"][1]))
    )
    config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
    config["z(km)"] = (0, 20)
    config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}

    if args.eikonal:
        eikonal = initialize_eikonal(config["eikonal"])
    else:
        eikonal = None

    # %%
    stations = pd.read_csv(data_path / "stations.csv", delimiter="\t")
    picks = pd.read_csv(data_path / "picks_gamma.csv", delimiter="\t", parse_dates=["phase_time"])
    events = pd.read_csv(data_path / "catalog_gamma.csv", delimiter="\t", parse_dates=["time"])

    events = events[events["event_index"] < 100]
    picks = picks[picks["event_index"] < 100]

    # %%
    proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z_km"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)
    # starttime = events["time"].min()
    # events["time"] = (events["time"] - starttime).dt.total_seconds()
    # picks["phase_time"] = (picks["phase_time"] - starttime).dt.total_seconds()
    events[["x_km", "y_km"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events["z_km"] = events["depth(m)"].apply(lambda x: x / 1e3)

    # %%
    num_event = len(events)
    num_station = len(stations)
    vp = 6.0
    vs = vp / 1.73

    stations.reset_index(inplace=True, drop=True)
    stations["index"] = stations.index.values
    stations.set_index("station", inplace=True)
    station_loc = stations[["x_km", "y_km", "z_km"]].values
    station_dt = None

    events.reset_index(inplace=True, drop=True)
    events["index"] = events.index.values
    event_loc = events[["x_km", "y_km", "z_km"]].values
    event_time = events["time"].values  # [:, np.newaxis]

    event_index_map = {x: i for i, x in enumerate(events["event_index"])}
    picks = picks[picks["event_index"] != -1]
    picks["index"] = picks["event_index"].apply(lambda x: event_index_map[x])
    picks["phase_time"] = picks.apply(lambda x: (x["phase_time"] - event_time[x["index"]]).total_seconds(), axis=1)

    # %%
    plt.figure()
    plt.scatter(stations["x_km"], stations["y_km"], s=10, marker="^")
    plt.scatter(events["x_km"], events["y_km"], s=1)
    plt.axis("scaled")
    plt.savefig(figure_path / "station_event_v2.png", dpi=300, bbox_inches="tight")

    # %%
    utils.init_distributed_mode(args)
    print(args)

    phase_dataset = PhaseDataset(picks, events, stations, double_difference=False, config=args)
    phase_dataset_dd = PhaseDataset(picks, events, stations, double_difference=True, config=args)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(phase_dataset, shuffle=False)
        sampler_dd = torch.utils.data.distributed.DistributedSampler(phase_dataset_dd, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(phase_dataset)
        sampler_dd = torch.utils.data.SequentialSampler(phase_dataset_dd)

    data_loader = DataLoader(phase_dataset, batch_size=None, sampler=sampler, num_workers=args.workers, collate_fn=None)
    data_loader_dd = DataLoader(
        phase_dataset_dd, batch_size=None, sampler=sampler_dd, num_workers=args.workers, collate_fn=None
    )

    #####################################
    # %%
    event_index = []
    station_index = []
    phase_score = []
    phase_time = []
    phase_type = []

    for i in range(len(events)):
        phase_time.append(picks[picks["event_index"] == events.loc[i, "event_index"]]["phase_time"].values)
        phase_score.append(picks[picks["event_index"] == events.loc[i, "event_index"]]["phase_score"].values)
        phase_type.extend(picks[picks["event_index"] == events.loc[i, "event_index"]]["phase_type"].values.tolist())
        event_index.extend([i] * len(picks[picks["event_index"] == events.loc[i, "event_index"]]))
        station_index.append(
            stations.loc[picks[picks["event_index"] == events.loc[i, "event_index"]]["station_id"], "index"].values
        )

    phase_time = np.concatenate(phase_time)
    phase_score = np.concatenate(phase_score)
    phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
    event_index = np.array(event_index)
    station_index = np.concatenate(station_index)

    # %%
    station_index = torch.tensor(station_index, dtype=torch.long)
    event_index = torch.tensor(event_index, dtype=torch.long)
    phase_weight = torch.tensor(phase_score, dtype=torch.float32)
    phase_time = torch.tensor(phase_time, dtype=torch.float32)
    phase_type = torch.tensor(phase_type, dtype=torch.long)

    travel_time = TravelTime(
        num_event,
        num_station,
        station_loc,
        # event_loc=event_loc,
        # event_time=event_time,
        velocity={"P": vp, "S": vs},
        eikonal=eikonal,
    )
    init_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    init_event_time = travel_time.event_time.weight.clone().detach().numpy()

    # optimizer = optim.LBFGS(params=travel_time.parameters(), max_iter=1000, line_search_fn="strong_wolfe")
    optimizer = optim.Adam(params=travel_time.parameters(), lr=0.1)
    epoch = 1000
    for i in range(epoch):
        optimizer.zero_grad()

        loss = 0
        loss_dd = 0
        for meta in data_loader:
            station_index = meta["station_index"]
            event_index = meta["event_index"]
            phase_time = meta["phase_time"]
            phase_type = meta["phase_type"]
            phase_weight = meta["phase_weight"]

            # def closure():
            #     loss = travel_time(station_index, event_index, phase_type, phase_time, phase_weight)["loss"]
            #     loss.backward()
            #     return loss

            # optimizer.step(closure)

            loss = travel_time(
                station_index,
                event_index,
                phase_type,
                phase_time,
                phase_weight,
                double_difference=False,
            )["loss"]
            loss.backward()

            # print(f"{station_index[:10] = }, {event_index[:10] = }, {phase_type[:10] = }, {phase_weight[:10] = }")
            # print(f"{travel_time.event_loc.weight[:10] = }")
            # print(f"{travel_time.event_time.weight[:10] = }")
            # print(f"{loss = }")
            # raise

        if args.double_difference:
            for meta in data_loader_dd:
                station_index = meta["station_index"]
                event_index = meta["event_index"]
                phase_time = meta["phase_time"]
                phase_type = meta["phase_type"]
                phase_weight = meta["phase_weight"]

                # def closure():
                #     loss = travel_time(station_index, event_index, phase_type, phase_time, phase_weight)["loss"]
                #     loss.backward()
                #     return loss

                # optimizer.step(closure)

                loss_dd = travel_time(
                    station_index,
                    event_index,
                    phase_type,
                    phase_time,
                    phase_weight,
                    double_difference=True,
                )["loss"]
                (loss_dd * args.dd_weight).backward()

        if i % 100 == 0:
            print(f"Loss: {loss+loss_dd}:  {loss} + {loss_dd}")

        # optimizer.step(closure)
        optimizer.step()

    # %%
    tt = travel_time(
        station_index, event_index, phase_type, phase_weight=phase_weight, double_difference=args.double_difference
    )["phase_time"]
    print("Loss using invert location", F.mse_loss(tt, phase_time))
    station_dt = travel_time.station_dt.weight.clone().detach().numpy()
    print(f"station_dt: max = {np.max(station_dt)}, min = {np.min(station_dt)}, mean = {np.mean(station_dt)}")
    invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
    invert_event_time = travel_time.event_time.weight.clone().detach().numpy()
    invert_station_dt = travel_time.station_dt.weight.clone().detach().numpy()

    # %%
    plt.figure()
    # plt.scatter(station_loc[:,0], station_loc[:,1], c=tp[idx_event,:])
    plt.plot(event_loc[:, 0], event_loc[:, 1], "x", markersize=1, color="blue", label="True locations")
    plt.scatter(station_loc[:, 0], station_loc[:, 1], c=station_dt[:, 0], marker="o", linewidths=0, alpha=0.6)
    plt.scatter(station_loc[:, 0], station_loc[:, 1] + 2, c=station_dt[:, 1], marker="o", linewidths=0, alpha=0.6)
    plt.axis("scaled")
    plt.colorbar()
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(init_event_loc[:, 0], init_event_loc[:, 1], "x", markersize=1, color="green", label="Initial locations")
    plt.plot(invert_event_loc[:, 0], invert_event_loc[:, 1], "x", markersize=1, color="red", label="Inverted locations")
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    plt.legend()
    plt.savefig(figure_path / "invert_location_v3.png", dpi=300, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    main(args)

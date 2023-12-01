# %%
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pyproj import Proj
from torch import nn
import torch.optim as optim
from tqdm.auto import tqdm
import time
from sklearn.neighbors import NearestNeighbors

torch.manual_seed(0)
np.random.seed(0)

# %%
# !rm -rf test_data
# !wget https://github.com/zhuwq0/ADLoc/releases/download/test_data/test_data.zip
# !unzip test_data.zip

# %%
data_path = Path("test_data")
figure_path = Path("figures")
figure_path.mkdir(exist_ok=True)

config = {
    "center": (-117.504, 35.705),
    "xlim_degree": [-118.004, -117.004],
    "ylim_degree": [35.205, 36.205],
    "degree2km": 111.19492474777779,
    "starttime": datetime(2019, 7, 4, 17, 0),
    "endtime": datetime(2019, 7, 5, 0, 0),
}

# %%
stations = pd.read_csv(data_path / "stations.csv", delimiter="\t")
picks = pd.read_csv(data_path / "picks_gamma.csv", delimiter="\t", parse_dates=["phase_time"])
events = pd.read_csv(data_path / "catalog_gamma.csv", delimiter="\t", parse_dates=["time"])
print(f"Number of stations: {len(stations)}")
print(f"Number of events: {len(events)}")
print(f"Number of picks: {len(picks)}")

# %%
# events = events[events["event_index"] < 100]
# picks = picks[picks["event_index"] < 100]

# %%
proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
stations[["x_km", "y_km"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
stations["z_km"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)
events[["x_km", "y_km"]] = events.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
events["z_km"] = events["depth(m)"].apply(lambda x: x / 1e3)

# %%
num_event = len(events)
num_station = len(stations)
vp = 6.0
vs = vp / 1.73
MIN_PAIR_DIST = 3.0
MAX_TIME_RES = 0.3

stations.reset_index(inplace=True, drop=True)
stations["index"] = stations.index.values
stations.set_index("station", inplace=True)
station_loc = stations[["x_km", "y_km", "z_km"]].values
station_dt = None

events.reset_index(inplace=True, drop=True)
events["index"] = events.index.values
event_loc = events[["x_km", "y_km", "z_km"]].values
event_time = events["time"].values

event_index_map = {x: i for i, x in enumerate(events["event_index"])}
picks = picks[picks["event_index"] != -1]
picks["index"] = picks["event_index"].apply(lambda x: event_index_map[x])
# picks["phase_time"] = picks.apply(lambda x: (x["phase_time"] - events.loc[x["index"], "time"]).total_seconds(), axis=1)
picks["phase_time"] = picks.apply(lambda x: (x["phase_time"] - event_time[x["index"]]).total_seconds(), axis=1)


# %%
plt.figure()
plt.scatter(stations["x_km"], stations["y_km"], s=10, marker="^", c="k")
plt.scatter(events["x_km"], events["y_km"], s=1)
plt.axis("scaled")
plt.savefig(figure_path / "station_event_v1.png", dpi=300, bbox_inches="tight")


# %%
def generate_absolute_time(picks, stations):
    event_index = []
    station_index = []
    phase_score = []
    phase_time = []
    phase_type = []

    picks_by_event = picks.groupby("index")
    for key, group in picks_by_event:
        if key == -1:
            continue
        phase_time.append(group["phase_time"].values)
        phase_score.append(group["phase_score"].values)
        phase_type.extend(group["phase_type"].values.tolist())
        event_index.extend([[key]] * len(group))
        station_index.append(stations.loc[group["station_id"], "index"].values)

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
    return event_index, station_index, phase_time, phase_type, phase_weight


def generate_relative_time(picks, stations):
    event_index = []
    station_index = []
    phase_score = []
    phase_time = []
    phase_type = []

    neigh = NearestNeighbors(radius=MIN_PAIR_DIST)
    neigh.fit(event_loc)

    picks_by_event = picks.groupby("index")

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
            station_index.append(stations.loc[common["station_id"], "index"].values)

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
    return event_index, station_index, phase_time, phase_type, phase_weight


event_index, station_index, phase_time, phase_type, phase_weight = generate_absolute_time(picks, stations)

event_index_dd, station_index_dd, phase_time_dd, phase_type_dd, phase_weight_dd = generate_relative_time(
    picks, stations
)


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
        # self.register_buffer("station_loc", torch.tensor(station_loc, dtype=dtype))
        self.velocity = [velocity["P"], velocity["S"]]
        self.reg = reg
        if event_loc is not None:
            self.event_loc.weight = torch.nn.Parameter(torch.tensor(event_loc, dtype=dtype).contiguous())
        if event_time is not None:
            self.event_time.weight = torch.nn.Parameter(torch.tensor(event_time, dtype=dtype).contiguous())

    def calc_time(self, event_loc, station_loc, phase_type):
        dist = torch.linalg.norm(event_loc - station_loc, axis=-1, keepdim=False)
        tt = dist / self.velocity[phase_type]
        return tt

    def forward(
        self,
        station_index,
        event_index=None,
        phase_type=None,
        phase_weight=None,
        phase_time=None,
        double_difference=False,
    ):
        loss = 0.0
        pred_time = torch.zeros(len(phase_type), dtype=torch.float32)
        for type in [0, 1]:
            station_index_ = station_index[phase_type == type]  # (nb,)
            event_index_ = event_index[phase_type == type]  # (nb,)
            phase_weight_ = phase_weight[phase_type == type]  # (nb,)

            station_loc_ = self.station_loc(station_index_)
            station_loc_ = station_loc_.unsqueeze(1)  # (nb, 1, 3)
            station_dt_ = self.station_dt(station_index_)[:, type]  # (nb, )

            event_loc_ = self.event_loc(event_index_)  # (nb, 1/2, 3)
            event_time_ = self.event_time(event_index_).squeeze(-1)  # (nb, 1, 1)

            tt_ = self.calc_time(event_loc_, station_loc_, type)  # (nb, 1/2)
            t_ = event_time_ + tt_ + station_dt_.unsqueeze(1)  # (nb, 1/2)

            if not double_difference:
                pred_time[phase_type == type] = t_.squeeze(1)  # (nb, 1) -> (nb,)
            else:
                pred_time[phase_type == type] = t_[:, 0] - t_[:, 1]  # (nb, 2) -> (nb,)

            if phase_time is not None:
                phase_time_ = phase_time[phase_type == type]  # (nb, )

                if not double_difference:
                    # loss = torch.mean(phase_weight * (t - phase_time) ** 2)
                    loss += torch.mean(
                        F.huber_loss(
                            tt_.squeeze(1) + station_dt_, phase_time_ - event_time_.squeeze(1), reduction="none"
                        )
                        * phase_weight_
                    )
                    loss += self.reg * torch.mean(
                        torch.abs(station_dt_)
                    )  ## prevent the trade-off between station_dt and event_time
                else:
                    dt = t_[:, 0] - t_[:, 1]
                    loss += torch.mean(F.huber_loss(dt, phase_time_, reduction="none") * phase_weight_)

        return {"phase_time": pred_time, "loss": loss}


################################################## Absolute location  #############################################################

# %%
travel_time = TravelTime(
    num_event,
    num_station,
    station_loc,
    station_dt=station_dt,
    event_loc=event_loc,
    velocity={"P": vp, "S": vs},
)

tt = travel_time(station_index, event_index, phase_type, phase_weight=phase_weight, double_difference=False)[
    "phase_time"
]
loss = travel_time(
    station_index, event_index, phase_type, phase_weight=phase_weight, phase_time=phase_time, double_difference=False
)["loss"]
print("Loss using true location: ", loss.item())

tt_dd = travel_time(
    station_index_dd, event_index_dd, phase_type_dd, phase_weight=phase_weight_dd, double_difference=True
)["phase_time"]
loss_dd = travel_time(
    station_index_dd,
    event_index_dd,
    phase_type_dd,
    phase_weight=phase_weight_dd,
    phase_time=phase_time_dd,
    double_difference=True,
)["loss"]
print("Loss using true location (double difference): ", loss_dd.item())

# %%
# travel_time = TravelTime(num_event, num_station, station_loc, event_time=event_time, velocity={"P": vp, "S": vs})
# tt = travel_time(station_index, event_index, phase_type, phase_weight=phase_weight, double_difference=True)["phase_time"]
# print("Loss using init location", F.mse_loss(tt, phase_time))
# init_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
# init_event_time = travel_time.event_time.weight.clone().detach().numpy()

# # %%
# optimizer = optim.LBFGS(params=travel_time.parameters(), max_iter=1000, line_search_fn="strong_wolfe")

# def closure():
#     optimizer.zero_grad()
#     loss = travel_time(station_index, event_index, phase_type, phase_time, phase_weight)["loss"]
#     loss.backward()
#     return loss

# optimizer.step(closure)

# %%
optimizer = optim.Adam(params=travel_time.parameters(), lr=0.1)
gamma_dd = num_event
t0 = time.time()
## TODO: regenerate pairs after N iterations
for i in range(301):
    optimizer.zero_grad()
    loss = travel_time(
        station_index,
        event_index,
        phase_type,
        phase_weight=phase_weight,
        phase_time=phase_time,
        double_difference=False,
    )["loss"]
    loss.backward()
    loss_dd = travel_time(
        station_index_dd,
        event_index_dd,
        phase_type_dd,
        phase_weight=phase_weight_dd,
        phase_time=phase_time_dd,
        double_difference=True,
    )["loss"]
    (loss_dd * gamma_dd).backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Loss: {loss.item()} {loss_dd.item()} using {time.time() - t0:.0f} seconds")

print(f"Inversion using {time.time() - t0:.0f} seconds")

# %%
tt = travel_time(station_index, event_index, phase_type, phase_weight=phase_weight, double_difference=False)[
    "phase_time"
]
loss = travel_time(
    station_index, event_index, phase_type, phase_weight=phase_weight, phase_time=phase_time, double_difference=False
)["loss"]
print("Loss using invert location:", loss.item())

tt_dd = travel_time(
    station_index_dd, event_index_dd, phase_type_dd, phase_weight=phase_weight_dd, double_difference=True
)["phase_time"]
loss_dd = travel_time(
    station_index_dd,
    event_index_dd,
    phase_type_dd,
    phase_weight=phase_weight_dd,
    phase_time=phase_time_dd,
    double_difference=True,
)["loss"]
print("Loss using invert location (double difference):", loss_dd.item())

station_dt = travel_time.station_dt.weight.clone().detach().numpy()
print(f"station_dt: max = {np.max(station_dt)}, min = {np.min(station_dt)}, mean = {np.mean(station_dt)}")
invert_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
invert_event_time = travel_time.event_time.weight.clone().detach().numpy()
invert_station_dt = travel_time.station_dt.weight.clone().detach().numpy()

# %%
pred_phase_time = travel_time(
    station_index,
    event_index,
    phase_type,
    phase_weight=phase_weight,
    double_difference=False,
)["phase_time"]

res_event = torch.zeros(num_event)
res_phase = pred_phase_time - phase_time
tmp = event_index.squeeze()
for i in tqdm(range(num_event)):
    res = res_phase[tmp == i]
    if len(res) == 0:
        res_event[i] = torch.inf
    else:
        res_event[i] = torch.mean(torch.abs(res))


# %%
plt.figure()
idx = res_event < MAX_TIME_RES
# plt.scatter(station_loc[:,0], station_loc[:,1], c=tp[idx_event,:])
plt.plot(event_loc[:, 0], event_loc[:, 1], ".", markersize=0.5, color="blue", label="Initial locations")
plt.scatter(
    station_loc[:, 0], station_loc[:, 1], c=station_dt[:, 0], marker="^", linewidths=0, alpha=0.6, cmap="viridis_r"
)
plt.scatter(
    station_loc[:, 0], station_loc[:, 1] + 2, c=station_dt[:, 1], marker="^", linewidths=0, alpha=0.6, cmap="viridis_r"
)
plt.axis("scaled")
plt.colorbar()
xlim = plt.xlim()
ylim = plt.ylim()
# plt.plot(init_event_loc[:, 0], init_event_loc[:, 1], "x", markersize=1, color="green", label="Initial locations")
plt.plot(
    invert_event_loc[idx, 0], invert_event_loc[idx, 1], ".", markersize=0.5, color="red", label="Inverted locations"
)
# plt.xlim(xlim)
# plt.ylim(ylim)
plt.legend()
plt.savefig(figure_path / "invert_location_dd_v1_1.png", dpi=300, bbox_inches="tight")
# %%
fig, ax = plt.subplots(1, 2)
ax[0].scatter(
    station_loc[:, 0], station_loc[:, 1], c=station_dt[:, 0], marker="^", linewidths=0, alpha=0.6, cmap="viridis_r"
)
ax[0].scatter(
    station_loc[:, 0], station_loc[:, 1] + 2, c=station_dt[:, 1], marker="^", linewidths=0, alpha=0.6, cmap="viridis_r"
)
ax[0].scatter(
    event_loc[idx, 0],
    event_loc[idx, 1],
    s=min(1000 / len(event_loc), 10),
    marker=".",
    color="blue",
    linewidths=0,
    alpha=0.6,
)
ax[0].axis("scaled")
xlim = ax[0].get_xlim()
ylim = ax[0].get_ylim()
ax[0].set_title("Initial location")

ax[1].scatter(
    station_loc[:, 0],
    station_loc[:, 1],
    c=invert_station_dt[:, 0],
    marker="^",
    linewidths=0,
    alpha=0.6,
    cmap="viridis_r",
)
ax[1].scatter(
    station_loc[:, 0],
    station_loc[:, 1] + 2,
    c=invert_station_dt[:, 1],
    marker="^",
    linewidths=0,
    alpha=0.6,
    cmap="viridis_r",
)
ax[1].scatter(
    invert_event_loc[idx, 0],
    invert_event_loc[idx, 1],
    s=min(1000 / len(event_loc), 10),
    marker=".",
    color="red",
    linewidths=0,
    alpha=0.6,
)
ax[1].axis("scaled")
ax[1].set_xlim(xlim)
ax[1].set_ylim(ylim)
ax[1].set_title("Inverted location")
plt.savefig(figure_path / "invert_location_dd_v1_2.png", dpi=300, bbox_inches="tight")

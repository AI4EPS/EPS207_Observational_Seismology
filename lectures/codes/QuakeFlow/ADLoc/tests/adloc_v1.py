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

# %%
events = events[events["event_index"] < 100]
picks = picks[picks["event_index"] < 100]

# %%
proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
stations[["x_km", "y_km"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
stations["z_km"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)
starttime = events["time"].min()
events["time"] = (events["time"] - starttime).dt.total_seconds()
picks["phase_time"] = (picks["phase_time"] - starttime).dt.total_seconds()
events[["x_km", "y_km"]] = events.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
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
event_time = events["time"].values[:, np.newaxis]

# %%
plt.figure()
plt.scatter(stations["x_km"], stations["y_km"], s=10, marker="^")
plt.scatter(events["x_km"], events["y_km"], s=1)
plt.axis("scaled")
plt.savefig(figure_path / "station_event_v1.png", dpi=300, bbox_inches="tight")

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
phase_time = torch.tensor(phase_time[:, np.newaxis], dtype=torch.float32)
phase_type = torch.tensor(phase_type, dtype=torch.long)

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
        self.station_dt = nn.Embedding(num_station, 2) # vp, vs
        self.station_loc.weight = torch.nn.Parameter(torch.tensor(station_loc, dtype=dtype), requires_grad=False)
        if station_dt is not None:
            self.station_dt.weight = torch.nn.Parameter(torch.tensor(station_dt, dtype=dtype))#, requires_grad=False)
        else:
            self.station_dt.weight = torch.nn.Parameter(torch.zeros(num_station, 2, dtype=dtype))#, requires_grad=False)
        # self.register_buffer("station_loc", torch.tensor(station_loc, dtype=dtype))
        self.velocity = [velocity["P"], velocity["S"]]
        self.reg = reg
        if event_loc is not None:
            self.event_loc.weight = torch.nn.Parameter(torch.tensor(event_loc, dtype=dtype).contiguous())
        if event_time is not None:
            self.event_time.weight = torch.nn.Parameter(torch.tensor(event_time, dtype=dtype).contiguous())

    def calc_time(self, event_loc, station_loc, phase_type):
        dist = torch.linalg.norm(event_loc - station_loc, axis=-1, keepdim=True)
        # velocity = torch.tensor([self.velocity[p] for p in phase_type]).unsqueeze(-1)
        # tt = dist / velocity
        # if isinstance(self.velocity, dict):
        #     self.velocity = torch.tensor([vel[p.upper()] for p in phase_type]).unsqueeze(-1)
        # tt = dist / self.velocity
        tt = dist / self.velocity[phase_type]
        return tt

    def forward(
        self, station_index, event_index=None, phase_type=None, phase_time=None, phase_weight=None, double_difference=False
    ):
        loss = 0.0
        pred_time = torch.zeros(len(phase_type), 1, dtype=torch.float32)
        for type in [0, 1]:
            station_index_ = station_index[phase_type == type]
            event_index_ = event_index[phase_type == type]                
            phase_weight_ = phase_weight[phase_type == type]

            station_loc_ = self.station_loc(station_index_)
            station_dt_ = self.station_dt(station_index_)[:, type].unsqueeze(-1)

            event_loc_ = self.event_loc(event_index_)
            event_time_ = self.event_time(event_index_)

            tt_ = self.calc_time(event_loc_, station_loc_, type)
            t_ = event_time_ + tt_ + station_dt_
            pred_time[phase_type == type] = t_

            if double_difference:
                t_ = t_[0] - t_[1]

            if phase_time is not None:
                phase_time_ = phase_time[phase_type == type]
                # loss = torch.mean(phase_weight * (t - phase_time) ** 2)
                loss += torch.mean(F.huber_loss(tt_ + station_dt_, phase_time_ - event_time_, reduction="none") * phase_weight_)
                loss += self.reg * torch.mean(torch.abs(station_dt_)) ## prevent the trade-off between station_dt and event_time

        return {"phase_time": pred_time, "loss": loss}


################################################## Absolute location  #############################################################

# %%
travel_time = TravelTime(
    num_event,
    num_station,
    station_loc,
    station_dt=station_dt,
    event_loc=event_loc,
    event_time=event_time,
    velocity={"P": vp, "S": vs},
)
tt = travel_time(station_index, event_index, phase_type, phase_weight=phase_weight)["phase_time"]
print("Loss using true location: ", F.mse_loss(tt, phase_time))

# %%
# travel_time = TravelTime(num_event, num_station, station_loc, velocity={"P": vp, "S": vs})
travel_time = TravelTime(num_event, num_station, station_loc, event_time=event_time, velocity={"P": vp, "S": vs})
tt = travel_time(station_index, event_index, phase_type, phase_weight=phase_weight)["phase_time"]
print("Loss using init location", F.mse_loss(tt, phase_time))
init_event_loc = travel_time.event_loc.weight.clone().detach().numpy()
init_event_time = travel_time.event_time.weight.clone().detach().numpy()

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
for i in range(1000):
    optimizer.zero_grad()
    loss = travel_time(station_index, event_index, phase_type, phase_time, phase_weight)["loss"]
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Loss: {loss.item()}")

# %%
tt = travel_time(station_index, event_index, phase_type, phase_weight=phase_weight)["phase_time"]
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
plt.scatter(station_loc[:, 0], station_loc[:, 1], c=station_dt[:,0], marker="o", linewidths=0, alpha=0.6)
plt.scatter(station_loc[:, 0], station_loc[:, 1]+2, c=station_dt[:,1], marker="o", linewidths=0, alpha=0.6)
plt.axis("scaled")
plt.colorbar()
xlim = plt.xlim()
ylim = plt.ylim()
plt.plot(init_event_loc[:, 0], init_event_loc[:, 1], "x", markersize=1, color="green", label="Initial locations")
plt.plot(invert_event_loc[:, 0], invert_event_loc[:, 1], "x", markersize=1, color="red", label="Inverted locations")
# plt.xlim(xlim)
# plt.ylim(ylim)
plt.legend()
plt.savefig(figure_path / "invert_location_v1.png", dpi=300, bbox_inches="tight")
# %%

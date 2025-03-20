# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def clamp(input, min, max):
    return Clamp.apply(input, min, max)


def interp2d(time_table, x, y, xgrid, ygrid, h):
    nx = len(xgrid)
    ny = len(ygrid)
    assert time_table.shape == (nx, ny)

    with torch.no_grad():
        ix0 = torch.floor((x - xgrid[0]) / h).clamp(0, nx - 2).long()
        iy0 = torch.floor((y - ygrid[0]) / h).clamp(0, ny - 2).long()
        ix1 = ix0 + 1
        iy1 = iy0 + 1

    # x = (torch.clamp(x, xgrid[0], xgrid[-1]) - xgrid[0]) / h
    # y = (torch.clamp(y, ygrid[0], ygrid[-1]) - ygrid[0]) / h

    x = (clamp(x, xgrid[0], xgrid[-1]) - xgrid[0]) / h
    y = (clamp(y, ygrid[0], ygrid[-1]) - ygrid[0]) / h

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation

    Q00 = time_table[ix0, iy0]
    Q01 = time_table[ix0, iy1]
    Q10 = time_table[ix1, iy0]
    Q11 = time_table[ix1, iy1]

    t = (
        Q00 * (ix1 - x) * (iy1 - y)
        + Q10 * (x - ix0) * (iy1 - y)
        + Q01 * (ix1 - x) * (y - iy0)
        + Q11 * (x - ix0) * (y - iy0)
    )

    return t


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
        velocity={"P": 6.0, "S": 6.0 / 1.73},
        eikonal=None,
        zlim=[0, 30],
        dtype=torch.float32,
        grad_type="auto",
    ):
        super().__init__()
        self.num_event = num_event
        self.event_loc = nn.Embedding(num_event, 3)
        self.event_time = nn.Embedding(num_event, 1)
        self.station_loc = nn.Embedding(num_station, 3)
        self.station_dt = nn.Embedding(num_station, 1)  # same statioin term for P and S

        ## check initialization
        station_loc = torch.tensor(station_loc, dtype=dtype)
        if station_dt is not None:
            station_dt = torch.tensor(station_dt, dtype=dtype)
        else:
            station_dt = torch.zeros(num_station, 1, dtype=dtype)
        if event_loc is not None:
            event_loc = torch.tensor(event_loc, dtype=dtype).contiguous()
        else:
            event_loc = torch.zeros(num_event, 3, dtype=dtype).contiguous()
        if event_time is not None:
            event_time = torch.tensor(event_time, dtype=dtype).contiguous()
        else:
            event_time = torch.zeros(num_event, 1, dtype=dtype).contiguous()

        self.station_loc.weight = torch.nn.Parameter(station_loc, requires_grad=False)
        self.station_dt.weight = torch.nn.Parameter(station_dt, requires_grad=False)
        self.event_loc.weight = torch.nn.Parameter(event_loc, requires_grad=True)
        self.event_time.weight = torch.nn.Parameter(event_time, requires_grad=True)

        self.velocity = [velocity["P"], velocity["S"]]
        self.eikonal = eikonal
        self.zlim = zlim
        self.grad_type = grad_type
        if self.eikonal is not None:
            self.timetable_p = torch.tensor(
                np.reshape(self.eikonal["up"], (self.eikonal["nr"], self.eikonal["nz"])), dtype=dtype
            )
            self.timetable_s = torch.tensor(
                np.reshape(self.eikonal["us"], (self.eikonal["nr"], self.eikonal["nz"])), dtype=dtype
            )
            self.rgrid = self.eikonal["rgrid"]
            self.zgrid = self.eikonal["zgrid"]
            self.h = self.eikonal["h"]

    def calc_time(self, event_loc, station_loc, phase_type):

        if self.eikonal is None:
            dist = torch.linalg.norm(event_loc - station_loc, axis=-1, keepdim=True)
            tt = dist / self.velocity[phase_type]
            tt = tt.float()
        else:
            # r = torch.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)  ## nb, 3
            x = event_loc[:, 0] - station_loc[:, 0]
            y = event_loc[:, 1] - station_loc[:, 1]
            z = event_loc[:, 2] - station_loc[:, 2]
            r = torch.sqrt(x**2 + y**2)

            # timetable = self.eikonal["up"] if phase_type == 0 else self.eikonal["us"]
            # timetable_grad = self.eikonal["grad_up"] if phase_type == 0 else self.eikonal["grad_us"]
            # timetable_grad_r = timetable_grad[0]
            # timetable_grad_z = timetable_grad[1]
            # rgrid0 = self.eikonal["rgrid"][0]
            # zgrid0 = self.eikonal["zgrid"][0]
            # nr = self.eikonal["nr"]
            # nz = self.eikonal["nz"]
            # h = self.eikonal["h"]
            # tt = CalcTravelTime.apply(r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid0, zgrid0, nr, nz, h)

            if phase_type in [0, "P"]:
                timetable = self.timetable_p
            elif phase_type in [1, "S"]:
                timetable = self.timetable_s
            else:
                raise ValueError("phase_type should be 0 or 1. for P and S, respectively.")

            tt = interp2d(timetable, r, z, self.rgrid, self.zgrid, self.h)

            tt = tt.float().unsqueeze(-1)

        return tt

    def forward(
        self,
        station_index,
        event_index=None,
        phase_type=None,
        phase_time=None,
        phase_weight=None,
    ):
        loss = 0.0
        pred_time = torch.zeros(len(phase_type), dtype=torch.float32)
        resisudal = torch.zeros(len(phase_type), dtype=torch.float32)

        # for type in [0, 1]:  # phase_type: 0 for P, 1 for S
        for type in np.unique(phase_type):

            if len(phase_type[phase_type == type]) == 0:
                continue

            station_index_ = station_index[phase_type == type]  # (nb,)
            event_index_ = event_index[phase_type == type]  # (nb,)
            phase_weight_ = phase_weight[phase_type == type]  # (nb,)

            station_loc_ = self.station_loc(station_index_)  # (nb, 3)
            station_dt_ = self.station_dt(station_index_)  # (nb, 1)

            event_loc_ = self.event_loc(event_index_)  # (nb, 3)
            event_time_ = self.event_time(event_index_)  # (nb, 1)

            tt_ = self.calc_time(event_loc_, station_loc_, type)  # (nb, 1)

            t_ = event_time_ + tt_ + station_dt_  # (nb, 1)
            t_ = t_.squeeze(1)  # (nb, )

            pred_time[phase_type == type] = t_  # (nb, )

            if phase_time is not None:
                phase_time_ = phase_time[phase_type == type]
                resisudal[phase_type == type] = phase_time_ - t_
                # loss += torch.sum(F.huber_loss(t_, phase_time_, reduction="none") * phase_weight_)
                # loss += torch.sum(F.l1_loss(t_, phase_time_, reduction="none") * phase_weight_)
                loss += torch.sum(torch.abs(t_ - phase_time_) * phase_weight_)

        return {"phase_time": pred_time, "residual": resisudal, "loss": loss}


# %%
class TravelTimeDD(TravelTime):

    def __init__(
        self,
        num_event,
        num_station,
        station_loc,
        station_dt=None,
        event_loc=None,
        event_time=None,
        velocity={"P": 6.0, "S": 6.0 / 1.73},
        eikonal=None,
        zlim=[0, 30],
        dtype=torch.float32,
        grad_type="auto",
    ):
        super().__init__(
            num_event,
            num_station,
            station_loc,
            station_dt=station_dt,
            event_loc=event_loc,
            event_time=event_time,
            velocity=velocity,
            eikonal=eikonal,
            zlim=zlim,
            dtype=dtype,
            grad_type=grad_type,
        )

        self.event_loc.weight.requires_grad = True
        self.event_time.weight.requires_grad = False
        # self.event_time.weight.requires_grad = True

    def calc_time(self, event_loc, station_loc, phase_type):
        if self.eikonal is None:
            dist = torch.linalg.norm(event_loc - station_loc, axis=-1, keepdim=True)
            tt = dist / self.velocity[phase_type]
            tt = tt.float()
        else:
            nb1, ne1, nc1 = event_loc.shape  # batch, event, xyz
            nb2, ne2, nc2 = station_loc.shape
            assert ne1 % ne2 == 0
            assert nb1 == nb2
            station_loc = torch.repeat_interleave(station_loc, ne1 // ne2, dim=1)
            event_loc = event_loc.view(nb1 * ne1, nc1)
            station_loc = station_loc.view(nb1 * ne1, nc2)

            # r = torch.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)  ## nb, 2 (pair), 3
            x = event_loc[:, 0] - station_loc[:, 0]
            y = event_loc[:, 1] - station_loc[:, 1]
            z = event_loc[:, 2] - station_loc[:, 2]
            r = torch.sqrt(x**2 + y**2)

            # timetable = self.eikonal["up"] if phase_type == 0 else self.eikonal["us"]
            # timetable_grad = self.eikonal["grad_up"] if phase_type == 0 else self.eikonal["grad_us"]
            # timetable_grad_r = timetable_grad[0]
            # timetable_grad_z = timetable_grad[1]
            # rgrid0 = self.eikonal["rgrid"][0]
            # zgrid0 = self.eikonal["zgrid"][0]
            # nr = self.eikonal["nr"]
            # nz = self.eikonal["nz"]
            # h = self.eikonal["h"]
            # tt = CalcTravelTime.apply(r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid0, zgrid0, nr, nz, h)

            if phase_type in [0, "P"]:
                timetable = self.timetable_p
            elif phase_type in [1, "S"]:
                timetable = self.timetable_s
            else:
                raise ValueError("phase_type should be 0 or 1. for P and S, respectively.")

            tt = interp2d(timetable, r, z, self.rgrid, self.zgrid, self.h)

            tt = tt.float().view(nb1, ne1, 1)

        return tt

    def forward(
        self,
        station_index,
        event_index=None,
        phase_type=None,
        phase_time=None,
        phase_weight=None,
    ):
        loss = 0.0
        pred_time = torch.zeros(len(phase_type), dtype=torch.float32)

        # for type in [0, 1]:  # phase_type: 0 for P, 1 for S
        for type in np.unique(phase_type):

            if len(phase_type[phase_type == type]) == 0:
                continue

            station_index_ = station_index[phase_type == type]  # (nb,)
            event_index_ = event_index[phase_type == type]  # (nb,)
            phase_weight_ = phase_weight[phase_type == type]  # (nb,)

            station_loc_ = self.station_loc(station_index_)  # (nb, 3)
            station_dt_ = self.station_dt(station_index_)  # (nb, 1)

            event_loc_ = self.event_loc(event_index_)  # (nb, 2, 3)
            event_time_ = self.event_time(event_index_)  # (nb, 2, 1)

            station_loc_ = station_loc_.unsqueeze(1)  # (nb, 1, 3)
            station_dt_ = station_dt_.unsqueeze(1)  # (nb, 1, 1)

            tt_ = self.calc_time(event_loc_, station_loc_, type)  # (nb, 2)

            t_ = event_time_ + tt_ + station_dt_  # (nb, 2, 1)

            t_ = t_[:, 0] - t_[:, 1]  # (nb, 1)
            t_ = t_.squeeze(1)  # (nb, )

            pred_time[phase_type == type] = t_  # (nb, )

            if phase_time is not None:
                phase_time_ = phase_time[phase_type == type]
                # loss += torch.sum(F.huber_loss(t_, phase_time_, reduction="none") * phase_weight_)
                # loss += torch.sum(F.l1_loss(t_, phase_time_, reduction="none") * phase_weight_)
                loss += torch.sum(torch.abs(t_ - phase_time_) * phase_weight_)

        if loss == 0.0:
            return None
        return {"phase_time": pred_time, "loss": loss}


# %%
if __name__ == "__main__":

    # %%
    import json
    import os
    from datetime import datetime, timedelta

    import matplotlib.pyplot as plt
    import pandas as pd
    from eikonal2d import eikonal_solve
    from torch import optim

    ######################################## Create Synthetic Data #########################################
    np.random.seed(0)
    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    nx = 10
    ny = 10
    nr = int(np.sqrt(nx**2 + ny**2))
    nz = 10
    h = 3.0
    eikonal_config = {"nr": nr, "nz": nz, "h": h}
    with open(f"{data_path}/config.json", "w") as f:
        json.dump(eikonal_config, f)
    xgrid = np.arange(0, nx) * h
    ygrid = np.arange(0, ny) * h
    rgrid = np.arange(0, nr) * h
    zgrid = np.arange(0, nz) * h
    eikonal_config.update({"rgrid": rgrid, "zgrid": zgrid})
    num_station = 10
    num_event = 50
    stations = []
    for i in range(num_station):
        x = np.random.uniform(xgrid[0], xgrid[-1])
        y = np.random.uniform(ygrid[0], ygrid[-1])
        z = np.random.uniform(zgrid[0], zgrid[0] + 3 * h)
        stations.append({"station_id": f"STA{i:02d}", "x_km": x, "y_km": y, "z_km": z, "dt_s": 0.0})
    stations = pd.DataFrame(stations)
    stations["station_index"] = stations.index
    stations.to_csv(f"{data_path}/stations.csv", index=False)
    events = []
    reference_time = pd.to_datetime("2021-01-01T00:00:00.000")
    for i in range(num_event):
        x = np.random.uniform(xgrid[0], xgrid[-1])
        y = np.random.uniform(ygrid[0], ygrid[-1])
        z = np.random.uniform(zgrid[0], zgrid[-1])
        t = i * 5
        events.append(
            {"event_id": i, "event_time": reference_time + pd.Timedelta(seconds=t), "x_km": x, "y_km": y, "z_km": z}
        )
    events = pd.DataFrame(events)
    events["event_index"] = events.index
    events["event_time"] = events["event_time"].apply(lambda x: x.isoformat(timespec="milliseconds"))
    events.to_csv(f"{data_path}/events.csv", index=False)
    vpvs_ratio = 1.73
    vp = np.ones((nr, nz)) * 6.0
    vs = vp / vpvs_ratio

    ## eikonal solver
    up = 1000 * np.ones((nr, nz), dtype=np.float64)
    us = 1000 * np.ones((nr, nz), dtype=np.float64)
    ir0 = np.around((0 - rgrid[0]) / h).astype(np.int64)
    iz0 = np.around((0 - zgrid[0]) / h).astype(np.int64)
    up[ir0, iz0] = 0.0
    us[ir0, iz0] = 0.0
    up = eikonal_solve(up, vp, h)
    us = eikonal_solve(us, vs, h)
    up = torch.tensor(up, dtype=torch.float64)
    us = torch.tensor(us, dtype=torch.float64)

    # %%
    picks = []
    for j, station in stations.iterrows():
        for i, event in events.iterrows():
            r = np.linalg.norm([event["x_km"] - station["x_km"], event["y_km"] - station["y_km"]])
            z = event["z_km"] - station["z_km"]
            r = torch.tensor(r, dtype=torch.float64)
            z = torch.tensor(z, dtype=torch.float64)
            if np.random.rand() < 0.5:
                tt = interp2d(up, r, z, rgrid, zgrid, h).item()
                picks.append(
                    {
                        "event_id": event["event_id"],
                        "station_id": station["station_id"],
                        "phase_type": "P",
                        "phase_time": pd.to_datetime(event["event_time"]) + pd.Timedelta(seconds=tt),
                        "phase_score": 1.0,
                        "travel_time": tt,
                    }
                )
            if np.random.rand() < 0.5:
                tt = interp2d(us, r, z, rgrid, zgrid, h).item()
                picks.append(
                    {
                        "event_id": event["event_id"],
                        "station_id": station["station_id"],
                        "phase_type": "S",
                        "phase_time": pd.to_datetime(event["event_time"]) + pd.Timedelta(seconds=tt),
                        "phase_score": 1.0,
                        "travel_time": tt,
                    }
                )
    picks = pd.DataFrame(picks)
    # use picks,  stations.index, events.index to set station_index and
    picks["phase_time"] = picks["phase_time"].apply(lambda x: x.isoformat(timespec="milliseconds"))
    picks["event_index"] = picks["event_id"].map(events.set_index("event_id")["event_index"])
    picks["station_index"] = picks["station_id"].map(stations.set_index("station_id")["station_index"])
    picks.to_csv(f"{data_path}/picks.csv", index=False)
    # %%
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/true2d_vp_vs.png")

    # %%
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
    ax[0, 0].scatter(stations["x_km"], stations["y_km"], c=stations["z_km"], marker="^", label="Station")
    ax[0, 0].scatter(events["x_km"], events["y_km"], c=events["z_km"], marker=".", label="Event")
    ax[0, 0].set_xlabel("x (km)")
    ax[0, 0].set_ylabel("y (km)")
    ax[0, 0].legend()
    ax[0, 0].set_title("Station and Event Locations")
    plt.savefig(f"{data_path}/station_event_3d.png")
    # %%
    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(10, 15))
    picks = picks.merge(stations, on="station_id")
    mapping_color = lambda x: f"C{int(x)}"
    ax[0, 0].scatter(pd.to_datetime(picks["phase_time"]), picks["x_km"], c=picks["event_index"].apply(mapping_color))
    ax[0, 0].scatter(
        pd.to_datetime(events["event_time"]), events["x_km"], c=events["event_index"].apply(mapping_color), marker="x"
    )
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("x (km)")
    ax[1, 0].scatter(pd.to_datetime(picks["phase_time"]), picks["y_km"], c=picks["event_index"].apply(mapping_color))
    ax[1, 0].scatter(
        pd.to_datetime(events["event_time"]), events["y_km"], c=events["event_index"].apply(mapping_color), marker="x"
    )
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("y (km)")
    ax[2, 0].scatter(pd.to_datetime(picks["phase_time"]), picks["z_km"], c=picks["event_index"].apply(mapping_color))
    ax[2, 0].scatter(
        pd.to_datetime(events["event_time"]), events["z_km"], c=events["event_index"].apply(mapping_color), marker="x"
    )
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 0].set_ylabel("z (km)")
    plt.savefig(f"{data_path}/picks_3d.png")
    # %%
    ######################################### Load Synthetic Data #########################################
    data_path = "data"
    events = pd.read_csv(f"{data_path}/events.csv")
    stations = pd.read_csv(f"{data_path}/stations.csv")
    picks = pd.read_csv(f"{data_path}/picks.csv")
    picks = picks.merge(events[["event_index", "event_time"]], on="event_index")

    #### make the time values relative to event time in seconds
    picks["phase_time_origin"] = picks["phase_time"].copy()
    picks["phase_time"] = (
        pd.to_datetime(picks["phase_time"]) - pd.to_datetime(picks["event_time"])
    ).dt.total_seconds()  # relative to event time (arrival time)
    picks.drop(columns=["event_time"], inplace=True)
    events["event_time_origin"] = events["event_time"].copy()
    events["event_time"] = np.zeros(len(events))  # relative to event time
    ####

    with open(f"{data_path}/config.json", "r") as f:
        eikonal_config = json.load(f)
    events = events.sort_values("event_index").set_index("event_index")
    stations = stations.sort_values("station_index").set_index("station_index")
    num_event = len(events)
    num_station = len(stations)

    ## eikonal solver
    nr, nz, h = eikonal_config["nr"], eikonal_config["nz"], eikonal_config["h"]
    rgrid = np.arange(0, nr) * h
    zgrid = np.arange(0, nz) * h
    eikonal_config.update({"rgrid": rgrid, "zgrid": zgrid})
    vp = np.ones((nr, nz), dtype=np.float64) * 6.0
    vs = vp / 1.73
    up = 1000 * np.ones((nr, nz), dtype=np.float64)
    us = 1000 * np.ones((nr, nz), dtype=np.float64)
    ir0 = np.around((0 - rgrid[0]) / h).astype(np.int64)
    iz0 = np.around((0 - zgrid[0]) / h).astype(np.int64)
    up[ir0, iz0] = 0.0
    us[ir0, iz0] = 0.0
    up = eikonal_solve(up, vp, h)
    us = eikonal_solve(us, vs, h)
    eikonal_config.update({"up": up, "us": us})

    ## initial event location
    event_loc = (
        events[["x_km", "y_km", "z_km"]].values * 0.0
    )  # + stations[["x_km", "y_km", "z_km"]].values.mean(axis=0)
    event_time = events[["event_time"]].values * 0.0

    # %%
    traveltime = TravelTime(
        num_event,
        num_station,
        stations[["x_km", "y_km", "z_km"]].values,
        stations[["dt_s"]].values,
        event_loc,
        event_time,
        eikonal=eikonal_config,
    )

    # %%
    output = traveltime(
        torch.tensor(picks["station_index"].values, dtype=torch.long),
        torch.tensor(picks["event_index"].values, dtype=torch.long),
        picks["phase_type"].values,
        torch.tensor(picks["phase_time"].values, dtype=torch.float32),
        torch.tensor(picks["phase_score"].values, dtype=torch.float32),
    )
    loss = output["loss"]

    ######################################### Optimize #########################################
    # %%
    print(
        "Optimizing parameters:\n"
        + "\n".join(
            [f"{name}: {param.size()}" for name, param in traveltime.named_parameters() if param.requires_grad]
        ),
    )
    params = [param for param in traveltime.parameters() if param.requires_grad]
    optimizer = optim.LBFGS(params=params, max_iter=1000, line_search_fn="strong_wolfe")
    print("Initial loss:", loss.item())

    picks_station_index = torch.tensor(picks["station_index"].values, dtype=torch.long)
    picks_event_index = torch.tensor(picks["event_index"].values, dtype=torch.long)
    picks_phase_type = picks["phase_type"].values
    picks_phase_time = torch.tensor(picks["phase_time"].values, dtype=torch.float32)
    picks_phase_score = torch.tensor(picks["phase_score"].values, dtype=torch.float32)

    def closure():

        optimizer.zero_grad()

        output = traveltime(
            picks_station_index,
            picks_event_index,
            picks_phase_type,
            picks_phase_time,
            picks_phase_score,
        )
        loss = output["loss"]
        loss.backward()

        with torch.no_grad():
            traveltime.event_loc.weight.data[:, 2].clamp_(min=3.0)

        return loss

    optimizer.step(closure)

    output = traveltime(
        picks_station_index,
        picks_event_index,
        picks_phase_type,
        picks_phase_time,
        picks_phase_score,
    )
    loss = output["loss"]
    print("Final loss:", loss.item())

    # %%
    event_loc_init = event_loc.copy()
    event_loc = traveltime.event_loc.weight.detach().numpy()

    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(15, 5))
    # ax[0, 0].plot(stations["x_km"], stations["y_km"], "^", label="Station")
    ax[0, 0].plot(events["x_km"], events["y_km"], ".", label="True Events")
    ax[0, 0].plot(event_loc_init[:, 0], event_loc_init[:, 1], "x", label="Initial Events")
    for i in range(len(event_loc_init)):
        ax[0, 0].plot(
            [events["x_km"].iloc[i], event_loc_init[i, 0]],
            [events["y_km"].iloc[i], event_loc_init[i, 1]],
            "k--",
            alpha=0.5,
        )
    ax[0, 0].plot(event_loc[:, 0], event_loc[:, 1], "x", label="Inverted Events")
    for i in range(len(event_loc)):
        ax[0, 0].plot(
            [events["x_km"].iloc[i], event_loc[i, 0]], [events["y_km"].iloc[i], event_loc[i, 1]], "r--", alpha=0.5
        )
    ax[0, 0].set_xlabel("x (km)")
    ax[0, 0].set_ylabel("y (km)")
    ax[0, 0].legend()
    # ax[0, 0].set_title("Station and Event Locations")

    ax[0, 1].plot(events["x_km"], events["z_km"], ".", label="True Events")
    ax[0, 1].plot(event_loc_init[:, 0], event_loc_init[:, 2], "x", label="Initial Events")
    for i in range(len(event_loc_init)):
        ax[0, 1].plot(
            [events["x_km"].iloc[i], event_loc_init[i, 0]],
            [events["z_km"].iloc[i], event_loc_init[i, 2]],
            "k--",
            alpha=0.5,
        )
    ax[0, 1].plot(event_loc[:, 0], event_loc[:, 2], "x", label="Inverted Events")
    for i in range(len(event_loc)):
        ax[0, 1].plot(
            [events["x_km"].iloc[i], event_loc[i, 0]], [events["z_km"].iloc[i], event_loc[i, 2]], "r--", alpha=0.5
        )
    ax[0, 1].set_xlabel("x (km)")
    ax[0, 1].set_ylabel("z (km)")
    ax[0, 1].legend()

    ax[0, 2].plot(events["y_km"], events["z_km"], ".", label="True Events")
    ax[0, 2].plot(event_loc_init[:, 1], event_loc_init[:, 2], "x", label="Initial Events")
    for i in range(len(event_loc_init)):
        ax[0, 2].plot(
            [events["y_km"].iloc[i], event_loc_init[i, 1]],
            [events["z_km"].iloc[i], event_loc_init[i, 2]],
            "k--",
            alpha=0.5,
        )
    ax[0, 2].plot(event_loc[:, 1], event_loc[:, 2], "x", label="Inverted Events")
    for i in range(len(event_loc)):
        ax[0, 2].plot(
            [events["y_km"].iloc[i], event_loc[i, 1]], [events["z_km"].iloc[i], event_loc[i, 2]], "r--", alpha=0.5
        )
    ax[0, 2].set_xlabel("y (km)")
    ax[0, 2].set_ylabel("z (km)")
    ax[0, 2].legend()

    plt.savefig(f"{data_path}/inverted_station_event.png")

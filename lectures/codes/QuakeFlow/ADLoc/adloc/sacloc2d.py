# %%
import numpy as np
import scipy
from sklearn.base import BaseEstimator

from ._ransac import RANSACRegressor
from .eikonal2d import grad_traveltime, init_eikonal2d, traveltime
from .gmpe import calc_amp, calc_mag

# from sklearn.linear_model import RANSACRegressor

np.random.seed(0)


class ADLoc(BaseEstimator):
    def __init__(self, config, stations, num_event=1, events=None, eikonal=None):
        """
        events: [x, y, z, t]
        """
        xlim = config["xlim_km"]
        ylim = config["ylim_km"]
        zlim = config["zlim_km"]
        if "vel" in config:
            vel = config["vel"]
        else:
            vel = {"P": 6.0, "S": 6.0 / 1.73}
            if eikonal is None:
                print(f"Using default velocity: {vel}")
        self.config = config
        self.stations = stations
        self.vel = vel

        self.num_event = num_event

        self.eikonal = eikonal

        if events is not None:
            assert events.shape == (num_event, 4)
            self.events = events
        else:
            self.events = np.array(
                [[(xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2, (zlim[0] + zlim[1]) / 2, 0]] * num_event
            )
        if ("use_amplitude" in config) and (config["use_amplitude"]):
            self.use_amplitude = True
            self.magnitudes = np.ones(num_event) * -999.0
        else:
            self.use_amplitude = False

    @staticmethod
    def l2_loss_grad(event, X, y, vel={0: 6.0, 1: 6.0 / 1.75}, stations=None, eikonal=None):
        """
        X: data_frame with columns ["timestamp", "x_km", "y_km", "z_km", "type"]
        """
        station_index = X[:, 0].astype(int)
        phase_type = X[:, 1].astype(int)
        phase_weight = X[:, 2]

        if eikonal is None:
            v = np.array([vel[t] for t in phase_type])
            tt = np.linalg.norm(event[:3] - stations[station_index, :3], axis=-1) / v + event[3]
        else:
            tt = traveltime(0, station_index, phase_type, event[np.newaxis, :3], stations, eikonal) + event[3]
        loss = 0.5 * np.sum((tt - y) ** 2 * phase_weight)

        J = np.ones((len(X), 4))
        if eikonal is None:
            J[:, :3] = (
                (event[:3] - stations[station_index, :3])
                / np.linalg.norm(event[:3] - stations[station_index, :3], axis=-1, keepdims=True)
                / v[:, np.newaxis]
            )
        else:
            grad = grad_traveltime(0, station_index, phase_type, event[np.newaxis, :3], stations, eikonal)
            J[:, :3] = grad

        J = np.sum((tt - y)[:, np.newaxis] * J * phase_weight[:, np.newaxis], axis=0)
        return loss, J

    @staticmethod
    def huber_loss_grad(event, X, y, vel={0: 6.0, 1: 6.0 / 1.75}, stations=None, eikonal=None, sigma=1.0):

        station_index = X[:, 0].astype(int)
        phase_type = X[:, 1].astype(int)
        phase_weight = X[:, 2]

        if eikonal is None:
            v = np.array([vel[t] for t in phase_type])
            tt = np.linalg.norm(event[:3] - stations[station_index, :3], axis=-1) / v + event[3]
        else:
            tt = traveltime(0, station_index, phase_type, event[np.newaxis, :3], stations, eikonal) + event[3]

        t_diff = tt - y
        l1 = np.squeeze((np.abs(t_diff) > sigma))
        l2 = np.squeeze((np.abs(t_diff) <= sigma))

        loss = np.sum((sigma * np.abs(t_diff[l1]) - 0.5 * sigma**2) * phase_weight[l1]) + np.sum(
            0.5 * t_diff[l2] ** 2 * phase_weight[l2]
        )

        J = np.ones((len(X), 4))
        if eikonal is None:
            J[:, :3] = (
                (event[:3] - stations[station_index, :3])
                / np.linalg.norm(event[:3] - stations[station_index, :3], axis=-1, keepdims=True)
                / v[:, np.newaxis]
            )
        else:
            grad = grad_traveltime(0, station_index, phase_type, event[np.newaxis, :3], stations, eikonal)
            J[:, :3] = grad

        J = np.sum(
            sigma * np.sign(t_diff[l1, np.newaxis]) * J[l1] * phase_weight[l1, np.newaxis], axis=0, keepdims=True
        ) + np.sum(t_diff[l2, np.newaxis] * J[l2] * phase_weight[l2, np.newaxis], axis=0, keepdims=True)

        return loss, J

    def fit(self, X, y=None, event_index=0):

        if self.use_amplitude:
            yt = y[:, 0]  # time
            ya = y[:, 1]  # amplitude
        else:
            yt = y[:, 0]

        opt = scipy.optimize.minimize(
            self.huber_loss_grad,
            # self.l2_loss_grad,
            x0=self.events[event_index],
            method="L-BFGS-B",
            jac=True,
            args=(X, yt, self.vel, self.stations, self.eikonal),
            # bounds=[
            #     (self.config["xlim_km"][0], self.config["xlim_km"][1]),
            #     (self.config["ylim_km"][0], self.config["ylim_km"][1]),
            #     (self.config["zlim_km"][0], self.config["zlim_km"][1]),
            #     (None, None),
            # ],
            bounds=self.config["bfgs_bounds"],
        )

        self.events[event_index, :] = opt.x
        self.is_fitted_ = True

        if self.use_amplitude:
            station_index = X[:, 0].astype(int)
            phase_weight = X[:, 2:3]
            self.magnitudes[event_index] = calc_mag(
                ya[:, np.newaxis], self.events[event_index, :3], self.stations[station_index, :3], phase_weight
            )

        return self

    def predict(self, X, event_index=0):
        """
        X: data_frame with columns ["timestamp", "x_km", "y_km", "z_km", "type"]
        """
        station_index = X[:, 0].astype(int)
        phase_type = X[:, 1].astype(int)

        if self.eikonal is None:
            v = np.array([self.vel[t] for t in phase_type])
            tt = (
                np.linalg.norm(self.events[event_index, :3] - self.stations[station_index, :3], axis=-1) / v
                + self.events[event_index, 3]
            )
        else:
            tt = (
                traveltime(event_index, station_index, phase_type, self.events, self.stations, self.eikonal)
                + self.events[event_index, 3]
            )

        if self.use_amplitude:
            amp = calc_amp(
                self.magnitudes[event_index], self.events[event_index, :3], self.stations[station_index, :3]
            ).squeeze()
            return np.array([tt, amp]).T
        else:
            return np.array([tt]).T

    def weight(self, X, event_index=0):
        """
        X: data_frame with columns ["timestamp", "x_km", "y_km", "z_km", "type"]
        """
        station_index = X[:, 0].astype(int)
        phase_type = X[:, 1].astype(int)

        if self.eikonal is None:
            v = np.array([self.vel[t] for t in phase_type])
            tt = np.linalg.norm(self.events[event_index, :3] - self.stations[station_index, :3], axis=-1) / v
        else:
            tt = traveltime(event_index, station_index, phase_type, self.events, self.stations, self.eikonal)

        return np.array([tt]).T

    def score(self, X, y=None, event_index=0):
        """
        X: idx_sta, type, score, amp
        """
        if len(y) <= 1:
            return -np.inf

        output = self.predict(X, event_index)
        if self.use_amplitude:
            tt = output[:, 0]
            amp = output[:, 1]
            yt = y[:, 0]
            ya = y[:, 1]
            R2_t = 1 - np.sum((yt - tt) ** 2) / (np.sum((y[:, 0] - np.mean(y[:, 0])) ** 2) + 1e-6)
            R2_a = 1 - np.sum((ya - amp) ** 2) / (np.sum((y[:, 1] - np.mean(y[:, 1])) ** 2) + 1e-6)
            # weight = X[:, 2]
            # mu_yt = np.average(yt, weights=weight)
            # mu_ya = np.average(ya, weights=weight)
            # R2_t = 1 - np.sum(weight * (yt - tt) ** 2) / (np.sum(weight * (yt - mu_yt) ** 2) + 1e-6)
            # R2_a = 1 - np.sum(weight * (ya - amp) ** 2) / (np.sum(weight * (ya - mu_ya) ** 2) + 1e-6)
            R2 = (R2_t + R2_a) / 2
        else:
            tt = output
            yt = y
            R2 = 1 - np.sum((yt - tt) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-6)
            # weight = X[:, 2]
            # mu_y = np.average(yt, weights=weight)
            # R2 = 1 - np.sum(weight * (yt - tt) ** 2) / (np.sum(weight * (yt - mu_y) ** 2) + 1e-6)

        return R2


# %%
if __name__ == "__main__":

    # %%
    import json
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    from eikonal2d import _interp, eikonal_solve

    ######################################## Create Synthetic Data #########################################
    np.random.seed(0)
    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    nx = 10
    ny = 10
    nr = int(np.sqrt(nx**2 + ny**2))
    nz = 10
    h = 20.0
    xgrid = np.arange(0, nx) * h
    ygrid = np.arange(0, ny) * h
    rgrid = np.arange(0, nr) * h
    zgrid = np.arange(0, nz) * h
    xlim = [xgrid[0].item(), xgrid[-1].item()]
    ylim = [ygrid[0].item(), ygrid[-1].item()]
    zlim = [zgrid[0].item(), zgrid[-1].item()]
    eikonal_config = {"nr": nr, "nz": nz, "h": h, "xlim_km": xlim, "ylim_km": ylim, "zlim_km": zlim}
    with open(f"{data_path}/config.json", "w") as f:
        json.dump(eikonal_config, f)
    eikonal_config.update({"rgrid": rgrid, "zgrid": zgrid})
    num_station = 10
    num_event = 50
    stations = []
    for i in range(num_station):
        x = np.random.uniform(xgrid[0], xgrid[-1])
        y = np.random.uniform(ygrid[0], ygrid[-1])
        # z = np.random.uniform(zgrid[0], zgrid[0] + 3 * h)
        z = 0.0
        stations.append({"station_id": f"STA{i:02d}", "x_km": x, "y_km": y, "z_km": z, "dt_s": 0.0})
    stations = pd.DataFrame(stations)
    stations["station_index"] = stations.index
    stations.to_csv(f"{data_path}/stations.csv", index=False)
    events = []
    reference_time = pd.to_datetime("2021-01-01T00:00:00.000")
    for i in range(num_event):
        x = np.random.uniform(xgrid[0], xgrid[-1])
        y = np.random.uniform(ygrid[0], ygrid[-1])
        # z = np.random.uniform(zgrid[0], zgrid[-1])
        z = 0.0
        t = i * 30
        # mag = np.random.uniform(1.0, 5.0)
        mag = 4.0
        events.append(
            {
                "event_id": i,
                "event_time": reference_time + pd.Timedelta(seconds=t),
                "x_km": x,
                "y_km": y,
                "z_km": z,
                "magnitude": mag,
            }
        )
    events = pd.DataFrame(events)
    events["event_index"] = events.index
    events["event_time"] = events["event_time"].apply(lambda x: x.isoformat(timespec="milliseconds"))
    events.to_csv(f"{data_path}/events.csv", index=False)
    vpvs_ratio = 1.73
    vp = np.ones((nr, nz)) * 6.0
    vs = vp / vpvs_ratio

    ## eikonal solver
    up = 1000.0 * np.ones((nr, nz), dtype=np.float64)
    us = 1000.0 * np.ones((nr, nz), dtype=np.float64)
    ir0 = np.around((0.0 - rgrid[0]) / h).astype(np.int64)
    iz0 = np.around((0.0 - zgrid[0]) / h).astype(np.int64)
    up[ir0, iz0] = 0.0
    us[ir0, iz0] = 0.0
    up = eikonal_solve(up, vp, h)
    us = eikonal_solve(us, vs, h)
    up = up.ravel()
    us = us.ravel()

    # %%
    picks = []
    for j, station in stations.iterrows():
        for i, event in events.iterrows():
            r = np.array([np.linalg.norm([event["x_km"] - station["x_km"], event["y_km"] - station["y_km"]])])
            z = np.array([event["z_km"] - station["z_km"]])
            if np.random.rand() < 0.5:
                tt = _interp(up, r, z, rgrid[0], zgrid[0], nr, nz, h)[0]
                picks.append(
                    {
                        "event_id": event["event_id"],
                        "station_id": station["station_id"],
                        "phase_type": "P",
                        "phase_time": pd.to_datetime(event["event_time"]) + pd.Timedelta(seconds=tt),
                        "phase_amplitude": calc_amp(
                            event["magnitude"],
                            event[["x_km", "y_km", "z_km"]].values.astype(np.float32),
                            station[["x_km", "y_km", "z_km"]].values.astype(np.float32),
                        ).item(),
                        "phase_score": 1.0,
                        "travel_time": tt,
                    }
                )
            if np.random.rand() < 0.5:
                tt = _interp(us, r, z, rgrid[0], zgrid[0], nr, nz, h)[0]
                picks.append(
                    {
                        "event_id": event["event_id"],
                        "station_id": station["station_id"],
                        "phase_type": "S",
                        "phase_time": pd.to_datetime(event["event_time"]) + pd.Timedelta(seconds=tt),
                        "phase_amplitude": calc_amp(
                            event["magnitude"],
                            event[["x_km", "y_km", "z_km"]].values.astype(np.float32),
                            station[["x_km", "y_km", "z_km"]].values.astype(np.float32),
                        ).item(),
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
    ax[0, 0].scatter(
        pd.to_datetime(picks["phase_time"]),
        picks["x_km"],
        c=picks["event_index"].apply(mapping_color),
        s=20,
    )
    ax[0, 0].scatter(
        pd.to_datetime(events["event_time"]), events["x_km"], c=events["event_index"].apply(mapping_color), marker="x"
    )
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("x (km)")
    ax[1, 0].scatter(
        pd.to_datetime(picks["phase_time"]),
        picks["y_km"],
        c=picks["event_index"].apply(mapping_color),
        s=20,
    )
    ax[1, 0].scatter(
        pd.to_datetime(events["event_time"]), events["y_km"], c=events["event_index"].apply(mapping_color), marker="x"
    )
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("y (km)")
    ax[2, 0].scatter(
        pd.to_datetime(picks["phase_time"]),
        picks["z_km"],
        c=picks["event_index"].apply(mapping_color),
        s=20,
    )
    ax[2, 0].scatter(
        pd.to_datetime(events["event_time"]), events["z_km"], c=events["event_index"].apply(mapping_color), marker="x"
    )
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 0].set_ylabel("z (km)")
    ax[2, 0].invert_yaxis()
    plt.savefig(f"{data_path}/picks_3d.png")

    fig, ax = plt.subplots(1, 1)
    picks = picks.merge(events, on="event_index", suffixes=("_station", "_event"))
    picks["dist_km"] = np.linalg.norm(
        picks[["x_km_station", "y_km_station", "z_km_station"]].values
        - picks[["x_km_event", "y_km_event", "z_km_event"]].values,
        axis=1,
    )
    ax.scatter(np.log10(picks["dist_km"]), picks["phase_amplitude"], c=picks["event_index"].apply(mapping_color), s=30)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Amplitude")
    plt.savefig(f"{data_path}/picks_dist_amp.png")

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
    num_event = len(events)
    num_station = len(stations)
    ## eikonal solver
    vpvs_ratio = 1.73
    vel = {
        "Z": [0, 100.0],
        "P": [6.0, 6.0],
        "S": [6.0 / vpvs_ratio, 6.0 / vpvs_ratio],
    }
    eikonal_config["vel"] = vel
    eikonal_config = init_eikonal2d(eikonal_config)

    # %%
    stations["idx_sta"] = stations.index  # reindex in case the index does not start from 0 or is not continuous
    events["idx_eve"] = events.index  # reindex in case the index does not start from 0 or is not continuous
    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    # %%
    event_index = 1
    num_event = len(events)
    # for idx_eve, picks_event in picks.groupby("idx_eve"):
    picks_event = picks[picks["idx_eve"] == event_index]
    event_loc = events.loc[event_index]
    print(f"Event {event_loc['event_index']} at ({event_loc['x_km']}, {event_loc['y_km']})")
    for _, pick in picks_event.iterrows():
        station_loc = stations.loc[pick["idx_sta"]]
        print(f"Station {station_loc['station_id']} at ({station_loc['x_km']}, {station_loc['y_km']})")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    tmp = picks_event.merge(stations[["x_km", "y_km", "z_km", "station_id"]], on="station_id")
    tmp["dist_km"] = tmp[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - event_loc[["x_km", "y_km", "z_km"]]), axis=1
    )
    im = ax[0, 0].scatter(tmp["x_km"], tmp["y_km"], c=tmp["phase_time"], cmap="viridis_r", marker="^")
    ax[0, 0].scatter(event_loc["x_km"], event_loc["y_km"], c="r", marker="x")
    ax[0, 0].set_xlabel("x (km)")
    ax[0, 0].set_ylabel("y (km)")
    ax[0, 0].set_aspect("equal")
    plt.colorbar(im, ax=ax[0, 0])

    colors = lambda x: "r" if x == "P" else "b"
    ax[0, 1].scatter(tmp["dist_km"], tmp["phase_time"], c=tmp["phase_type"].apply(colors), s=30)
    ax[0, 1].set_ylabel("Time (s)")
    ax[0, 1].set_xlabel("Distance (km)")
    plt.savefig(f"{data_path}/picks_event_{event_index}.png")

    # %%
    config = {}
    config["xlim_km"] = eikonal_config["xlim_km"]
    config["ylim_km"] = eikonal_config["ylim_km"]
    config["zlim_km"] = eikonal_config["zlim_km"]
    config["bfgs_bounds"] = [
        (config["xlim_km"][0], config["xlim_km"][1]),
        (config["ylim_km"][0], config["ylim_km"][1]),
        (config["zlim_km"][0], config["zlim_km"][1]),
        (None, None),
    ]
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    config["use_amplitude"] = True
    X = picks_event.merge(
        stations[["x_km", "y_km", "z_km", "station_id"]],
        on="station_id",
    )
    # t0 = X["phase_time"].min() ## already convert to travel time in seconds

    X.rename(
        columns={"phase_type": "type", "phase_time": "t_s", "phase_score": "score", "phase_amplitude": "amp"},
        inplace=True,
    )
    # X["t_s"] = (X["t_s"] - t0).dt.total_seconds()
    X = X[["x_km", "y_km", "z_km", "t_s", "type", "score", "idx_sta", "amp"]]
    mapping_int = {"P": 0, "S": 1}
    config["vel"] = {mapping_int[k]: v for k, v in config["vel"].items()}
    X["type"] = X["type"].apply(lambda x: mapping_int[x.upper()])

    estimator = ADLoc(
        config, stations=stations[["x_km", "y_km", "z_km"]].values, num_event=num_event, eikonal=eikonal_config
    )
    output = estimator.predict(X[["idx_sta", "type"]].values, event_index=event_index)
    tt_init = output[:, 0]
    amp_init = output[:, 1]
    estimator.score(X[["idx_sta", "type"]].values, y=X[["t_s", "amp"]].values, event_index=event_index)

    print(f"Init event loc: {estimator.events[event_index].round(3)}")

    # %%
    estimator.fit(X[["idx_sta", "type", "score", "amp"]].values, y=X[["t_s", "amp"]].values, event_index=event_index)
    estimator.score(X[["idx_sta", "type"]].values, y=X[["t_s", "amp"]].values, event_index=event_index)
    output = estimator.predict(X[["idx_sta", "type"]].values, event_index=event_index)
    tt = output[:, 0]
    amp = output[:, 1]
    print("Basic:")
    print(f"True event loc: {event_loc[['x_km', 'y_km', 'z_km']].values.astype(float).round(3)}")
    print(f"Invt event loc: {estimator.events[event_index].round(3)}")

    # %%
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
    X["dist_km"] = X[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - estimator.events[event_index, :3]), axis=1
    )
    colors = lambda x: "r" if x == 0 else "b"
    ax[0, 0].scatter(X["dist_km"], X["t_s"], s=20, marker="o", label="Picks")
    ax[0, 0].scatter(X["dist_km"], tt_init, s=40, marker="x", label="Init")
    ax[0, 0].scatter(X["dist_km"], tt, s=40, marker="x", label="Invert")
    ax[0, 0].set_xlabel("Distance (km)")
    ax[0, 0].set_ylabel("Time (s)")
    ax[0, 0].legend()

    ax[0, 1].scatter(np.log10(X["dist_km"]), X["amp"], s=20, marker="o", label="Picks")
    # ax[0, 1].scatter(np.log10(X["dist_km"]), amp_init, s=40, marker="x", label="Init")
    ax[0, 1].scatter(np.log10(X["dist_km"]), amp, s=40, marker="x", label="Invert")
    ax[0, 1].set_xlabel("Distance (km)")
    ax[0, 1].set_ylabel("Amplitude")
    ax[0, 1].legend()
    fig.savefig(f"{data_path}/picks_xt_{event_index}_adloc.png")

    # %%
    reg = RANSACRegressor(
        estimator=ADLoc(
            config, stations=stations[["x_km", "y_km", "z_km"]].values, num_event=num_event, eikonal=eikonal_config
        ),
        random_state=0,
        min_samples=4,
        residual_threshold=[1.0, 0.5],
    )
    reg.fit(X[["idx_sta", "type", "score", "amp"]].values, X[["t_s", "amp"]].values, event_index=event_index)
    mask_invt = reg.inlier_mask_
    estimator = reg.estimator_
    estimator.score(X[["idx_sta", "type"]].values, y=X[["t_s", "amp"]].values, event_index=event_index)
    output = estimator.predict(X[["idx_sta", "type"]].values, event_index=event_index)
    tt = output[:, 0]
    amp = output[:, 1]
    print("RANSAC:")
    print(f"True event loc: {event_loc[['x_km', 'y_km', 'z_km']].values.astype(float).round(3)}")
    print(f"Invt event loc: {estimator.events[event_index].round(3)}")

    # %% add random picks
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    X = picks_event.merge(stations[["x_km", "y_km", "z_km", "station_id"]], on="station_id")
    # t0 = X["phase_time"].min() ## already convert to travel time in seconds
    X.rename(
        columns={"phase_type": "type", "phase_time": "t_s", "phase_score": "score", "phase_amplitude": "amp"},
        inplace=True,
    )
    # X["t_s"] = (X["t_s"] - t0).dt.total_seconds()
    X = X[["x_km", "y_km", "z_km", "t_s", "type", "score", "idx_sta", "amp"]]
    mapping_int = {"P": 0, "S": 1}
    config["vel"] = {mapping_int[k]: v for k, v in config["vel"].items()}
    X["type"] = X["type"].apply(lambda x: mapping_int[x.upper()])

    num_noise = len(X) * 2
    noise = pd.DataFrame(
        {
            "x_km": np.random.rand(num_noise) * (X["x_km"].max() - X["x_km"].min()) + X["x_km"].min(),
            "y_km": np.random.rand(num_noise) * (X["y_km"].max() - X["y_km"].min()) + X["y_km"].min(),
            "z_km": np.random.rand(num_noise) * (X["z_km"].max() - X["z_km"].min()) + X["z_km"].min(),
            "t_s": np.random.rand(num_noise) * (X["t_s"].max() - X["t_s"].min()) + X["t_s"].min(),
            "type": np.random.choice([0, 1], num_noise),
            "score": 1.0,
            "idx_sta": np.random.choice(X["idx_sta"], num_noise),
            "amp": np.random.rand(num_noise) * (X["amp"].max() - X["amp"].min()) * 2 + X["amp"].min(),
            "mask": [0] * num_noise,
        }
    )
    X["mask"] = 1
    X = pd.concat([X, noise], ignore_index=True)
    X["dist_km"] = X[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - event_loc[["x_km", "y_km", "z_km"]]), axis=1
    )

    # %%
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
    colors = lambda x: "r" if x == 0 else "b"
    # ax.scatter(X["t_s"], X["dist_km"], c=X["type"].apply(colors), marker="o", label="Picks")
    # ax.scatter(X["t_s"][X["mask"] == 0], X["dist_km"][X["mask"] == 0], c="k", marker="o", alpha=0.6, label="Noise")
    ax[0, 0].scatter(X["dist_km"], X["t_s"], c=X["type"].apply(colors), s=30, marker="o", label="Picks")
    ax[0, 0].scatter(
        X["dist_km"][X["mask"] == 0], X["t_s"][X["mask"] == 0], c="k", s=40, marker="o", alpha=0.6, label="Noise"
    )
    ax[0, 0].set_ylabel("Time (s)")
    ax[0, 0].set_xlabel("Distance (km)")
    ax[0, 0].legend()

    ax[0, 1].scatter(np.log10(X["dist_km"]), X["amp"], c=X["type"].apply(colors), s=30, marker="o", label="Picks")
    ax[0, 1].scatter(
        np.log10(X["dist_km"][X["mask"] == 0]), X["amp"][X["mask"] == 0], c="k", s=30, marker="o", label="Noise"
    )
    ax[0, 1].set_xlabel("Distance (km)")
    ax[0, 1].set_ylabel("Amplitude")
    plt.savefig(f"{data_path}/picks_xt_{event_index}_noise.png")

    # %%
    reg = RANSACRegressor(
        estimator=ADLoc(
            config, stations=stations[["x_km", "y_km", "z_km"]].values, num_event=num_event, eikonal=eikonal_config
        ),
        random_state=0,
        min_samples=4,
        residual_threshold=[1.0, 0.5],
    )
    reg.fit(X[["idx_sta", "type", "score", "amp"]].values, X[["t_s", "amp"]].values, event_index=event_index)
    mask_invt = reg.inlier_mask_
    estimator = reg.estimator_
    estimator.score(X[["idx_sta", "type"]].values, y=X[["t_s", "amp"]].values, event_index=event_index)
    output = estimator.predict(X[["idx_sta", "type"]].values, event_index=event_index)
    tt = output[:, 0]
    amp = output[:, 1]
    print("RANSAC with noise picks:")
    print(f"True event loc: {event_loc[['x_km', 'y_km', 'z_km']].values.astype(float).round(3)}")
    print(f"Invt event loc: {estimator.events[event_index].round(3)}")

    # %%
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), squeeze=False)
    X["dist_km"] = X[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - event_loc[["x_km", "y_km", "z_km"]]),
        axis=1,
    )
    colors = lambda x: "r" if x == 0 else "b"
    ax[0, 0].scatter(X["dist_km"], X["t_s"], c=X["type"].apply(colors), s=30, marker="o", label="Picks")
    mask_true = X["mask"] == 1
    ax[0, 0].scatter(X["dist_km"][~mask_true], X["t_s"][~mask_true], s=30, c="k", marker="o", alpha=0.6, label="Noise")
    ax[0, 0].scatter(X["dist_km"][mask_invt], tt[mask_invt], s=50, c="g", marker="x", label="Invert")
    ax[0, 0].set_ylabel("Time (s)")
    ax[0, 0].set_xlabel("Distance (km)")
    ax[0, 0].legend()

    ax[0, 1].scatter(np.log10(X["dist_km"]), X["amp"], c=X["type"].apply(colors), s=30, marker="o", label="Picks")
    ax[0, 1].scatter(np.log10(X["dist_km"][~mask_true]), X["amp"][~mask_true], c="k", s=30, marker="o", label="Noise")
    ax[0, 1].scatter(np.log10(X["dist_km"][mask_invt]), amp[mask_invt], c="g", s=50, marker="x", label="Invert")

    X["dist_km"] = X[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - estimator.events[event_index, :3]), axis=1
    )
    colors = lambda x: "r" if x == 0 else "b"
    ax[1, 0].scatter(X["dist_km"], X["t_s"], c=X["type"].apply(colors), s=30, marker="o", label="Picks")
    mask_true = X["mask"] == 1
    ax[1, 0].scatter(X["dist_km"][~mask_true], X["t_s"][~mask_true], s=30, c="k", marker="o", alpha=0.6, label="Noise")
    ax[1, 0].scatter(X["dist_km"][mask_invt], tt[mask_invt], s=50, c="g", marker="x", label="Invert")
    ax[1, 0].set_ylabel("Time (s)")
    ax[1, 0].set_xlabel("Distance (km)")
    ax[1, 0].legend()

    ax[1, 1].scatter(np.log10(X["dist_km"]), X["amp"], c=X["type"].apply(colors), s=30, marker="o", label="Picks")
    ax[1, 1].scatter(np.log10(X["dist_km"][~mask_true]), X["amp"][~mask_true], c="k", s=30, marker="o", label="Noise")
    ax[1, 1].scatter(np.log10(X["dist_km"][mask_invt]), amp[mask_invt], c="g", s=50, marker="x", label="Invert")

    fig.savefig(f"{data_path}/picks_xt_{event_index}_ransac.png")

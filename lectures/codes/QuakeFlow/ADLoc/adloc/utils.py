import multiprocessing as mp
from contextlib import nullcontext

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._ransac import RANSACRegressor


def invert(picks, stations, config, estimator, event_index, event_init):
    """
    Earthquake location for a single event using RANSAC.
    """

    def is_model_valid(estimator, X, y):
        """
        X: idx_sta, type, score, amp
        """
        # n0 = np.sum(X[:, 1] == 0)  # P
        # n1 = np.sum(X[:, 1] == 1)  # S
        n0 = np.sum(X[X[:, 1] == 0, 2])
        n1 = np.sum(X[X[:, 1] == 1, 2])
        if not (n0 >= config["min_p_picks"] and n1 >= config["min_s_picks"]):
            return False

        if estimator.events[0][2] < 1.0:  # depth > 1.0 km
            return False

        if estimator.score(X, y) < config["min_score"]:
            return False

        return True

    def is_data_valid(X, y):
        """
        X: idx_sta, type, score, amp
        y: t_s
        """
        # n0 = np.sum(X[:, 1] == 0)  # P
        # n1 = np.sum(X[:, 1] == 1)  # S
        n0 = np.sum(X[X[:, 1] == 0, 2])
        n1 = np.sum(X[X[:, 1] == 1, 2])
        return n0 >= config["min_p_picks"] and n1 >= config["min_s_picks"]  # At least min P and S picks

    MIN_PICKS = config["min_picks"]
    MIN_PICKS_RATIO = config["min_picks_ratio"]
    if config["use_amplitude"]:
        MAX_RESIDUAL = [config["max_residual_time"], config["max_residual_amplitude"]]
    else:
        MAX_RESIDUAL = config["max_residual_time"]

    if "station_term_time_p" not in stations.columns:
        stations["station_term_time_p"] = 0.0
    if "station_term_time_s" not in stations.columns:
        stations["station_term_time_s"] = 0.0
    if config["use_amplitude"] and ("station_term_amplitude" not in stations.columns):
        stations["station_term_amplitude"] = 0.0

    if config["use_amplitude"]:
        X = picks.merge(
            # stations[["x_km", "y_km", "z_km", "station_id", "station_term_time", "station_term_amplitude"]],
            stations[
                [
                    "x_km",
                    "y_km",
                    "z_km",
                    "station_id",
                    "station_term_time_p",
                    "station_term_time_s",
                    "station_term_amplitude",
                ]
            ],  ## Separate P and S station term
            on="station_id",
        )
    else:
        X = picks.merge(
            # stations[["x_km", "y_km", "z_km", "station_id", "station_term_time"]],
            stations[
                ["x_km", "y_km", "z_km", "station_id", "station_term_time_p", "station_term_time_s"]
            ],  ## Separate P and S station term
            on="station_id",
        )
    t0 = X["phase_time"].min()

    if (event_init is None) or len(X) == 0:
        event_init = np.array([[np.median(X["x_km"]), np.median(X["y_km"]), 5.0, 0.0]])
    else:
        event_init = np.array([[event_init[0], event_init[1], event_init[2], (event_init[3] - t0).total_seconds()]])

    # xstd = np.std(X["x_km"])
    # ystd = np.std(X["y_km"])
    # rstd = np.sqrt(xstd**2 + ystd**2)

    X.rename(columns={"phase_type": "type", "phase_score": "score", "phase_time": "t_s"}, inplace=True)
    X["t_s"] = (X["t_s"] - t0).dt.total_seconds()
    X["station_term_time"] = X.apply(
        lambda x: x["station_term_time_p"] if x["type"] == 0 else x["station_term_time_s"], axis=1
    )  ## Separate P and S station term
    X["t_s"] = X["t_s"] - X["station_term_time"]
    if config["use_amplitude"]:
        X.rename(columns={"phase_amplitude": "amp"}, inplace=True)
        X["amp"] = X["amp"] - X["station_term_amplitude"]

    estimator.set_params(**{"events": event_init})

    # ## Location using ADLoc
    # if config["use_amplitude"]:
    #     estimator.fit(X[["idx_sta", "type", "score", "amp"]].values, y=X[["t_s", "amp"]].values)
    # else:
    #     estimator.fit(X[["idx_sta", "type", "score"]].values, y=X[["t_s"]].values)
    # inlier_mask = np.ones(len(X)).astype(bool)

    ## Location using RANSAC
    num_picks = len(X)
    reg = RANSACRegressor(
        estimator=estimator,
        random_state=0,
        min_samples=max(MIN_PICKS, int(MIN_PICKS_RATIO * num_picks)),
        # residual_threshold=MAX_RESIDUAL * (1.0 - np.exp(-rstd / 60.0)),  # not sure which one is better
        residual_threshold=MAX_RESIDUAL,
        is_model_valid=is_model_valid,
        is_data_valid=is_data_valid,
    )
    try:
        if config["use_amplitude"]:
            reg.fit(X[["idx_sta", "type", "score", "amp"]].values, X[["t_s", "amp"]].values)
        else:
            reg.fit(X[["idx_sta", "type", "score"]].values, X[["t_s"]].values)
    except Exception as e:
        print(f"No valid model for event_index {event_index}.")
        message = "RANSAC could not find a valid consensus set."
        if str(e)[: len(message)] != message:
            print(e)
        picks["mask"] = 0
        picks["residual_time"] = 0.0
        if config["use_amplitude"]:
            picks["residual_amplitude"] = 0.0
        return picks, None

    estimator = reg.estimator_
    inlier_mask = reg.inlier_mask_

    ## Predict travel time
    output = estimator.predict(X[["idx_sta", "type"]].values)
    if config["use_amplitude"]:
        tt = output[:, 0]
        amp = output[:, 1]
    else:
        tt = output[:, 0]
    if config["use_amplitude"]:
        score = estimator.score(
            X[["idx_sta", "type", "score", "amp"]].values[inlier_mask], y=X[["t_s", "amp"]].values[inlier_mask]
        )
    else:
        score = estimator.score(X[["idx_sta", "type", "score"]].values[inlier_mask], y=X[["t_s"]].values[inlier_mask])

    if (np.sum(inlier_mask) > MIN_PICKS) and (score > config["min_score"]):
        mean_residual_time = np.sum(np.abs(X["t_s"].values - tt) * inlier_mask) / np.sum(inlier_mask)
        if config["use_amplitude"]:
            mean_residual_amp = np.sum(np.abs(X["amp"].values - amp) * inlier_mask) / np.sum(inlier_mask)
        else:
            mean_residual_amp = 0.0
        x, y, z, t = estimator.events[0]
        if config["use_amplitude"]:
            mag = estimator.magnitudes[0]

        event = {
            "idx_eve": event_index,  ## inside adloc, idx_eve is used which starts from 0 to N events
            "x_km": x,
            "y_km": y,
            "z_km": z,
            "time": t0 + pd.Timedelta(t, unit="s"),
            "adloc_score": score,
            "adloc_residual_time": mean_residual_time,
            "num_picks": np.sum(inlier_mask),
        }
        if config["use_amplitude"]:
            event["magnitude"] = mag
            event["adloc_residual_amplitude"] = mean_residual_amp
        event = pd.DataFrame([event])
    else:
        inlier_mask = np.zeros(len(inlier_mask), dtype=bool)
        event = None

    picks["residual_time"] = X["t_s"].values - tt
    if config["use_amplitude"]:
        picks["residual_amplitude"] = X["amp"].values - amp
    else:
        picks["residual_amplitude"] = 0.0
    picks["mask"] = inlier_mask.astype(int)

    # ####################################### Debug #######################################
    # import os

    # import matplotlib.pyplot as plt

    # print(f"{max(MIN_PICKS, int(MIN_PICKS_RATIO * num_picks))=}")
    # print(f"{MAX_RESIDUAL=}")
    # fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(7, 4))
    # event = estimator.events[0]
    # X["dist"] = np.linalg.norm(X[["x_km", "y_km", "z_km"]].values - event[:3], axis=1)
    # ax[0, 0].scatter(X["dist"], X["t_s"], c="r", s=10, label="Picks")
    # ax[0, 0].scatter(X["dist"][~inlier_mask], X["t_s"][~inlier_mask], s=10, c="k", label="Noise")
    # ax[0, 0].scatter(X["dist"][inlier_mask], tt[inlier_mask], c="g", s=30, marker="x", label="Predicted")
    # ax[0, 0].set_xlabel("Distance (km)")
    # ax[0, 0].set_ylabel("Time (s)")
    # ax[0, 0].legend()
    # ax[0, 1].scatter(np.log10(X["dist"]), X["amp"], c="r", s=10, label="Picks")
    # ax[0, 1].scatter(np.log10(X["dist"][~inlier_mask]), X["amp"][~inlier_mask], s=10, c="k", label="Noise")
    # ax[0, 1].scatter(np.log10(X["dist"][inlier_mask]), amp[inlier_mask], c="g", s=30, marker="x", label="Predicted")
    # ax[0, 1].set_xlabel("Distance (km)")
    # ax[0, 1].set_ylabel("Amplitude")
    # if not os.path.exists("debug"):
    #     os.makedirs("debug")
    # fig.savefig(f"debug/event_{event_index}.png")
    # plt.close(fig)
    # raise
    # ####################################### Debug #######################################

    return picks, event


def invert_location(picks, stations, config, estimator, events_init=None, iter=0):

    if "ncpu" in config:
        NCPU = config["ncpu"]
    else:
        NCPU = min(64, mp.cpu_count() * 2 - 1)

    jobs = []
    events_inverted = []
    picks_inverted = []
    pbar = tqdm(total=len(picks.groupby("idx_eve")), desc=f"Iter {iter}")
    with mp.get_context("spawn").Pool(NCPU) as pool:
        for idx_eve, picks_by_event in picks.groupby("idx_eve"):
            if events_init is not None:
                event_init = events_init[events_init["idx_eve"] == idx_eve]
                if len(event_init) == 0:
                    event_init = None
                elif len(event_init) == 1:
                    event_init = event_init[["x_km", "y_km", "z_km", "time"]].values[0]
                else:
                    event_init = event_init[["x_km", "y_km", "z_km", "time"]].values[0]
                    print(f"Multiple initial locations for event_index {idx_eve}.")
            else:
                event_init = None

            # picks_, event_ = invert(picks_by_event, stations, config, estimator, idx_eve, event_init)
            # if event_ is not None:
            #     events_inverted.append(event_)
            # picks_inverted.append(picks_)
            # pbar.update()

            job = pool.apply_async(
                invert,
                (picks_by_event, stations, config, estimator, idx_eve, event_init),
                callback=lambda _: pbar.update(),
            )
            jobs.append(job)

        pool.close()
        pool.join()

        for job in jobs:
            picks_, event_ = job.get()
            if event_ is not None:
                events_inverted.append(event_)
            picks_inverted.append(picks_)

    if len(events_inverted) == 0:
        return None, None

    events_inverted = pd.concat(events_inverted, ignore_index=True)
    picks_inverted = pd.concat(picks_inverted)
    if events_init is not None:
        events_inverted = events_inverted.merge(events_init[["event_index", "idx_eve"]], on="idx_eve")

    print(f"ADLoc using {len(picks_inverted[picks_inverted['mask'] == 1])} picks outof {len(picks_inverted)} picks")

    return picks_inverted, events_inverted


def invert_location_iter(picks, stations, config, estimator, events_init=None, iter=0):

    events_inverted = []
    picks_inverted = []

    for idx_eve, picks_by_event in tqdm(picks.groupby("idx_eve"), desc=f"Iter {iter}"):
        if events_init is not None:
            event_init = events_init[events_init["idx_eve"] == idx_eve]
            if len(event_init) > 0:
                event_init = event_init[["x_km", "y_km", "z_km", "time"]].values[0]
        else:
            event_init = None

        picks_, event_ = invert(picks_by_event, stations, config, estimator, idx_eve, event_init)

        if event_ is not None:
            events_inverted.append(event_)
        picks_inverted.append(picks_)

    if len(events_inverted) == 0:
        return None, None
    events_inverted = pd.concat(events_inverted, ignore_index=True)
    picks_inverted = pd.concat(picks_inverted)
    if events_init is not None:
        events_inverted = events_inverted.merge(events_init[["event_index", "idx_eve"]], on="idx_eve")
    else:
        events_inverted["event_index"] = events_inverted["idx_eve"]

    print(f"ADLoc using {len(picks_inverted[picks_inverted['mask'] == 1])} picks outof {len(picks_inverted)} picks")
    return picks_inverted, events_inverted

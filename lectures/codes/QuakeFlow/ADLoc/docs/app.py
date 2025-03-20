# %%
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pyproj import Proj

import adloc
from adloc.eikonal2d import init_eikonal2d
from adloc.sacloc2d import ADLoc
from adloc.utils import invert_location, invert_location_iter

app = FastAPI()


@app.get("/")
def greet_json():
    return {"Hello": "ADLoc!"}


@app.post("/predict/")
def predict(picks: dict, stations: dict, config: dict):
    picks = picks["data"]
    stations = stations["data"]
    picks = pd.DataFrame(picks)
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    stations = pd.DataFrame(stations)
    events_, picks_ = run_adloc(picks, stations, config)
    if events_ is None:
        return {"events": None, "picks": None}
    events_ = events_.to_dict(orient="records")
    picks_ = picks_.to_dict(orient="records")

    return {"events": events_, "picks": picks_}


def set_config(region="ridgecrest"):

    config = {
        "min_picks": 8,
        "min_picks_ratio": 0.2,
        "max_residual_time": 1.0,
        "max_residual_amplitude": 1.0,
        "min_score": 0.6,
        "min_s_picks": 2,
        "min_p_picks": 2,
        "use_amplitude": False,
    }

    # ## Domain
    if region.lower() == "ridgecrest":
        config.update(
            {
                "region": "ridgecrest",
                "minlongitude": -118.004,
                "maxlongitude": -117.004,
                "minlatitude": 35.205,
                "maxlatitude": 36.205,
                "mindepth_km": 0.0,
                "maxdepth_km": 30.0,
            }
        )

    lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
    lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")
    xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
    xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
    zmin, zmax = config["mindepth_km"], config["maxdepth_km"]
    xlim_km = (xmin, xmax)
    ylim_km = (ymin, ymax)
    zlim_km = (zmin, zmax)

    config.update(
        {
            "xlim_km": xlim_km,
            "ylim_km": ylim_km,
            "zlim_km": zlim_km,
            "proj": proj,
        }
    )

    ## Eikonal for 1D velocity model
    zz = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0]
    vp = [4.746, 4.793, 4.799, 5.045, 5.721, 5.879, 6.504, 6.708, 6.725, 7.800]
    vs = [2.469, 2.470, 2.929, 2.930, 3.402, 3.403, 3.848, 3.907, 3.963, 4.500]
    h = 0.3

    vel = {"Z": zz, "P": vp, "S": vs}
    eikonal = {
        "vel": vel,
        "h": h,
        "xlim_km": xlim_km,
        "ylim_km": ylim_km,
        "zlim_km": zlim_km,
    }
    eikonal = init_eikonal2d(eikonal)
    config["eikonal"] = eikonal

    config["bfgs_bounds"] = (
        (xlim_km[0] - 1, xlim_km[1] + 1),  # x
        (ylim_km[0] - 1, ylim_km[1] + 1),  # y
        (0, zlim_km[1] + 1),  # z
        (None, None),  # t
    )

    config["event_index"] = 0

    return config


config = set_config()


# %%
def run_adloc(picks, stations, config_):

    # %%
    config.update(config_)

    proj = config["proj"]

    # %%
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z_km"] = stations["elevation_m"].apply(lambda x: -x / 1e3)

    # %%
    mapping_phase_type_int = {"P": 0, "S": 1}
    picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)
    if "phase_amplitude" in picks.columns:
        picks["phase_amplitude"] = picks["phase_amplitude"].apply(lambda x: np.log10(x) + 2.0)  # convert to log10(cm/s)

    # %%
    # reindex in case the index does not start from 0 or is not continuous
    stations["idx_sta"] = np.arange(len(stations))
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")
    picks["idx_eve"] = config["event_index"]

    # %%
    estimator = ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, eikonal=config["eikonal"])

    # %%
    picks, events = invert_location_iter(picks, stations, config, estimator, events_init=None, iter=0)

    if (picks is None) or (events is None):
        return None, None

    # %%
    if "event_index" not in events.columns:
        events["event_index"] = events.merge(picks[["idx_eve", "event_index"]], on="idx_eve")["event_index"]
    events[["longitude", "latitude"]] = events.apply(
        lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
    )
    events["depth_km"] = events["z_km"]
    events = events.drop(["idx_eve", "x_km", "y_km", "z_km"], axis=1, errors="ignore")
    events.sort_values(["time"], inplace=True)

    picks.rename({"mask": "adloc_mask", "residual_time": "adloc_residual_time", "residual_amplitude": "adloc_residual_amplitude"}, axis=1, inplace=True)
    picks["phase_type"] = picks["phase_type"].map({0: "P", 1: "S"})
    picks = picks.drop(["idx_eve", "idx_sta"], axis=1, errors="ignore")
    picks.sort_values(["phase_time"], inplace=True)

    return events, picks

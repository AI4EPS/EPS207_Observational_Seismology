# %%
import json
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Proj
from tqdm import tqdm

from adloc.eikonal2d import calc_traveltime, init_eikonal2d

np.random.seed(111)

# %%
# result_path = "results"
result_path = "test_data/synthetic"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# %%
config = {
    "minlatitude": 30.0,
    "maxlatitude": 32.0,
    "minlongitude": 130.0,
    "maxlongitude": 132.0,
    "mindepth": 0.0,
    "maxdepth": 20.0,
    "degree2km": 111.19,
}


# %%
time0 = datetime.fromisoformat("2019-01-01T00:00:00")
lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0} +lat_ts={lat0} +units=km")

min_x_km, min_y_km = proj(longitude=config["minlongitude"], latitude=config["minlatitude"])
max_x_km, max_y_km = proj(longitude=config["maxlongitude"], latitude=config["maxlatitude"])
config["xlim_km"] = [min_x_km, max_x_km]
config["ylim_km"] = [min_y_km, max_y_km]
config["zlim_km"] = [config["mindepth"], config["maxdepth"]]
config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
config["eikonal"] = None

with open(f"{result_path}/config.json", "w") as f:
    json.dump(config, f, indent=4)

# %%
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
config["eikonal"] = init_eikonal2d(config["eikonal"])
# config["eikonal"] = None

# %%
mapping_phase_type_int = {"P": 0, "S": 1}


# %%
def calc_time(event_loc, station_loc, phase_type, velocity={"P": 6.0, "S": 6.0 / 1.73}, eikonal=None):
    # def calc_time(event_loc, station_loc, phase_type, velocity={0: 6.0, 1: 6.0 / 1.73}, eikonal=None):
    if eikonal is None:
        dist = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False)
        tt = dist / np.array([velocity[x] for x in phase_type])
    else:
        tt = calc_traveltime(event_loc, station_loc, [mapping_phase_type_int[x] for x in phase_type], eikonal)
    return tt


# %%
num_station = 10
stations = []
for i in range(num_station):
    station_id = f"NC.{i:02d}"
    latitude = lat0 + (np.random.rand() - 0.5) * 1
    longitude = lon0 + (np.random.rand() - 0.5) * 1
    elevation_m = np.random.rand() * 1000
    depth_km = -elevation_m / 1000
    # station_term = np.random.rand() - 0.5
    station_term = 0.0
    stations.append([station_id, latitude, longitude, elevation_m, depth_km, station_term])

stations = pd.DataFrame(
    stations, columns=["station_id", "latitude", "longitude", "elevation_m", "depth_km", "station_term"]
)
stations[["x_km", "y_km"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
stations["z_km"] = stations["depth_km"]
stations.to_csv(f"{result_path}/stations.csv", index=False)


# %%
stations_json = stations.copy()
stations_json.set_index("station_id", inplace=True)
stations_json = stations_json.to_dict(orient="index")
with open(f"{result_path}/stations.json", "w") as f:
    json.dump(stations_json, f, indent=4)

# %%
event_index = 0
events = []
picks = []

num_event = 30
theta = np.deg2rad(30)
R = 60
for depth in [5.0, 10.0, 15.0]:

    for i, theta in enumerate(np.linspace(0, 2 * np.pi, num_event, endpoint=False)):
        event_index += 1
        origin_time = time0 + timedelta(seconds=i * 10)
        x = np.cos(theta) * R
        y = np.sin(theta) * R
        longitude, latitude = proj(x, y, inverse=True)
        events.append([origin_time.strftime("%Y-%m-%dT%H:%M:%S.%f"), latitude, longitude, depth, event_index])
        x_km, y_km = proj(longitude=longitude, latitude=latitude)
        z_km = depth
        for j, station in stations.iterrows():
            for phase_type in ["P", "S"]:
                # for phase_type in [0, 1]:
                travel_time = calc_time(
                    np.array([[x_km, y_km, z_km]]),
                    np.array([[station["x_km"], station["y_km"], station["z_km"]]]),
                    [phase_type],
                    velocity=config["vel"],
                    eikonal=config["eikonal"],
                )[0]
                arrival_time = (
                    origin_time + timedelta(seconds=float(travel_time)) + timedelta(seconds=station["station_term"])
                )
                pick = [
                    station["station_id"],
                    arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    phase_type,
                    1.0,
                    event_index,
                ]
                picks.append(pick)

    y = (config["xlim_km"][0] + config["xlim_km"][1]) / 2
    for i, x in enumerate(np.linspace(config["xlim_km"][0], config["xlim_km"][1], num_event)):
        event_index += 1
        origin_time = time0 + timedelta(seconds=i * 10)
        longitude, latitude = proj(x, y, inverse=True)
        events.append([origin_time.strftime("%Y-%m-%dT%H:%M:%S.%f"), latitude, longitude, depth, event_index])
        x_km, y_km = proj(longitude=longitude, latitude=latitude)
        z_km = depth
        for j, station in stations.iterrows():
            for phase_type in ["P", "S"]:
                # for phase_type in [0, 1]:
                travel_time = calc_time(
                    np.array([[x_km, y_km, z_km]]),
                    np.array([[station["x_km"], station["y_km"], station["z_km"]]]),
                    [phase_type],
                    velocity=config["vel"],
                    eikonal=config["eikonal"],
                )[0]
                arrival_time = (
                    origin_time + timedelta(seconds=float(travel_time)) + timedelta(seconds=station["station_term"])
                )
                pick = [
                    station["station_id"],
                    arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    phase_type,
                    1.0,
                    event_index,
                ]
                picks.append(pick)

    x = (config["xlim_km"][0] + config["xlim_km"][1]) / 2
    for i, y in enumerate(np.linspace(config["ylim_km"][0], config["ylim_km"][1], num_event)):
        event_index += 1
        origin_time = time0 + timedelta(seconds=i * 10)
        longitude, latitude = proj(x, y, inverse=True)
        events.append([origin_time.strftime("%Y-%m-%dT%H:%M:%S.%f"), latitude, longitude, depth, event_index])
        x_km, y_km = proj(longitude=longitude, latitude=latitude)
        z_km = depth
        for j, station in stations.iterrows():
            for phase_type in ["P", "S"]:
                # for phase_type in [0, 1]:
                travel_time = calc_time(
                    np.array([[x_km, y_km, z_km]]),
                    np.array([[station["x_km"], station["y_km"], station["z_km"]]]),
                    [phase_type],
                    velocity=config["vel"],
                    eikonal=config["eikonal"],
                )[0]
                arrival_time = (
                    origin_time + timedelta(seconds=float(travel_time)) + timedelta(seconds=station["station_term"])
                )
                pick = [
                    station["station_id"],
                    arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    phase_type,
                    1.0,
                    event_index,
                ]
                picks.append(pick)


# %%
events = pd.DataFrame(events, columns=["time", "latitude", "longitude", "depth_km", "event_index"])
events[["x_km", "y_km"]] = events.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
events["z_km"] = events["depth_km"]
events.to_csv(f"{result_path}/events.csv", index=False)
picks = pd.DataFrame(picks, columns=["station_id", "phase_time", "phase_type", "phase_score", "event_index"])
picks.to_csv(f"{result_path}/picks.csv", index=False)


# %%
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# axs[0].scatter(stations["longitude"], stations["latitude"], s=10, marker="^", label="stations")
# axs[0].scatter(events["longitude"], events["latitude"], s=10, marker=".", label="events")
# axs[0].axis("scaled")
# axs[0].legend()

# axs[1].scatter(stations["longitude"], stations["depth_km"], s=10, marker="^", label="stations")
# axs[1].scatter(events["longitude"], events["depth_km"], s=10, marker=".", label="events")
# axs[1].invert_yaxis()
# axs[1].legend()

# axs[2].scatter(stations["latitude"], stations["depth_km"], s=10, marker="^", label="stations")
# axs[2].scatter(events["latitude"], events["depth_km"], s=10, marker=".", label="events")
# axs[2].invert_yaxis()
# axs[2].legend()

# plt.tight_layout()
# plt.savefig(f"{result_path}/events.png", dpi=300, bbox_inches="tight")

fig, axs = plt.subplots(1, 3, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1, 1]})

im = axs[0].scatter(
    stations["x_km"],
    stations["y_km"],
    c=stations["station_term"],
    s=30,
    cmap="viridis_r",
    marker="^",
    label="stations",
)
axs[0].scatter(events["x_km"], events["y_km"], s=10, marker=".", label="events")
axs[0].axis("scaled")
axs[0].legend()
fig.colorbar(im, ax=axs[0])

axs[1].scatter(stations["x_km"], stations["z_km"], s=10, marker="^", label="stations")
axs[1].scatter(events["x_km"], events["z_km"], s=10, marker=".", label="events")
axs[1].invert_yaxis()
axs[1].legend()

axs[2].scatter(stations["y_km"], stations["z_km"], s=10, marker="^", label="stations")
axs[2].scatter(events["y_km"], events["z_km"], s=10, marker=".", label="events")
axs[2].invert_yaxis()
axs[2].legend()

plt.tight_layout()
plt.savefig(f"{result_path}/events.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(12, 5))
picks = picks.merge(stations[["station_id", "x_km", "y_km", "z_km"]], on="station_id")
picks["phase_time"] = pd.to_datetime(picks["phase_time"])
mapping_color = lambda x: f"C{x}"
plt.scatter(picks["phase_time"], picks["x_km"], c=picks["event_index"].map(mapping_color), s=1)
plt.xlabel("phase_time")
plt.ylabel("x_km")
plt.savefig(f"{result_path}/picks.png", dpi=300, bbox_inches="tight")

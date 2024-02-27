# %%
import json
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Proj
from tqdm import tqdm

np.random.seed(0)

# %%
result_path = "results"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# %%
config = {
    "minlatitude": 30.0,
    "maxlatitude": 35.0,
    "minlongitude": 130.0,
    "maxlongitude": 135.0,
    "mindepth": 0.0,
    "maxdepth": 20.0,
    "degree2km": 111.19,
}


# %%
time0 = datetime.fromisoformat("2019-01-01T00:00:00")
depth0 = (config["mindepth"] + config["maxdepth"]) / 2
latitude0 = (config["minlatitude"] + config["maxlatitude"]) / 2
longitude0 = (config["minlongitude"] + config["maxlongitude"]) / 2
config["latitude0"] = latitude0
config["longitude0"] = longitude0
config["depth0"] = depth0

proj = Proj(f"+proj=sterea +lon_0={config['longitude0']} +lat_0={config['latitude0']} +units=km")

with open(f"{result_path}/config.json", "w") as f:
    json.dump(config, f)

# %%
# def calc_travel_time(
#     event_time,
#     # event_latitude,
#     # event_longitude,
#     event_x_km,
#     event_y_km,
#     ev
#     # event_depth,
#     # station_latitude,
#     # station_longitude,
#     station_x_km,
#     station_y_km,
#     station_depth,
#     phase="P",
#     velocity={"P": 6.0, "S": 6.0 / 1.73},
# ):
#     # r = (
#     #     np.sqrt(
#     #         (event_latitude - station_latitude) ** 2
#     #         + (event_longitude - station_longitude) ** 2 * np.cos(np.deg2rad(event_latitude)) ** 2
#     #     )
#     #     * 111.19
#     # )
#     # r = np.sqrt((event_x_km - station_x_km) ** 2 + (event_y_km - station_y_km) ** 2)
#     # z = event_depth - station_depth
#     # t = np.sqrt(r**2 + z**2) / velocity[phase]

#     dist = np.linalg.norm
#     return t


# copy from model.py
def calc_time(event_loc, station_loc, phase_type, velocity={"P": 6.0, "S": 6.0 / 1.73}):
    dist = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=True)
    tt = dist / velocity[phase_type]
    return tt


# %%
num_station = 5
stations = []
for i in range(num_station):
    station_id = f"NC.{i:02d}"
    latitude = latitude0 + (np.random.rand() - 0.5) * 1
    longitude = longitude0 + (np.random.rand() - 0.5) * 1
    elevation_m = np.random.rand() * 1000 * 0.0
    depth_km = -elevation_m / 1000
    stations.append([station_id, latitude, longitude, elevation_m, depth_km])


stations = pd.DataFrame(stations, columns=["station_id", "latitude", "longitude", "elevation_m", "depth_km"])
stations[["x_km", "y_km"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
stations["z_km"] = stations["depth_km"]
stations.to_csv(f"{result_path}/stations.csv", index=False)

# %%
plt.figure()
plt.scatter(stations["longitude"], stations["latitude"], s=10, marker="^", label="stations")
plt.axis("scaled")
plt.savefig(f"{result_path}/stations_xy.png", dpi=300)

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

# num_event = 100
# for i in range(num_event):
#     event_index += 1
#     # time = time0 + timedelta(seconds=np.random.randint(0, 86400))
#     time = time0
#     latitude = latitude0 + (np.random.rand() - 0.5) * 1
#     longitude = longitude0 + (np.random.rand() - 0.5) * 1
#     depth = np.random.rand() * 20
#     events.append([time.strftime("%Y-%m-%dT%H:%M:%S"), latitude, longitude, depth, event_index])

#     for j, station in stations.iterrows():
#         phase_type = np.random.choice(["P", "S"])
#         travel_time = calc_travel_time(
#             time,
#             latitude,
#             longitude,
#             depth,
#             station["latitude"],
#             station["longitude"],
#             station["depth_km"],
#             phase=phase_type,
#         )
#         arrival_time = time + timedelta(seconds=travel_time)
#         pick = [station["station_id"], arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%f"), phase_type, 1.0, event_index]
#         picks.append(pick)

num_event = 100
theta = np.deg2rad(30)
d = 0.5
for i in range(num_event):
    event_index += 1
    time = time0
    latitude = latitude0 + (i / num_event - 0.5) * np.sin(theta) * d
    longitude = longitude0 + (i / num_event - 0.5) * np.cos(theta) * d
    depth = depth0 + (i / num_event - 0.5) * (config["maxdepth"] - config["mindepth"])
    events.append([time.strftime("%Y-%m-%dT%H:%M:%S.%f"), latitude, longitude, depth, event_index])
    x_km, y_km = proj(longitude=longitude, latitude=latitude)
    z_km = depth
    for j, station in stations.iterrows():
        phase_type = np.random.choice(["P", "S"])
        # travel_time = calc_travel_time(
        #     time,
        #     # latitude,
        #     # longitude,
        #     x_km,
        #     y_km,
        #     depth,
        #     # station["latitude"],
        #     # station["longitude"],
        #     station["x_km"],
        #     station["y_km"],
        #     station["depth_km"],
        #     phase=phase_type,
        # )
        travel_time = calc_time(
            np.array([[x_km, y_km, z_km]]), np.array([[station["x_km"], station["y_km"], station["z_km"]]]), phase_type
        )[0, 0]
        arrival_time = time + timedelta(seconds=travel_time)
        pick = [station["station_id"], arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%f"), phase_type, 1.0, event_index]
        picks.append(pick)

# num_event = 100
# r = 0.1
# for i, theta in enumerate(np.linspace(0, 2 * np.pi, num_event)):
#     event_index += 1
#     time = time0
#     latitude = latitude0 + np.sin(theta) * r
#     longitude = longitude0 + np.cos(theta) * r
#     depth = (config["maxdepth"] - config["mindepth"]) / 2
#     events.append([time.strftime("%Y-%m-%dT%H:%M:%S.%f"), latitude, longitude, depth, event_index])
#     for j, station in stations.iterrows():
#         phase_type = np.random.choice(["P", "S"])
#         travel_time = calc_travel_time(
#             time,
#             latitude,
#             longitude,
#             depth,
#             station["latitude"],
#             station["longitude"],
#             station["depth_km"],
#             phase=phase_type,
#         )
#         arrival_time = time + timedelta(seconds=travel_time)
#         pick = [station["station_id"], arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%f"), phase_type, 1.0, event_index]
#         picks.append(pick)


# %%
events = pd.DataFrame(events, columns=["time", "latitude", "longitude", "depth_km", "event_index"])
events[["x_km", "y_km"]] = events.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
events["z_km"] = events["depth_km"]
events.to_csv(f"{result_path}/events.csv", index=False)
picks = pd.DataFrame(picks, columns=["station_id", "phase_time", "phase_type", "phase_score", "event_index"])
picks.to_csv(f"{result_path}/picks.csv", index=False)


# %%
plt.figure()
plt.scatter(stations["longitude"], stations["latitude"], s=10, marker="^", label="stations")
plt.scatter(events["longitude"], events["latitude"], s=10, marker=".", label="events")
plt.axis("scaled")
plt.legend()
# plt.show()
plt.savefig(f"{result_path}/events_xy.png", dpi=300)
# %%
plt.figure()
plt.scatter(stations["longitude"], stations["depth_km"], s=10, marker="^", label="stations")
plt.scatter(events["longitude"], events["depth_km"], s=10, marker=".", label="events")
plt.gca().invert_yaxis()
plt.legend()
# plt.show()
plt.savefig(f"{result_path}/events_xz.png", dpi=300)
# %%
plt.figure()
plt.scatter(stations["latitude"], stations["depth_km"], s=10, marker="^", label="stations")
plt.scatter(events["latitude"], events["depth_km"], s=10, marker=".", label="events")
plt.gca().invert_yaxis()
plt.legend()
# plt.show()
plt.savefig(f"{result_path}/events_yz.png", dpi=300)

# %%

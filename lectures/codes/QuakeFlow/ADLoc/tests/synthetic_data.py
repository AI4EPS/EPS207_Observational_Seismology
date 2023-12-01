# %%
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(0)

# %%
# stations = pd.read_csv("../test_data/stations.csv", sep="\t")
# stations["station_id"] = stations["station"].apply(lambda x: ".".join(x.split(".")[:3]) + ".")
# stations = stations.groupby("station_id").first().reset_index()
# stations["depth_km"] = stations["elevation(m)"].apply(lambda x: round(x / 1000, 3))
# stations = stations[["station_id", "latitude", "longitude", "depth_km"]]
# stations.to_csv("stations.csv", index=False)
stations = pd.read_csv("stations.csv")

# %%
latitude0 = stations["latitude"].mean()
longitude0 = stations["longitude"].mean()
time0 = datetime.fromisoformat("2019-01-01T00:00:00")


# %%
def calc_travel_time(
    event_time,
    event_latitude,
    event_longitude,
    event_depth,
    station_latitude,
    station_longitude,
    station_depth,
    phase="P",
    velocity={"P": 6.0, "S": 6.0 / 1.73},
):
    r = (
        np.sqrt(
            (event_latitude - station_latitude) ** 2
            + (event_longitude - station_longitude) ** 2 * np.cos(np.deg2rad(event_latitude)) ** 2
        )
        * 111.19
    )
    z = event_depth - station_depth
    t = np.sqrt(r**2 + z**2) / velocity[phase]
    return t


# %%
num_event = 100
events = []
picks = []
for i in range(num_event):
    # time = time0 + timedelta(seconds=np.random.randint(0, 86400))
    time = time0
    latitude = latitude0 + (np.random.rand() - 0.5) * 1
    longitude = longitude0 + (np.random.rand() - 0.5) * 1
    depth = np.random.rand() * 20
    events.append([time.isoformat(), latitude, longitude, depth, i])

    for j, station in stations.iterrows():
        phase_type = np.random.choice(["P", "S"])
        travel_time = calc_travel_time(
            time,
            latitude,
            longitude,
            depth,
            station["latitude"],
            station["longitude"],
            station["depth_km"],
            phase=phase_type,
        )
        arrival_time = time + timedelta(seconds=travel_time)
        pick = [station["station_id"], arrival_time.isoformat(), phase_type, 1.0, i]
        picks.append(pick)

events = pd.DataFrame(events, columns=["time", "latitude", "longitude", "depth_km", "event_index"])
events.to_csv("events.csv", index=False)
picks = pd.DataFrame(picks, columns=["station_id", "phase_time", "phase_type", "phase_score", "event_index"])
picks.to_csv("picks.csv", index=False)

# %%
plt.figure()
plt.scatter(stations["longitude"], stations["latitude"], s=10, marker="^", label="stations")
plt.scatter(events["longitude"], events["latitude"], s=10, marker=".", label="events")
plt.axis("scaled")
plt.legend()
plt.show()
# %%

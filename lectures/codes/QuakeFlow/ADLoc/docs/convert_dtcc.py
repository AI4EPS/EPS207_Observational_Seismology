# %%
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
data_path = "stanford"
result_path = "stanford"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# %%
stations = pd.read_csv(f"{data_path}/station.csv")
# events = pd.read_csv(f"{data_path}/true_event.csv")
events = pd.read_csv(f"results/{data_path}/adloc_events_sst.csv")
events["event_index"] = [f"{i}" for i in range(1, len(events) + 1)]
events.rename({"Lat": "latitude", "Lon": "longitude", "Dep": "depth_km"}, axis=1, inplace=True)

stations.rename({"elevation": "elevation_m"}, axis=1, inplace=True)
stations["depth_km"] = -stations["elevation_m"] / 1000
stations["station_id"] = stations["station"]

stations["idx_sta"] = np.arange(len(stations))  # reindex in case the index does not start from 0 or is not continuous
events["idx_eve"] = np.arange(len(events))  # reindex in case the index does not start from 0 or is not continuous
mapping_phase_type_int = {"P": 0, "S": 1}

# %%
with open(f"{data_path}/dt.cc", "r") as f:
    lines = f.readlines()


# %%
event_index1 = []
event_index2 = []
station_index = []
phase_type = []
phase_score = []
phase_dtime = []

stations.set_index("station_id", inplace=True)
events.set_index("event_index", inplace=True)

for line in tqdm(lines):
    if line[0] == "#":
        evid1, evid2, _ = line[1:].split()
    else:
        stid, dt, weight, phase = line.split()
        event_index1.append(events.loc[evid1, "idx_eve"])
        event_index2.append(events.loc[evid2, "idx_eve"])
        station_index.append(stations.loc[stid, "idx_sta"])
        phase_type.append(mapping_phase_type_int[phase])
        phase_score.append(weight)
        phase_dtime.append(dt)


dtypes = np.dtype(
    [
        ("event_index1", np.int32),
        ("event_index2", np.int32),
        ("station_index", np.int32),
        ("phase_type", np.int32),
        ("phase_score", np.float32),
        ("phase_dtime", np.float32),
    ]
)
pairs_array = np.memmap(
    os.path.join(result_path, "pair_dt.dat"),
    mode="w+",
    shape=(len(phase_dtime),),
    dtype=dtypes,
)
pairs_array["event_index1"] = event_index1
pairs_array["event_index2"] = event_index2
pairs_array["station_index"] = station_index
pairs_array["phase_type"] = phase_type
pairs_array["phase_score"] = phase_score
pairs_array["phase_dtime"] = phase_dtime
with open(os.path.join(result_path, "pair_dtypes.pkl"), "wb") as f:
    pickle.dump(dtypes, f)


# %%
events.to_csv(os.path.join(result_path, "pair_events.csv"), index=True, index_label="event_index")
stations.to_csv(os.path.join(result_path, "pair_stations.csv"), index=True, index_label="station_id")

# %%

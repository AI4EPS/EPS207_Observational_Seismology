# %%
import pandas as pd

# %%
picks = pd.read_csv("results/picks.csv", parse_dates=["phase_time"])

# %%
stations = pd.read_csv("results/stations.csv")
stations["idx_sta"] = stations.index # reindex in case the index does not start from 0 or is not continuous 

# %%
events = pd.read_csv("results/events.csv", parse_dates=["time"])
events["idx_eve"] = events.index # reindex in case the index does not start from 0 or is not continuous

# %%
picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")

# %%
picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

# %%
for idx_sta, picks_station in picks.groupby("idx_sta"):
    station_loc = stations.loc[idx_sta]
    print(f"Station {station_loc['station_id']} at ({station_loc['latitude']}, {station_loc['longitude']})")
    for _, pick in picks_station.iterrows():
        event_loc = events.loc[pick["idx_eve"]]
        print(f"Event {event_loc['event_index']} at ({event_loc['latitude']}, {event_loc['longitude']})")
    raise
# %%

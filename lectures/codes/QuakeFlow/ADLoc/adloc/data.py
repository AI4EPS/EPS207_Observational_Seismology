import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tqdm import tqdm

# class PhaseDataset(Dataset):
#     def __init__(self, picks, events, stations, batch_size=100, double_difference=False, config=None):
#         self.picks = picks
#         self.events = events
#         self.stations = stations
#         self.config = config
#         self.double_difference = double_difference
#         self.batch_size = batch_size
#         if double_difference:
#             self.read_data_dd()
#         else:
#             self.read_data()

#     def __len__(self):
#         ## TODO: return batch
#         return 1
#         # return (len(self.events) - 1) // self.batch_size + 1

#     def read_data(self):
#         event_index = []
#         station_index = []
#         phase_score = []
#         phase_time = []
#         phase_type = []

#         picks_by_event = self.picks.groupby("index")
#         for key, group in picks_by_event:
#             if key == -1:
#                 continue
#             phase_time.append(group["phase_time"].values)
#             phase_score.append(group["phase_score"].values)
#             phase_type.extend(group["phase_type"].values.tolist())
#             event_index.extend([key] * len(group))
#             station_index.append(self.stations.loc[group["station_id"], "index"].values)

#         phase_time = np.concatenate(phase_time)
#         phase_score = np.concatenate(phase_score)
#         phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
#         event_index = np.array(event_index)
#         station_index = np.concatenate(station_index)

#         self.station_index = torch.tensor(station_index, dtype=torch.long)
#         self.event_index = torch.tensor(event_index, dtype=torch.long)
#         self.phase_weight = torch.tensor(phase_score, dtype=torch.float32)
#         self.phase_time = torch.tensor(phase_time, dtype=torch.float32)
#         self.phase_type = torch.tensor(phase_type, dtype=torch.long)

#     def read_data_dd(self):
#         event_index = []
#         station_index = []
#         phase_score = []
#         phase_time = []
#         phase_type = []

#         event_loc = self.events[["x_km", "y_km", "z_km"]].values
#         event_time = self.events["time"].values[:, np.newaxis]

#         neigh = NearestNeighbors(radius=self.config.min_pair_dist, metric="euclidean")
#         neigh.fit(event_loc)

#         picks_by_event = self.picks.groupby("index")

#         for key1, group1 in tqdm(picks_by_event, total=len(picks_by_event), desc="Generating pairs"):
#             if key1 == -1:
#                 continue

#             for key2 in neigh.radius_neighbors([event_loc[key1]], return_distance=False)[0]:
#                 if key1 >= key2:
#                     continue

#                 common = group1.merge(picks_by_event.get_group(key2), on=["station_id", "phase_type"], how="inner")
#                 phase_time.append(common["phase_time_x"].values - common["phase_time_y"].values)
#                 phase_score.append(common["phase_score_x"].values * common["phase_score_y"].values)
#                 phase_type.extend(common["phase_type"].values.tolist())
#                 event_index.extend([[key1, key2]] * len(common))
#                 station_index.append(self.stations.loc[common["station_id"], "index"].values)

#         if len(phase_time) > 0:
#             phase_time = np.concatenate(phase_time)
#             phase_score = np.concatenate(phase_score)
#             phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
#             event_index = np.array(event_index)
#             station_index = np.concatenate(station_index)

#             # %%
#             self.station_index = torch.tensor(station_index, dtype=torch.long)
#             self.event_index = torch.tensor(event_index, dtype=torch.long)
#             self.phase_weight = torch.tensor(phase_score, dtype=torch.float32)
#             self.phase_time = torch.tensor(phase_time, dtype=torch.float32)
#             self.phase_type = torch.tensor(phase_type, dtype=torch.long)
#         else:
#             self.station_index = torch.tensor([], dtype=torch.long)
#             self.event_index = torch.tensor([], dtype=torch.long)
#             self.phase_weight = torch.tensor([], dtype=torch.float32)
#             self.phase_time = torch.tensor([], dtype=torch.float32)
#             self.phase_type = torch.tensor([], dtype=torch.long)

#     def __getitem__(self, i):
#         # phase_time = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_time"].values
#         # phase_score = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_score"].values
#         # phase_type = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]][
#         #     "phase_type"
#         # ].values.tolist()
#         # event_index = np.array([i] * len(self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]))
#         # station_index = self.stations.loc[
#         #     self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["station_id"], "index"
#         # ].values

#         return {
#             "event_index": self.event_index,
#             "station_index": self.station_index,
#             "phase_time": self.phase_time,
#             "phase_weight": self.phase_weight,
#             "phase_type": self.phase_type,
#         }


class PhaseDataset(Dataset):
    def __init__(
        self,
        picks,
        events,
        stations,
        batch_size=1000,
        config=None,
        rank=0,
        world_size=1,
    ):
        self.picks = picks
        self.events = events
        self.stations = stations
        self.config = config
        self.batch_size = batch_size

        # preprocess
        self.picks_by_event = picks.groupby("idx_eve")
        self.idx_eve_batch = np.array_split(self.events["idx_eve"], (len(self.events) - 1) // self.batch_size + 1)[
            rank::world_size
        ]
        self.read_data()

    def __len__(self):
        return len(self.idx_eve_batch)

    def read_data(self):
        meta = {}
        for i, index_batch in enumerate(self.idx_eve_batch):
            idx_eve = []
            idx_sta = []
            phase_score = []
            phase_time = []
            phase_type = []

            for key in index_batch:
                if key == -1:
                    continue
                group = self.picks_by_event.get_group(key)
                phase_time.append(group["travel_time"].values)  # seconds
                # phase_time.append(group["phase_time"].values) # datetime not supported
                phase_score.append(group["phase_score"].values)
                phase_type.extend(group["phase_type"].values.tolist())
                idx_eve.extend(group["idx_eve"].values.tolist())
                idx_sta.append(group["idx_sta"].values.tolist())

            phase_time = np.concatenate(phase_time)
            phase_score = np.concatenate(phase_score)
            # phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
            # if ("P" in phase_type) and ("S" in phase_type):
            #     phase_type = np.array([{"P": 0, "S": 1}[x] for x in phase_type])
            idx_eve = np.array(idx_eve)
            idx_sta = np.concatenate(idx_sta)

            idx_sta = torch.tensor(idx_sta, dtype=torch.long)
            idx_eve = torch.tensor(idx_eve, dtype=torch.long)
            phase_weight = torch.tensor(phase_score, dtype=torch.float32)
            phase_time = torch.tensor(phase_time, dtype=torch.float32)
            phase_type = np.array(phase_type)

            meta[i] = {
                "idx_eve": idx_eve,
                "idx_sta": idx_sta,
                "phase_time": phase_time,
                "phase_weight": phase_weight,
                "phase_type": phase_type,
            }
        self.meta = meta

    def __getitem__(self, i):
        return self.meta[i]


class PhaseDatasetDT(Dataset):
    def __init__(
        self,
        pairs,
        picks,
        events,
        stations,
        batch_size=10000,
        valid_index=None,
        config=None,
        rank=0,
        world_size=1,
    ):
        self.pairs = pairs
        self.picks = picks
        self.events = events
        self.stations = stations
        self.config = config
        self.batch_size = batch_size
        if valid_index is None:
            valid_index = np.ones(len(self.pairs), dtype=bool)
        self.valid_index = valid_index

        self.idx_batch = np.array_split(np.arange(len(self.pairs)), (len(self.pairs) - 1) // self.batch_size + 1)[
            rank::world_size
        ]

        print(f"Generated {len(self.pairs)} pairs")
        print(f"Split into {len(self.idx_batch)} batches of size {self.batch_size} at rank {rank}/{world_size}")

    def __len__(self):
        return len(self.idx_batch)

    def __getitem__(self, i):

        idx = self.idx_batch[i][self.valid_index[self.idx_batch[i]]]

        if len(idx) == 0:
            return None  # skip empty batch

        idx1_eve = self.pairs["idx_eve1"][idx]
        idx2_eve = self.pairs["idx_eve2"][idx]
        idx_eve = np.stack([idx1_eve, idx2_eve], axis=1)
        idx_sta = self.pairs["idx_sta"][idx]
        phase_weight = self.pairs["phase_score"][idx]
        phase_type = self.pairs["phase_type"][idx]
        phase_time = self.pairs["phase_dtime"][idx]

        return {
            "idx_eve": torch.tensor(idx_eve, dtype=torch.long),
            "idx_sta": torch.tensor(idx_sta, dtype=torch.long),
            "phase_type": phase_type,
            "phase_weight": torch.tensor(phase_weight, dtype=torch.float32),
            "phase_time": torch.tensor(phase_time, dtype=torch.float32),
        }


class PhaseDatasetDTCC(PhaseDatasetDT):

    def __getitem__(self, i):

        idx = self.idx_batch[i][self.valid_index[self.idx_batch[i]]]
        idx1_eve = self.pairs["idx_eve1"][idx].values
        idx2_eve = self.pairs["idx_eve2"][idx].values
        idx_eve = np.stack([idx1_eve, idx2_eve], axis=1)
        idx_sta = self.pairs["idx_sta"][idx].values
        phase_weight = self.pairs["weight"][idx].values
        phase_type = self.pairs["phase_type"][idx].values
        phase_time = self.pairs["dt"][idx].values

        return {
            "idx_eve": torch.tensor(idx_eve, dtype=torch.long),
            "idx_sta": torch.tensor(idx_sta, dtype=torch.long),
            "phase_type": phase_type,
            "phase_weight": torch.tensor(phase_weight, dtype=torch.float32),
            "phase_time": torch.tensor(phase_time, dtype=torch.float32),
        }

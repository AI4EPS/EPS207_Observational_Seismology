import itertools
import logging
from collections import defaultdict
from pathlib import Path

import fsspec
import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import scipy.signal
import torch
import torch.nn.functional as F
import torchaudio
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm


class CCDataset(Dataset):
    def __init__(
        self,
        config=None,
        pair_list=None,
        data_list1=None,
        data_list2=None,
        data_path1="./",
        data_path2="./",
        data_format1="h5",
        data_format2="h5",
        block_size1=1,
        block_size2=1,
        device="cpu",
        transforms=None,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super(CCDataset).__init__()
        ## TODO: extract this part into a function; keep this temporary until TM and AN are implemented
        ## pair_list has the highest priority
        if pair_list is not None:
            self.pair_list, self.data_list1, self.data_list2 = read_pair_list(pair_list)
        ## use data_list1 if exits and use pair_list as filtering
        if data_list1 is not None:
            self.data_list1 = pd.unique(pd.read_csv(data_list1, header=None)[0]).tolist()
            if data_list2 is not None:
                self.data_list2 = pd.unique(pd.read_csv(data_list2, header=None)[0]).tolist()
            elif pair_list is None:
                self.data_list2 = self.data_list1
            ## generate pair_list if not provided
            if pair_list is None:
                self.pair_list = generate_pairs(self.data_list1, self.data_list2, config.auto_xcorr)

        self.mode = config.mode
        self.config = config
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        block_num1 = int(np.ceil(len(self.data_list1) / block_size1))
        block_num2 = int(np.ceil(len(self.data_list2) / block_size2))
        self.group1 = [list(x) for x in np.array_split(self.data_list1, block_num1) if len(x) > 0]
        self.group2 = [list(x) for x in np.array_split(self.data_list2, block_num2) if len(x) > 0]
        self.block_index = generate_block_index(self.group1, self.group2, self.pair_list)
        self.data_path1 = Path(data_path1)
        self.data_path2 = Path(data_path2)
        self.data_format1 = data_format1
        self.data_format2 = data_format2
        self.transforms = transforms
        self.device = device

    def __getitem__(self, idx):
        i, j = self.block_index[idx]
        event1, event2 = self.group1[i], self.group2[j]

        index_dict = {}
        data, info, pair_index = [], [], []
        for ii in range(len(event1)):
            for jj in range(len(event2)):
                if (event1[ii], event2[jj]) not in self.pair_list:
                    continue

                if event1[ii] not in index_dict:
                    data_dict = read_data(event1[ii], self.data_path1, self.data_format1)
                    data.append(torch.tensor(data_dict["data"]))
                    info.append(data_dict["info"])
                    index_dict[event1[ii]] = len(data) - 1
                idx1 = index_dict[event1[ii]]

                if event2[jj] not in index_dict:
                    data_dict = read_data(event2[jj], self.data_path2, self.data_format2)
                    data.append(torch.tensor(data_dict["data"]))
                    info.append(data_dict["info"])
                    index_dict[event2[jj]] = len(data) - 1
                idx2 = index_dict[event2[jj]]

                pair_index.append([idx1, idx2])

        if len(data) > 0:
            data = torch.stack(data, dim=0).to(self.device)
            pair_index = torch.tensor(pair_index).to(self.device)

            if self.transforms is not None:
                data = self.transforms(data)
        else:
            data = torch.empty((1, 1, 1), dtype=torch.float32).to(self.device)
            pair_index = torch.tensor([[0, 0]], dtype=torch.int64).to(self.device)

        return {"data": data, "info": info, "pair_index": pair_index}

    def __len__(self):
        return len(self.block_index)


class CCIterableDataset(IterableDataset):
    def __init__(
        self,
        config=None,
        pair_list=None,
        data_list1=None,
        data_list2=None,
        data_path1="./",
        data_path2="./",
        data_format1="h5",
        data_format2="h5",
        block_size1=1,
        block_size2=1,
        dtype=torch.float32,
        device="cpu",
        transforms=None,
        batch_size=32,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super(CCIterableDataset).__init__()

        self.mode = config.mode
        self.config = config
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        self.data_path1 = Path(data_path1)
        self.data_path2 = Path(data_path2)
        self.data_format1 = data_format1
        self.data_format2 = data_format2
        self.transforms = transforms
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.num_batch = None

        self.pair_matrix, self.row_matrix, self.col_matrix, unique_row, unique_col = self.read_pairs(pair_list)
        if self.mode in ["CC", "TM"]:
            self.time_before = {"P": config.time_before_p, "S": config.time_before_s}
        if self.mode == "CC":
            self.symmetric = True
            self.data_format2 = self.data_format1
            self.data_path2 = self.data_path1
            self.data_list1 = pd.read_csv(data_list1).set_index("idx_pick")
            self.data_list2 = self.data_list1

        if self.mode == "TM":
            if data_list1 is not None:
                if data_list1.endswith(".txt"):
                    with open(data_list1, "r") as fp:
                        self.data_list1 = fp.read().splitlines()
                else:
                    self.data_list1 = pd.read_csv(data_list1)
                    if "idx_pick" in self.data_list1.columns:
                        self.data_list1 = self.data_list1.set_index("idx_pick")
            else:
                self.data_list1 = None
            if data_list2 is not None:
                if data_list2.endswith(".txt"):
                    with open(data_list2, "r") as fp:
                        self.data_list2 = fp.read().splitlines()
                else:
                    self.data_list2 = pd.read_csv(data_list2)
                    if "idx_pick" in self.data_list2.columns:
                        self.data_list2 = self.data_list2.set_index("idx_pick")
            else:
                self.data_list2 = None

        if self.mode == "AN":
            self.data_list1 = pd.read_csv(data_list1)
            self.data_list2 = self.data_list1

        block_num1 = int(np.ceil(len(unique_row) / block_size1))
        block_num2 = int(np.ceil(len(unique_col) / block_size2))
        self.group1 = [list(x) for x in np.array_split(unique_row, block_num1) if len(x) > 0]
        self.group2 = [list(x) for x in np.array_split(unique_col, block_num2) if len(x) > 0]

        blocks = list(itertools.product(range(len(self.group1)), range(len(self.group2))))[rank::world_size]
        self.block_index, self.num_batch = self.count_blocks(blocks)

        print(
            f"pair_matrix: {self.pair_matrix.shape}, blocks: {len(self.block_index)}, block_size: {self.block_size1} x {self.block_size2}"
        )

        if (self.data_format1 == "memmap") or (self.data_format2 == "memmap"):
            self.templates = np.memmap(
                config.template_file,
                dtype=np.float32,
                mode="r",
                shape=tuple(config.template_shape),
            )
            self.traveltime = np.memmap(
                config.traveltime_file,
                dtype=np.float32,
                mode="r",
                shape=tuple(config.traveltime_shape),
            )
            self.traveltime_index = np.memmap(
                config.traveltime_index_file,
                dtype=np.int32,
                mode="r",
                shape=tuple(config.traveltime_shape),
            )
            self.traveltime_mask = np.memmap(
                config.traveltime_mask_file,
                dtype=bool,
                mode="r",
                shape=tuple(config.traveltime_shape),
            )
            # config.stations = pd.read_csv(
            #     config.station_index_file, header=None, names=["index", "station_id", "component"], index_col=0
            # )

    def read_pairs(self, pair_list):
        """
        Assume pair_list is a list of pairs of event indices
        """
        pair_list = np.loadtxt(pair_list, delimiter=",", dtype=np.int64)
        # For TEST
        # pair_list = np.array(list(itertools.product(range(6000), range(6000))))
        # pair_list = pair_list[pair_list[:, 0] < pair_list[:, 1]]
        # pair_list = pair_list[pair_list[:, 1] - pair_list[:, 0] < 10]
        unique_row = np.sort(np.unique(pair_list[:, 0]))
        unique_col = np.sort(np.unique(pair_list[:, 1]))
        print(f"Number of pairs: {len(pair_list)}, list1: {len(unique_row)}, list2: {len(unique_col)}")

        rows, cols = pair_list[:, 0], pair_list[:, 1]
        data = [True] * len(pair_list)
        shape = (max(rows) + 1, max(cols) + 1)
        pair_matrix = coo_matrix((data, (rows, cols)), shape=shape, dtype=bool)
        pair_matrix = pair_matrix.tocsr()

        row_index = coo_matrix((rows, (rows, cols)), shape=shape, dtype=int)
        row_index = row_index.tocsr()
        col_index = coo_matrix((cols, (rows, cols)), shape=shape, dtype=int)
        col_index = col_index.tocsr()

        return pair_matrix, row_index, col_index, unique_row, unique_col

    def count_blocks(self, blocks):
        num_batch = 0
        non_empty = []
        for i, j in tqdm(blocks, desc="Counting batch"):
            index1, index2 = self.group1[i], self.group2[j]
            count = (self.pair_matrix[index1, :][:, index2]).sum()
            if count > 0:
                non_empty.append((i, j))
                num_batch += (count - 1) // self.batch_size + 1

        return non_empty, num_batch

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        return iter(self.sample(self.block_index[worker_id::num_workers]))

    def sample(self, block_index):
        for i, j in block_index:
            local_dict = {}
            row_index, col_index = self.group1[i], self.group2[j]
            row_matrix = self.row_matrix[row_index, :][:, col_index].tocoo()
            col_matrix = self.col_matrix[row_index, :][:, col_index].tocoo()

            data1, index1, info1, data2, index2, info2 = [], [], [], [], [], []
            num = 0

            for ii, jj in zip(row_matrix.data, col_matrix.data):

                if self.data_format1 == "memmap":
                    if ii not in local_dict:
                        meta1 = {
                            "data": self.templates[ii],
                            "index": ii,
                            "info": {
                                "idx_eve": self.data_list1.loc[ii, "idx_eve"],
                                "idx_sta": self.data_list1.loc[ii, "idx_sta"],
                                "phase_type": self.data_list1.loc[ii, "phase_type"],
                                "traveltime": self.traveltime[ii],
                                "traveltime_mask": self.traveltime_mask[ii],
                                "traveltime_index": self.traveltime_index[ii],
                                "time_before": self.time_before[self.data_list1.loc[ii, "phase_type"]],
                            },
                        }
                        data = torch.tensor(meta1["data"], dtype=self.dtype).to(self.device)
                        if self.transforms is not None:
                            data = self.transforms(data)
                        meta1["data"] = data
                        local_dict[ii] = meta1
                    else:
                        meta1 = local_dict[ii]
                else:
                    if self.data_list1.loc[ii, "file_name"] not in local_dict:
                        data, info = read_data(
                            self.data_list1.loc[ii, "file_name"],
                            self.data_path1,
                            self.data_format1,
                            mode=self.mode,
                            config=self.config,
                        )
                        info.update({"file_name": self.data_list1.loc[ii, "file_name"]})
                        if "channel_index" in self.data_list1.columns:  # AN
                            info.update({"channel_index": self.data_list1.loc[ii, "channel_index"]})
                        data = torch.tensor(data, dtype=self.dtype).to(self.device)
                        if self.transforms is not None:
                            data = self.transforms(data)
                        meta1 = {
                            "data": data,
                            "index": ii,
                            "info": info,
                        }
                        local_dict[self.data_list1.loc[ii, "file_name"]] = meta1
                    else:
                        meta1 = local_dict[self.data_list1.loc[ii, "file_name"]]

                if self.data_format2 == "memmap":
                    if jj not in local_dict:
                        meta2 = {
                            "data": self.templates[jj],
                            "index": jj,
                            "info": {
                                "idx_eve": self.data_list2.loc[jj, "idx_eve"],
                                "idx_sta": self.data_list2.loc[jj, "idx_sta"],
                                "phase_type": self.data_list2.loc[jj, "phase_type"],
                                "traveltime": self.traveltime[jj],
                                "traveltime_mask": self.traveltime_mask[jj],
                                "traveltime_index": self.traveltime_index[jj],
                                "time_before": self.time_before[self.data_list2.loc[jj, "phase_type"]],
                            },
                        }
                        data = torch.tensor(meta2["data"], dtype=self.dtype).to(self.device)
                        if self.transforms is not None:
                            data = self.transforms(data)
                        meta2["data"] = data
                        local_dict[jj] = meta2
                    else:
                        meta2 = local_dict[jj]
                else:
                    if self.data_list2.loc[jj, "file_name"] not in local_dict:
                        data, info = read_data(
                            self.data_list2.loc[jj, "file_name"],
                            self.data_path2,
                            self.data_format2,
                            mode=self.mode,
                            config=self.config,
                        )
                        info.update({"file_name": self.data_list2.loc[jj, "file_name"]})
                        if "channel_index" in self.data_list2.columns:  # AN
                            info.update({"channel_index": self.data_list2.loc[jj, "channel_index"]})
                        data = torch.tensor(data, dtype=self.dtype).to(self.device)
                        if self.transforms is not None:
                            data = self.transforms(data)
                        meta2 = {
                            "data": data,
                            "index": jj,
                            "info": info,
                        }
                        local_dict[self.data_list2.loc[jj, "file_name"]] = meta2
                    else:
                        meta2 = local_dict[self.data_list2.loc[jj, "file_name"]]

                if self.mode == "AN":
                    data1.append(meta1["data"][:, :, self.data_list1.loc[ii, "channel_index"]])
                    index1.append(self.data_list1.loc[ii, "channel_index"])
                    info1.append({"file_name": self.data_list1.loc[ii, "file_name"]})
                    data2.append(meta2["data"][:, :, self.data_list2.loc[jj, "channel_index"]])
                    index2.append(self.data_list2.loc[jj, "channel_index"])
                    info2.append({"file_name": self.data_list2.loc[jj, "file_name"]})
                else:
                    data1.append(meta1["data"])
                    index1.append(meta1["index"])
                    info1.append(meta1["info"])
                    data2.append(meta2["data"])
                    index2.append(meta2["index"])
                    info2.append(meta2["info"])

                num += 1
                if num == self.batch_size:
                    # if len(np.unique(index1)) == 1:  ## TM mode
                    #     data_batch1 = data1[0].unsqueeze(0)
                    #     data_batch2 = torch.stack(data2)
                    # else:
                    data_batch1 = torch.stack(data1)
                    data_batch2 = torch.stack(data2)
                    # if (
                    #     (self.mode == "TM")
                    #     and (data_batch2.shape[1] != data_batch1.shape[1])
                    #     and (data_batch2.shape[1] % data_batch1.shape[1] == 0)
                    # ):
                    #     data_batch1 = data_batch1.repeat(1, data_batch2.shape[1] // data_batch1.shape[1], 1, 1)

                    info_batch1 = {k: [x[k] for x in info1] for k in info1[0].keys()}
                    info_batch2 = {k: [x[k] for x in info2] for k in info2[0].keys()}
                    if "traveltime" in info_batch1:
                        info_batch1["traveltime"] = np.stack(info_batch1["traveltime"])
                        info_batch1["traveltime_mask"] = np.stack(info_batch1["traveltime_mask"])
                        info_batch1["traveltime_index"] = np.stack(info_batch1["traveltime_index"])
                    if "traveltime" in info_batch2:
                        info_batch2["traveltime"] = np.stack(info_batch2["traveltime"])
                        info_batch2["traveltime_mask"] = np.stack(info_batch2["traveltime_mask"])
                        info_batch2["traveltime_index"] = np.stack(info_batch2["traveltime_index"])
                    if "begin_time" in info_batch1:
                        info_batch1["begin_time"] = np.stack(info_batch1["begin_time"])

                    yield {
                        "data": data_batch1,
                        "index": index1,
                        "info": info_batch1,
                    }, {
                        "data": data_batch2,
                        "index": index2,
                        "info": info_batch2,
                    }

                    num = 0
                    data1, index1, info1, data2, index2, info2 = [], [], [], [], [], []

            ## yield the last batch
            if num > 0:
                data_batch1 = torch.stack(data1)
                data_batch2 = torch.stack(data2)
                # if (
                #     (self.mode == "TM")
                #     and (data_batch2.shape[1] != data_batch1.shape[1])
                #     and (data_batch2.shape[1] % data_batch1.shape[1] == 0)
                # ):
                #     data_batch1 = data_batch1.repeat(1, data_batch2.shape[1] // data_batch1.shape[1], 1, 1)
                info_batch1 = {k: [x[k] for x in info1] for k in info1[0].keys()}
                info_batch2 = {k: [x[k] for x in info2] for k in info2[0].keys()}
                if "traveltime" in info_batch1:
                    info_batch1["traveltime"] = np.stack(info_batch1["traveltime"])
                    info_batch1["traveltime_mask"] = np.stack(info_batch1["traveltime_mask"])
                    info_batch1["traveltime_index"] = np.stack(info_batch1["traveltime_index"])
                if "traveltime" in info_batch2:
                    info_batch2["traveltime"] = np.stack(info_batch2["traveltime"])
                    info_batch2["traveltime_mask"] = np.stack(info_batch2["traveltime_mask"])
                    info_batch2["traveltime_index"] = np.stack(info_batch2["traveltime_index"])
                if "begin_time" in info_batch1:
                    info_batch1["begin_time"] = np.stack(info_batch1["begin_time"])

                yield {
                    "data": data_batch1,
                    "index": index1,
                    "info": info_batch1,
                }, {
                    "data": data_batch2,
                    "index": index2,
                    "info": info_batch2,
                }


def read_data(file_name, data_path, format="h5", mode="CC", config={}):
    if mode == "CC":
        if format == "h5":
            data_list, info_list = read_das_phase_data_h5(
                data_path / file_name, phase="P", event=True, dataset_keys=["shift_index"]
            )
            ## TODO: check with Jiaxuan; why do we need to read a list but return the first one
            data = data_list[0]
            info = info_list[0]

    elif mode == "AN":
        if format == "h5":
            data, info = read_das_continuous_data_h5(data_path / file_name, dataset_keys=[])

    elif mode == "TM":
        if format == "mseed":
            data, info = read_mseed(file_name, config=config)
            # data, info = read_mseed_3c(file_name, config=config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # return {"data": data, "info": info}
    return data, info


def read_mseed(fname, highpass_filter=False, sampling_rate=100, config=None):
    try:
        stream = obspy.Stream()
        for tmp in fname.split("_"):
            with fsspec.open(tmp, "rb") as fs:
                if tmp.endswith(".sac"):
                    meta = obspy.read(fs, format="SAC")
                else:
                    meta = obspy.read(fs, format="MSEED")
                stream += meta
            # stream += obspy.read(tmp)
        stream = stream.merge(fill_value="latest")

    except Exception as e:
        print(f"Error reading {fname}:\n{e}")
        return None

    tmp_stream = obspy.Stream()
    for trace in stream:
        if len(trace.data) < 10:
            continue

        ## interpolate to 100 Hz
        if trace.stats.sampling_rate != sampling_rate:
            logging.warning(f"Resampling {trace.id} from {trace.stats.sampling_rate} to {sampling_rate} Hz")
            try:
                trace = trace.interpolate(sampling_rate, method="linear")
            except Exception as e:
                print(f"Error resampling {trace.id}:\n{e}")

        trace = trace.detrend("demean")

        ## detrend
        # try:
        #     trace = trace.detrend("spline", order=2, dspline=5 * trace.stats.sampling_rate)
        # except:
        #     logging.error(f"Error: spline detrend failed at file {fname}")
        #     trace = trace.detrend("demean")

        ## highpass filtering > 1Hz
        if highpass_filter:
            trace = trace.filter("highpass", freq=1.0)

        tmp_stream.append(trace)

    if len(tmp_stream) == 0:
        return None
    stream = tmp_stream

    begin_time = min([st.stats.starttime for st in stream])
    end_time = max([st.stats.endtime for st in stream])
    stream = stream.trim(begin_time, end_time, pad=True, fill_value=0)

    comp = ["3", "2", "1", "E", "N", "Z"]
    comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}

    station_ids = defaultdict(list)
    for tr in stream:
        station_ids[tr.id[:-1]].append(tr.id[-1])
        if tr.id[-1] not in comp:
            print(f"Unknown component {tr.id[-1]}")

    station_keys = sorted(list(station_ids.keys()))
    nx = len(station_ids)
    nt = max([len(tr.data) for tr in stream])
    data = np.zeros([3, nx, nt], dtype=np.float32)
    for i, sta in enumerate(station_keys):
        for c in station_ids[sta]:
            j = comp2idx[c]

            if len(stream.select(id=sta + c)) == 0:
                print(f"Empty trace: {sta+c} {begin_time}")
                continue

            trace = stream.select(id=sta + c)[0]

            ## accerleration to velocity
            if sta[-1] == "N":
                trace = trace.integrate().filter("highpass", freq=1.0)

            tmp = trace.data.astype("float32")
            data[j, i, : len(tmp)] = tmp[:nt]

    # return data, {
    #     "begin_time": begin_time.datetime,  # .strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    #     "end_time": end_time.datetime,  # .strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    # }
    return data, {
        "begin_time": np.datetime64(begin_time.datetime),
        "end_time": np.datetime64(end_time.datetime),
    }


def read_mseed_3c(fname, response=None, highpass_filter=0.0, sampling_rate=100, config=None):
    try:
        # stream = obspy.read(fname)
        files = fname.rstrip("\n").split(",")

        traces = []
        station_ids = []
        for file in files:
            # with fsspec.open(file, "rb", anon=True) as fp:
            #     stream += obspy.read(fp)
            stream = obspy.read(file)
            trace = stream.merge(fill_value="latest")[0]
            # station_ids.append(trace.id[:-1])
            station_ids.append(trace.id.rstrip("B")[:-1])  # Hardcode for station N.WJMF.EB

            ## interpolate to 100 Hz
            if abs(trace.stats.sampling_rate - sampling_rate) > 0.1:
                logging.warning(f"Resampling {trace.id} from {trace.stats.sampling_rate} to {sampling_rate} Hz")
                try:
                    trace = trace.interpolate(sampling_rate, method="linear")
                except Exception as e:
                    print(f"Error resampling {trace.id}:\n{e}")

            trace = trace.detrend("demean")
            if highpass_filter > 0.0:
                trace = trace.filter("highpass", freq=highpass_filter)

            traces.append(trace)

        station_ids = list(set(station_ids))
        if len(station_ids) > 1:
            print(f"{station_ids = }")
            raise
        assert len(station_ids) == 1, f"Error: {fname} has multiple stations {station_ids}"

        begin_time = min([st.stats.starttime for st in traces])
        end_time = max([st.stats.endtime for st in traces])
        [trace.trim(begin_time, end_time, pad=True, fill_value=0) for trace in traces]

    except Exception as e:
        print(f"Error reading {fname}:\n{e}")
        return {}

    nt = len(traces[0].data)
    data = np.zeros([3, nt], dtype=np.float32)
    for i, trace in enumerate(traces):
        tmp = trace.data.astype("float32")
        data[i, : len(tmp)] = tmp[:nt]

    data = data[:, np.newaxis, :]  # (nc, nt) -> (nc, nx, nt)

    return data, {
        "begin_time": np.datetime64(begin_time.datetime),
        "end_time": np.datetime64(end_time.datetime),
    }


def read_das_continuous_data_h5(fn, dataset_keys=[]):
    with h5py.File(fn, "r") as f:
        if "Data" in f:
            data = f["Data"][:]
        elif "data" in f:
            data = f["data"][:]
        else:
            raise ValueError("Cannot find data in the file")
        info = {}
        for key in dataset_keys:
            info[key] = f[key][:]
    return data, info


# helper reading functions
def read_das_phase_data_h5(fn, phase=None, event=False, dataset_keys=None, attrs_only=False):
    """
    read event phase data from hdf5 file
    Args:
        fn:  hdf5 filename
        phase: phase name list, e.g. ['P', 'S']
        dataset_keys: event phase data attributes, e.g. ['snr', 'traveltime', 'shift_index']
        event: if True, return event dict in info_list[0]
    Returns:
        data_list: list of event phase data
        info_list: list of event phase info
    """
    if isinstance(phase, str):
        phase = [phase]
    data_list = []
    info_list = []
    with h5py.File(fn, "r") as fid:
        g_phases = fid["data"]
        phase_avail = g_phases.keys()
        if phase is None:
            phase = list(phase_avail)
        for phase_name in phase:
            if not phase_name in g_phases.keys():
                raise (f"{fn} does not have phase: {phase_name}")
            g_phase = g_phases[phase_name]
            if attrs_only:
                data = []
            else:
                data = g_phase["data"][:]
            info = {}
            for key in g_phase["data"].attrs.keys():
                info[key] = g_phases[phase_name]["data"].attrs[key]
            if dataset_keys is not None:
                for key in dataset_keys:
                    if key in g_phase.keys():
                        info[key] = g_phase[key][:]
                        for kk in g_phase[key].attrs.keys():
                            info[kk] = g_phase[key].attrs[kk]
            data_list.append(data)
            info_list.append(info)
        if event:
            event_dict = dict((key, fid["data"].attrs[key]) for key in fid["data"].attrs.keys())
            info_list[0]["event"] = event_dict
    return data_list, info_list

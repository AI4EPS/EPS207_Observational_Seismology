import itertools
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import scipy.signal
import torch
import torch.nn.functional as F
import torchaudio
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
        self.pair_list, self.data_list1, self.data_list2 = self.init_pairs(pair_list, data_list1, data_list2, config)

        if self.mode == "CC":
            self.symmetric = True
            self.data_format2 = self.data_format1
            self.data_path2 = self.data_path1

        if self.mode == "AN":
            ## For ambient noise, we split chunks in the sampling function
            self.data_list1 = self.data_list1[rank::world_size]
        else:
            block_num1 = int(np.ceil(len(self.data_list1) / block_size1))
            block_num2 = int(np.ceil(len(self.data_list2) / block_size2))
            self.group1 = [list(x) for x in np.array_split(self.data_list1, block_num1) if len(x) > 0]
            self.group2 = [list(x) for x in np.array_split(self.data_list2, block_num2) if len(x) > 0]
            # self.block_index = generate_block_index(
            #     self.group1,
            #     self.group2,
            #     auto_xcorr=config.auto_xcorr,
            #     pair_list=self.pair_list,
            #     symmetric=self.symmetric,
            #     min_sample_per_block=1,
            # )[rank::world_size]
            # self.block_index = list(itertools.product(range(len(self.group1)), range(len(self.group2))))[
            #     rank::world_size
            # ]
            blocks = list(itertools.product(range(len(self.group1)), range(len(self.group2))))
            self.block_index = self.filt_empty_block(blocks)[rank::world_size]

            print(f"Pairs: {len(self.pair_list)}, Blocks: {len(self.group1)} x {len(self.group2)}")
            print(
                f"data_list1: {len(self.data_list1)}, data_list2: {len(self.data_list2)}, block_size1: {block_size1}, block_size2: {block_size2}"
            )

        if (self.data_format1 == "memmap") or (self.data_format2 == "memmap"):
            self.templates = np.memmap(
                config.template_file,
                dtype=np.float32,
                mode="r",
                shape=tuple(config.template_shape),
            )
            self.traveltime_index = np.memmap(
                config.traveltime_index_file,
                dtype=np.int32,
                mode="r",
                shape=tuple(config.traveltime_shape),
            )
            config.stations = pd.read_csv(
                config.station_index_file, header=None, names=["index", "station_id", "component"], index_col=0
            )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        if self.mode == "AN":
            return iter(self.sample_ambient_noise(self.data_list1[worker_id::num_workers]))
        else:
            return iter(self.sample(self.block_index[worker_id::num_workers]))

    def init_pairs(self, pair_list, data_list1, data_list2, config):
        if data_list1 is not None:
            data_list1 = pd.unique(pd.read_csv(data_list1, header=None)[0]).tolist()

        if data_list2 is not None:
            data_list2 = pd.unique(pd.read_csv(data_list2, header=None)[0]).tolist()
        else:
            data_list2 = data_list1

        if pair_list is not None:
            pair_list, data_list1, data_list2 = read_pair_list(pair_list)

        return pair_list, data_list1, data_list2

    def sample(self, block_index):
        for i, j in block_index:
            local_dict = {}
            event1, event2 = self.group1[i], self.group2[j]
            pairs = generate_pairs(event1, event2, self.config.auto_xcorr, self.symmetric)
            data1, index1, info1, data2, index2, info2 = [], [], [], [], [], []
            num = 0

            for ii, jj in pairs:
                if self.pair_list is not None:
                    if (ii, jj) not in self.pair_list:
                        continue

                if ii not in local_dict:
                    if self.data_format1 == "memmap":
                        meta1 = {
                            "data": self.templates[ii],
                            "index": ii,
                            "info": {"shift_index": self.traveltime_index[ii]},
                        }
                    else:
                        meta1 = read_data(ii, self.data_path1, self.data_format1, mode=self.mode, config=self.config)
                        meta1["index"] = ii
                    data = torch.tensor(meta1["data"], dtype=self.dtype).to(self.device)
                    if self.transforms is not None:
                        data = self.transforms(data)

                    meta1["data"] = data
                    local_dict[ii] = meta1
                else:
                    meta1 = local_dict[ii]

                if jj not in local_dict:
                    if self.data_format2 == "memmap":
                        meta2 = {
                            "data": self.templates[jj],
                            "index": jj,
                            "info": {"shift_index": self.traveltime_index[jj]},
                        }
                    else:
                        meta2 = read_data(jj, self.data_path2, self.data_format2, mode=self.mode, config=self.config)
                        meta2["index"] = jj
                    data = torch.tensor(meta2["data"], dtype=self.dtype).to(self.device)
                    if self.transforms is not None:
                        data = self.transforms(data)

                    meta2["data"] = data
                    local_dict[jj] = meta2
                else:
                    meta2 = local_dict[jj]

                data1.append(meta1["data"])
                index1.append(meta1["index"])
                info1.append(meta1["info"])
                data2.append(meta2["data"])
                index2.append(meta2["index"])
                info2.append(meta2["info"])

                num += 1
                if num == self.batch_size:
                    data_batch1 = torch.stack(data1)
                    data_batch2 = torch.stack(data2)
                    if (
                        (self.mode == "TM")
                        and (data_batch2.shape[1] != data_batch1.shape[1])
                        and (data_batch2.shape[1] % data_batch1.shape[1] == 0)
                    ):
                        data_batch1 = data_batch1.repeat(1, data_batch2.shape[1] // data_batch1.shape[1], 1, 1)

                    info_batch1 = {k: [x[k] for x in info1] for k in info1[0].keys()}
                    info_batch2 = {k: [x[k] for x in info2] for k in info2[0].keys()}
                    if "shift_index" in info_batch1:
                        info_batch1["shift_index"] = torch.tensor(
                            np.stack(info_batch1["shift_index"]), dtype=torch.int64
                        )
                    if "shift_index" in info_batch2:
                        info_batch2["shift_index"] = torch.tensor(
                            np.stack(info_batch2["shift_index"]), dtype=torch.int64
                        )
                    yield {"data": data_batch1, "index": index1, "info": info_batch1}, {
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
                if (
                    (self.mode == "TM")
                    and (data_batch2.shape[1] != data_batch1.shape[1])
                    and (data_batch2.shape[1] % data_batch1.shape[1] == 0)
                ):
                    data_batch1 = data_batch1.repeat(1, data_batch2.shape[1] // data_batch1.shape[1], 1, 1)
                info_batch1 = {k: [x[k] for x in info1] for k in info1[0].keys()}
                info_batch2 = {k: [x[k] for x in info2] for k in info2[0].keys()}
                if "shift_index" in info_batch1:
                    info_batch1["shift_index"] = torch.tensor(np.stack(info_batch1["shift_index"]), dtype=torch.int64)
                if "shift_index" in info_batch2:
                    info_batch2["shift_index"] = torch.tensor(np.stack(info_batch2["shift_index"]), dtype=torch.int64)
                yield {"data": data_batch1, "index": index1, "info": info_batch1}, {
                    "data": data_batch2,
                    "index": index2,
                    "info": info_batch2,
                }

    def sample_ambient_noise(self, data_list):
        for fd in data_list:
            meta = read_data(fd, self.data_path1, self.data_format1, mode=self.mode)  # (nch, nt)
            data = meta["data"].float().unsqueeze(0).unsqueeze(0)  # (1, 1, nx, nt)

            if (self.config.transform_on_file) and (self.transforms is not None):
                data = self.transforms(data)

            # plt.figure()
            # tmp = data[0, 0, :, :].cpu().numpy()
            # vmax = np.std(tmp[:, -1000:]) * 5
            # plt.imshow(tmp[:, -1000:], aspect="auto", vmin=-vmax, vmax=vmax, cmap="seismic")
            # plt.colorbar()
            # plt.savefig(f"cctorch_step1_{fd.split('/')[-1]}.png", dpi=300)
            # raise

            nb, nc, nx, nt = data.shape

            ## cut blocks
            min_channel = self.config.min_channel if self.config.min_channel is not None else 0
            max_channel = self.config.max_channel if self.config.max_channel is not None else nx
            left_channel = self.config.left_channel if self.config.left_channel is not None else -nx
            right_channel = self.config.right_channel if self.config.right_channel is not None else nx

            if self.config.fixed_channels is not None:
                ## only process channels passed by "--fixed-channels" as source
                lists_1 = (
                    self.config.fixed_channels
                    if isinstance(self.config.fixed_channels, list)
                    else [self.fixed_channels]
                )
            else:
                ## using delta_channel to down-sample channels needed for ambient noise
                ## using min_channel and max_channel to selected channels that are within a range
                lists_1 = range(min_channel, max_channel, self.config.delta_channel)
            lists_2 = range(min_channel, max_channel, self.config.delta_channel)
            block_num1 = int(np.ceil(len(lists_1) / self.block_size1))
            block_num2 = int(np.ceil(len(lists_2) / self.block_size2))
            group_1 = [list(x) for x in np.array_split(lists_1, block_num1) if len(x) > 0]
            group_2 = [list(x) for x in np.array_split(lists_2, block_num2) if len(x) > 0]
            block_index = list(itertools.product(range(len(group_1)), range(len(group_2))))

            ## loop each block
            for i, j in block_index:
                block1 = group_1[i]
                block2 = group_2[j]
                index_i = []
                index_j = []
                for ii, jj in itertools.product(block1, block2):
                    if (jj < (ii + left_channel)) or (jj > (ii + right_channel)):
                        continue
                    index_i.append(ii)
                    index_j.append(jj)

                data_i = data[:, :, index_i, :].to(self.device)
                data_j = data[:, :, index_j, :].to(self.device)

                if (self.config.transform_on_batch) and (self.transforms is not None):
                    data_i = self.transforms(data_i)
                    data_j = self.transforms(data_j)

                yield {
                    "data": data_i,
                    "index": [index_i],
                    "info": {},
                }, {"data": data_j, "index": [index_j], "info": {}}

    def __len__(self):
        if self.num_batch is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                num_workers = 1
                worker_id = 0
            else:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id

            if self.mode == "AN":
                num_samples = self.count_sample_ambient_noise(num_workers, worker_id)
            else:
                num_samples = self.count_sample(num_workers, worker_id)
            self.num_batch = num_samples

        return self.num_batch

    def count_sample(self, num_workers, worker_id):
        # if self.pair_list is not None:
        #     num_samples = (
        #         len(self.pair_list) // min(int((self.block_size1 - 1) * (self.block_size2 - 1) / 2), self.batch_size)
        #         + 1
        #     )
        # else:
        #     if self.symmetric:
        #         num_samples = (
        #             len(self.data_list1)
        #             * (len(self.data_list1) - 1)
        #             / 2
        #             // min(self.batch_size, int((self.block_size1 - 1) * (self.block_size2 - 1) / 2))
        #             + 1
        #         )
        #     else:
        #         num_samples = (
        #             len(self.data_list1)
        #             * len(self.data_list2)
        #             // min(self.batch_size, int((self.block_size1 - 1) * (self.block_size2 - 1) / 2))
        #             + 1
        #         )

        if self.mode == "CC":
            num_samples = 0
            for i, j in tqdm(self.block_index[worker_id::num_workers], desc="Counting batches"):
                event1, event2 = self.group1[i], self.group2[j]
                num = 0
                for x, y in itertools.product(event1, event2):
                    if (x < y) and ((x, y) in self.pair_list):
                        num += 1
                num_samples += (num - 1) // self.batch_size + 1
        else:
            num_samples = 0
            for i, j in tqdm(self.block_index[worker_id::num_workers], desc="Counting batches"):
                event1, event2 = self.group1[i], self.group2[j]
                num_samples += (len(event1) * len(event2) - 1) // self.batch_size + 1

        return num_samples

    def filt_empty_block(self, blocks):
        non_empty_blocks = []
        if self.mode == "CC":
            for i, j in tqdm(blocks, desc="Filtering empty blocks"):
                event1, event2 = self.group1[i], self.group2[j]
                for x, y in itertools.product(event1, event2):
                    if (x < y) and ((x, y) in self.pair_list):
                        non_empty_blocks.append((i, j))
                        break
        else:
            non_empty_blocks = blocks
        return non_empty_blocks

    def count_sample_ambient_noise(self, num_workers, worker_id):
        num_samples = 0
        for fd in self.data_list1:
            nx, nt = get_shape_das_continuous_data_h5(self.data_path1 / fd)  # (nch, nt)

            ## cut blocks
            min_channel = self.config.min_channel if self.config.min_channel is not None else 0
            max_channel = self.config.max_channel if self.config.max_channel is not None else nx
            left_channel = self.config.left_channel if self.config.left_channel is not None else -nx
            right_channel = self.config.right_channel if self.config.right_channel is not None else nx

            if self.config.fixed_channels is not None:
                ## only process channels passed by "--fixed-channels" as source
                lists_1 = (
                    self.config.fixed_channels
                    if isinstance(self.config.fixed_channels, list)
                    else [self.fixed_channels]
                )
            else:
                ## using delta_channel to down-sample channels needed for ambient noise
                ## using min_channel and max_channel to selected channels that are within a range
                lists_1 = range(min_channel, max_channel, self.config.delta_channel)
            lists_2 = range(min_channel, max_channel, self.config.delta_channel)
            block_num1 = int(np.ceil(len(lists_1) / self.block_size1))
            block_num2 = int(np.ceil(len(lists_2) / self.block_size2))
            group_1 = [list(x) for x in np.array_split(lists_1, block_num1) if len(x) > 0]
            group_2 = [list(x) for x in np.array_split(lists_2, block_num2) if len(x) > 0]
            block_index = list(itertools.product(range(len(group_1)), range(len(group_2))))

            ## loop each block
            for i, j in block_index:
                num_samples += 1

        return num_samples


def generate_pairs(event1, event2, auto_xcorr=False, symmetric=False):
    event1 = set(event1)
    event2 = set(event2)
    event_inner = event1 & event2
    event_outer1 = event1 - event_inner
    event_outer2 = event2 - event_inner
    event_inner = list(event_inner)
    event_outer1 = list(event_outer1)
    event_outer2 = list(event_outer2)

    if symmetric:
        if auto_xcorr:
            condition = lambda evt1, evt2: evt1 <= evt2
        else:
            condition = lambda evt1, evt2: evt1 < evt2
    else:
        condition = lambda evt1, evt2: True

    pairs = []
    if len(event_inner) > 0:
        # pairs += [(evt1, evt2) for i1, evt1 in enumerate(event_inner) for evt2 in event_inner[i1 + xcor_offset :]]
        pairs += [(evt1, evt2) for evt1 in event_inner for evt2 in event_inner if condition(evt1, evt2)]
        if len(event_outer1) > 0:
            pairs += [(evt1, evt2) for evt1 in event_outer1 for evt2 in event_inner]  # if condition(evt1, evt2)]
        if len(event_outer2) > 0:
            pairs += [(evt1, evt2) for evt1 in event_inner for evt2 in event_outer2]  # if condition(evt1, evt2)]
    if len(event_outer1) > 0 and len(event_outer2) > 0:
        pairs += [(evt1, evt2) for evt1 in event_outer1 for evt2 in event_outer2]  # if condition(evt1, evt2)]

    # print(f"Total number of pairs: {len(pairs)}")
    return pairs


def read_pair_list(file_pair_list):
    # read pair ids from a text file
    # pairs_df = pd.read_csv(file_pair_list, header=None, names=["event1", "event2"])
    # # pair_list = {(x["event1"], x["event2"]) for _, x in pairs_df.iterrows()}
    # pair_list = pairs_df[["event1", "event2"]].values.tolist()
    # data_list1 = sorted(list(set(pairs_df["event1"].tolist())))
    # data_list2 = sorted(list(set(pairs_df["event2"].tolist())))

    pair_list = np.loadtxt(file_pair_list, delimiter=",", dtype=np.int64)
    # pair_list = pair_list[:1_000_000]
    data_list1 = np.unique(pair_list[:, 0]).tolist()
    data_list2 = np.unique(pair_list[:, 1]).tolist()
    pair_list = pair_list.tolist()
    pair_list = set(map(tuple, pair_list))
    return pair_list, data_list1, data_list2


def generate_block_index(group1, group2, pair_list=None, auto_xcorr=False, symmetric=False, min_sample_per_block=1):
    block_index = [(i, j) for i in range(len(group1)) for j in range(len(group2))]
    num_empty_index = []
    for i, j in tqdm(block_index, desc="Generating blocks"):
        num_samples = 0
        event1, event2 = group1[i], group2[j]
        pairs = generate_pairs(event1, event2, auto_xcorr=auto_xcorr, symmetric=symmetric)
        if pair_list is None:
            num_samples = len(pairs)
        else:
            for pair in pairs:
                if pair in pair_list:
                    num_samples += 1
        if num_samples >= min_sample_per_block:
            num_empty_index.append((i, j))
    return num_empty_index


def read_data(file_name, data_path, format="h5", mode="CC", config={}):
    if mode == "CC":
        if format == "h5":
            data_list, info_list = read_das_eventphase_data_h5(
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
            data, info = read_mseed(file_name, config.stations, config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {"data": data, "info": info}


def read_mseed(file_name, stations, config):
    meta = obspy.read(file_name)
    meta.merge(fill_value="latest")
    for tr in meta:
        if tr.stats.sampling_rate != config.sampling_rate:
            tr.resample(config.sampling_rate)
    begin_time = min([tr.stats.starttime for tr in meta])
    end_time = max([tr.stats.endtime for tr in meta])
    meta.detrend("constant")
    meta.trim(begin_time, end_time, pad=True, fill_value=0)
    nt = meta[0].stats.npts
    data = np.zeros([3, len(stations), nt])
    component_mapping = {"1": 2, "2": 1, "3": 0, "E": 0, "N": 1, "Z": 2}
    for i, sta in stations.iterrows():
        if len(sta["component"]) == 3:
            for j, c in enumerate(sta["component"]):
                st = meta.select(id=f"{sta['station_id']}{c}")
                data[j, i, :] = st[0].data
        else:
            j = component_mapping[sta["component"]]
            st = meta.select(id=f"{sta['station_id']}{c}")
            data[j, i, :] = st[0].data

    return data, {
        "begin_time": begin_time.datetime,
        "end_time": end_time.datetime,
        "station_id": stations["station_id"].tolist(),
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


def get_shape_das_continuous_data_h5(file):
    with h5py.File(file, "r") as f:
        if "Data" in f:
            data_shape = f["Data"].shape
        elif "data" in f:
            data_shape = f["data"].shape
        else:
            raise ValueError("Cannot find data in the file")
    return data_shape


# helper reading functions
def read_das_eventphase_data_h5(fn, phase=None, event=False, dataset_keys=None, attrs_only=False):
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

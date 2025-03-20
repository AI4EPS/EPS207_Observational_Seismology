import json
import logging
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
import torchvision.transforms as T
import utils
from cctorch import CCDataset, CCIterableDataset, CCModel
from cctorch.transforms import *
from cctorch.utils import write_ambient_noise
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Correlation using Pytorch", add_help=add_help)
    parser.add_argument(
        "--mode",
        default="CC",
        type=str,
        help="mode for tasks of CC (cross-correlation), TM (template matching), and AN (ambient noise)",
    )
    parser.add_argument("--pair_list", default=None, type=str, help="pair list")
    parser.add_argument("--data_list1", default=None, type=str, help="data list 1")
    parser.add_argument("--data_list2", default=None, type=str, help="data list 1")
    parser.add_argument("--data_path1", default="./", type=str, help="data path")
    parser.add_argument("--data_path2", default="./", type=str, help="data path")
    parser.add_argument("--data_format1", default="h5", type=str, help="data type in {h5, memmap}")
    parser.add_argument("--data_format2", default="h5", type=str, help="data type in {h5, memmap}")
    parser.add_argument("--config", default=None, type=str, help="config file")
    parser.add_argument("--result_path", default="./results", type=str, help="results path")
    parser.add_argument("--dataset_type", default="iterable", type=str, help="data loader type in {map, iterable}")
    parser.add_argument(
        "--block_size1", default=1024, type=int, help="Number of sample for the 1st data pair dimension"
    )
    parser.add_argument(
        "--block_size2", default=1024, type=int, help="Number of sample for the 2nd data pair dimension"
    )
    parser.add_argument("--auto_xcorr", action="store_true", help="do auto-correlation for data list")

    ## common
    parser.add_argument("--dt", default=0.01, type=float, help="time sampling interval")
    parser.add_argument("--sampling_rate", default=100, type=float, help="sampling frequency")
    parser.add_argument("--domain", default="time", type=str, help="domain in {time, freqency, stft}")
    parser.add_argument("--maxlag", default=0.5, type=float, help="maximum time lag during cross-correlation")
    parser.add_argument("--batch_size", default=1024, type=int, help="batch size")
    parser.add_argument("--buffer_size", default=10, type=int, help="buffer size for writing to h5 file")
    parser.add_argument("--workers", default=16, type=int, help="data loading workers")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cpu/cuda/mps, Default: cpu)")
    parser.add_argument("--dataset_cpu", action="store_true", help="Load data to cpu")
    parser.add_argument(
        "--dtype", default="float32", type=str, help="data type (Use float32 or float64, Default: float32)"
    )
    parser.add_argument("--normalize", action="store_true", help="normalized cross-correlation (pearson correlation)")

    ## template matching parameters
    parser.add_argument("--shift_t", action="store_true", help="shift continuous waveform to align with template time")

    ## ambient noise parameters
    parser.add_argument("--min_channel", default=0, type=int, help="minimum channel index")
    parser.add_argument("--max_channel", default=None, type=int, help="maximum channel index")
    parser.add_argument("--delta_channel", default=1, type=int, help="channel interval")
    parser.add_argument("--left_channel", default=None, type=int, help="channel index of the left end from the source")
    parser.add_argument(
        "--right_channel", default=None, type=int, help="channel index of the right end from the source"
    )
    parser.add_argument(
        "--fixed_channels",
        nargs="+",
        default=None,
        type=int,
        help="fixed channel index, if specified, min and max are ignored",
    )
    parser.add_argument("--temporal_gradient", action="store_true", help="use temporal gradient")

    # cross-correlation parameters
    parser.add_argument("--picks_csv", default="cctorch_picks.csv", type=str, help="picks file")
    parser.add_argument("--events_csv", default="cctorch_events.csv", type=str, help="events file")
    parser.add_argument("--stations_csv", default="cctorch_stations.csv", type=str, help="stations file")
    parser.add_argument("--taper", action="store_true", help="taper two data window")
    parser.add_argument("--interp", action="store_true", help="interpolate the data window along time axs")
    parser.add_argument("--scale_factor", default=10, type=int, help="interpolation scale up factor")
    parser.add_argument(
        "--channel_shift", default=0, type=int, help="channel shift of 2nd window for cross-correlation"
    )
    parser.add_argument("--reduce_t", action="store_true", help="reduce the time axis of xcor data")
    parser.add_argument(
        "--reduce_x",
        action="store_true",
        help="reduce the station axis of xcor data: only have effect when reduce_t is true",
    )
    parser.add_argument("--reduce_c", action="store_true", help="reduce the channel axis of xcor data")
    parser.add_argument(
        "--mccc", action="store_true", help="use mccc to reduce time axis: only have effect when reduce_t is true"
    )
    parser.add_argument("--phase_type1", default="P", type=str, help="Phase type of the 1st data window")
    parser.add_argument("--phase_type2", default="S", type=str, help="Phase type of the 2nd data window")
    parser.add_argument(
        "--path_xcor_data", default="", type=str, help="path to save xcor data output: path_{channel_shift}"
    )
    parser.add_argument(
        "--path_xcor_pick", default="", type=str, help="path to save xcor pick output: path_{channel_shift}"
    )
    parser.add_argument(
        "--path_xcor_matrix", default="", type=str, help="path to save xcor matrix output: path_{channel_shift}"
    )
    parser.add_argument("--path_dasinfo", default="", type=str, help="csv file with das channel info")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    return parser


def main(args):
    logging.basicConfig(filename="cctorch.log", level=logging.INFO)
    utils.init_distributed_mode(args)
    rank = utils.get_rank() if args.distributed else 0
    world_size = utils.get_world_size() if args.distributed else 1

    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        print(json.dumps(config, indent=4))
    else:
        config = None

    @dataclass
    class CCConfig:
        ## common
        mode = args.mode
        domain = args.domain
        dtype = torch.float32 if args.dtype == "float32" else torch.float64
        device = args.device
        dt = args.dt
        fs = args.sampling_rate
        if dt != 0.01:
            fs = 1 / dt
        if fs != 100:
            dt = 1 / fs
        maxlag = args.maxlag
        nlag = int(maxlag / dt)
        pre_fft = False  ## if true, do fft in dataloader
        auto_xcorr = args.auto_xcorr

        ## ambinet noise
        spectral_whitening = True
        max_channel = args.max_channel
        min_channel = args.min_channel
        delta_channel = args.delta_channel
        left_channel = args.left_channel
        right_channel = args.right_channel
        fixed_channels = args.fixed_channels
        ### preprocessing for ambient noise
        transform_on_file = True
        transform_on_batch = False
        transform_device = "cpu"
        window_size = 64
        #### bandpass filter
        fmin = 0.1
        fmax = 10
        ftype = "bandpass"
        alpha = 0.05  # tukey window parameter
        order = 2
        #### Decimate
        decimate_factor = 2

        ## cross-correlation
        nma = (20, 0)
        reduce_t = args.reduce_t
        reduce_x = args.reduce_x
        reduce_c = args.reduce_c
        channel_shift = args.channel_shift
        mccc = args.mccc
        use_pair_index = True if args.dataset_type == "map" else False
        # filtering
        min_cc = 0.5
        max_shift = {"P": int(0.5 * fs), "S": int(0.85 * fs)}
        max_obs = 100
        min_obs = 8

        ## template matching
        shift_t = args.shift_t
        reduce_c = args.reduce_c
        normalize = args.normalize

        def __init__(self, config):
            if config is not None:
                for k, v in config.items():
                    setattr(self, k, v)

    ccconfig = CCConfig(config)

    ## Sanity check
    if args.mode == "TM":
        pass

    if rank == 0:
        #     if os.path.exists(args.result_path):
        #         print(f"Remove existing result path: {args.result_path}")
        #         if os.path.exists(args.result_path.rstrip("/") + "_backup"):
        #             shutil.rmtree(args.result_path.rstrip("/") + "_backup")
        #         shutil.move(args.result_path.rstrip("/"), args.result_path.rstrip("/") + "_backup")
        #     os.makedirs(args.result_path)

        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)

    preprocess = []
    if args.mode == "CC":
        # preprocess.append(Filtering(1, 15, 100, 0.1, ccconfig.dtype, args.device))
        if args.taper:
            preprocess.append(T.Lambda(taper_time))
        if args.domain == "time":
            preprocess.append(T.Lambda(normalize))
        elif args.domain == "frequency":
            preprocess.append(T.Lambda(fft_real_normalize))
        else:
            raise ValueError(f"domain {args.domain} not supported")
    elif args.mode == "TM":
        ## TODO add preprocess for template matching
        pass
    elif args.mode == "AN":
        ## TODO add preprocess for ambient noise
        if args.temporal_gradient:  ## convert to strain rate
            preprocess.append(TemporalGradient(ccconfig.fs))
        preprocess.append(TemporalMovingNormalization(int(ccconfig.maxlag * ccconfig.fs)))  # 30s for 25Hz
        preprocess.append(
            Filtering(
                ccconfig.fmin,
                ccconfig.fmax,
                ccconfig.fs,
                ccconfig.ftype,
                ccconfig.alpha,
                ccconfig.dtype,
                ccconfig.transform_device,
            )
        )  # 50Hz # not working on M1
        preprocess.append(Decimation(ccconfig.decimate_factor))  # 25Hz
        preprocess.append(T.Lambda(remove_spatial_median))
        preprocess.append(TemporalMovingNormalization(int(2 * ccconfig.fs // ccconfig.decimate_factor)))  # 2s for 25Hz

    preprocess = T.Compose(preprocess)

    postprocess = []
    if args.mode == "CC":
        ## TODO: add postprocess for cross-correlation
        postprocess.append(DetectPeaksCC(kernel=3, stride=1, topk=2))
    elif args.mode == "TM":
        postprocess.append(
            DetectPeaksTM(vmin=0.6, kernel=301, stride=1, topk=3600 // 5)
        )  # assume 100Hz and 1 hour file
    elif args.mode == "AN":
        ## TODO: add postprocess for ambient noise
        pass
    postprocess = T.Compose(postprocess)

    if args.dataset_type == "map":
        dataset = CCDataset(
            config=ccconfig,
            pair_list=args.pair_list,
            data_list1=args.data_list1,
            data_list2=args.data_list2,
            block_size1=args.block_size1,
            block_size2=args.block_size2,
            data_path1=args.data_path1,
            data_path2=args.data_path2,
            data_format1=args.data_format1,
            data_format2=args.data_format2,
            device="cpu" if args.workers > 0 else args.device,
            transforms=preprocess,
            rank=rank,
            world_size=world_size,
        )
    elif args.dataset_type == "iterable":  ## prefered
        dataset = CCIterableDataset(
            config=ccconfig,
            pair_list=args.pair_list,
            data_list1=args.data_list1,
            data_list2=args.data_list2,
            block_size1=args.block_size1,
            block_size2=args.block_size2,
            data_path1=args.data_path1,
            data_path2=args.data_path2,
            data_format1=args.data_format1,
            data_format2=args.data_format2,
            device="cpu" if args.dataset_cpu else args.device,
            transforms=preprocess,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
        )
    else:
        raise ValueError(f"dataset_type {args.dataset_type} not supported")
    # if len(dataset) < world_size:
    #     raise ValueError(f"dataset size {len(dataset)} is smaller than world size {world_size}")

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        # num_workers=args.workers if args.dataset_type == "map" else 0,
        num_workers=args.workers if args.dataset_cpu else 0,
        sampler=sampler if args.dataset_type == "map" else None,
        pin_memory=False,
        collate_fn=lambda x: x,
    )

    ccmodel = CCModel(
        config=ccconfig,
        batch_size=args.batch_size,  ## only useful for dataset_type == map
        # to_device=False,  ## to_device is done in dataset in default
        to_device=args.dataset_cpu,
        device=args.device,
        transforms=postprocess,
    )
    ccmodel.to(args.device)

    if args.mode in ["CC", "TM"]:
        picks = pd.read_csv(args.picks_csv)
        picks.set_index("idx_pick", inplace=True)
        events = pd.read_csv(args.events_csv)
        stations = pd.read_csv(args.stations_csv)

        result_df = []
        for i, data in enumerate(tqdm(dataloader, position=rank, desc=f"{rank}/{world_size}: computing")):

            if args.mode == "CC":
                idx_eve1 = data[0]["info"]["idx_eve"]
                idx_eve2 = data[1]["info"]["idx_eve"]
            if args.mode == "TM":
                idx_mseed = data[0]["index"]
                idx_eve = data[1]["info"]["idx_eve"]
            idx_sta = data[1]["info"]["idx_sta"]
            phase_type = data[1]["info"]["phase_type"]

            result = ccmodel(data)

            if args.mode == "CC":
                cc_max = result["cc_max"]
                cc_weight = result["cc_weight"]
                cc_shift = result["cc_shift"]  ## shift of cc window
                cc_dt = result["cc_dt"]
                tt_dt = result["tt_dt"] if "tt_dt" in result else 0.0  ## travel time difference
                for ii in range(len(idx_sta)):
                    result_df.append(
                        {
                            "idx_eve1": idx_eve1[ii],
                            "idx_eve2": idx_eve2[ii],
                            "idx_sta": idx_sta[ii],
                            "phase_type": phase_type[ii],
                            "tt_dt": tt_dt[ii].squeeze().item(),
                            "dt": cc_dt[ii].squeeze().item(),
                            "shift": cc_shift[ii].squeeze().item(),
                            "cc": cc_max[ii].squeeze().item(),
                            "weight": cc_weight[ii].squeeze().item(),
                        }
                    )

            if args.mode == "TM":
                origin_time = result["origin_time"][:, 0, 0, :]
                phase_time = result["phase_time"][:, 0, 0, :]
                max_cc = result["max_cc"][:, 0, 0, :]
                for ii in range(len(idx_sta)):
                    for jj in range(len(origin_time[ii])):
                        if max_cc[ii][jj].item() > ccconfig.min_cc:
                            result_df.append(
                                {
                                    "idx_mseed": idx_mseed[ii],
                                    "idx_eve": idx_eve[ii],
                                    "idx_sta": idx_sta[ii],
                                    "phase_type": phase_type[ii],
                                    "phase_time": phase_time[ii][jj].item(),
                                    "origin_time": origin_time[ii][jj].item(),
                                    "cc": max_cc[ii][jj].item(),
                                }
                            )

        if ccconfig.mode == "CC":

            # %%
            if len(result_df) > 0:

                result_df = pd.DataFrame(result_df)
                result_df.to_csv(
                    os.path.join(args.result_path, f"{ccconfig.mode}_{rank:03d}_{world_size:03d}_origin.csv"),
                    index=False,
                )

                ##### More accurate by merging all results
                # if world_size > 1:
                #     dist.barrier()

                # if rank == 0:
                #     result_df = []
                #     for i in tqdm(range(world_size), desc="Merging"):
                #         if os.path.exists(
                #             os.path.join(args.result_path, f"{ccconfig.mode}_{i:03d}_{world_size:03d}_origin.csv")
                #         ):
                #             result_df.append(
                #                 pd.read_csv(
                #                     os.path.join(args.result_path, f"{ccconfig.mode}_{i:03d}_{world_size:03d}_origin.csv")
                #                 )
                #             )
                #     result_df = pd.concat(result_df)

                ### Efficient but less accurate when event pairs split into different files
                # %% filter based on cc values
                result_df = result_df[
                    (result_df["cc"] >= ccconfig.min_cc)
                    & (result_df["shift"].abs() <= result_df["phase_type"].map(ccconfig.max_shift))
                ]

                # %% merge different instrument types of the same stations
                stations["network_station"] = stations["network"] + "." + stations["station"]
                result_df = result_df.merge(stations[["network_station", "idx_sta"]], on="idx_sta", how="left")
                result_df.sort_values("weight", ascending=False, inplace=True)
                result_df = (
                    result_df.groupby(["idx_eve1", "idx_eve2", "network_station", "phase_type"]).first().reset_index()
                )
                result_df.drop(columns=["network_station"], inplace=True)

                # %% filter based on cc observations
                result_df = (
                    result_df.groupby(["idx_eve1", "idx_eve2"])
                    .apply(lambda x: (x.nlargest(ccconfig.max_obs, "weight") if len(x) >= ccconfig.min_obs else None))
                    .reset_index(drop=True)
                )

                # %%
                event_idx_dict = events["event_index"].to_dict()  ##  faster than using .loc
                station_id_dict = stations["station"].to_dict()

                # %%
                result_df.to_csv(
                    os.path.join(args.result_path, f"{ccconfig.mode}_{rank:03d}_{world_size:03d}.csv"), index=False
                )

                # %% write to cc file
                with open(
                    os.path.join(args.result_path, f"{ccconfig.mode}_{rank:03d}_{world_size:03d}_dt.cc"), "w"
                ) as fp:
                    for (i, j), record in tqdm(
                        result_df.groupby(["idx_eve1", "idx_eve2"]), desc=f"{rank}/{world_size} writing"
                    ):
                        event_idx1 = event_idx_dict[i]
                        event_idx2 = event_idx_dict[j]
                        fp.write(f"# {event_idx1} {event_idx2} 0.000\n")
                        for k, record_ in record.iterrows():
                            idx_sta = record_["idx_sta"]
                            station_id = station_id_dict[idx_sta]
                            phase_type = record_["phase_type"]
                            fp.write(f"{station_id} {record_['dt']: .4f} {record_['weight']:.4f} {phase_type}\n")

        # Leave merging to the postprocess script
        # if world_size > 1:
        #     dist.barrier()

        # if rank == 0:
        #     for rank in range(world_size):
        #         if not os.path.exists(
        #             os.path.join(args.result_path, f"{ccconfig.mode}_{rank:03d}_{world_size:03d}.csv")
        #         ):
        #             continue
        #         if rank == 0:
        #             cmd = f"cat {args.result_path}/CC_{rank:03d}_{world_size:03d}.csv > {args.result_path}/CC_{world_size:03d}.csv"
        #         else:
        #             cmd = f"tail -n +2 {args.result_path}/CC_{rank:03d}_{world_size:03d}.csv >> {args.result_path}/CC_{world_size:03d}.csv"
        #         print(cmd)
        #         os.system(cmd)

        # if rank == 0:
        #     cmd = f"cat {args.result_path}/CC_*_{world_size:03d}_dt.cc > {args.result_path}/CC_{world_size:03d}_dt.cc"
        #     print(cmd)
        #     os.system(cmd)

    if ccconfig.mode == "TM":

        if len(result_df) > 0:
            result_df = pd.DataFrame(result_df)
            result_df.to_csv(
                os.path.join(args.result_path, f"{ccconfig.mode}_{rank:03d}_{world_size:03d}.csv"), index=False
            )

        # if world_size > 1:
        #     dist.barrier()

        # if rank == 0:
        result_df = []
        for i in tqdm(range(world_size), desc="Merging"):
            if os.path.exists(os.path.join(args.result_path, f"{ccconfig.mode}_{i:03d}_{world_size:03d}.csv")):
                result_df.append(
                    pd.read_csv(os.path.join(args.result_path, f"{ccconfig.mode}_{i:03d}_{world_size:03d}.csv"))
                )
        if len(result_df) == 0:
            return None

        result_df = pd.concat(result_df)
        print(f"Number of picks: {len(result_df)}")

        result_df["origin_time"] = pd.to_datetime(result_df["origin_time"])
        t0 = result_df["origin_time"].min()
        result_df["timestamp"] = result_df["origin_time"].apply(lambda x: (x - t0).total_seconds())
        # clustering = DBSCAN(eps=2, min_samples=3).fit(result_df[["timestamp"]].values)
        clustering = DBSCAN(eps=0.2, min_samples=3).fit(
            result_df[["timestamp"]].values, sample_weight=result_df["cc"].values
        )
        print(f"Number of events (merge picks): {len(set(clustering.labels_))}")
        result_df["event_index"] = clustering.labels_
        result_df["event_time"] = result_df.groupby("event_index")["timestamp"].transform("median")
        result_df["event_time"] = result_df["event_time"].apply(lambda x: t0 + pd.Timedelta(seconds=x))

        picks_df = result_df.copy()
        events_df = result_df[["event_index", "event_time", "cc"]].copy()
        events_df = events_df.groupby("event_index").agg(
            {"event_time": "first", "cc": "median", "event_index": "count"}
        )
        events_df = events_df.rename(columns={"event_index": "num_picks"})
        events_df["event_index"] = events_df.index
        events_df = events_df[events_df["event_index"] != -1]

        events_df["timestamp"] = (events_df["event_time"] - t0).dt.total_seconds()
        clustering = DBSCAN(eps=2, min_samples=1).fit(events_df[["timestamp"]].values)
        events_df["cluster_index"] = clustering.labels_
        events_df = events_df[events_df["cluster_index"] != -1]
        print(f"Number of events (merge events): {len(events_df['cluster_index'].unique())}")
        events_df = events_df.groupby("cluster_index").agg(
            {"event_time": "first", "cc": "median", "num_picks": "sum", "event_index": lambda x: x.tolist()}
        )
        mapping = {
            idx: i for i, row in events_df.iterrows() for idx in row["event_index"]
        }  # mapping from event_index to cluster_index
        mapping[-1] = -1
        picks_df["event_index"] = picks_df["event_index"].map(mapping)
        events_df["event_index"] = events_df.index

        picks_df.sort_values(by="phase_time", inplace=True)
        events_df.sort_values(by="event_time", inplace=True)
        picks_df.to_csv(os.path.join(args.result_path, f"{ccconfig.mode}_{world_size:03d}_pick.csv"), index=False)
        events_df.to_csv(os.path.join(args.result_path, f"{ccconfig.mode}_{world_size:03d}_event.csv"), index=False)

    if args.mode == "AN":
        MAX_THREADS = 32
        with h5py.File(os.path.join(args.result_path, f"{ccconfig.mode}_{rank:03d}_{world_size:03d}.h5"), "w") as fp:
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = set()
                lock = threading.Lock()
                for data in tqdm(dataloader, position=rank, desc=f"{args.mode}: {rank}/{world_size}"):
                    result = ccmodel(data)
                    thread = executor.submit(write_ambient_noise, [result], fp, ccconfig, lock)
                    futures.add(thread)
                    if len(futures) >= MAX_THREADS:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                executor.shutdown(wait=True)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

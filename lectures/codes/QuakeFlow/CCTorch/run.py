import functools
import json
import logging
import multiprocessing as mp
import os
import pickle
import shelve
import shutil
import threading
from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import utils
from cctorch import CCDataset, CCIterableDataset, CCModel
from cctorch.transforms import *
from cctorch.utils import write_cc_pairs, write_results
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
    parser.add_argument("--workers", default=4, type=int, help="data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu, Default: cuda)")
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
        help="reduce the channel axis of xcor data: only have effect when reduce_t is true",
    )
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
        channel_shift = args.channel_shift
        mccc = args.mccc
        use_pair_index = True if args.dataset_type == "map" else False
        min_cc_score = 0.6
        min_cc_ratio = 0.0  ## ratio is defined as the portion of channels with cc score larger than min_cc_score
        min_cc_diff = 0.0  ## the weight is defined as the difference between largest and second largest cc score

        ## template matching
        shift_t = args.shift_t
        reduce_x = args.reduce_x
        normalize = args.normalize

        def __init__(self, config):
            if config is not None:
                for k, v in config.items():
                    setattr(self, k, v)

    ccconfig = CCConfig(config)

    ## Sanity check
    if args.mode == "TM":
        assert ccconfig.shift_t
        assert ccconfig.nlag == 0

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
        preprocess.append(TemporalMovingNormalization(int(30 * ccconfig.fs)))  # 30s for 25Hz
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
        )  # 50Hz
        preprocess.append(Decimation(ccconfig.decimate_factor))  # 25Hz
        preprocess.append(T.Lambda(remove_spatial_median))
        preprocess.append(TemporalMovingNormalization(int(2 * ccconfig.fs // ccconfig.decimate_factor)))  # 2s for 25Hz

    preprocess = T.Compose(preprocess)

    postprocess = []
    if args.mode == "CC":
        ## add postprocess for cross-correlation
        postprocess.append(DetectPeaks())
        postprocess.append(Reduction())
    elif args.mode == "TM":
        postprocess.append(DetectTM())
    elif args.mode == "AN":
        ## TODO add postprocess for ambient noise
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
            device=args.device,
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
        num_workers=args.workers if args.dataset_type == "map" else 0,
        sampler=sampler if args.dataset_type == "map" else None,
        pin_memory=False,
    )

    ccmodel = CCModel(
        config=ccconfig,
        batch_size=args.batch_size,  ## only useful for dataset_type == map
        to_device=False,  ## to_device is done in dataset in default
        device=args.device,
        transforms=postprocess,
    )
    ccmodel.to(args.device)

    num = 0
    results = []
    threads = []
    fp = h5py.File(os.path.join(args.result_path, f"{ccconfig.mode}_{rank:03d}_{world_size:03d}.h5"), "w")

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # log_freq = max(1, 10240 // args.batch_size) if args.mode == "CC" else 1
    # for data in metric_logger.log_every(dataloader, log_freq, ""):
    for data in tqdm(dataloader, position=rank, desc=f"CC {rank}/{world_size}"):
        result = ccmodel(data)

        thread = threading.Thread(
            target=write_cc_pairs,
            # args=([result], args.result_path, ccconfig, rank, world_size),
            args=([result], fp, ccconfig),
        )
        thread.start()
        threads.append(thread)

        if len(threads) >= 8:
            for thread in threads:
                thread.join()
            threads = []

    for thread in threads:
        thread.join()
    fp.close()

    # write_results([result], args.result_path, ccconfig, rank=rank, world_size=world_size)
    # results.append(result)
    # num += 1

    # topk_index = meta["topk_index"]
    # topk_score = meta["topk_score"]
    # neighbor_score = meta["neighbor_score"]
    # pair_index = meta["pair_index"]
    # for i, pair_index in enumerate(result["pair_index"]):
    #     topk_index = result["topk_index"].cpu().numpy()
    #     topk_score = result["topk_score"].cpu().numpy()
    #     neighbor_score = result["neighbor_score"].cpu().numpy()
    #     cc_quality = result["cc_quality"].cpu().numpy()
    # print(topk_index[i].dtype)
    # print(topk_score[i].dtype)
    # print(neighbor_score[i].dtype)
    # db[pair_index] = {
    #     "topk_index": topk_index[i],
    #     "topk_score": topk_score[i],
    #     "neighbor_score": neighbor_score[i],
    #     "cc_quality": cc_quality[i],
    # }
    # db[str(pair_index)] = {
    #     "topk_index": pickle.dumps(topk_index[i]),
    #     "topk_score": pickle.dumps(topk_score[i]),
    #     "neighbor_score": pickle.dumps(neighbor_score[i]),
    #     "cc_quality": pickle.dumps(cc_quality[i]),
    # }
    #     if num % args.buffer_size == 0:
    #         write_results(results, args.result_path, ccconfig, rank=rank, world_size=world_size)
    #         num = 0
    #         results = []
    # if num > 0:
    #     write_results(results, args.result_path, ccconfig, rank=rank, world_size=world_size)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    args = get_args_parser().parse_args()
    main(args)

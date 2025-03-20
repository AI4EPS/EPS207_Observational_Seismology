from multiprocessing import Manager
from pathlib import Path

import h5py
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from cctorch import CCDataset, CCModel, fft_normalize, write_xcor_to_csv


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Cross-correlation using Pytorch", add_help=add_help)
    parser.add_argument(
        "--pair-list", default="/home/jxli/packages/CCTorch/tests/pair_more.txt", type=str, help="pair list"
    )
    parser.add_argument(
        "--data-path", default="/kuafu/jxli/Data/DASEventData/Ridgecrest_South/temp3", type=str, help="data path"
    )
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--workers", default=16, type=int, help="data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--output-dir", default="tests/ridgecrest", type=str, help="path to save outputs")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    ## TODO: Add more arguments for visualization, data processing, etc
    return parser


def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    manager = Manager()
    shared_dict = manager.dict()
    transform = T.Compose([T.Lambda(fft_normalize)])
    # transform = get_transform()

    pair_list = args.pair_list
    data_path = args.data_path
    dataset = CCDataset(pair_list, data_path, shared_dict, device=args.device, transform=transform)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        # batch_size=args.batch_size,
        batch_size=None,
        # num_workers=args.workers,
        # num_workers=2 * args.batch_size,
        num_workers=0,
        sampler=sampler,
        # collate_fn=None,
        # sampler=None,
        # pin_memory=True,
        pin_memory=False,
    )

    ## TODO: check if DataParallel is better for dataset memory
    ccmodel = CCModel(device=args.device, dt=0.01, maxlag=0.3)
    ccmodel.to(device)
    if args.distributed:
        # ccmodel = torch.nn.parallel.DistributedDataParallel(ccmodel, device_ids=[args.gpu])
        # model_without_ddp = ccmodel.module
        pass
    else:
        ccmodel = nn.DataParallel(ccmodel)

    metric_logger = utils.MetricLogger(delimiter="  ")

    # for x in metric_logger.log_every(dataloader, 10, "CC: "):
    #     tmp.append(x)
    # for x in tqdm(tmp):
    # print(x[0]["data"].shape)
    # print(x[1]["data"].shape)

    for i in tqdm(range(1000)):
        dum = dataset[i]

    tmp1 = []
    for i in tqdm(range(5000), desc="dataset"):
        tmp1.append(dataset[i])

    tmp2 = []
    for i, x in enumerate(tqdm(dataloader, desc="dataloader", total=5000 // args.batch_size)):
        tmp2.append(x)
        if i >= 5000 // args.batch_size:
            break

    for x in tqdm(tmp2, desc="preload memory"):
        # result = ccmodel(x)
        dum = x

    cc_list = pd.read_csv(args.pair_list, header=None, names=["event1", "event2"])
    for i in tqdm(range(5000), desc="shared_dict"):
        event1, event2 = cc_list.iloc[i]
        data1 = shared_dict[event1]
        data2 = shared_dict[event2]
        x = {"event": event1, "data": data1.unsqueeze(0)}, {"event": event2, "data": data2.unsqueeze(0)}
        # result = ccmodel(x)
        dum = x

    shared_dict_cuda = {}
    for i in tqdm(range(5000), desc="to cuda"):
        event1, event2 = cc_list.iloc[i]
        data1 = shared_dict[event1]
        data2 = shared_dict[event2]
        shared_dict_cuda[event1] = data1.cuda()
        shared_dict_cuda[event2] = data2.cuda()

    ccmodel2 = CCModel(device=args.device, to_device=False, batching=False, dt=0.01, maxlag=0.3)
    ccmodel2.to(device)
    for i in tqdm(range(5000), desc="shared_dict cuda"):
        event1, event2 = cc_list.iloc[i]
        data1 = shared_dict_cuda[event1]
        data2 = shared_dict_cuda[event2]
        shared_dict_cuda[event1] = data1.cuda()
        x = {"event": event1, "data": data1}, {"event": event2, "data": data2}
        result = ccmodel2(x)

    for i, x in enumerate(tqdm(dataloader, desc="normal", total=5000 // args.batch_size)):
        result = ccmodel(x)
        if i >= 5000 // args.batch_size:
            break

        # write_xcor_to_csv(result, args.output_dir)
        ## TODO: ADD post-processing
        ## TODO: Add visualization


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args_parser().parse_args()
    main(args)

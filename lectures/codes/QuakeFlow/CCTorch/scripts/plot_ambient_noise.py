# %%
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_args_parser(add_help=True):

    import argparse

    parser = argparse.ArgumentParser(description="Read CCTorch Results", add_help=add_help)
    parser.add_argument("--result_path", type=str, default="results", help="path to results")
    parser.add_argument("--figure_path", type=str, default="figures", help="path to figures")
    parser.add_argument(
        "--fixed_channels",
        nargs="+",
        default=None,
        type=int,
        help="fixed channel index, if specified, min and max are ignored",
    )
    return parser


# %%
if __name__ == "__main__":

    args = get_args_parser().parse_args()

    result_path = Path(args.result_path)
    figure_path = Path(args.figure_path)
    if not figure_path.exists():
        figure_path.mkdir(parents=True)

    h5_files = sorted(result_path.glob("*.h5"))
    print(f"{len(h5_files)} hdf5 files found")

    tmp = []
    for ch1 in args.fixed_channels:
        data = []
        index = []
        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as fp:
                ch2 = sorted([int(x) for x in fp[f"/{ch1}"].keys()])
                for c in ch2:
                    data.append(fp[f"/{ch1}/{c}"]["xcorr"][:])
                    index.append(c)

        index = np.array(index)
        data = np.stack(data)
        sorted_idx = np.argsort(index)
        index = index[sorted_idx]
        data = data[sorted_idx]

        np.savez(figure_path / f"result_{ch1}.npz", data=data, index=index)
        plt.figure()
        vmax = np.std(data)
        plt.imshow(data, vmin=-vmax, vmax=vmax, aspect="auto", cmap="RdBu")
        plt.colorbar()
        plt.savefig(figure_path / f"result_{ch1}.png", dpi=300, bbox_inches="tight")

# %%
import argparse
import os
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

data_path = "results"
result_path = f"results/adloc"
if not os.path.exists(f"{result_path}"):
    os.makedirs(f"{result_path}")

# %%
epochs = 1
bootstrap = 10
batch = 100
double_difference = False
base_cmd = f"../run.py --config {data_path}/config.json --stations {data_path}/stations.json --events {data_path}/events.csv --picks {data_path}/picks.csv --result_path {result_path} --batch_size {batch} --bootstrap {bootstrap}"
if double_difference:
    base_cmd += " --double_difference"
os.system(f"python {base_cmd} --device=cpu --epochs={epochs}")

# %%
events_true = pd.read_csv(f"{data_path}/events.csv")
events_invert = pd.read_csv(f"{result_path}/adloc_events.csv")
stations = pd.read_json(f"{data_path}/stations.json", orient="index")

# %%
plt.figure()
plt.scatter(stations["longitude"], stations["latitude"], marker="^", s=10)
plt.scatter(events_true["longitude"], events_true["latitude"], s=1, label="true")
plt.scatter(events_invert["longitude"], events_invert["latitude"], s=1, label="invert")
plt.legend()
plt.savefig(f"{result_path}/events_xy.png", dpi=300)

plt.figure()
plt.scatter(stations["longitude"], stations["depth_km"], marker="^", s=10)
plt.scatter(events_true["longitude"], events_true["depth_km"], s=1, label="true")
plt.scatter(events_invert["longitude"], events_invert["depth_km"], s=1, label="invert")
plt.gca().invert_yaxis()
plt.legend()
plt.savefig(f"{result_path}/events_xz.png", dpi=300)

plt.figure()
plt.scatter(stations["latitude"], stations["depth_km"], marker="^", s=10)
plt.scatter(events_true["latitude"], events_true["depth_km"], s=1, label="true")
plt.scatter(events_invert["latitude"], events_invert["depth_km"], s=1, label="invert")
plt.gca().invert_yaxis()
plt.legend()
plt.savefig(f"{result_path}/events_yz.png", dpi=300)

# %%
if os.path.exists(f"{result_path}/adloc_events_bootstrap.csv"):
    events_invert_bootstrap = pd.read_csv(f"{result_path}/adloc_events_bootstrap.csv")
    plt.figure()
    plt.scatter(stations["x_km"], stations["y_km"], marker="^", s=10)
    plt.scatter(events_true["x_km"], events_true["y_km"], s=1, label="true")
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.scatter(events_invert["x_km"], events_invert["y_km"], s=1, label="invert")
    plt.scatter(events_invert_bootstrap["x_km"], events_invert_bootstrap["y_km"], s=1, label="bootstrap")
    plt.errorbar(
        events_invert_bootstrap["x_km"],
        events_invert_bootstrap["y_km"],
        events_invert_bootstrap["std_x_km"] / 2,
        events_invert_bootstrap["std_y_km"] / 2,
        linestyle="",
        marker=".",
        alpha=0.5,
    )
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.savefig(f"{result_path}/events_xy_bootstrap.png", dpi=300)

    plt.figure()
    plt.scatter(stations["x_km"], stations["z_km"], marker="^", s=10)
    plt.scatter(events_true["x_km"], events_true["z_km"], s=1, label="true")
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.scatter(events_invert["x_km"], events_invert["z_km"], s=1, label="invert")
    plt.scatter(events_invert_bootstrap["x_km"], events_invert_bootstrap["z_km"], s=1, label="bootstrap")
    plt.errorbar(
        events_invert_bootstrap["x_km"],
        events_invert_bootstrap["z_km"],
        events_invert_bootstrap["std_x_km"] / 2,
        events_invert_bootstrap["std_z_km"] / 2,
        linestyle="",
        marker=".",
        alpha=0.5,
    )
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(f"{result_path}/events_xz_bootstrap.png", dpi=300)

    plt.figure()
    plt.scatter(stations["y_km"], stations["z_km"], marker="^", s=10)
    plt.scatter(events_true["y_km"], events_true["z_km"], s=1, label="true")
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.scatter(events_invert["y_km"], events_invert["z_km"], s=1, label="invert")
    plt.scatter(events_invert_bootstrap["y_km"], events_invert_bootstrap["z_km"], s=1, label="bootstrap")
    plt.errorbar(
        events_invert_bootstrap["y_km"],
        events_invert_bootstrap["z_km"],
        events_invert_bootstrap["std_y_km"] / 2,
        events_invert_bootstrap["std_z_km"] / 2,
        linestyle="",
        marker=".",
        alpha=0.5,
    )
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(f"{result_path}/events_yz_bootstrap.png", dpi=300)

# %%

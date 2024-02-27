# %%
import numpy as np
import json
import pandas as pd
import obspy
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datetime import datetime, timezone
import fsspec

# %%
root_path = "local"
region = "demo"
protocal = "gs"
bucket = "quakeflow_share"
fs = fsspec.filesystem(protocol=protocal)
fs.get(
    f"{bucket}/{region}/waveforms/",
    f"{root_path}/{region}/waveforms/",
    recursive=True,
)
fs.get(
    f"{bucket}/{region}/cctorch/",
    f"{root_path}/{region}/cctorch/",
    recursive=True,
)

# %%
with open(f"{root_path}/{region}/cctorch/config.json", "r") as template:
    config = json.load(template)

print(json.dumps(config, indent=4, sort_keys=True))

# %% Load template
template = np.memmap(
    f"{root_path}/{region}/cctorch/template.dat", dtype=np.float32, mode="r", shape=tuple(config["template_shape"])
)
traveltime_type = np.memmap(
    f"{root_path}/{region}/cctorch/traveltime_type.dat",
    dtype=np.int32,
    mode="r",
    shape=tuple(config["traveltime_shape"]),
)
traveltime_index = np.memmap(
    f"{root_path}/{region}/cctorch/traveltime_index.dat",
    dtype=np.int32,
    mode="r",
    shape=tuple(config["traveltime_shape"]),
)
arrivaltime_index = np.memmap(
    f"{root_path}/{region}/cctorch/arrivaltime_index.dat",
    dtype=np.int64,
    mode="r",
    shape=tuple(config["traveltime_shape"]),
)

# %% Load station/event index
station_index = pd.read_csv(
    f"{root_path}/{region}/cctorch/station_index.txt", header=None, names=["index", "station_id", "component"]
)
event_index = pd.read_csv(f"{root_path}/{region}/cctorch/event_index.txt", header=None, names=["index", "event_index"])

# %% Plot template
ieve = 1
ich = 0  # 0-2: E, N, Z for P wave; 3-5: E, N, Z for S wave
normalize = lambda x: (x - np.mean(x)) / (np.std(x) + np.finfo(np.float32).eps)
plt.figure(figsize=(10, 10))
for ista in range(template.shape[2]):
    plt.plot(normalize(template[ieve, ich, ista, :]) / 3 + ista, "b", linewidth=0.5)
shift = ista + 1
for ista in range(template.shape[2]):
    plt.plot(normalize(template[ieve, ich + 3, ista, :]) / 3 + ista + shift, "r", linewidth=0.5)
plt.show()

# %% Load continuous waveform
year = "2019"
jday = "185"
hour = "17"

meta = obspy.read(f"{root_path}/{region}/waveforms/{year}-{jday}/{hour}/*.mseed")
meta.merge(fill_value="latest")
for tr in meta:
    if tr.stats.sampling_rate != config["sampling_rate"]:
        if tr.stats.sampling_rate % config["sampling_rate"] == 0:
            tr.decimate(int(tr.stats.sampling_rate / config["sampling_rate"]))
        else:
            tr.resample(config["sampling_rate"])
begin_time = min([tr.stats.starttime for tr in meta])
end_time = max([tr.stats.endtime for tr in meta])
meta.detrend("constant")
meta.trim(begin_time, end_time, pad=True, fill_value=0)


nt = meta[0].stats.npts
data = np.zeros([3, len(station_index), nt])
for i, sta in station_index.iterrows():
    if len(sta["component"]) == 3:
        for j, c in enumerate(sta["component"]):
            tr = meta.select(id=f"{sta['station_id']}{c}")
            data[j, i, :] = tr[0].data
    else:
        j = config["component_mapping"][sta["component"]]
        tr = meta.select(id=f"{sta['station_id']}{c}")
        data[j, i, :] = tr[0].data


# %% Utility functions
def detect_peaks(scores, ratio=10, maxpool_kernel=101, median_kernel=1000, K=100):
    # kernel = 101
    # median_window = 1000
    # stride = 1
    # scores = xcorr
    # ratio = 10
    # K = 100
    maxpool_stride = 1
    median_stride = median_kernel // 2
    nb, nc, nx, nt = scores.shape
    smax = F.max_pool2d(scores, (1, maxpool_kernel), stride=(1, maxpool_stride), padding=(0, maxpool_kernel // 2))[
        :, :, :, :nt
    ]
    scores_ = F.pad(scores, (0, median_kernel, 0, 0), mode="reflect", value=0)
    ## MAD = median(|x_i - median(x)|)
    unfolded = scores_.unfold(-1, median_kernel, median_stride)
    mad = (unfolded - unfolded.median(dim=-1, keepdim=True).values).abs().median(dim=-1).values
    mad = F.interpolate(mad, scale_factor=(1, median_stride), mode="bilinear", align_corners=False)[:, :, :, :nt]
    keep = (smax == scores).float() * (scores > ratio * mad).float()
    scores = scores * keep

    if K == 0:
        K = max(round(nt * 10.0 / 3000.0), 3)
    topk_scores, topk_inds = torch.topk(scores, K, dim=-1, sorted=True)
    topk_scores = topk_scores.flatten()
    topk_inds = topk_inds.flatten()
    topk_inds = topk_inds[topk_scores > 0]
    topk_scores = topk_scores[topk_scores > 0]

    return topk_scores, topk_inds, mad


# %% Run QTM
input = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
input = input.repeat(1, 2, 1, 1)  # repeat to match both P and S wave templates
weight = torch.tensor(template[ieve], dtype=torch.float32).unsqueeze(0)
shift_index = torch.tensor(traveltime_index[ieve], dtype=torch.int64).unsqueeze(0)
shift_index = shift_index.repeat_interleave(3, dim=1)  # repeat to match three channels for P and S wave templates

data1 = input[:, :, :, :]
data2 = weight[:, :, :, :]
shift_index = shift_index[:, :, :]

nb1, nc1, nx1, nt1 = data1.shape
nb2, nc2, nx2, nt2 = data2.shape

## shift continuous waveform to align with template time
nt_index = torch.arange(nt1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
adjusted_index = (nt_index + shift_index.unsqueeze(-1)) % nt1
data1 = data1.gather(-1, adjusted_index)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    data1.to(device)
    data2.to(device)

data1 = data1.view(1, nb1 * nc1 * nx1, nt1)
data2 = data2.view(nb2 * nc2 * nx2, 1, nt2)

## calculate pearson correlation coefficient: xcorr = (x - mu_x) * y / (sigma_x * sigma_y)
eps = torch.finfo(data1.dtype).eps
data2 = (data2 - torch.mean(data2, dim=-1, keepdim=True)) / (torch.std(data2, dim=-1, keepdim=True) + eps)
data1_ = F.pad(data1, (nt2 // 2, nt2 - 1 - nt2 // 2), mode="reflect")
local_mean = F.avg_pool1d(data1_, nt2, stride=1)
local_std = F.lp_pool1d(data1 - local_mean, norm_type=2, kernel_size=nt2, stride=1) * np.sqrt(nt2)

xcorr = F.conv1d(data1, data2, stride=1, groups=nb1 * nc1 * nx1)
xcorr = xcorr / (local_std + eps)

xcorr = xcorr.view(nb1, nc1, nx1, -1)

## sum over all channels and stations
xcorr_1d = torch.sum(xcorr, dim=(-3, -2), keepdim=True)

event_score, event_index, mad = detect_peaks(xcorr_1d, ratio=10, maxpool_kernel=101, median_kernel=6000, K=100)
event_time = [begin_time + x / config["sampling_rate"] for x in event_index.numpy()]

# %% Visualize CC results
plt.figure(figsize=(20, 5))
plt.plot(xcorr_1d[0, 0, 0, :].numpy())
plt.plot(10 * mad[0, 0, 0, :].numpy())
plt.show()

# %% Visualize event results
data1 = data1.view(nb1, nc1, nx1, nt1)
ich = 0
for i in event_index:
    plt.figure(figsize=(10, 10))
    for j in range(input.shape[2]):
        plt.plot(normalize(data1[0, ich, j, i - 500 : i + 1000].numpy()) / 3 + j, "b", linewidth=0.5)
    shift = j + 1
    for j in range(input.shape[2]):
        plt.plot(normalize(data1[0, ich + 3, j, i - 500 : i + 1000].numpy()) / 3 + j + shift, "r", linewidth=0.5)
    plt.show()

# %%

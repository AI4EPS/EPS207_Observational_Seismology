# %%
import dataclasses
import json
import math
import multiprocessing as mp
import os
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from multiprocessing import shared_memory
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

# import gamma
import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import scipy
import scipy.signal
import torch
from tqdm.auto import tqdm


# %%
def write_results(results, result_path, ccconfig, rank=0, world_size=1):
    if ccconfig.mode == "CC":
        write_cc_pairs(results, result_path, ccconfig, rank=rank, world_size=world_size)
    elif ccconfig.mode == "TM":
        write_tm_events(results, result_path, ccconfig, rank=rank, world_size=world_size)
    elif ccconfig.mode == "AN":
        write_ambient_noise(results, result_path, ccconfig, rank=rank, world_size=world_size)
    else:
        raise ValueError(f"{ccconfig.mode} not supported")


def write_tm_events(results, result_path, ccconfig, rank=0, world_size=1):
    if not isinstance(result_path, Path):
        result_path = Path(result_path)

    events = []
    for meta in results:
        for event_time, event_score in zip(meta["event_time"], meta["event_score"]):
            events.append({"event_time": event_time.isoformat(), "event_score": round(event_score, 3)})
    if len(events) > 0:
        events = pd.DataFrame(events)
        events = events.sort_values(by="event_time", ascending=True)
        events.to_csv(result_path / f"cctorch_events_{rank:03d}_{world_size:03d}.csv", index=False)


def write_tm_detects(results, fp, ccconfig, lock=nullcontext(), plot_figure=False):
    """
    Write cross-correlation results to disk.
    Parameters
    ----------
    results : list of dict
        List of results from cross-correlation.
        e.g. [{
            "topk_index": topk_index,
            "topk_score": topk_score,
            "neighbor_score": neighbor_score,
            "pair_index": pair_index}]
    """

    for meta in results:
        topk_index = meta["topk_index"].numpy()
        topk_score = meta["topk_score"].numpy()
        pair_index = meta["pair_index"]

        nb, nch, nx, nk = topk_index.shape

        for i in range(nb):
            if topk_score[i].max() < ccconfig.min_cc_score:
                continue

            pair_id = pair_index[i]
            id1, id2 = pair_id

            if f"{id1}/{id2}" not in fp:
                with lock:
                    gp = fp.create_group(f"{id1}/{id2}")
            else:
                gp = fp[f"{id1}/{id2}"]

            with lock:
                idx = np.where(topk_score[i] >= ccconfig.min_cc_score)
                gp.create_dataset(f"cc_index", data=topk_index[i][idx])
                gp.create_dataset(f"cc_score", data=topk_score[i][idx])

    return 0


def write_cc_pairs(results, fp, ccconfig, lock=nullcontext(), plot_figure=False):
    """
    Write cross-correlation results to disk.
    Parameters
    ----------
    results : list of dict
        List of results from cross-correlation.
        e.g. [{
            "topk_index": topk_index,
            "topk_score": topk_score,
            "neighbor_score": neighbor_score,
            "pair_index": pair_index}]
    """

    for meta in results:
        topk_index = meta["topk_index"].numpy()
        topk_score = meta["topk_score"].numpy()
        neighbor_score = meta["neighbor_score"].numpy()
        pair_index = meta["pair_index"]
        cc_diff = topk_score[:, :, :, 0] - topk_score[:, :, :, 1]
        if "cc_sum" in meta:
            cc_sum = meta["cc_sum"].numpy()
        else:
            cc_sum = None

        nb, nch, nx, nk = topk_index.shape

        for i in range(nb):
            if (topk_score[i].max() < ccconfig.min_cc_score) or (cc_diff[i].max() < ccconfig.min_cc_diff):
                continue

            pair_id = pair_index[i]
            id1, id2 = pair_id
            if int(id1) > int(id2):
                id1, id2 = id2, id1
                topk_index = -topk_index

            if f"{id1}/{id2}" not in fp:
                with lock:
                    gp = fp.create_group(f"{id1}/{id2}")
            else:
                gp = fp[f"{id1}/{id2}"]

            with lock:
                gp.create_dataset(f"cc_index", data=topk_index[i])
                gp.create_dataset(f"cc_score", data=topk_score[i])
                gp.create_dataset(f"cc_diff", data=cc_diff[i])
                gp.create_dataset(f"neighbor_score", data=neighbor_score[i])
                if cc_sum is not None:
                    gp.create_dataset(f"cc_sum", data=cc_sum[i])

            # if id2 != id1:
            #     fp[f"{id2}/{id1}"] = h5py.SoftLink(f"/{id1}/{id2}")

            if plot_figure:
                for j in range(nch):
                    fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(10, 20), sharey=True)
                    ax[0, 0].imshow(meta["xcorr"][i, j, :, :], cmap="seismic", vmax=1, vmin=-1, aspect="auto")
                    for k in range(nx):
                        ax[0, 1].plot(
                            meta["data1"][i, j, k, :] / np.max(np.abs(meta["data1"][i, j, k, :])) + k,
                            linewidth=1,
                            color="k",
                        )
                        ax[0, 2].plot(
                            meta["data2"][i, j, k, :] / np.max(np.abs(meta["data2"][i, j, k, :])) + k,
                            linewidth=1,
                            color="k",
                        )

                    try:
                        fig.savefig(f"debug/test_{pair_id[0]}_{pair_id[1]}_{j}.png", dpi=300)
                    except:
                        os.mkdir("debug")
                        fig.savefig(f"debug/test_{pair_id[0]}_{pair_id[1]}_{j}.png", dpi=300)
                    print(f"debug/test_{pair_id[0]}_{pair_id[1]}_{j}.png")
                    plt.close(fig)

    return 0

    # with h5py.File(result_path / f"{ccconfig.mode}_{rank:03d}_{world_size:03d}.h5", "a") as fp:
    #     for meta in results:
    #         topk_index = meta["topk_index"]
    #         topk_score = meta["topk_score"]
    #         neighbor_score = meta["neighbor_score"]
    #         pair_index = meta["pair_index"]

    #         nb, nch, nx, nk = topk_index.shape

    #         for i in range(nb):
    #             cc_score = topk_score[i, :, :, 0]
    #             cc_diff = topk_score[i, :, :, 0] - topk_score[i, :, :, 1]

    #             if (
    #                 (cc_score.max() >= min_cc_score)
    #                 and (cc_diff.max() >= min_cc_diff)
    #                 and (torch.sum((cc_score > min_cc_score) & (cc_diff > min_cc_diff)) >= nch * nx * min_cc_ratio)
    #             ):
    #                 pair_id = pair_index[i]
    #                 id1, id2 = pair_id
    #                 if int(id1) > int(id2):
    #                     id1, id2 = id2, id1
    #                     topk_index = -topk_index

    #                 if f"{id1}/{id2}" not in fp:
    #                     gp = fp.create_group(f"{id1}/{id2}")
    #                 else:
    #                     gp = fp[f"{id1}/{id2}"]

    #                 if f"cc_index" in gp:
    #                     del gp["cc_index"]
    #                 gp.create_dataset(f"cc_index", data=topk_index[i].cpu())
    #                 if f"cc_score" in gp:
    #                     del gp["cc_score"]
    #                 gp.create_dataset(f"cc_score", data=topk_score[i].cpu())
    #                 if f"cc_diff" in gp:
    #                     del gp["cc_diff"]
    #                 gp.create_dataset(f"cc_diff", data=cc_diff.cpu())
    #                 if f"neighbor_score" in gp:
    #                     del gp["neighbor_score"]
    #                 gp.create_dataset(f"neighbor_score", data=neighbor_score[i].cpu())

    #                 if id2 != id1:
    #                     if f"{id2}/{id1}" not in fp:
    #                         # fp[f"{id2}/{id1}"] = h5py.SoftLink(f"/{id1}/{id2}")
    #                         gp = fp.create_group(f"{id2}/{id1}")
    #                     else:
    #                         gp = fp[f"{id2}/{id1}"]

    #                     if f"cc_index" in gp:
    #                         del gp["cc_index"]
    #                     gp.create_dataset(f"cc_index", data=-topk_index[i].cpu())
    #                     if f"neighbor_score" in gp:
    #                         del gp["neighbor_score"]
    #                     gp.create_dataset(f"neighbor_score", data=neighbor_score[i].cpu().flip(-1))
    #                     if f"cc_score" in gp:
    #                         del gp["cc_score"]
    #                     gp["cc_score"] = fp[f"{id1}/{id2}/cc_score"]
    #                     if f"cc_diff" in gp:
    #                         del gp["cc_diff"]
    #                     gp["cc_diff"] = fp[f"{id1}/{id2}/cc_diff"]

    #                 if plot_figure:
    #                     for j in range(nch):
    #                         fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(10, 20), sharey=True)
    #                         ax[0, 0].imshow(
    #                             meta["xcorr"][i, j, :, :].cpu().numpy(), cmap="seismic", vmax=1, vmin=-1, aspect="auto"
    #                         )
    #                         for k in range(nx):
    #                             ax[0, 1].plot(
    #                                 meta["data1"][i, j, k, :].cpu().numpy()
    #                                 / np.max(np.abs(meta["data1"][i, j, k, :].cpu().numpy()))
    #                                 + k,
    #                                 linewidth=1,
    #                                 color="k",
    #                             )
    #                             ax[0, 2].plot(
    #                                 meta["data2"][i, j, k, :].cpu().numpy()
    #                                 / np.max(np.abs(meta["data2"][i, j, k, :].cpu().numpy()))
    #                                 + k,
    #                                 linewidth=1,
    #                                 color="k",
    #                             )

    #                         try:
    #                             fig.savefig(f"debug/test_{pair_id[0]}_{pair_id[1]}_{j}.png", dpi=300)
    #                         except:
    #                             os.mkdir("debug")
    #                             fig.savefig(f"debug/test_{pair_id[0]}_{pair_id[1]}_{j}.png", dpi=300)
    #                         print(f"debug/test_{pair_id[0]}_{pair_id[1]}_{j}.png")
    #                         plt.close(fig)


def write_ambient_noise(results, fp, ccconfig, lock=nullcontext(), plot_figure=False):
    """
    Write ambient noise results to disk.
    """
    for meta in results:
        xcorr = meta["xcorr"].cpu().numpy()
        nb, nch, nx, nt = xcorr.shape
        for i in range(nb):
            data = np.squeeze(np.nan_to_num(xcorr[i, :, :, :]))
            id1, id2 = meta["pair_index"][i]

            if f"{id1}/{id2}" not in fp:
                gp = fp.create_group(f"{id1}/{id2}")
                ds = gp.create_dataset("xcorr", data=data)
                ds.attrs["count"] = 1
            else:
                gp = fp[f"{id1}/{id2}"]
                ds = gp["xcorr"]
                count = ds.attrs["count"]
                ds[:] = count / (count + 1) * ds[:] + data / (count + 1)
                ds.attrs["count"] = count + 1


def write_xcor_data_to_h5(result, path_result, phase1="P", phase2="P"):
    """
    Write full xcor to hdf5 file. No reduce in time and channel axis
    """
    nbatch = result["xcor"].shape[0]
    dt = result["dt"]
    xcor = result["xcor"].cpu().numpy()
    channel_shift = int(result["channel_shift"])
    for ibatch in range(nbatch):
        id1 = int(result["id1"][ibatch].cpu().numpy())
        id2 = int(result["id2"][ibatch].cpu().numpy())
        fn = f"{path_result}/{id1}_{id2}.h5"
        xcor_dict = {
            "event_id1": id1,
            "event_id2": id2,
            "dt": dt,
            "channel_shift": channel_shift,
            "phase1": phase1,
            "phase2": phase2,
        }
        write_h5(fn, "data", xcor[ibatch, :, :], xcor_dict)


def write_xcor_mccc_pick_to_csv(result, x, path_result, dt=0.01, channel_index=None):
    """ """
    event_id1 = x[0]["event"].cpu().numpy()
    event_id2 = x[1]["event"].cpu().numpy()
    cc_main_lobe = result["cc_main"].cpu().numpy()
    cc_side_lobe = result["cc_side"].cpu().numpy()
    cc_dt = result["cc_dt"].cpu().numpy()
    fn_save = f"{path_result}/{event_id1}_{event_id2}.csv"
    phase_time1 = datetime.fromisoformat(x[0]["event_time"]) + timedelta(seconds=dt) * x[0]["shift_index"].numpy()
    phase_time2 = datetime.fromisoformat(x[1]["event_time"]) + timedelta(seconds=dt) * x[1]["shift_index"].numpy()
    if channel_index is None:
        channel_index = np.arange(len(cc_dt))
    pd.DataFrame(
        {
            "channel_index": channel_index,
            "event_id1": event_id1,
            "event_id2": event_id2,
            "phase_time1": [t.isoformat() for t in phase_time1],
            "phase_time2": [t.isoformat() for t in phase_time2],
            "cc_dt": cc_dt,
            "cc_main": cc_main_lobe,
            "cc_side": cc_side_lobe,
        }
    ).to_csv(fn_save, index=False)


def write_xcor_to_csv(result, path_result):
    """
    Write xcor to csv file. Reduce in time axis
    """
    nbatch = result["cc"].shape[0]
    cc = result["cc"].cpu().numpy()
    cc_dt = result["cc_dt"].cpu().numpy()
    for ibatch in range(nbatch):
        id1 = int(result["id1"][ibatch].cpu().numpy())
        id2 = int(result["id2"][ibatch].cpu().numpy())
        fn = f"{path_result}/{id1}_{id2}.csv"
        pd.DataFrame({"cc": cc[ibatch, :], "dt": cc_dt[ibatch, :]}).to_csv(fn, index=False)


def write_xcor_to_ccmat(result, ccmat, id_row, id_col):
    """
    Write single xcor value to a matrix. Reduce in both time and channel axis
    """
    nbatch = result["xcor"].shape[0]
    for ibatch in range(nbatch):
        """"""
        id1 = result["id1"][ibatch]
        id2 = result["id2"][ibatch]
        irow = torch.where(id_row == id1)
        icol = torch.where(id_col == id2)
        ccmat[irow, icol] = result["cc_mean"]


def reduce_ccmat(file_cc_matrix, channel_shift, nrank, clean=True):
    """
    reduce the cc matrix calculated from different cores
    """
    for rank in range(nrank):
        data = np.load(f"{file_cc_matrix}_{channel_shift}_{rank}.npz")
        if rank == 0:
            cc = data["cc"]
            id_row = data["id_row"]
            id_col = data["id_col"]
            id_pair = data["id_pair"]
        else:
            cc += data["cc"]
    np.savez(f"{file_cc_matrix}_{channel_shift}.npz", cc=cc, id_row=id_row, id_col=id_col, id_pair=id_pair)
    if clean:
        for rank in range(nrank):
            os.remove(f"{file_cc_matrix}_{channel_shift}_{rank}.npz")


# helper functions
def write_h5(fn, dataset_name, data, attrs_dict):
    """
    write dataset to hdf5 file
    """
    with h5py.File(fn, "a") as fid:
        if dataset_name in fid.keys():
            del fid[dataset_name]
        fid.create_dataset(dataset_name, data=data)
        for key, val in attrs_dict.items():
            fid[dataset_name].attrs.modify(key, val)


# # %%
# @dataclass
# class Config:
#     sampling_rate: int = 100
#     time_before: float = 2
#     time_after: float = 2
#     component: str = "ENZ123"
#     degree2km: float = 111.2

#     def __init__(self) -> None:
#         self.nt = int((self.time_before + self.time_after) * self.sampling_rate)
#         pass


# # %%
# def resample(data, sampling_rate, new_sampling_rate):
#     """
#     data is a 1D numpy array
#     implement resampling using numpy
#     """
#     if sampling_rate == new_sampling_rate:
#         return data
#     else:
#         # resample
#         n = data.shape[0]
#         t = np.linspace(0, 1, n)
#         t_interp = np.linspace(0, 1, int(n * new_sampling_rate / sampling_rate))
#         data_interp = np.interp(t_interp, t, data)
#         return data_interp


# def detrend(data):
#     """
#     data is a 1D numpy array
#     implement detrending using scipy to remove a linear trend
#     """
#     return scipy.signal.detrend(data, type="linear")


# def taper(data, taper_type="hann", taper_fraction=0.05):
#     """
#     data is a 1D numpy array
#     implement tapering using scipy
#     """
#     if taper_type == "hann":
#         taper = scipy.signal.hann(int(data.shape[0] * taper_fraction))
#     elif taper_type == "hamming":
#         taper = scipy.signal.hamming(int(data.shape[0] * taper_fraction))
#     elif taper_type == "blackman":
#         taper = scipy.signal.blackman(int(data.shape[0] * taper_fraction))
#     else:
#         raise ValueError("Unknown taper type")
#     taper = taper[: len(taper) // 2]
#     taper = np.hstack((taper, np.ones(data.shape[0] - taper.shape[0] * 2), taper[::-1]))
#     return data * taper


# def filter(data, type="highpass", freq=1.0, sampling_rate=100.0):
#     """
#     data is a 1D numpy array
#     implement filtering using scipy
#     """
#     if type == "highpass":
#         b, a = scipy.signal.butter(2, freq, btype="highpass", fs=sampling_rate)
#     elif type == "lowpass":
#         b, a = scipy.signal.butter(2, freq, btype="lowpass", fs=sampling_rate)
#     elif type == "bandpass":
#         b, a = scipy.signal.butter(2, freq, btype="bandpass", fs=sampling_rate)
#     elif type == "bandstop":
#         b, a = scipy.signal.butter(2, freq, btype="bandstop", fs=sampling_rate)
#     else:
#         raise ValueError("Unknown filter type")
#     return scipy.signal.filtfilt(b, a, data)


# # %%
# def extract_template(year_dir, jday, events, stations, picks, config, mseed_path, output_path, figure_path):

#     # %%
#     waveforms_dict = {}
#     for station_id in tqdm(stations["station_id"], desc=f"Loading: "):
#         net, sta, loc, chn = station_id.split(".")
#         key = f"{net}.{sta}.{chn}[{config.component}].mseed"
#         try:
#             stream = obspy.read(jday / key)
#             stream.merge(method=1, interpolation_samples=0)
#             waveforms_dict[key] = stream
#         except Exception as e:
#             print(e)
#             continue

#     # %%
#     picks["station_component_index"] = picks.apply(lambda x: f"{x.station_id}.{x.phase_type}", axis=1)

#     # %%
#     with h5py.File(output_path / f"{year_dir.name}-{jday.name}.h5", "w") as fp:

#         begin_time = datetime.strptime(f"{year_dir.name}-{jday.name}", "%Y-%j").replace(tzinfo=timezone.utc)
#         end_time = begin_time + timedelta(days=1)
#         events_ = events[(events["event_time"] > begin_time) & (events["event_time"] < end_time)]

#         num_event = 0
#         for event_index in tqdm(events_["event_index"], desc=f"Cutting event {year_dir.name}-{jday.name}.h5"):

#             picks_ = picks.loc[event_index]
#             picks_ = picks_.set_index("station_component_index")

#             event_loc = events_.loc[event_index][["x_km", "y_km", "z_km"]].to_numpy().astype(np.float32)
#             event_loc = np.hstack((event_loc, [0]))[np.newaxis, :]
#             station_loc = stations[["x_km", "y_km", "z_km"]].to_numpy()

#             h5_event = fp.create_group(f"{event_index}")

#             for i, phase_type in enumerate(["P", "S"]):

#                 travel_time = gamma.seismic_ops.calc_time(
#                     event_loc,
#                     station_loc,
#                     [phase_type.lower() for _ in range(len(station_loc))],
#                 ).squeeze()

#                 predicted_phase_timestamp = events_.loc[event_index]["event_timestamp"] + travel_time
#                 # predicted_phase_time = [events_.loc[event_index]["event_time"] + pd.Timedelta(seconds=x) for x in travel_time]

#                 for c in config.component:

#                     h5_template = h5_event.create_group(f"{phase_type}_{c}")

#                     data = np.zeros((len(stations), config.nt))
#                     label = []
#                     snr = []
#                     empty_data = True

#                     # fig, axis = plt.subplots(1, 1, squeeze=False, figsize=(6, 10))
#                     for j, station_id in enumerate(stations["station_id"]):

#                         if f"{station_id}_{phase_type}" in picks_.index:
#                             ## TODO: check if multiple phases for the same station
#                             phase_timestamp = picks_.loc[f"{station_id}_{phase_type}"]["phase_timestamp"]
#                             predicted_phase_timestamp[j] = phase_timestamp
#                             label.append(1)
#                         else:
#                             label.append(0)

#                         net, sta, loc, chn = station_id.split(".")
#                         key = f"{net}.{sta}.{chn}[{config.component}].mseed"

#                         if key in waveforms_dict:

#                             trace = waveforms_dict[key]
#                             trace = trace.select(channel=f"*{c}")
#                             if len(trace) == 0:
#                                 continue
#                             if len(trace) > 1:
#                                 print(f"More than one trace: {trace}")
#                             trace = trace[0]

#                             begin_time = (
#                                 predicted_phase_timestamp[j]
#                                 - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
#                                 - config.time_before
#                             )
#                             end_time = (
#                                 predicted_phase_timestamp[j]
#                                 - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
#                                 + config.time_after
#                             )

#                             trace_data = trace.data[
#                                 int(begin_time * trace.stats.sampling_rate) : int(end_time * trace.stats.sampling_rate)
#                             ].astype(np.float32)
#                             if len(trace_data) < (config.nt // 2):
#                                 continue
#                             std = np.std(trace_data)
#                             if std == 0:
#                                 continue

#                             if trace.stats.sampling_rate != config.sampling_rate:
#                                 # print(f"Resampling {trace.id}: {trace.stats.sampling_rate}Hz -> {config.sampling_rate}Hz")
#                                 trace_data = resample(trace_data, trace.stats.sampling_rate, config.sampling_rate)

#                             trace_data = detrend(trace_data)
#                             trace_data = taper(trace_data, taper_type="hann", taper_fraction=0.05)
#                             trace_data = filter(trace_data, type="highpass", freq=1, sampling_rate=config.sampling_rate)

#                             empty_data = False
#                             data[j, : config.nt] = trace_data[: config.nt]
#                             snr.append(np.std(trace_data[config.nt // 2 :]) / np.std(trace_data[: config.nt // 2]))

#                     #         # axis[0, 0].plot(
#                     #         #     np.arange(len(trace_data)) / config.sampling_rate - config.time_before,
#                     #         #     trace_data / std / 3.0 + j,
#                     #         #     c="k",
#                     #         #     linewidth=0.5,
#                     #         #     label=station_id,
#                     #         # )

#                     if not empty_data:
#                         data = np.array(data)
#                         data_ds = h5_template.create_dataset("data", data=data, dtype=np.float32)
#                         data_ds.attrs["nx"] = data.shape[0]
#                         data_ds.attrs["nt"] = data.shape[1]
#                         data_ds.attrs["dt_s"] = 1.0 / config.sampling_rate
#                         data_ds.attrs["time_before_s"] = config.time_before
#                         data_ds.attrs["time_after_s"] = config.time_after
#                         tt_ds = h5_template.create_dataset("travel_time", data=travel_time, dtype=np.float32)
#                         tti_ds = h5_template.create_dataset(
#                             "travel_time_index", data=np.round(travel_time * config.sampling_rate), dtype=np.int32
#                         )
#                         ttt_ds = h5_template.create_dataset("travel_time_type", data=label, dtype=np.int32)
#                         ttt_ds.attrs["label"] = ["predicted", "auto_picks", "manual_picks"]
#                         sta_ds = h5_template.create_dataset(
#                             "station_id",
#                             data=stations["station_id"].to_numpy(),
#                             dtype=h5py.string_dtype(encoding="utf-8"),
#                         )
#                         snr_ds = h5_template.create_dataset("snr", data=snr, dtype=np.float32)

#                     # if has_data:
#                     #     fig.savefig(figure_path / f"{event_index}_{phase_type}_{c}.png")
#                     #     plt.close(fig)

#                 num_event += 1
#                 if num_event > 20:
#                     break


# # %%
# if __name__ == "__main__":

#     # %%
#     config = Config()

#     min_longitude, max_longitude, min_latitude, max_latitude = [34.7 + 0.4, 39.7 - 0.4, 35.5, 39.5 - 0.1]
#     center = [(min_longitude + max_longitude) / 2, (min_latitude + max_latitude) / 2]
#     config.center = center
#     config.xlim_degree = [min_longitude, max_longitude]
#     config.ylim_degree = [min_latitude, max_latitude]

#     stations = pd.read_json("../../EikoLoc/stations.json", orient="index")
#     stations["station_id"] = stations.index
#     stations = stations[
#         (stations["longitude"] > config.xlim_degree[0])
#         & (stations["longitude"] < config.xlim_degree[1])
#         & (stations["latitude"] > config.ylim_degree[0])
#         & (stations["latitude"] < config.ylim_degree[1])
#     ]
#     # stations["distance_km"] = stations.apply(
#     #     lambda x: math.sqrt((x.latitude - config.center[1]) ** 2 + (x.longitude - config.center[0]) ** 2)
#     #     * config.degree2km,
#     #     axis=1,
#     # )
#     # stations.sort_values(by="distance_km", inplace=True)
#     # stations.drop(columns=["distance_km"], inplace=True)
#     # stations.sort_values(by="latitude", inplace=True)
#     stations["x_km"] = stations.apply(
#         lambda x: (x.longitude - config.center[0]) * np.cos(np.deg2rad(config.center[1])) * config.degree2km, axis=1
#     )
#     stations["y_km"] = stations.apply(lambda x: (x.latitude - config.center[1]) * config.degree2km, axis=1)
#     stations["z_km"] = stations.apply(lambda x: -x.elevation_m / 1e3, axis=1)

#     # %%
#     events = pd.read_csv(
#         "../../EikoLoc/eikoloc_catalog.csv", parse_dates=["time"], date_parser=lambda x: pd.to_datetime(x, utc=True)
#     )
#     events = events[events["time"].notna()]
#     events.sort_values(by="time", inplace=True)
#     events.rename(columns={"time": "event_time"}, inplace=True)
#     events["event_timestamp"] = events["event_time"].apply(lambda x: x.timestamp())
#     events["x_km"] = events.apply(
#         lambda x: (x.longitude - config.center[0]) * np.cos(np.deg2rad(config.center[1])) * config.degree2km, axis=1
#     )
#     events["y_km"] = events.apply(lambda x: (x.latitude - config.center[1]) * config.degree2km, axis=1)
#     events["z_km"] = events.apply(lambda x: x.depth_km, axis=1)
#     event_index = list(events["event_index"])

#     # %%
#     picks = pd.read_csv(
#         "../../EikoLoc/gamma_picks.csv", parse_dates=["phase_time"], date_parser=lambda x: pd.to_datetime(x, utc=True)
#     )
#     picks = picks[picks["event_index"] != -1]
#     picks["phase_timestamp"] = picks["phase_time"].apply(lambda x: x.timestamp())
#     picks = picks.merge(stations, on="station_id")
#     picks = picks.merge(events, on="event_index", suffixes=("_station", "_event"))

#     # %%
#     events["index"] = events["event_index"]
#     events = events.set_index("index")
#     picks["index"] = picks["event_index"]
#     picks = picks.set_index("index")

#     # %%
#     mseed_path = Path("../../convert_format/wf/")
#     figure_path = Path("./figures/")
#     output_path = Path("./templates/")
#     if not figure_path.exists():
#         figure_path.mkdir()
#     if not output_path.exists():
#         output_path.mkdir()

#     # %%
#     ncpu = mp.cpu_count()
#     with mp.Pool(ncpu) as pool:
#         pool.starmap(
#             extract_template,
#             [
#                 (
#                     year_dir,
#                     jday,
#                     events,
#                     stations,
#                     picks,
#                     config,
#                     mseed_path,
#                     output_path,
#                     figure_path,
#                 )
#                 for year_dir in mseed_path.iterdir()
#                 for jday in year_dir.iterdir()
#             ],
#         )

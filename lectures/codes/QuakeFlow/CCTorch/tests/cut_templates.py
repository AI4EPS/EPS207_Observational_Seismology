# %%
def resample(data, sampling_rate, new_sampling_rate):
    """
    data is a 1D numpy array
    implement resampling using numpy
    """
    if sampling_rate == new_sampling_rate:
        return data
    elif (sampling_rate % new_sampling_rate) == 0:
        return data[:: int(sampling_rate / new_sampling_rate)]
    else:
        n = data.shape[0]
        t = np.linspace(0, 1, n)
        f = scipy.interpolate.interp1d(t, data, kind="linear")
        t_interp = np.linspace(0, 1, int(n * new_sampling_rate / sampling_rate))
        data_interp = f(t_interp)
        # t_interp = np.linspace(0, 1, int(n * new_sampling_rate / sampling_rate))
        # data_interp = np.interp(t_interp, t, data)
        return data_interp


def detrend(data):
    """
    data is a 1D numpy array
    implement detrending using scipy to remove a linear trend
    """
    return scipy.signal.detrend(data, type="linear")


def taper(data, taper_type="hann", taper_fraction=0.05):
    """
    data is a 1D numpy array
    implement tapering using scipy
    """
    if taper_type == "hann":
        taper = scipy.signal.hann(int(data.shape[0] * taper_fraction))
    elif taper_type == "hamming":
        taper = scipy.signal.hamming(int(data.shape[0] * taper_fraction))
    elif taper_type == "blackman":
        taper = scipy.signal.blackman(int(data.shape[0] * taper_fraction))
    else:
        raise ValueError("Unknown taper type")
    taper = taper[: len(taper) // 2]
    taper = np.hstack((taper, np.ones(data.shape[0] - taper.shape[0] * 2), taper[::-1]))
    return data * taper


def filter(data, type="highpass", freq=1.0, sampling_rate=100.0):
    """
    data is a 1D numpy array
    implement filtering using scipy
    """
    if type == "highpass":
        b, a = scipy.signal.butter(2, freq, btype="highpass", fs=sampling_rate)
    elif type == "lowpass":
        b, a = scipy.signal.butter(2, freq, btype="lowpass", fs=sampling_rate)
    elif type == "bandpass":
        b, a = scipy.signal.butter(2, freq, btype="bandpass", fs=sampling_rate)
    elif type == "bandstop":
        b, a = scipy.signal.butter(2, freq, btype="bandstop", fs=sampling_rate)
    else:
        raise ValueError("Unknown filter type")
    return scipy.signal.filtfilt(b, a, data)


# %%
def extract_template_h5(year_dir, jday, events, stations, picks, config, mseed_path, output_path, figure_path):

    # %%
    waveforms_dict = {}
    for station_id in tqdm(stations["station_id"], desc=f"Loading: "):
        net, sta, loc, chn = station_id.split(".")
        for c in config.components:
            key = f"{net}.{sta}.{chn}{c}.mseed"
            try:
                stream = obspy.read(jday / key)
                stream.merge(method=1, interpolation_samples=0)
                waveforms_dict[key] = stream
            except Exception as e:
                # print(e)
                continue

    # %%
    picks["station_component_index"] = picks.apply(lambda x: f"{x.station_id}.{x.phase_type}", axis=1)

    # %%
    with h5py.File(output_path / f"{year_dir.name}-{jday.name}.h5", "w") as fp:

        begin_time = datetime.strptime(f"{year_dir.name}-{jday.name}", "%Y-%j").replace(tzinfo=timezone.utc)
        end_time = begin_time + timedelta(days=1)
        events_ = events[(events["event_time"] > begin_time) & (events["event_time"] < end_time)]

        num_event = 0
        for event_index in tqdm(events_["event_index"], desc=f"Cutting event {year_dir.name}-{jday.name}.h5"):

            picks_ = picks.loc[[event_index]]
            picks_ = picks_.set_index("station_component_index")

            event_loc = events_.loc[event_index][["x_km", "y_km", "z_km"]].to_numpy().astype(np.float32)
            event_loc = np.hstack((event_loc, [0]))[np.newaxis, :]
            station_loc = stations[["x_km", "y_km", "z_km"]].to_numpy()

            h5_event = fp.create_group(f"{event_index}")

            for i, phase_type in enumerate(["P", "S"]):

                travel_time = gamma.seismic_ops.calc_time(
                    event_loc,
                    station_loc,
                    [phase_type.lower() for _ in range(len(station_loc))],
                ).squeeze()

                predicted_phase_timestamp = events_.loc[event_index]["event_timestamp"] + travel_time
                # predicted_phase_time = [events_.loc[event_index]["event_time"] + pd.Timedelta(seconds=x) for x in travel_time]

                for c in config.components:
                    
                    data = np.zeros((len(stations), config.nt))
                    label = []
                    snr = []
                    empty_data = True

                    # fig, axis = plt.subplots(1, 1, squeeze=False, figsize=(6, 10))
                    for j, station_id in enumerate(stations["station_id"]):

                        if f"{station_id}.{phase_type}" in picks_.index:
                            ## TODO: check if multiple phases for the same station
                            phase_timestamp = picks_.loc[f"{station_id}.{phase_type}"]["phase_timestamp"]
                            predicted_phase_timestamp[j] = phase_timestamp
                            label.append(1)
                        else:
                            label.append(0)

                        net, sta, loc, chn = station_id.split(".")
                        key = f"{net}.{sta}.{chn}{c}.mseed"

                        if key in waveforms_dict:

                            trace = waveforms_dict[key]
                            # trace = trace.select(channel=f"*{c}")
                            if len(trace) == 0:
                                continue

                            if len(trace) > 1:
                                print(f"More than one trace: {trace}")
                            trace = trace[0]

                            begin_time = (
                                predicted_phase_timestamp[j]
                                - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
                                - config.time_before
                            )
                            end_time = (
                                predicted_phase_timestamp[j]
                                - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
                                + config.time_after
                            )

                            trace_data = trace.data[
                                max(0, int(begin_time * trace.stats.sampling_rate)) : int(
                                    end_time * trace.stats.sampling_rate
                                )
                            ].astype(np.float32)
                            if len(trace_data) < config.nt:
                                continue
                            std = np.std(trace_data)
                            if std == 0:
                                continue

                            if trace.stats.sampling_rate != config.sampling_rate:
                                # print(f"Resampling {trace.id}: {trace.stats.sampling_rate}Hz -> {config.sampling_rate}Hz")
                                trace_data = resample(trace_data, trace.stats.sampling_rate, config.sampling_rate)

                            if len(trace_data) < config.nt:
                                continue
                            trace_data -= np.mean(trace_data)
                            # trace_data = detrend(trace_data)
                            # trace_data = taper(trace_data, taper_type="hann", taper_fraction=0.05)
                            trace_data = filter(trace_data, type="highpass", freq=1, sampling_rate=config.sampling_rate)

                            empty_data = False
                            data[j, : config.nt] = trace_data[: config.nt]
                            s = np.std(trace_data[config.nt // 2 :])
                            n = np.std(trace_data[: config.nt // 2])
                            if n == 0:
                                snr.append(0)
                            else:
                                snr.append(s/n)

                    #         # axis[0, 0].plot(
                    #         #     np.arange(len(trace_data)) / config.sampling_rate - config.time_before,
                    #         #     trace_data / std / 3.0 + j,
                    #         #     c="k",
                    #         #     linewidth=0.5,
                    #         #     label=station_id,
                    #         # )

                    if not empty_data:
                        data = np.array(data)
                        h5_template = h5_event.create_group(f"{phase_type}_{config.component_mapping[c]}")
                        data_ds = h5_template.create_dataset("data", data=data, dtype=np.float32)
                        data_ds.attrs["nx"] = data.shape[0]
                        data_ds.attrs["nt"] = data.shape[1]
                        data_ds.attrs["dt_s"] = 1.0 / config.sampling_rate
                        data_ds.attrs["time_before_s"] = config.time_before
                        data_ds.attrs["time_after_s"] = config.time_after
                        tt_ds = h5_template.create_dataset("travel_time", data=travel_time, dtype=np.float32)
                        tti_ds = h5_template.create_dataset(
                            "travel_time_index", data=np.round(travel_time * config.sampling_rate), dtype=np.int32
                        )
                        ttt_ds = h5_template.create_dataset("travel_time_type", data=label, dtype=np.int32)
                        ttt_ds.attrs["label"] = ["predicted", "auto_picks", "manual_picks"]
                        sta_ds = h5_template.create_dataset(
                            "station_id",
                            data=stations["station_id"].to_numpy(),
                            dtype=h5py.string_dtype(encoding="utf-8"),
                        )
                        snr_ds = h5_template.create_dataset("snr", data=snr, dtype=np.float32)

                    # if has_data:
                    #     fig.savefig(figure_path / f"{event_index}_{phase_type}_{c}.png")
                    #     plt.close(fig)

                # num_event += 1
                # if num_event > 20:
                #     break

# %%
def merge_h5(h5_path: Path, h5_out):
    """Merge h5 files into one file"""
    h5_files = list(h5_path.glob("*.h5"))
    h5_files.sort()
    print(h5_files)
    with h5py.File(h5_out, "w") as h5_out:
        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as h5_in:
                for event_index in tqdm(h5_in, desc=f"Merging {h5_file.name}"):
                    h5_out.create_group(event_index)
                    for phase_type in h5_in[event_index]:
                        h5_out[event_index].create_group(phase_type)
                        for key in h5_in[event_index][phase_type]:
                            h5_out[event_index][phase_type].create_dataset(
                                key, data=h5_in[event_index][phase_type][key][()]
                            )
                            for attr in h5_in[event_index][phase_type][key].attrs:
                                h5_out[event_index][phase_type][key].attrs[attr] = h5_in[event_index][phase_type][key].attrs[
                                    attr
                                ]
    return 0



def extract_template_numpy(template_array, travel_time_array, travel_time_index_array, travel_time_type_array, snr_array,
                           mseed_path, events, stations, picks, config, output_path, figure_path, lock, ibar):

    # %%
    tmp = str(mseed_path).split("/")
    year_jday, hour = tmp[-2], tmp[-1]
    begin_time = datetime.strptime(f"{year_jday}T{hour}", "%Y-%jT%H").replace(tzinfo=timezone.utc)
    end_time = begin_time + timedelta(hours=1)
    events_ = events[(events["event_time"] > begin_time) & (events["event_time"] < end_time)]

    if len(events_) == 0:
        return 0

    # %%
    waveforms_dict = {}
    for station_id in tqdm(stations["station_id"], desc=f"Loading: ", position=ibar%6, nrows=7, mininterval=5, leave=False):
        for c in config.components:
            if (mseed_path / f"{station_id}{c}.mseed").exists():
                try:
                    stream = obspy.read(mseed_path / f"{station_id}{c}.mseed")
                    # stream.merge(method=1, interpolation_samples=0)
                    stream.merge(fill_value="latest")
                    if len(stream) > 1:
                        print(f"More than one trace: {stream}")
                    trace = stream[0]
                    if trace.stats.sampling_rate != config.sampling_rate:
                        if trace.stats.sampling_rate % config.sampling_rate == 0:
                            trace.decimate(int(trace.stats.sampling_rate / config.sampling_rate))
                        else:
                            trace.resample(config.sampling_rate)
                    trace.detrend("linear")
                    # trace.taper(max_percentage=0.05, type="cosine")
                    trace.filter("bandpass", freqmin=1.0, freqmax=15.0, corners=2, zerophase=True)
                    waveforms_dict[f"{station_id}{c}"] = trace
                except Exception as e:
                    print(e)
                    continue

    # %%
    picks["station_component_index"] = picks.apply(lambda x: f"{x.station_id}.{x.phase_type}", axis=1)

    # %%
    num_event = 0
    for event_index in tqdm(events_["event_index"], desc=f"Cutting event {year_jday}T{hour}", position=ibar%6, nrows=7, mininterval=5, leave=False):

        if event_index not in picks.index:
            continue

        picks_ = picks.loc[[event_index]]
        picks_ = picks_.set_index("station_component_index")

        event_loc = events_.loc[event_index][["x_km", "y_km", "z_km"]].to_numpy().astype(np.float32)
        event_loc = np.hstack((event_loc, [0]))[np.newaxis, :]
        station_loc = stations[["x_km", "y_km", "z_km"]].to_numpy()

        template_ = np.zeros((6, len(stations), config.nt), dtype=np.float32)
        snr_ = np.zeros((6, len(stations)), dtype=np.float32)
        travel_time_ = np.zeros((2, len(stations)), dtype=np.float32)
        travel_time_type_ = np.zeros((2, len(stations)), dtype=np.int32)

        for i, phase_type in enumerate(["P", "S"]):

            travel_time = gamma.seismic_ops.calc_time(
                event_loc,
                station_loc,
                [phase_type.lower() for _ in range(len(station_loc))],
                vel={"p": 6.0, "s": 6.0 / 1.73},
            ).squeeze()

            phase_timestamp_pred = events_.loc[event_index]["event_timestamp"] + travel_time
            # predicted_phase_time = [events_.loc[event_index]["event_time"] + pd.Timedelta(seconds=x) for x in travel_time]

            mean_shift = []
            for j, station_id in enumerate(stations["station_id"]):
                if f"{station_id}.{phase_type}" in picks_.index:
                    ## TODO: check if multiple phases for the same station
                    phase_timestamp = picks_.loc[f"{station_id}.{phase_type}"]["phase_timestamp"]
                    phase_timestamp_pred[j] = phase_timestamp
                    travel_time[j] = phase_timestamp - events_.loc[event_index]["event_timestamp"]
                    travel_time_type_[i, j] = 1
                    mean_shift.append(phase_timestamp - (events_.loc[event_index]["event_timestamp"] - travel_time[j]))
                else:
                    travel_time_type_[i, j] = 0
            if len(mean_shift) > 0:
                mean_shift = float(np.median(mean_shift))
            else:
                mean_shift = 0
            phase_timestamp_pred[travel_time_type_[i, :] == 0] += mean_shift
            travel_time[travel_time_type_[i, :] == 0] += mean_shift
            travel_time_[i, :] = travel_time

            for c in config.components:
                
                c_index = i*3 + config.component_mapping[c]
                empty_data = True

                # fig, axis = plt.subplots(1, 1, squeeze=False, figsize=(6, 10))
                for j, station_id in enumerate(stations["station_id"]):

                    if f"{station_id}{c}" in waveforms_dict:

                        trace = waveforms_dict[f"{station_id}{c}"]

                        begin_time = (
                            phase_timestamp_pred[j]
                            - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
                            - config.time_before
                        )
                        end_time = (
                            phase_timestamp_pred[j]
                            - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
                            + config.time_after
                        )
                        
                        # if begin_time < 0:
                        #     print(f"{events_.loc[event_index]['event_time'] = }")
                        #     print(f"{predicted_phase_timestamp[j] = }")
                        #     print(f"{trace.stats.starttime.datetime = }")
                        #     print(f"{trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp() = }")
                        #     print(f"{begin_time = }")

                        
                        trace_data = trace.data[
                            max(0, int(begin_time * trace.stats.sampling_rate)) : max(0, int(
                                end_time * trace.stats.sampling_rate)
                            )
                        ].astype(np.float32)


                        if len(trace_data) < config.nt:
                            continue
                        std = np.std(trace_data)
                        if std == 0:
                            continue

                        # if trace.stats.sampling_rate != config.sampling_rate:
                        #     # print(f"Resampling {trace.id}: {trace.stats.sampling_rate}Hz -> {config.sampling_rate}Hz")
                        #     trace_data = resample(trace_data, trace.stats.sampling_rate, config.sampling_rate)
                        #     if len(trace_data) < config.nt:
                        #         continue
                        # if len(trace_data) > config.nt:
                        #     print(f"{len(trace_data) = }")
                        #     print(f"{begin_time = }")
                        #     print(f"{end_time = }")
                        #     print(f"{max(0, int(begin_time * trace.stats.sampling_rate)) = }")
                        #     print(f"{int(end_time * trace.stats.sampling_rate) = }")

                        # trace_data -= np.mean(trace_data)
                        # trace_data = detrend(trace_data)
                        # trace_data = taper(trace_data, taper_type="hann", taper_fraction=0.05)
                        # # trace_data = filter(trace_data, type="highpass", freq=1, sampling_rate=config.sampling_rate)
                        # trace_data = filter(trace_data, type="bandpass", freq=(1, 15), sampling_rate=config.sampling_rate)

                        # if travel_time_type_[i, j] == 1:

                        empty_data = False
                        template_[c_index, j, : config.nt] = trace_data[: config.nt]
                        s = np.std(trace_data[-int(config.time_after * config.sampling_rate) :])
                        n = np.std(trace_data[: int(config.time_before * config.sampling_rate)])
                        if n == 0:
                            snr_[c_index, j] = 0
                        else:
                            snr_[c_index, j] = s / n

                #         if travel_time_type_[i, j] == 1:
                #             color = "r"
                #         else:
                #             color = "k"
                #         axis[0, 0].plot(
                #             np.arange(len(trace_data)) / config.sampling_rate - config.time_before,
                #             trace_data / std / 3.0 + j,
                #             c=color,
                #             linewidth=0.5,
                #             label=station_id,
                #         )
                        
                # if not empty_data:
                #     fig.savefig(figure_path / f"{event_index}_{phase_type}_{c}.png", dpi=300)
                #     plt.close(fig)

        template_array[event_index] = template_
        travel_time_array[event_index] = travel_time_
        travel_time_index_array[event_index] = np.round(travel_time_ * config.sampling_rate).astype(np.int32)
        travel_time_type_array[event_index] = travel_time_type_
        snr_array[event_index] = snr_

        with lock:
            template_array.flush()
            travel_time_array.flush()
            travel_time_index_array.flush()
            travel_time_type_array.flush()
            snr_array.flush()

        # num_event += 1
        # if num_event > 20:
        #     break

# %%
if __name__ == "__main__":

    # %%
    config = Config()

    min_longitude, max_longitude, min_latitude, max_latitude = [34.7 + 0.4, 39.7 - 0.4, 35.5, 39.5 - 0.1]
    center = [round((min_longitude + max_longitude) / 2, 2), round((min_latitude + max_latitude) / 2, 2)]
    config.center = center
    config.xlim_degree = [round(min_longitude, 2), round(max_longitude, 2)]
    config.ylim_degree = [round(min_latitude, 2), round(max_latitude, 2)]

    stations = pd.read_json("../../EikoLoc/stations.json", orient="index")
    stations["station_id"] = stations.index
    stations = stations[
        (stations["longitude"] > config.xlim_degree[0])
        & (stations["longitude"] < config.xlim_degree[1])
        & (stations["latitude"] > config.ylim_degree[0])
        & (stations["latitude"] < config.ylim_degree[1])
    ]
    # stations["distance_km"] = stations.apply(
    #     lambda x: math.sqrt((x.latitude - config.center[1]) ** 2 + (x.longitude - config.center[0]) ** 2)
    #     * config.degree2km,
    #     axis=1,
    # )
    # stations.sort_values(by="distance_km", inplace=True)
    # stations.drop(columns=["distance_km"], inplace=True)
    # stations.sort_values(by="latitude", inplace=True)
    stations["x_km"] = stations.apply(
        lambda x: (x.longitude - config.center[0]) * np.cos(np.deg2rad(config.center[1])) * config.degree2km, axis=1
    )
    stations["y_km"] = stations.apply(lambda x: (x.latitude - config.center[1]) * config.degree2km, axis=1)
    stations["z_km"] = stations.apply(lambda x: -x.elevation_m / 1e3, axis=1)

    # %%
    events = pd.read_csv(
        "../../EikoLoc/eikoloc_catalog.csv", parse_dates=["time"], date_parser=lambda x: pd.to_datetime(x, utc=True)
    )
    events = events[events["time"].notna()]
    events.sort_values(by="time", inplace=True)
    events.rename(columns={"time": "event_time"}, inplace=True)
    events["event_timestamp"] = events["event_time"].apply(lambda x: x.timestamp())
    events["x_km"] = events.apply(
        lambda x: (x.longitude - config.center[0]) * np.cos(np.deg2rad(config.center[1])) * config.degree2km, axis=1
    )
    events["y_km"] = events.apply(lambda x: (x.latitude - config.center[1]) * config.degree2km, axis=1)
    events["z_km"] = events.apply(lambda x: x.depth_km, axis=1)
    event_index = list(events["event_index"])

    with open("../event_index.txt", "w") as f:
        for i in event_index:
            f.write(f"{i}\n")

    # %%
    picks = pd.read_csv(
        "../../EikoLoc/gamma_picks.csv", parse_dates=["phase_time"], date_parser=lambda x: pd.to_datetime(x, utc=True)
    )
    picks = picks[picks["event_index"] != -1]
    picks["phase_timestamp"] = picks["phase_time"].apply(lambda x: x.timestamp())
    picks = picks.merge(stations, on="station_id")
    picks = picks.merge(events, on="event_index", suffixes=("_station", "_event"))

    # %%
    events["index"] = events["event_index"]
    events = events[events["event_index"].isin([1452114, 1452121])]
    events = events.set_index("index")
    picks["index"] = picks["event_index"]
    picks = picks.set_index("index")

    # %%
    mseed_path = Path("../../convert_format/waveforms/")
    figure_path = Path("./figures/")
    output_path = Path("./templates/")
    if not figure_path.exists():
        figure_path.mkdir()
    if not output_path.exists():
        output_path.mkdir()


    # %%
    nt = config.nt
    nch = 6 ## For [P,S] phases and [E,N,Z] components
    nev = int(events.index.max()) + 1
    nst = len(stations)
    config.template_shape = [nev, nch, nst, nt]
    print(config.template_shape)
    template_array = np.memmap(output_path/"template.dat", dtype=np.float32, mode="w+", shape=(nev, nch, nst, nt))
    travel_time_array = np.memmap(output_path/"travel_time.dat", dtype=np.float32, mode="w+", shape=(nev, nch//3, nst))
    travel_time_index_array = np.memmap(output_path/"travel_time_index.dat", dtype=np.int32, mode="w+", shape=(nev, nch//3, nst))
    travel_time_type_array = np.memmap(output_path/"travel_time_type.dat", dtype=np.int32, mode="w+", shape=(nev, nch//3, nst))
    snr_array = np.memmap(output_path/"snr.dat", dtype=np.float32, mode="w+", shape=(nev, nch, nst))
    # fp = shared_memory.SharedMemory(name="templates.dat", create=True, size=fp.nbytes)
    
    with open(output_path/"config.json", "w") as f:
        json.dump(config.__dict__, f, indent=4)

    # %%
    dirs = [hour_dir  for jday_dir in sorted(list(mseed_path.iterdir()))[::-1] for hour_dir in sorted(list(jday_dir.iterdir()))]
    ncpu = mp.cpu_count()//2
    lock = mp.Lock()
    processes = []
    for i, d in enumerate(dirs):
        # add process of extract_template_numpy to list
        proc = mp.Process(target=extract_template_numpy, args=(
            template_array, travel_time_array, travel_time_index_array, travel_time_type_array, snr_array,
            d, events, stations, picks, config, output_path, figure_path, lock, i%ncpu))
        processes.append(proc)
        proc.start()
        if len(processes) == ncpu:
            for proc in processes:
                proc.join()
            processes = []
    for proc in processes:
        proc.join()
    
    # %%
    # fp = np.memmap("template.dat", dtype=np.float32, mode="r", shape=(nev, nch, nst, nt))

    # %%
    # ncpu = mp.cpu_count()
    # # ncpu = 4
    # with mp.Pool(ncpu) as pool:
    #     pool.starmap(
    #         extract_template,
    #         [
    #             (
    #                 year_dir,
    #                 jday,
    #                 events,
    #                 stations,
    #                 picks,
    #                 config,
    #                 mseed_path,
    #                 output_path,
    #                 figure_path,
    #             )
    #             for year_dir in sorted(list(mseed_path.iterdir()))[::-1]
    #             for jday in sorted(list(year_dir.iterdir()))[::-1]
    #             # for jday in [Path("2023/041")]
    #         ],
    #     )

    # %%
    # for year_dir in sorted(list(mseed_path.iterdir()))[::-1]:
    #     for jday in sorted(list(year_dir.iterdir()))[::-1]:
    #         extract_template(
    #             year_dir,
    #             jday,
    #             events,
    #             stations,
    #             picks,
    #             config,
    #             mseed_path,
    #             output_path,
    #             figure_path,
    #         )

    # %%
    # merge_h5(output_path, "templates.h5")
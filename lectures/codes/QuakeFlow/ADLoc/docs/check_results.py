# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
true_events = pd.read_csv("stanford/true_event.csv")
# pred_events = pd.read_csv("results/stanford/adloc_events.csv")
# pred_events = pd.read_csv("results/stanford/adloc_events_sst.csv")
# pred_events = pd.read_csv("results/stanford/adloc_events_grid_search.csv")
# pred_events = pd.read_csv("results/stanford/adloc_dd_events.csv")
pred_events = pd.read_csv("results/stanford/ransac_events_sst.csv")

# %%
pred_picks = pd.read_csv("results/stanford/ransac_picks_sst.csv")
stations = pd.read_csv("results/stanford/ransac_stations_sst.csv")
pred_picks = pred_picks.merge(stations, on="station_id")
pred_picks = pred_picks.merge(pred_events, on="event_index", suffixes=("_station", "_event"))
deg2km = 111.32
pred_picks["dist_km"] = pred_picks.apply(
    lambda x: np.sqrt(
        ((x["latitude_station"] - x["latitude_event"]) * deg2km) ** 2
        + ((x["longitude_station"] - x["longitude_event"]) * deg2km * np.cos(np.radians(x["latitude_event"]))) ** 2
        + ((x["depth_km_station"] - x["depth_km_event"]) ** 2)
    ),
    axis=1,
)
pred_picks["time"] = pd.to_datetime(pred_picks["time"])
pred_picks["phase_time"] = pd.to_datetime(pred_picks["phase_time"])
pred_picks["travel_time"] = (pred_picks["phase_time"] - pred_picks["time"]).dt.total_seconds()

# %%
print("P outliers:", len(pred_picks[(pred_picks["adloc_mask"] == 0) & (pred_picks["phase_type"] == "P")]))
print("S outliers:", len(pred_picks[(pred_picks["adloc_mask"] == 0) & (pred_picks["phase_type"] == "S")]))

# %%
event_index = (
    pred_picks[pred_picks["adloc_mask"] == 0]
    .groupby("event_index")
    .filter(lambda x: len(x) > 2)["event_index"]
    .unique()
)
fig, ax = plt.subplots(2, 2, figsize=(8, 8))

event_index = event_index[[1, 2, 3, 5]]
ii = 0
for i in range(2):
    for j in range(2):
        picks = pred_picks[pred_picks["event_index"] == event_index[ii]]
        ii += 1

        picks_true = picks[picks["adloc_mask"] == 1]
        picks_false = picks[picks["adloc_mask"] == 0]
        # ax[i, j].scatter(picks_true["dist_km"], picks_true["travel_time"], c="C0", s=10)
        # color p phase c0 and s phase c1
        ax[i, j].scatter(
            picks_true[picks_true["phase_type"] == "P"]["travel_time"],
            picks_true[picks_true["phase_type"] == "P"]["dist_km"],
            c="C0",
            s=10,
        )
        ax[i, j].scatter(
            picks_true[picks_true["phase_type"] == "S"]["travel_time"],
            picks_true[picks_true["phase_type"] == "S"]["dist_km"],
            c="C2",
            s=10,
        )

        ax[i, j].scatter(picks_false["travel_time"], picks_false["dist_km"], c="k", s=10)

        if i == 1:
            ax[i, j].set_xlabel("Travel time (s)")
        if j == 0:
            ax[i, j].set_ylabel("Distance (km)")
        # ax[i, j].set_title(f"Event {event_index[ii]}")

        # set 0, 0 as start
        # ax[i, j].set_xlim(left=0)
        # ax[i, j].set_ylim(bottom=0)

        xlim = ax[i, j].get_xlim()
        ylim = ax[i, j].get_ylim()
        y = np.linspace(ylim[0], ylim[1], 100)

        # Linear regression for P and S waves
        picks_p = picks_true[picks_true["phase_type"] == "P"]
        picks_s = picks_true[picks_true["phase_type"] == "S"]

        # P wave regression
        if len(picks_p) > 0:
            fit = np.polyfit(picks_p["travel_time"], picks_p["dist_km"], 1)
            kp = fit[0]
            bp = fit[1]
            tp = (y - bp) / kp

        # S wave regression
        if len(picks_s) > 0:
            fit = np.polyfit(picks_s["travel_time"], picks_s["dist_km"], 1)
            ks = fit[0]
            bs = fit[1]
            ts = (y - bs) / ks

        ax[i, j].plot(tp, y, "k--", linewidth=0.5, alpha=0.5)
        ax[i, j].plot(ts, y, "k--", linewidth=0.5, alpha=0.5)

        # ax[i, j].scatter([], [], c="C0", s=10, label="Inlier picks")
        # ax[i, j].scatter([], [], c="C0", s=10, label="Inlier picks")
        ax[i, j].scatter([], [], c="C0", s=10, label="Inliers (P)")
        ax[i, j].scatter([], [], c="C2", s=10, label="Inliers (S)")
        ax[i, j].scatter([], [], c="k", s=10, label="Outliers")
        ax[i, j].legend()


plt.savefig("filter_picks.png", dpi=300, bbox_inches="tight")
plt.savefig("filter_picks.pdf", dpi=300, bbox_inches="tight")

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True, gridspec_kw={"hspace": 0.1, "wspace": 0.0})
# vmax = max(stations["station_term_time_p"].max(), stations["station_term_time_s"].max())
# vmin = min(stations["station_term_time_p"].min(), stations["station_term_time_s"].min())
# vmax = max(abs(vmax), abs(vmin))
# vmin = -vmax
im = ax[0].scatter(stations["longitude"], stations["latitude"], c=stations["station_term_time_p"], s=50, marker="^")
cbar = plt.colorbar(im, ax=ax[0])
cbar.set_label("(s)", labelpad=-37, y=1.05, rotation=0)
# ax[0].set_title("Static station term")
# ax[0].set_xlabel("Longitude")
# ax[0].set_ylabel("Latitude")
ax[0].text(
    0.95,
    0.95,
    "P-wave",
    transform=ax[0].transAxes,
    ha="right",
    va="top",
    fontsize=plt.rcParams["legend.fontsize"],
    bbox=dict(facecolor="white", edgecolor="black", alpha=0.5, boxstyle="round"),
)
ax[0].set_aspect("equal")
ax[0].grid(True, linestyle=":")
ax[0].set_yticks([35.5, 35.7, 35.9, 36.1, 36.3])


im = ax[1].scatter(stations["longitude"], stations["latitude"], c=stations["station_term_time_s"], s=50, marker="^")
cbar = plt.colorbar(im, ax=ax[1])
cbar.set_label("(s)", labelpad=-37, y=1.05, rotation=0)
# ax[1].set_title("Static station term (S)")
# ax[1].set_xlabel("Longitude")
# ax[1].set_ylabel("Latitude")
ax[1].text(
    0.95,
    0.95,
    "S-wave",
    transform=ax[1].transAxes,
    ha="right",
    va="top",
    fontsize=plt.rcParams["legend.fontsize"],
    bbox=dict(facecolor="white", edgecolor="black", alpha=0.5, boxstyle="round"),
)
ax[1].set_aspect("equal")
ax[1].grid(True, linestyle=":")
ax[1].set_yticks([35.5, 35.7, 35.9, 36.1, 36.3])

plt.savefig("station_term.png", dpi=300, bbox_inches="tight")
plt.savefig("station_term.pdf", dpi=300, bbox_inches="tight")


# %%
true_events["event_index"] = np.arange(1, len(true_events) + 1)
true_events.rename({"Lat": "latitude", "Lon": "longitude", "Dep": "depth_km"}, axis=1, inplace=True)

# %%
true_events.set_index("event_index", inplace=True)
pred_events.set_index("event_index", inplace=True)

# %%
s = 5
fig, ax = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 0.5]})

# ax[0, 0].scatter(true_events["longitude"], true_events["latitude"], c=true_events["depth_km"], s=s, label="True events")
ax[0, 0].scatter(
    pred_events["longitude"],
    pred_events["latitude"],
    c=pred_events["depth_km"],
    s=s,
    label="Predicted events",
    vmin=0,
    vmax=15,
    cmap="viridis_r",
)
ax[0, 0].set_title(f"{len(true_events)} true events and {len(pred_events)} predicted events")

ax[0, 1].scatter(true_events["longitude"], true_events["latitude"], c="r", s=s, label="True events")
ax[0, 1].scatter(pred_events["longitude"], pred_events["latitude"], c="b", s=s, label="Predicted events")
for i in true_events.index:
    if i in pred_events.index:
        ax[0, 1].plot(
            [true_events.loc[i, "longitude"], pred_events.loc[i, "longitude"]],
            [true_events.loc[i, "latitude"], pred_events.loc[i, "latitude"]],
            "k--",
            linewidth=0.5,
            alpha=0.5,
        )
    else:
        ax[0, 1].plot(
            true_events.loc[i, "longitude"],
            true_events.loc[i, "latitude"],
            "rx",
            markersize=5,
            alpha=0.5,
        )

ax[1, 0].scatter(true_events["longitude"], true_events["depth_km"], c="r", s=s, label="True events")
ax[1, 0].scatter(pred_events["longitude"], pred_events["depth_km"], c="b", s=s, label="Predicted events")
for i in true_events.index:
    if i in pred_events.index:
        ax[1, 0].plot(
            [true_events.loc[i, "longitude"], pred_events.loc[i, "longitude"]],
            [true_events.loc[i, "depth_km"], pred_events.loc[i, "depth_km"]],
            "k--",
            linewidth=0.5,
            alpha=0.5,
        )
    else:
        ax[1, 0].plot(
            true_events.loc[i, "longitude"],
            true_events.loc[i, "depth_km"],
            "rx",
            markersize=5,
            alpha=0.5,
        )
ax[1, 0].set_ylim([15, 0])


ax[1, 1].scatter(true_events["latitude"], true_events["depth_km"], c="r", s=s, label="True events")
ax[1, 1].scatter(pred_events["latitude"], pred_events["depth_km"], c="b", s=s, label="Predicted events")
for i in true_events.index:
    if i in pred_events.index:
        ax[1, 1].plot(
            [true_events.loc[i, "latitude"], pred_events.loc[i, "latitude"]],
            [true_events.loc[i, "depth_km"], pred_events.loc[i, "depth_km"]],
            "k--",
            linewidth=0.5,
            alpha=0.5,
        )
    else:
        ax[1, 1].plot(
            true_events.loc[i, "latitude"],
            true_events.loc[i, "depth_km"],
            "rx",
            markersize=5,
            alpha=0.5,
        )
ax[1, 1].set_ylim([15, 0])

# %%
km2deg = 1 / 111.32

if (
    ("sigma_x_km" in pred_events.columns)
    and ("sigma_y_km" in pred_events.columns)
    and ("sigma_z_km" in pred_events.columns)
):
    # plot error larger than median
    idx = (
        (pred_events["sigma_x_km"] > pred_events["sigma_x_km"].median())
        | (pred_events["sigma_y_km"] > pred_events["sigma_y_km"].median())
        | (pred_events["sigma_z_km"] > pred_events["sigma_z_km"].median())
    )
    fig, ax = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 0.5]})
    ax[0, 0].errorbar(
        pred_events["longitude"],
        pred_events["latitude"],
        xerr=pred_events["sigma_x_km"] * km2deg,
        yerr=pred_events["sigma_y_km"] * km2deg,
        fmt="none",
        c="k",
        alpha=0.5,
    )

    ax[0, 1].errorbar(
        pred_events[idx]["longitude"],
        pred_events[idx]["latitude"],
        xerr=pred_events[idx]["sigma_x_km"] * km2deg,
        yerr=pred_events[idx]["sigma_y_km"] * km2deg,
        fmt="none",
        c="k",
        alpha=0.5,
    )

    ax[1, 0].errorbar(
        pred_events[idx]["longitude"],
        pred_events[idx]["depth_km"],
        xerr=pred_events[idx]["sigma_x_km"] * km2deg,
        yerr=pred_events[idx]["sigma_z_km"],
        fmt="none",
        c="k",
        alpha=0.5,
    )

    ax[1, 1].errorbar(
        pred_events[idx]["latitude"],
        pred_events[idx]["depth_km"],
        xerr=pred_events[idx]["sigma_y_km"] * km2deg,
        yerr=pred_events[idx]["sigma_z_km"],
        fmt="none",
        c="k",
        alpha=0.5,
    )

    ax[1, 0].set_ylim([15, 0])
    ax[1, 1].set_ylim([15, 0])

    plt.show()

# %% Paper

# using cartopy
s = 2
fig, ax = plt.subplots(
    2,
    2,
    figsize=(8, 8),
    # subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"width_ratios": [1, 0.3], "height_ratios": [1, 0.3]},
)

ax[0, 0].scatter(
    true_events["longitude"],
    true_events["latitude"],
    c="C1",
    s=s,
    # label="True locations",
    # transform=ccrs.PlateCarree()
    rasterized=True,
)
ax[0, 0].scatter(
    pred_events["longitude"],
    pred_events["latitude"],
    c="C0",
    s=s,
    # label="Inverse locations",
    # transform=ccrs.PlateCarree(),
    rasterized=True,
)
for i in true_events.index:
    if i in pred_events.index:
        ax[0, 0].plot(
            [true_events.loc[i, "longitude"], pred_events.loc[i, "longitude"]],
            [true_events.loc[i, "latitude"], pred_events.loc[i, "latitude"]],
            "k--",
            linewidth=1.0,
            alpha=0.5,
            # transform=ccrs.PlateCarree(),
            rasterized=True,
        )
    else:
        ax[0, 0].plot(
            true_events.loc[i, "longitude"],
            true_events.loc[i, "latitude"],
            "rx",
            markersize=s,
            alpha=0.5,
            # transform=ccrs.PlateCarree(),
            rasterized=True,
        )

# ax[0, 0].set_xlabel("Longitude")
ax[0, 0].set_ylabel("Latitude")
ax[0, 0].grid(True, linestyle=":")
ax[0, 0].scatter([], [], c="C1", s=10, label="True location")
ax[0, 0].scatter([], [], c="C0", s=10, label="Inverted location")
ax[0, 0].legend()


ax[1, 0].scatter(true_events["longitude"], true_events["depth_km"], c="C1", s=s, rasterized=True)
ax[1, 0].scatter(pred_events["longitude"], pred_events["depth_km"], c="C0", s=s, rasterized=True)
for i in true_events.index:
    if i in pred_events.index:
        ax[1, 0].plot(
            [true_events.loc[i, "longitude"], pred_events.loc[i, "longitude"]],
            [true_events.loc[i, "depth_km"], pred_events.loc[i, "depth_km"]],
            "k--",
            linewidth=1.0,
            alpha=0.5,
            rasterized=True,
        )
    else:
        ax[1, 0].plot(
            true_events.loc[i, "longitude"],
            true_events.loc[i, "depth_km"],
            "rx",
            markersize=s,
            alpha=0.5,
            rasterized=True,
        )
ax[1, 0].set_ylim([15, 0])
ax[1, 0].set_ylabel("Depth (km)")
ax[1, 0].set_xlabel("Longitude")
ax[1, 0].grid(True, linestyle=":")

ax[0, 1].scatter(true_events["depth_km"], true_events["latitude"], c="C1", s=s, rasterized=True)
ax[0, 1].scatter(pred_events["depth_km"], pred_events["latitude"], c="C0", s=s, rasterized=True)
for i in true_events.index:
    if i in pred_events.index:
        ax[0, 1].plot(
            [true_events.loc[i, "depth_km"], pred_events.loc[i, "depth_km"]],
            [true_events.loc[i, "latitude"], pred_events.loc[i, "latitude"]],
            "k--",
            linewidth=1.0,
            alpha=0.5,
            rasterized=True,
        )
    else:
        ax[0, 1].plot(
            true_events.loc[i, "depth_km"],
            true_events.loc[i, "latitude"],
            "rx",
            markersize=s,
            alpha=0.5,
            rasterized=True,
        )
ax[0, 1].set_xlim([0, 15])
# ax[0, 1].set_ylabel("Latitude")
ax[0, 1].set_xlabel("Depth (km)")
ax[0, 1].grid(True, linestyle=":")

ax[1, 1].set_visible(False)

# plt.tight_layout()
# plt.show()
plt.savefig("stanford_location.png", dpi=300, bbox_inches="tight")
plt.savefig("stanford_location.pdf", dpi=300, bbox_inches="tight")

# %%

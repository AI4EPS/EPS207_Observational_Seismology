from datetime import timedelta

import numpy as np
import obspy
import pandas as pd
import point_cloud_utils as pcu
import pymap3d as pm
from obspy.geodetics.base import gps2dist_azimuth


def point_distance(lat1, lon1, dep1, lat2, lon2, dep2):
    h_e, az, baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    v_e = abs(dep1 - dep2)
    return h_e / 1000.0, v_e


# for location result estimation
# for accuracy error
def accuracy_error(truth, result):
    horizontal_error = np.zeros([len(truth), 1])
    vertical_error = np.zeros([len(truth), 1])
    for i in range(0, len(truth)):
        horizontal_error[i], vertical_error[i] = point_distance(
            truth.iloc[i]["Lat"],
            truth.iloc[i]["Lon"],
            truth.iloc[i]["Dep"],
            result.iloc[i]["Lat"],
            result.iloc[i]["Lon"],
            result.iloc[i]["Dep"],
        )
    return np.mean(horizontal_error), np.mean(vertical_error)


# for precision error
def pairup(threshold=2.0):
    df = pd.DataFrame(columns=["idx1", "idx2", "hori", "vert"])
    sources = np.load("./data/true_location.csv")
    for i in range(0, len(sources)):
        print(i)
        for j in range(i + 1, len(sources)):
            hori_dist, vert_dist = point_distance(
                sources.iloc[i]["Lat"],
                sources.iloc[i]["Lon"],
                sources.iloc[i]["Dep"],
                sources.iloc[j]["Lat"],
                sources.iloc[j]["Lon"],
                sources.iloc[j]["Dep"],
            )
            if (hori_dist < threshold) and (vert_dist < threshold):
                tmp = pd.DataFrame(data={"idx1": i, "idx2": j, "hori": hori_dist, "vert": vert_dist}, index=[0])
                df = pd.concat([df, tmp], ignore_index=True)
    df.to_csv("relative.csv", index=False)
    return df


def precision_error(truth, result):
    # pairup()
    horizontal_precision_error = np.zeros([len(truth), 1])
    vertical_precision_error = np.zeros([len(truth), 1])
    df = pd.read_csv("relative.csv")
    for i in range(0, len(truth)):
        tmp = df.loc[(df["idx1"] == i) | (df["idx2"] == i)]
        l = len(tmp)
        for j in range(0, l):
            idx1 = int(tmp.iloc[j]["idx1"])
            idx2 = int(tmp.iloc[j]["idx2"])
            o_hori = tmp.iloc[j]["hori"]
            o_vert = tmp.iloc[j]["vert"]
            hori, vert = point_distance(
                result.iloc[idx1]["Lat"],
                result.iloc[idx1]["Lon"],
                result.iloc[idx1]["Dep"],
                result.iloc[idx2]["Lat"],
                result.iloc[idx2]["Lon"],
                result.iloc[idx2]["Dep"],
            )
            horizontal_precision_error[i] += (hori - o_hori) ** 2 / l
            vertical_precision_error[i] += (o_vert - vert) ** 2 / l
    return (
        np.mean(np.sqrt(horizontal_precision_error)),
        np.mean(np.sqrt(vertical_precision_error)),
    )


# chamfer distance
def chamfer(truth, result):
    pc1 = truth[["Lat", "Lon", "Dep"]].to_numpy()
    pc2 = result[["Lat", "Lon", "Dep"]].to_numpy()
    o_lat = 35.4
    o_lon = -117.956
    new_pc1 = np.zeros_like(pc1)
    new_pc2 = np.zeros_like(pc2)
    for i in range(0, len(pc1)):
        new_pc1[i, 0], new_pc1[i, 1], new_pc1[i, 2] = pm.geodetic2enu(
            pc1[i, 0], pc1[i, 1], -1000 * pc1[i, 2], o_lat, o_lon, 0
        )
        new_pc2[i, 0], new_pc2[i, 1], new_pc2[i, 2] = pm.geodetic2enu(
            pc2[i, 0], pc2[i, 1], -1000 * pc2[i, 2], o_lat, o_lon, 0
        )
    new_pc2[:, 2] = -new_pc2[:, 2]
    new_pc1[:, 2] = -new_pc1[:, 2]
    return pcu.chamfer_distance(new_pc1 / 1000, new_pc2 / 1000)


# for uncertainty quantification
# calculate how many precentage of ground truths falls within the location result plus 95% confidence interval (Horizontal / Vertical)
def uncertainty_absolute(truth, result):
    count = 0
    for i in range(0, len(truth)):
        hori, dep = point_distance(
            truth.iloc[i]["Lat"],
            truth.iloc[i]["Lon"],
            truth.iloc[i]["Dep"],
            result.iloc[i]["Lat"],
            result.iloc[i]["Lon"],
            result.iloc[i]["Dep"],
        )
        if (hori < result.iloc[i]["unc_x"]) & (
            dep < result.iloc[i]["unc_z"]
        ):  # it has to be the 95% confidence interval
            count += 1
    return count / len(truth)


# we measure how many percentage of distances between event pairs falls within the combined measurement uncertanties
def uncertainty_relative(truth, result):
    count = 0
    count2 = 0
    df = pd.read_csv("relative.csv")
    for i in range(0, len(truth)):
        tmp = df.loc[(df["idx1"] == i) | (df["idx2"] == i)]
        l = len(tmp)
        for j in range(0, l):
            count += 1
            idx1 = int(tmp.iloc[j]["idx1"])
            idx2 = int(tmp.iloc[j]["idx2"])
            o_hori = tmp.iloc[j]["hori"]
            o_vert = tmp.iloc[j]["vert"]
            hori, vert = point_distance(
                result.iloc[idx1]["Lat"],
                result.iloc[idx1]["Lon"],
                result.iloc[idx1]["Dep"],
                result.iloc[idx2]["Lat"],
                result.iloc[idx2]["Lon"],
                result.iloc[idx2]["Dep"],
            )
            if (abs(hori - o_hori) <= np.sqrt(result.iloc[idx1]["unc_x"] ** 2 + result.iloc[idx2]["unc_x"] ** 2)) & (
                abs(vert - o_vert) <= np.sqrt(result.iloc[idx1]["unc_z"] ** 2 + result.iloc[idx2]["unc_z"] ** 2)
            ):
                count2 += 1
    return count2 / count


def calculate_time(row):
    index = row.name
    return obspy.UTCDateTime(row["time"]) + (index + 1) * 60


if __name__ == "__main__":
    # read ground truth
    true_location = pd.read_csv("./true_location.csv")
    # read program output (vary in different form !!! run prepare_result.py first)
    # Weiqaing's ADLoc
    # adloc = pd.read_csv('./share/adloc_events_sst.csv')
    adloc = pd.read_csv("../../results/stanford/ransac_events_sst.csv")
    # adloc = pd.read_csv("../../results/stanford/adloc_events_sst.csv")
    adloc = adloc[["latitude", "longitude", "depth_km"]].rename(
        columns={"latitude": "Lat", "longitude": "Lon", "depth_km": "Dep"}
    )
    print("accuracy error: %.3f %.3f" % accuracy_error(true_location, adloc))
    print("precision error: %.3f %.3f" % precision_error(true_location, adloc))
    print("Chamfer distance: %.3f" % chamfer(true_location, adloc))

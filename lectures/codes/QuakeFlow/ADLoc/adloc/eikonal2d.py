import itertools
import shelve
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from numba import njit
from numba.typed import List

np.random.seed(0)

###################################### Eikonal Solver ######################################

# |\nabla u| = f
# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2


@njit
def calculate_unique_solution(a, b, f, h):
    d = abs(a - b)
    if d >= f * h:
        return min([a, b]) + f * h
    else:
        return (a + b + np.sqrt(2.0 * f * f * h * h - (a - b) ** 2)) / 2.0


@njit
def sweeping_over_I_J_K(u, I, J, f, h):
    m = len(I)
    n = len(J)

    for i in I:
        for j in J:
            if i == 0:
                uxmin = u[i + 1, j]
            elif i == m - 1:
                uxmin = u[i - 1, j]
            else:
                uxmin = min([u[i - 1, j], u[i + 1, j]])

            if j == 0:
                uymin = u[i, j + 1]
            elif j == n - 1:
                uymin = u[i, j - 1]
            else:
                uymin = min([u[i, j - 1], u[i, j + 1]])

            u_new = calculate_unique_solution(uxmin, uymin, f[i, j], h)

            u[i, j] = min([u_new, u[i, j]])

    return u


@njit
def sweeping(u, v, h):
    f = 1.0 / v  ## slowness

    m, n = u.shape
    I = np.arange(m)
    iI = I[::-1]
    J = np.arange(n)
    iJ = J[::-1]

    u = sweeping_over_I_J_K(u, I, J, f, h)
    u = sweeping_over_I_J_K(u, iI, J, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, f, h)

    return u


def eikonal_solve(u, v, h):
    print("Eikonal Solver: ")
    t0 = time.time()
    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, v, h)

        err = np.max(np.abs(u - u_old))
        print(f"Iter {i}, error = {err:.3f}")
        if err < 1e-6:
            break
    print(f"Time: {time.time() - t0:.3f}")
    return u


###################################### Traveltime based on Eikonal Timetable ######################################
@njit
def _get_index(ir, iz, nr, nz, order="C"):
    if order == "C":
        return ir * nz + iz
    elif order == "F":
        return iz * nr + ir
    else:
        raise ValueError("order must be either C or F")


def test_get_index():
    vr, vz = np.meshgrid(np.arange(10), np.arange(20), indexing="ij")
    vr = vr.ravel()
    vz = vz.ravel()
    nr = 10
    nz = 20
    for ir in range(nr):
        for iz in range(nz):
            assert vr[_get_index(ir, iz, nr, nz)] == ir
            assert vz[_get_index(ir, iz, nr, nz)] == iz


@njit
def _interp(time_table, r, z, rgrid0, zgrid0, nr, nz, h):
    ir0 = np.floor((r - rgrid0) / h).clip(0, nr - 2).astype(np.int64)
    iz0 = np.floor((z - zgrid0) / h).clip(0, nz - 2).astype(np.int64)
    ir1 = ir0 + 1
    iz1 = iz0 + 1

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    r0 = ir0 * h + rgrid0
    r1 = ir1 * h + rgrid0
    z0 = iz0 * h + zgrid0
    z1 = iz1 * h + zgrid0

    Q00 = time_table[_get_index(ir0, iz0, nr, nz)]
    Q01 = time_table[_get_index(ir0, iz1, nr, nz)]
    Q10 = time_table[_get_index(ir1, iz0, nr, nz)]
    Q11 = time_table[_get_index(ir1, iz1, nr, nz)]

    t = (
        1.0
        / (r1 - r0)
        / (z1 - z0)
        * (
            Q00 * (r1 - r) * (z1 - z)
            + Q10 * (r - r0) * (z1 - z)
            + Q01 * (r1 - r) * (z - z0)
            + Q11 * (r - r0) * (z - z0)
        )
    )

    return t


# def traveltime(event_loc, station_loc, phase_type, eikonal):
def traveltime(event_index, station_index, phase_type, events, stations, eikonal, vel={0: 6.0, 1: 6.0 / 1.73}):
    """
    event_index: list of event index
    station_index: list of station index
    phase_type: list of phase type
    events: list of event location
    stations: list of station location
    """
    if eikonal is None:
        v = np.array([vel[x] for x in phase_type])
        tt = np.linalg.norm(events[event_index] - stations[station_index], axis=-1, keepdims=False) / v
    else:
        if isinstance(event_index, int):
            event_index = np.array([event_index] * len(phase_type))

        # r = np.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)
        # z = event_loc[:, 2] - station_loc[:, 2]
        x = events[event_index, 0] - stations[station_index, 0]
        y = events[event_index, 1] - stations[station_index, 1]
        z = events[event_index, 2] - stations[station_index, 2]
        r = np.sqrt(x**2 + y**2)

        rgrid0 = eikonal["rgrid"][0]
        zgrid0 = eikonal["zgrid"][0]
        nr = eikonal["nr"]
        nz = eikonal["nz"]
        h = eikonal["h"]

        if isinstance(phase_type, list):
            phase_type = np.array(phase_type)
        # if isinstance(station_index, list):
        #     station_index = np.array(station_index)

        tt = np.zeros(len(phase_type), dtype=np.float32)

        if isinstance(phase_type[0], str):
            p_index = phase_type == "P"
            s_index = phase_type == "S"
        elif isinstance(phase_type[0].item(), int):
            p_index = phase_type == 0
            s_index = phase_type == 1
        else:
            raise ValueError("phase_type must be either P/S or 0/1")

        if len(tt[p_index]) > 0:
            tt[p_index] = _interp(eikonal["up"], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
        if len(tt[s_index]) > 0:
            tt[s_index] = _interp(eikonal["us"], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

    return tt


def calc_traveltime(event_locs, station_locs, phase_type, eikonal):
    """
    event_locs: (num_event, 3) array of event locations
    station_locs: (num_station, 3) array of station locations
    phase_type: (num_event,) array of phase type
    eikonal: dictionary of eikonal solver
    """

    if eikonal is None:
        v = np.array([vel[x] for x in phase_type])
        tt = np.linalg.norm(event_locs - station_locs, axis=-1, keepdims=False) / v
    else:
        x = event_locs[:, 0] - station_locs[:, 0]
        y = event_locs[:, 1] - station_locs[:, 1]
        z = event_locs[:, 2] - station_locs[:, 2]
        r = np.sqrt(x**2 + y**2)

        rgrid0 = eikonal["rgrid"][0]
        zgrid0 = eikonal["zgrid"][0]
        nr = eikonal["nr"]
        nz = eikonal["nz"]
        h = eikonal["h"]

        if isinstance(phase_type, list):
            phase_type = np.array(phase_type)

        tt = np.zeros(len(phase_type), dtype=np.float32)
        if isinstance(phase_type[0], str):
            p_index = phase_type == "P"
            s_index = phase_type == "S"
        elif isinstance(phase_type[0].item(), int):
            p_index = phase_type == 0
            s_index = phase_type == 1
        else:
            raise ValueError("phase_type must be either P/S or 0/1")

        if len(tt[p_index]) > 0:
            tt[p_index] = _interp(eikonal["up"], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
        if len(tt[s_index]) > 0:
            tt[s_index] = _interp(eikonal["us"], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

    return tt


# def grad_traveltime(event_loc, station_loc, phase_type, eikonal):
def grad_traveltime(event_index, station_index, phase_type, events, stations, eikonal):

    if isinstance(event_index, int):
        event_index = np.array([event_index] * len(phase_type))

    # r = np.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)
    # z = event_loc[:, 2] - station_loc[:, 2]
    x = events[event_index, 0] - stations[station_index, 0]
    y = events[event_index, 1] - stations[station_index, 1]
    z = events[event_index, 2] - stations[station_index, 2]
    r = np.sqrt(x**2 + y**2)

    rgrid0 = eikonal["rgrid"][0]
    zgrid0 = eikonal["zgrid"][0]
    nr = eikonal["nr"]
    nz = eikonal["nz"]
    h = eikonal["h"]

    if isinstance(phase_type, list):
        phase_type = np.array(phase_type)
    # if isinstance(station_index, list):
    #     station_index = np.array(station_index)

    dt_dr = np.zeros(len(phase_type))
    dt_dz = np.zeros(len(phase_type))

    # p_index = phase_type == "p"
    # s_index = phase_type == "s"
    p_index = phase_type == 0
    s_index = phase_type == 1

    dt_dr[p_index] = _interp(eikonal["grad_up"][0], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    dt_dr[s_index] = _interp(eikonal["grad_us"][0], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)
    dt_dz[p_index] = _interp(eikonal["grad_up"][1], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    dt_dz[s_index] = _interp(eikonal["grad_us"][1], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

    # dr_dxy = (event_loc[:, :2] - station_loc[:, :2]) / (r[:, np.newaxis] + 1e-6)
    # dt_dxy = dt_dr[:, np.newaxis] * dr_dxy
    # grad = np.column_stack((dt_dxy, dt_dz[:, np.newaxis]))
    dt_dx = dt_dr * x / (r + 1e-6)
    dt_dy = dt_dr * y / (r + 1e-6)

    grad = np.column_stack((dt_dx, dt_dy, dt_dz))

    return grad


def init_eikonal2d(config):

    rlim = [
        0,
        np.sqrt(
            (config["xlim_km"][1] - config["xlim_km"][0]) ** 2 + (config["ylim_km"][1] - config["ylim_km"][0]) ** 2
        ),
    ]
    zlim = config["zlim_km"]
    h = config["h"]

    rgrid = np.arange(rlim[0], rlim[1] + h, h)
    zgrid = np.arange(zlim[0], zlim[1] + h, h)
    nr = len(rgrid)
    nz = len(zgrid)

    vel = config["vel"]
    zz, vp, vs = np.array(vel["Z"]), np.array(vel["P"]), np.array(vel["S"])

    # ##############################################
    # ## make the velocity staircase not linear
    # zz_grid = zz[1:] - h
    # vp_grid = vp[:-1]
    # vs_grid = vs[:-1]
    # zz = np.concatenate([zz, zz_grid])
    # vp = np.concatenate([vp, vp_grid])
    # vs = np.concatenate([vs, vs_grid])
    # idx = np.argsort(zz)
    # zz = zz[idx]
    # vp = vp[idx]
    # vs = vs[idx]
    # ##############################################

    vp1d = np.interp(zgrid, zz, vp)
    vs1d = np.interp(zgrid, zz, vs)
    vp = np.tile(vp1d, (nr, 1))
    vs = np.tile(vs1d, (nr, 1))
    # ir0 = np.floor(config["source_loc"][0] / h).astype(np.int64)
    # iz0 = np.floor(config["source_loc"][1] / h).astype(np.int64)
    ir0 = np.round(0 - rlim[0] / h).astype(np.int64)
    iz0 = np.round(0 - zlim[0] / h).astype(np.int64)
    up = 1000 * np.ones((nr, nz))
    # up[0, 0] = 0.0
    up[ir0, iz0] = 0.0

    up = eikonal_solve(up, vp, h)
    grad_up = np.gradient(up, h, edge_order=2)
    up = up.ravel()
    grad_up = [x.ravel() for x in grad_up]

    us = 1000 * np.ones((nr, nz))
    # us[0, 0] = 0.0
    us[ir0, iz0] = 0.0

    us = eikonal_solve(us, vs, h)
    grad_us = np.gradient(us, h, edge_order=2)
    us = us.ravel()
    grad_us = [x.ravel() for x in grad_us]

    config.update(
        {
            "up": up,
            "us": us,
            "grad_up": grad_up,
            "grad_us": grad_us,
            "rgrid": rgrid,
            "zgrid": zgrid,
            "nr": nr,
            "nz": nz,
            "h": h,
        }
    )

    return config


if __name__ == "__main__":

    import os

    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    nr = 21
    nz = 21
    vel = {"p": 6.0, "s": 6.0 / 1.73}
    vp = np.ones((nr, nz)) * vel["p"]
    vs = np.ones((nr, nz)) * vel["s"]
    h = 1.0

    up = 1000 * np.ones((nr, nz))
    # up[nr//2, nz//2] = 0.0
    up[0, 0] = 0.0

    up = eikonal_solve(up, vp, h)
    grad_up = np.gradient(up, h, edge_order=2)
    up = up.ravel()
    grad_up = [x.ravel() for x in grad_up]

    us = 1000 * np.ones((nr, nz))
    # us[nr//2, nz//2] = 0.0
    us[0, 0] = 0.0

    us = eikonal_solve(us, vs, h)
    grad_us = np.gradient(us, h, edge_order=2)
    us = us.ravel()
    grad_us = [x.ravel() for x in grad_us]

    num_event = 10
    event_loc = np.random.rand(num_event, 3) * np.array([nr * h / np.sqrt(2), nr * h / np.sqrt(2), nz * h])
    event_index = np.arange(num_event)
    print(f"{event_loc = }")
    # event_loc = np.round(event_loc, 0)
    # station_loc = np.random.rand(1, 3) * np.array([nr*h/np.sqrt(2), nr*h/np.sqrt(2), 0])
    station_loc = np.array([0, 0, 0])
    print(f"{station_loc = }")
    station_loc = np.tile(station_loc, (num_event, 1))
    station_index = [0] * num_event
    # phase_type = np.random.choice(["p", "s"], num_event, replace=True)
    # print(f"{list(phase_type) = }")
    phase_type = np.array(["p"] * (num_event // 2) + ["s"] * (num_event - num_event // 2))
    v = np.array([vel[x] for x in phase_type])
    t = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / v
    grad_t = (
        (event_loc - station_loc) / np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=True) / v[:, np.newaxis]
    )
    print(f"True traveltime: {t = }")
    print(f"True grad traveltime: {grad_t = }")

    # tp = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / vel["p"]
    # print(f"{tp = }")
    # ts = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / vel["s"]
    # print(f"{ts = }")

    config = {
        "up": up,
        "us": us,
        "grad_up": grad_up,
        "grad_us": grad_us,
        "rgrid": np.arange(nr) * h,
        "zgrid": np.arange(nz) * h,
        "nr": nr,
        "nz": nz,
        "h": h,
    }
    mapping_int = {"p": 0, "s": 1}
    phase_type = np.array([mapping_int[x] for x in phase_type])
    # t = traveltime(event_loc, station_loc, phase_type, config)
    # grad_t = grad_traveltime(event_loc, station_loc, phase_type, config)
    t = traveltime(event_index, station_index, phase_type, event_loc, station_loc, config)
    grad_t = grad_traveltime(event_index, station_index, phase_type, event_loc, station_loc, config)
    print(f"Computed traveltime: {t = }")
    print(f"Computed grad traveltime: {grad_t = }")

    up = up.reshape((nr, nz))
    plt.figure()
    plt.pcolormesh(up[:, :])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(f"{data_path}/slice_tp_2d.png")

    us = us.reshape((nr, nz))
    plt.figure()
    plt.pcolormesh(us[:, :])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(f"{data_path}/slice_ts_2d.png")

    grad_up = [x.reshape((nr, nz)) for x in grad_up]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cax0 = ax[0].pcolormesh(grad_up[0][:, :])
    fig.colorbar(cax0, ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title("grad_tp_x")
    cax1 = ax[1].pcolormesh(grad_up[1][:, :])
    fig.colorbar(cax1, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title("grad_tp_z")
    plt.savefig(f"{data_path}/slice_grad_tp_2d.png")

    grad_us = [x.reshape((nr, nz)) for x in grad_us]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cax0 = ax[0].pcolormesh(grad_us[0][:, :])
    fig.colorbar(cax0, ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title("grad_ts_x")
    cax1 = ax[1].pcolormesh(grad_us[1][:, :])
    fig.colorbar(cax1, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title("grad_ts_z")
    plt.savefig(f"{data_path}/slice_grad_ts_2d.png")

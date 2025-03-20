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
def calculate_unique_solution(a1, a2, a3, f, h):
    # a1, a2, a3 = np.sort([a1, a2, a3])
    if a1 > a2:
        a1, a2 = a2, a1
    if a2 > a3:
        a2, a3 = a3, a2
    if a1 > a2:
        a1, a2 = a2, a1

    x = a1 + f * h
    if x <= a2:
        return x
    else:
        B = -(a1 + a2)
        C = (a1**2 + a2**2 - f**2 * h**2) / 2
        x1 = (-B + np.sqrt(B**2 - 4 * C)) / 2.0
        x2 = (-B - np.sqrt(B**2 - 4 * C)) / 2.0
        if x1 > a2:
            x = x1
        else:
            x = x2
        if x <= a3:
            return x
        else:
            B = -2.0 * (a1 + a2 + a3) / 3.0
            C = (a1**2 + a2**2 + a3**2 - f**2 * h**2) / 3.0
            x1 = (-B + np.sqrt(B**2 - 4 * C)) / 2.0
            x2 = (-B - np.sqrt(B**2 - 4 * C)) / 2.0
            if x1 > a3:
                x = x1
            else:
                x = x2
        return x


@njit
def sweeping_over_I_J_K(u, I, J, K, f, h):
    m = len(I)
    n = len(J)
    l = len(K)

    for i in I:
        for j in J:
            for k in K:
                if i == 0:
                    uxmin = u[i + 1, j, k]
                elif i == m - 1:
                    uxmin = u[i - 1, j, k]
                else:
                    uxmin = min([u[i - 1, j, k], u[i + 1, j, k]])

                if j == 0:
                    uymin = u[i, j + 1, k]
                elif j == n - 1:
                    uymin = u[i, j - 1, k]
                else:
                    uymin = min([u[i, j - 1, k], u[i, j + 1, k]])

                if k == 0:
                    uzmin = u[i, j, k + 1]
                elif k == l - 1:
                    uzmin = u[i, j, k - 1]
                else:
                    uzmin = min([u[i, j, k - 1], u[i, j, k + 1]])

                u_new = calculate_unique_solution(uxmin, uymin, uzmin, f[i, j, k], h)

                u[i, j, k] = min([u_new, u[i, j, k]])
    return u


@njit
def sweeping(u, v, h):
    f = 1.0 / v  ## slowness

    m, n, l = u.shape
    I = np.arange(m)
    iI = I[::-1]
    J = np.arange(n)
    iJ = J[::-1]
    K = np.arange(l)
    iK = K[::-1]

    u = sweeping_over_I_J_K(u, I, J, K, f, h)
    u = sweeping_over_I_J_K(u, iI, J, K, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, K, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, K, f, h)

    u = sweeping_over_I_J_K(u, I, iJ, iK, f, h)
    u = sweeping_over_I_J_K(u, I, J, iK, f, h)
    u = sweeping_over_I_J_K(u, iI, J, iK, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, iK, f, h)

    return u


def eikonal_solve(u, v, h):
    print("Eikonal Solver: ")
    t0 = time.time()
    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, v, h)

        err = np.max(np.abs(u - u_old))
        print(f"Iteration {i}, Error = {err}")
        if err < 1e-6:
            break
    print(f"Time: {time.time() - t0:.3f}")
    return u


###################################### Traveltime based on Eikonal Timetable ######################################
@njit
def _get_index(ix, iy, iz, nx, ny, nz, order="C"):
    if order == "C":
        return (ix * ny + iy) * nz + iz
    elif order == "F":
        return (iz * ny + iy) * nx + ix
    else:
        raise ValueError("order must be either C or F")


def test_get_index():
    vx, vy, vz = np.meshgrid(np.arange(3), np.arange(4), np.arange(5), indexing="ij")
    vx = vx.ravel()
    vy = vy.ravel()
    vz = vz.ravel()
    nx = 3
    ny = 4
    nz = 5
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                assert vx[_get_index(ix, iy, iz, nx, ny, nz)] == ix
                assert vy[_get_index(ix, iy, iz, nx, ny, nz)] == iy
                assert vz[_get_index(ix, iy, iz, nx, ny, nz)] == iz


@njit
def _interp(time_table, x, y, z, xgrid0, ygrid0, zgrid0, nx, ny, nz, h):
    ix0 = np.floor((x - xgrid0) / h).clip(0, nx - 2).astype(np.int64)
    iy0 = np.floor((y - ygrid0) / h).clip(0, ny - 2).astype(np.int64)
    iz0 = np.floor((z - zgrid0) / h).clip(0, nz - 2).astype(np.int64)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    x0 = ix0 * h + xgrid0
    x1 = ix1 * h + xgrid0
    y0 = iy0 * h + ygrid0
    y1 = iy1 * h + ygrid0
    z0 = iz0 * h + zgrid0
    z1 = iz1 * h + zgrid0

    Q000 = time_table[_get_index(ix0, iy0, iz0, nx, ny, nz)]
    Q001 = time_table[_get_index(ix0, iy0, iz1, nx, ny, nz)]
    Q010 = time_table[_get_index(ix0, iy1, iz0, nx, ny, nz)]
    Q011 = time_table[_get_index(ix0, iy1, iz1, nx, ny, nz)]
    Q100 = time_table[_get_index(ix1, iy0, iz0, nx, ny, nz)]
    Q101 = time_table[_get_index(ix1, iy0, iz1, nx, ny, nz)]
    Q110 = time_table[_get_index(ix1, iy1, iz0, nx, ny, nz)]
    Q111 = time_table[_get_index(ix1, iy1, iz1, nx, ny, nz)]

    t = (
        1.0
        / (x1 - x0)
        / (y1 - y0)
        / (z1 - z0)
        * (
            Q000 * (x1 - x) * (y1 - y) * (z1 - z)
            + Q100 * (x - x0) * (y1 - y) * (z1 - z)
            + Q010 * (x1 - x) * (y - y0) * (z1 - z)
            + Q110 * (x - x0) * (y - y0) * (z1 - z)
            + Q001 * (x1 - x) * (y1 - y) * (z - z0)
            + Q101 * (x - x0) * (y1 - y) * (z - z0)
            + Q011 * (x1 - x) * (y - y0) * (z - z0)
            + Q111 * (x - x0) * (y - y0) * (z - z0)
        )
    )

    return t


# def traveltime(event_loc, station_index, phase_type, eikonal):
def traveltime(event_index, station_index, phase_type, events, stations, eikonal):
    """
    event_index: [N,]
    station_index: [N,]
    phase_type: [N,]
    events: [M, 3]
    stations: [K, 3]
    eikonal: dict
    """
    # x = event_loc[:, 0] - station_loc[:, 0]
    # y = event_loc[:, 1] - station_loc[:, 1]
    # z = event_loc[:, 2] - station_loc[:, 2]
    # x, y, z = event_loc[:, 0], event_loc[:, 1], event_loc[:, 2]
    if isinstance(event_index, int):
        event_index = np.array([event_index] * len(phase_type))
    x, y, z = events[event_index, 0], events[event_index, 1], events[event_index, 2]

    xgrid0 = eikonal["xgrid"][0]
    ygrid0 = eikonal["ygrid"][0]
    zgrid0 = eikonal["zgrid"][0]
    nx = eikonal["nx"]
    ny = eikonal["ny"]
    nz = eikonal["nz"]
    h = eikonal["h"]

    if isinstance(phase_type, list):
        phase_type = np.array(phase_type)
    if isinstance(station_index, list):
        station_index = np.array(station_index)

    tt = np.zeros(len(phase_type), dtype=np.float32)

    p_idx = phase_type == 0
    s_idx = phase_type == 1
    station_unique = np.unique(station_index)
    for i in station_unique:

        sta_idx = station_index == i
        sta_p_idx = sta_idx & p_idx
        sta_s_idx = sta_idx & s_idx

        tt[sta_p_idx] = _interp(
            eikonal["up"][i, :], x[sta_p_idx], y[sta_p_idx], z[sta_p_idx], xgrid0, ygrid0, zgrid0, nx, ny, nz, h
        )
        tt[sta_s_idx] = _interp(
            eikonal["us"][i, :], x[sta_s_idx], y[sta_s_idx], z[sta_s_idx], xgrid0, ygrid0, zgrid0, nx, ny, nz, h
        )

    return tt


def grad_traveltime(event_index, station_index, phase_type, events, stations, eikonal):
    # x = event_loc[:, 0] - station_loc[:, 0]
    # y = event_loc[:, 1] - station_loc[:, 1]
    # z = event_loc[:, 2] - station_loc[:, 2]
    # x, y, z = event_loc[:, 0], event_loc[:, 1], event_loc[:, 2]
    if isinstance(event_index, int):
        event_index = np.array([event_index] * len(phase_type))
    x, y, z = events[event_index, 0], events[event_index, 1], events[event_index, 2]

    xgrid0 = eikonal["xgrid"][0]
    ygrid0 = eikonal["ygrid"][0]
    zgrid0 = eikonal["zgrid"][0]
    nx = eikonal["nx"]
    ny = eikonal["ny"]
    nz = eikonal["nz"]
    h = eikonal["h"]

    if isinstance(phase_type, list):
        phase_type = np.array(phase_type)
    if isinstance(station_index, list):
        station_index = np.array(station_index)

    dt_dx = np.zeros(len(phase_type))
    dt_dy = np.zeros(len(phase_type))
    dt_dz = np.zeros(len(phase_type))

    p_idx = phase_type == 0
    s_idx = phase_type == 1
    station_unique = np.unique(station_index)
    for sta in station_unique:
        sta_idx = station_index == sta
        sta_p_idx = sta_idx & p_idx
        sta_s_idx = sta_idx & s_idx

        dt_dx[sta_p_idx] = _interp(
            eikonal["grad_up"][sta, 0], x[sta_p_idx], y[sta_p_idx], z[sta_p_idx], xgrid0, ygrid0, zgrid0, nx, ny, nz, h
        )
        dt_dx[sta_s_idx] = _interp(
            eikonal["grad_us"][sta, 0], x[sta_s_idx], y[sta_s_idx], z[sta_s_idx], xgrid0, ygrid0, zgrid0, nx, ny, nz, h
        )
        dt_dy[sta_p_idx] = _interp(
            eikonal["grad_up"][sta, 1], x[sta_p_idx], y[sta_p_idx], z[sta_p_idx], xgrid0, ygrid0, zgrid0, nx, ny, nz, h
        )
        dt_dy[sta_s_idx] = _interp(
            eikonal["grad_us"][sta, 1], x[sta_s_idx], y[sta_s_idx], z[sta_s_idx], xgrid0, ygrid0, zgrid0, nx, ny, nz, h
        )
        dt_dz[sta_p_idx] = _interp(
            eikonal["grad_up"][sta, 2], x[sta_p_idx], y[sta_p_idx], z[sta_p_idx], xgrid0, ygrid0, zgrid0, nx, ny, nz, h
        )
        dt_dz[sta_s_idx] = _interp(
            eikonal["grad_us"][sta, 2], x[sta_s_idx], y[sta_s_idx], z[sta_s_idx], xgrid0, ygrid0, zgrid0, nx, ny, nz, h
        )

    grad = np.column_stack((dt_dx, dt_dy, dt_dz))

    return grad


if __name__ == "__main__":

    import os

    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    # test_get_index()

    nx = 21
    ny = 21
    nz = 21
    vel = {"p": 6.0, "s": 6.0 / 1.73}
    vp = np.ones((nx, ny, nz)) * vel["p"]
    vs = np.ones((nx, ny, nz)) * vel["s"]
    h = 1.0

    sta_ix = 10
    sta_iy = 10
    sta_iz = 10
    station_loc = np.array([sta_ix, sta_iy, sta_iz]) * h
    up = 1000 * np.ones((nx, ny, nz))
    up[sta_ix, sta_iy, sta_iz] = 0.0

    up = eikonal_solve(up, vp, h)
    grad_up = np.gradient(up, h, edge_order=2)
    up = up.ravel()
    grad_up = np.array([x.ravel() for x in grad_up])
    up = up[np.newaxis, :]
    grad_up = grad_up[np.newaxis, :, :]

    us = 1000 * np.ones((nx, ny, nz))
    us[sta_ix, sta_iy, sta_iz] = 0.0

    us = eikonal_solve(us, vs, h)
    grad_us = np.gradient(us, h, edge_order=2)
    us = us.ravel()
    grad_us = np.array([x.ravel() for x in grad_us])
    us = us[np.newaxis, :]
    grad_us = grad_us[np.newaxis, :, :]

    num_event = 10
    event_loc = (
        np.random.rand(num_event, 3) * np.array([nx * h / np.sqrt(2), ny * h / np.sqrt(2), nz * h]) + station_loc
    )
    event_index = np.arange(num_event)
    print(f"{event_loc = }")
    print(f"{station_loc = }")
    # station_loc = np.random.rand(1, 3) * np.array([nx*h/np.sqrt(2), ny*h/np.sqrt(2), 0])
    station_loc = np.tile(station_loc, (num_event, 1))
    station_index = [0] * len(event_loc)
    # phase_type = np.random.choice(["p", "s"], num_event, replace=True)
    phase_type = np.array(["p"] * (num_event // 2) + ["s"] * (num_event - num_event // 2))
    v = np.array([vel[x] for x in phase_type])
    t = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / v
    grad_t = (
        (event_loc - station_loc) / np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=True) / v[:, np.newaxis]
    )
    print(f"True traveltime: {t = }")
    print(f"True gradient: {grad_t = }")

    config = {
        "up": up,
        "us": us,
        "grad_up": grad_up,
        "grad_us": grad_us,
        "xgrid": np.arange(nx) * h,
        "ygrid": np.arange(ny) * h,
        "zgrid": np.arange(nz) * h,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "h": h,
    }
    mapping_int = {"p": 0, "s": 1}
    phase_type = np.array([mapping_int[x] for x in phase_type])
    t = traveltime(event_index, station_index, phase_type, event_loc, station_loc, config)
    grad_t = grad_traveltime(event_index, station_index, phase_type, event_loc, station_loc, config)
    print(f"Computed traveltime: {t = }")
    print(f"Computed gradient: {grad_t = }")

    up = np.reshape(up, (nx, ny, nz))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cax0 = ax[0].pcolormesh(up[sta_ix, :, :])
    fig.colorbar(cax0, ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title("tp_x")
    cax1 = ax[1].pcolormesh(up[:, sta_iy, :])
    fig.colorbar(cax1, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title("tp_y")
    cax2 = ax[2].pcolormesh(up[:, :, sta_iz])
    fig.colorbar(cax2, ax=ax[2])
    # ax[2].invert_yaxis()
    ax[2].set_title("tp_z")
    plt.savefig(f"{data_path}/slice_tp_3d.png")

    us = np.reshape(us, (nx, ny, nz))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cax0 = ax[0].pcolormesh(us[sta_ix, :, :])
    fig.colorbar(cax0, ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title("ts_x")
    cax1 = ax[1].pcolormesh(us[:, sta_iy, :])
    fig.colorbar(cax1, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title("ts_y")
    cax2 = ax[2].pcolormesh(us[:, :, sta_iz])
    fig.colorbar(cax2, ax=ax[2])
    ax[2].invert_yaxis()
    ax[2].set_title("ts_z")
    plt.savefig(f"{data_path}/slice_ts_3d.png")

    grad_up = np.reshape(grad_up, (1, 3, nx, ny, nz))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cax0 = ax[0].pcolormesh(grad_up[0, 0][sta_ix, :, :])
    fig.colorbar(cax0, ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title("grad_tp_x")
    cax1 = ax[1].pcolormesh(grad_up[0, 1][sta_ix, :, :])
    fig.colorbar(cax1, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title("grad_tp_y")
    cax2 = ax[2].pcolormesh(grad_up[0, 2][sta_ix, :, :])
    fig.colorbar(cax2, ax=ax[2])
    ax[2].invert_yaxis()
    ax[2].set_title("grad_tp_z")
    plt.savefig(f"{data_path}/slice_grad_tp_3d.png")

    grad_us = np.reshape(grad_us, (1, 3, nx, ny, nz))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cax0 = ax[0].pcolormesh(grad_us[0, 0][sta_ix, :, :])
    fig.colorbar(cax0, ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title("grad_ts_x")
    cax1 = ax[1].pcolormesh(grad_us[0, 1][sta_ix, :, :])
    fig.colorbar(cax1, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title("grad_ts_y")
    cax2 = ax[2].pcolormesh(grad_us[0, 2][sta_ix, :, :])
    fig.colorbar(cax2, ax=ax[2])
    ax[2].invert_yaxis()
    ax[2].set_title("grad_ts_z")
    plt.savefig(f"{data_path}/slice_grad_ts_3d.png")

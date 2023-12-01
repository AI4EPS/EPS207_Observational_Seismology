# %%
import shelve
import time
from pathlib import Path

import numpy as np
import torch
from numba import njit
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

###################################### Eikonal Solver ######################################
# |\nabla u| = f
# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2


@njit
def calculate_unique_solution(a, b, f, h):
    d = abs(a - b)
    if d >= f * h:
        return min([a, b]) + f * h
    else:
        return (a + b + np.sqrt(2 * f * f * h * h - (a - b) ** 2)) / 2


@njit
def sweeping_over_I_J_K(u, I, J, f, h):
    m = len(I)
    n = len(J)

    # for i, j in itertools.product(I, J):
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
    # I = list(range(m))
    # I = List()
    # [I.append(i) for i in range(m)]
    I = np.arange(m)
    iI = I[::-1]
    # J = list(range(n))
    # J = List()
    # [J.append(j) for j in range(n)]
    J = np.arange(n)
    iJ = J[::-1]

    u = sweeping_over_I_J_K(u, I, J, f, h)
    u = sweeping_over_I_J_K(u, iI, J, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, f, h)

    return u


def eikonal_solve(u, f, h):
    print("Eikonal Solver: ")
    t0 = time.time()
    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, f, h)

        err = np.max(np.abs(u - u_old))
        print(f"Iter {i}, error = {err:.3f}")
        if err < 1e-6:
            break
    print(f"Time: {time.time() - t0:.3f}")
    return u


def initialize_eikonal(config):
    path = Path("./eikonal")
    path.mkdir(exist_ok=True)
    rlim = [0, np.sqrt((config["xlim"][1] - config["xlim"][0]) ** 2 + (config["ylim"][1] - config["ylim"][0]) ** 2)]
    zlim = config["zlim"]
    h = config["h"]

    filename = f"timetable_{rlim[0]:.0f}_{rlim[1]:.0f}_{zlim[0]:.0f}_{zlim[1]:.0f}_{h:.3f}"
    if (path / (filename + ".db")).is_file():
        print("Loading precomputed timetable...")
        with shelve.open(str(path / filename)) as db:
            up = db["up"]
            us = db["us"]
            grad_up = db["grad_up"]
            grad_us = db["grad_us"]
            rgrid = db["rgrid"]
            zgrid = db["zgrid"]
            nr = db["nr"]
            nz = db["nz"]
            h = db["h"]
    else:
        edge_grids = 0

        rgrid = np.arange(rlim[0] - edge_grids * h, rlim[1], h)
        zgrid = np.arange(zlim[0] - edge_grids * h, zlim[1], h)
        nr, nz = len(rgrid), len(zgrid)

        vel = config["vel"]
        zz, vp, vs = vel["z"], vel["p"], vel["s"]
        vp1d = np.interp(zgrid, zz, vp)
        vs1d = np.interp(zgrid, zz, vs)
        vp = np.ones((nr, nz)) * vp1d
        vs = np.ones((nr, nz)) * vs1d

        up = 1000.0 * np.ones((nr, nz))
        up[edge_grids, edge_grids] = 0.0
        up = eikonal_solve(up, vp, h)

        grad_up = np.gradient(up, h)

        us = 1000.0 * np.ones((nr, nz))
        us[edge_grids, edge_grids] = 0.0
        us = eikonal_solve(us, vs, h)

        grad_us = np.gradient(us, h)

        with shelve.open(str(path / filename)) as db:
            db["up"] = up
            db["us"] = us
            db["grad_up"] = grad_up
            db["grad_us"] = grad_us
            db["rgrid"] = rgrid
            db["zgrid"] = zgrid
            db["nr"] = nr
            db["nz"] = nz
            db["h"] = h

    up = up.flatten()
    us = us.flatten()
    grad_up = np.array([grad_up[0].flatten(), grad_up[1].flatten()])
    grad_us = np.array([grad_us[0].flatten(), grad_us[1].flatten()])
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


###################################### Travel-time Model ######################################


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
    vr = vr.flatten()
    vz = vz.flatten()
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
    # ir0 = torch.floor((r - rgrid0) / h).clamp(0, nr - 2).to(torch.int64)
    # iz0 = torch.floor((z - zgrid0) / h).clamp(0, nz - 2).to(torch.int64)
    ir1 = ir0 + 1
    iz1 = iz0 + 1

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    x1 = ir0 * h + rgrid0
    x2 = ir1 * h + rgrid0
    y1 = iz0 * h + zgrid0
    y2 = iz1 * h + zgrid0

    Q11 = time_table[_get_index(ir0, iz0, nr, nz)]
    Q12 = time_table[_get_index(ir0, iz1, nr, nz)]
    Q21 = time_table[_get_index(ir1, iz0, nr, nz)]
    Q22 = time_table[_get_index(ir1, iz1, nr, nz)]

    t = (
        1
        / (x2 - x1)
        / (y2 - y1)
        * (
            Q11 * (x2 - r) * (y2 - z)
            + Q21 * (r - x1) * (y2 - z)
            + Q12 * (x2 - r) * (z - y1)
            + Q22 * (r - x1) * (z - y1)
        )
    )

    return t


def grad(x, dim=0, h=1.0):
    if dim == 0:
        prepend = x[0:1, :]
        append = x[-1:, :]
    elif dim == 1:
        prepend = x[:, 0:1]
        append = x[:, -1:]
    grad_left = torch.diff(x, dim=dim, prepend=prepend) / h
    grad_right = torch.diff(x, dim=dim, append=append) / h
    grad = 0.5 * (grad_right + grad_left)
    return grad


class CalcTravelTime(Function):
    @staticmethod
    def forward(r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid0, zgrid0, nr, nz, h):
        # tt = _interp(timetable.numpy(), r.numpy(), z.numpy(), rgrid0, zgrid0, nr, nz, h)
        tt = _interp(timetable, r.numpy(), z.numpy(), rgrid0, zgrid0, nr, nz, h)
        tt = torch.from_numpy(tt)
        # tt = _interp(timetable, r, z, rgrid0, zgrid0, nr, nz, h)
        return tt

    @staticmethod
    def setup_context(ctx, inputs, output):
        r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid0, zgrid0, nr, nz, h = inputs
        ctx.save_for_backward(r, z)
        ctx.timetable = timetable
        ctx.timetable_grad_r = timetable_grad_r
        ctx.timetable_grad_z = timetable_grad_z
        ctx.rgrid0 = rgrid0
        ctx.zgrid0 = zgrid0
        ctx.nr = nr
        ctx.nz = nz
        ctx.h = h

    @staticmethod
    def backward(ctx, grad_output):
        timetable_grad_r = ctx.timetable_grad_r
        timetable_grad_z = ctx.timetable_grad_z
        rgrid0 = ctx.rgrid0
        zgrid0 = ctx.zgrid0
        nr = ctx.nr
        nz = ctx.nz
        h = ctx.h
        r, z = ctx.saved_tensors

        # grad_r = grad_z = grad_rgrid0 = grad_zgrid0 = grad_nr = grad_nz = grad_h = None

        # timetable = timetable.numpy().reshape(nr, nz)
        # timetable = timetable.reshape(nr, nz)
        # grad_time_r, grad_time_z = np.gradient(timetable, h, edge_order=2)
        # grad_time_r = grad_time_r.flatten()
        # grad_time_z = grad_time_z.flatten()

        grad_r = _interp(timetable_grad_r, r.numpy(), z.numpy(), rgrid0, zgrid0, nr, nz, h)
        grad_z = _interp(timetable_grad_z, r.numpy(), z.numpy(), rgrid0, zgrid0, nr, nz, h)

        grad_r = torch.from_numpy(grad_r) * grad_output
        grad_z = torch.from_numpy(grad_z) * grad_output

        # timetable = timetable.view(nr, nz)
        # grad_r = grad(timetable, dim=0, h=h).flatten()
        # grad_z = grad(timetable, dim=1, h=h).flatten()
        # grad_r = _interp(grad_r, r, z, rgrid0, zgrid0, nr, nz, h)
        # grad_z = _interp(grad_z, r, z, rgrid0, zgrid0, nr, nz, h)

        return grad_r, grad_z, None, None, None, None, None, None, None, None


# %%
class TravelTime(nn.Module):
    def __init__(
        self,
        num_event,
        num_station,
        station_loc,
        station_dt=None,
        event_loc=None,
        event_time=None,
        invert_station_dt=False,
        reg_station_dt=0.1,
        velocity={"P": 6.0, "S": 6.0 / 1.73},
        eikonal=None,
        zlim=[0, 30],
        dtype=torch.float32,
    ):
        super().__init__()
        self.num_event = num_event
        self.event_loc = nn.Embedding(num_event, 3)
        self.event_time = nn.Embedding(num_event, 1)
        self.station_loc = nn.Embedding(num_station, 3)
        self.station_dt = nn.Embedding(num_station, 2)  # vp, vs
        self.station_loc.weight = torch.nn.Parameter(torch.tensor(station_loc, dtype=dtype), requires_grad=False)
        if station_dt is not None:
            self.station_dt.weight = torch.nn.Parameter(
                torch.tensor(station_dt, dtype=dtype), requires_grad=invert_station_dt
            )
        else:
            self.station_dt.weight = torch.nn.Parameter(
                torch.zeros(num_station, 2, dtype=dtype), requires_grad=invert_station_dt
            )
        self.velocity = [velocity["P"], velocity["S"]]

        self.reg_station_dt = reg_station_dt
        if event_loc is not None:
            self.event_loc.weight = torch.nn.Parameter(torch.tensor(event_loc, dtype=dtype).contiguous())
        else:
            self.event_loc.weight = torch.nn.Parameter(torch.zeros(num_event, 3, dtype=dtype).contiguous())
        if event_time is not None:
            self.event_time.weight = torch.nn.Parameter(torch.tensor(event_time, dtype=dtype).contiguous())
        else:
            self.event_time.weight = torch.nn.Parameter(torch.zeros(num_event, 1, dtype=dtype).contiguous())

        self.eikonal = eikonal
        self.zlim = zlim

    def calc_time(self, event_loc, station_loc, phase_type, double_difference=False):
        if self.eikonal is None:
            dist = torch.linalg.norm(event_loc - station_loc, axis=-1, keepdim=True)
            # r = torch.linalg.norm(event_loc[..., :2] - station_loc[..., :2], axis=-1, keepdims=True)
            # z = event_loc[..., 2:3] - station_loc[..., 2:3]
            # # z = torch.clamp(z, min=0, max=30)
            # z = torch.clamp(z, min=self.zlim[0], max=self.zlim[1])
            # dist = torch.sqrt(r**2 + z**2)
            tt = dist / self.velocity[phase_type]
            tt = tt.float()
        else:
            if double_difference:
                nb1, ne1, nc1 = event_loc.shape  # batch, event, xyz
                nb2, ne2, nc2 = station_loc.shape
                assert ne1 % ne2 == 0
                assert nb1 == nb2
                station_loc = torch.repeat_interleave(station_loc, ne1 // ne2, dim=1)
                event_loc = event_loc.view(nb1 * ne1, nc1)
                station_loc = station_loc.view(nb1 * ne1, nc2)

            r = torch.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)  ## nb, 2 (pair), 3
            z = event_loc[:, 2] - station_loc[:, 2]
            # z = torch.clamp(z, min=0, max=30)

            timetable = self.eikonal["up"] if phase_type == 0 else self.eikonal["us"]
            timetable_grad = self.eikonal["grad_up"] if phase_type == 0 else self.eikonal["grad_us"]
            timetable_grad_r = timetable_grad[0]
            timetable_grad_z = timetable_grad[1]
            rgrid0 = self.eikonal["rgrid"][0]
            zgrid0 = self.eikonal["zgrid"][0]
            nr = self.eikonal["nr"]
            nz = self.eikonal["nz"]
            h = self.eikonal["h"]
            tt = CalcTravelTime.apply(r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid0, zgrid0, nr, nz, h)

            tt = tt.float()
            if double_difference:
                tt = tt.view(nb1, ne1, 1)
            else:
                tt = tt.unsqueeze(-1)

        return tt

    def forward(
        self,
        station_index,
        event_index=None,
        phase_type=None,
        phase_time=None,
        phase_weight=None,
        double_difference=False,
    ):
        loss = 0.0
        pred_time = torch.zeros(len(phase_type), dtype=torch.float32)
        for type in [0, 1]:
            station_index_ = station_index[phase_type == type]  # (nb,)
            event_index_ = event_index[phase_type == type]  # (nb,)
            phase_weight_ = phase_weight[phase_type == type]  # (nb,)

            station_loc_ = self.station_loc(station_index_)  # (nb, 3)
            station_dt_ = self.station_dt(station_index_)[:, [type]]  # (nb, 1)

            event_loc_ = self.event_loc(event_index_)  # (nb, 3)
            event_time_ = self.event_time(event_index_)  # (nb, 1)

            if double_difference:
                station_loc_ = station_loc_.unsqueeze(1)  # (nb, 1, 3)
                station_dt_ = station_dt_.unsqueeze(1)  # (nb, 1, 1)

            tt_ = self.calc_time(
                event_loc_, station_loc_, type, double_difference=double_difference
            )  # (nb, 1) or (nb, 2) for double_difference

            t_ = event_time_ + tt_ + station_dt_  # (nb, 1) or (nb, 2, 1) for double_difference

            if double_difference:
                t_ = t_[:, 0] - t_[:, 1]  # (nb, 1)

            t_ = t_.squeeze(1)  # (nb, )

            pred_time[phase_type == type] = t_  # (nb, )

            if phase_time is not None:
                phase_time_ = phase_time[phase_type == type]

                if double_difference:
                    loss += torch.sum(F.huber_loss(t_, phase_time_, reduction="none") * phase_weight_)
                else:
                    loss += torch.sum(F.huber_loss(t_, phase_time_, reduction="none") * phase_weight_)
                    loss += self.reg_station_dt * torch.mean(torch.abs(station_dt_)) * len(t_)
                    # loss += self.reg_station_dt * torch.abs(
                    #     torch.sum(station_dt_)
                    # )  ## prevent the trade-off between station_dt and event_time

        return {"phase_time": pred_time, "loss": loss}


class Test(nn.Module):
    def __init__(self, timetable, rgrid0, zgrid0, nr, nz, h):
        super().__init__()
        self.timetable = timetable
        self.rgrid0 = rgrid0
        self.zgrid0 = zgrid0
        self.nr = nr
        self.nz = nz
        self.h = h

    def forward(self, r, z):
        tt = CalcTravelTime.apply(r, z, self.timetable, self.rgrid0, self.zgrid0, self.nr, self.nz, self.h)
        return tt


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    starttime = time.time()
    rgrid0 = 0
    zgrid0 = 0
    nr0 = 20
    nz0 = 20
    h = 1
    r = rgrid0 + h * np.arange(0, nr0)
    z = zgrid0 + h * np.arange(0, nz0)
    r, z = np.meshgrid(r, z, indexing="ij")
    timetable = np.sqrt(r**2 + z**2)
    grad_r, grad_z = np.gradient(timetable, h, edge_order=2)
    # timetable = torch.from_numpy(timetable.flatten())
    timetable = timetable.flatten()

    nr = 1000
    nz = 1000
    r = torch.linspace(0, 20, nr)
    z = torch.linspace(0, 20, nz)
    r, z = torch.meshgrid(r, z, indexing="ij")
    r = r.flatten()
    z = z.flatten()

    test = Test(timetable, rgrid0, zgrid0, nr0, nz0, h)
    r.requires_grad = True
    z.requires_grad = True
    tt = test(r, z)
    tt.backward(torch.ones_like(tt))

    endtime = time.time()
    print(f"Time elapsed: {endtime - starttime} seconds.")
    tt = tt.detach().numpy()

    fig, ax = plt.subplots(3, 2)
    im = ax[0, 0].imshow(tt.reshape(nr, nz))
    fig.colorbar(im, ax=ax[0, 0])
    im = ax[0, 1].imshow(timetable.reshape(nr0, nz0))
    fig.colorbar(im, ax=ax[0, 1])
    im = ax[1, 0].imshow(r.grad.reshape(nr, nz))
    fig.colorbar(im, ax=ax[1, 0])
    im = ax[1, 1].imshow(grad_r.reshape(nr0, nz0))
    fig.colorbar(im, ax=ax[1, 1])
    im = ax[2, 0].imshow(z.grad.reshape(nr, nz))
    fig.colorbar(im, ax=ax[2, 0])
    im = ax[2, 1].imshow(grad_z.reshape(nr0, nz0))
    fig.colorbar(im, ax=ax[2, 1])
    plt.show()


# %%

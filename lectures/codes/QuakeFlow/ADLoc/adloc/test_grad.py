# %%
import numpy as np
import torch
import torch.nn as nn
from eikonal2d import _interp
from torch.autograd import Function


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def clamp(input, min, max):
    return Clamp.apply(input, min, max)


class CalcTravelTime(Function):
    @staticmethod
    def forward(r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid, zgrid, nr, nz, h):
        tt = _interp(timetable, r.numpy(), z.numpy(), rgrid[0], zgrid[0], nr, nz, h)
        tt = torch.from_numpy(tt)
        return tt

    @staticmethod
    def setup_context(ctx, inputs, output):
        r, z, timetable, timetable_grad_r, timetable_grad_z, rgrid, zgrid, nr, nz, h = inputs
        ctx.save_for_backward(r, z)
        ctx.timetable = timetable
        ctx.timetable_grad_r = timetable_grad_r
        ctx.timetable_grad_z = timetable_grad_z
        ctx.rgrid = rgrid
        ctx.zgrid = zgrid
        ctx.nr = nr
        ctx.nz = nz
        ctx.h = h

    @staticmethod
    def backward(ctx, grad_output):
        timetable_grad_r = ctx.timetable_grad_r
        timetable_grad_z = ctx.timetable_grad_z
        rgrid = ctx.rgrid
        zgrid = ctx.zgrid
        nr = ctx.nr
        nz = ctx.nz
        h = ctx.h
        r, z = ctx.saved_tensors

        grad_r = _interp(timetable_grad_r, r.numpy(), z.numpy(), rgrid[0], zgrid[0], nr, nz, h)
        grad_z = _interp(timetable_grad_z, r.numpy(), z.numpy(), rgrid[0], zgrid[0], nr, nz, h)

        grad_r = torch.from_numpy(grad_r) * grad_output
        grad_z = torch.from_numpy(grad_z) * grad_output

        return grad_r, grad_z, None, None, None, None, None, None, None, None


class Test(nn.Module):
    def __init__(self, timetable, rgrid, zgrid, grad_type="auto", timetable_grad_r=None, timetable_grad_z=None):
        super().__init__()
        self.timetable = timetable
        self.rgrid = rgrid
        self.zgrid = zgrid
        self.nr = len(rgrid)
        self.nz = len(zgrid)
        self.h = rgrid[1] - rgrid[0]
        assert self.h == zgrid[1] - zgrid[0]
        self.grad_type = grad_type
        self.timetable_grad_r = timetable_grad_r
        self.timetable_grad_z = timetable_grad_z

    def interp2d(self, time_table, r, z):
        nr = len(self.rgrid)
        nz = len(self.zgrid)
        assert time_table.shape == (nr, nz)

        ir0 = torch.floor((r - self.rgrid[0]) / self.h).clamp(0, nr - 2).long()
        iz0 = torch.floor((z - self.zgrid[0]) / self.h).clamp(0, nz - 2).long()
        ir1 = ir0 + 1
        iz1 = iz0 + 1

        r = (clamp(r, self.rgrid[0], self.rgrid[-1]) - self.rgrid[0]) / self.h
        z = (clamp(z, self.zgrid[0], self.zgrid[-1]) - self.zgrid[0]) / self.h
        # r = (torch.clamp(r, self.rgrid[0], self.rgrid[-1]) - self.rgrid[0]) / self.h
        # z = (torch.clamp(z, self.zgrid[0], self.zgrid[-1]) - self.zgrid[0]) / self.h

        ## https://en.wikipedia.org/wiki/Bilinear_interpolation
        Q00 = time_table[ir0, iz0]
        Q01 = time_table[ir0, iz1]
        Q10 = time_table[ir1, iz0]
        Q11 = time_table[ir1, iz1]

        t = (
            Q00 * (ir1 - r) * (iz1 - z)
            + Q10 * (r - ir0) * (iz1 - z)
            + Q01 * (ir1 - r) * (z - iz0)
            + Q11 * (r - ir0) * (z - iz0)
        )

        return t

    def forward(self, r, z):

        if self.grad_type == "auto":
            print(self.grad_type)
            tt = self.interp2d(self.timetable, r, z)
        else:
            tt = CalcTravelTime.apply(
                r,
                z,
                self.timetable,
                self.timetable_grad_r,
                self.timetable_grad_z,
                self.rgrid,
                self.zgrid,
                self.nr,
                self.nz,
                self.h,
            )

        return tt


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    starttime = time.time()
    rgrid0 = 0
    zgrid0 = 0
    nr0 = 20
    nz0 = 20
    h = 1
    rgrid = rgrid0 + h * np.arange(0, nr0)
    zgrid = zgrid0 + h * np.arange(0, nz0)
    r, z = np.meshgrid(rgrid, zgrid, indexing="ij")
    timetable = np.sqrt(r**2 + z**2)
    grad_r, grad_z = np.gradient(timetable, h, edge_order=2)
    timetable = torch.from_numpy(timetable)
    # timetable = timetable.flatten()
    # grad_r = grad_r.flatten()
    # grad_z = grad_z.flatten()

    nr = 1000
    nz = 1000
    r = torch.linspace(-2, 22, nr)
    z = torch.linspace(-2, 22, nz)
    r, z = torch.meshgrid(r, z, indexing="ij")
    r = r.flatten()
    z = z.flatten()

    test = Test(
        timetable,
        rgrid,
        zgrid,
        grad_type="auto",
        timetable_grad_r=grad_r,
        timetable_grad_z=grad_z,
    )
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

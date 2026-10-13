"""Focal mechanisms and moment tensors for EPS 207 session 3: the programs the field runs, in numpy.

Three programs are here in full, each checked against the original it reproduces
(`tools/topic03/_check_fpfit.py`, `_check_hash.py`, `_check_cap.py`):

  * FPFIT (Reasenberg & Oppenheimer 1985), from the NCSS build of the Fortran: the network's
    published 2016 Geysers mechanism exactly for 96.3 % of 410 earthquakes;
  * HASH v1.2 (Hardebeck & Shearer 2002): identical grid, identical acceptable sets for 22 of the
    25 events of its own example, preferred mechanisms within 1.7 deg;
  * gCAP (Zhu & Helmberger 1996; Zhu & Ben-Zion 2013), double couple: every window's error to the
    precision of gCAP's printout and every time shift identical.

The notebook writes out the part of each that teaches something, in a few lines per program, and
checks its cell against the function here once. Everything else -- the bookkeeping -- lives here.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

import geysers_loc as gl


# ================================================================================================
# FPFIT
# ================================================================================================
# FPFIT in numpy: a reading of Reasenberg & Oppenheimer (1985, USGS OFR 85-739), from the NCSS
# build of the Fortran (`fpfit_v1.5.f`, `search.f`, `hhog.f`, `shrflt.f`, `pexcf.f`, `rdeq3.f`,
# `input.f`, `refrmt.f`) that computes the Northern California Seismic Network's mechanisms.
#
# The whole method is one weighted misfit, searched on a grid:
#
#     F = sum_k |p_obs - p_th| w_obs w_th / sum_k w_obs w_th ,     p = +-0.5 ,  w_th = sqrt|A_k|
#
# `A_k` is the P radiation the trial mechanism sends toward reading k, so a reading near a nodal
# plane counts for little; `w_obs` comes from the pick quality through an assumed error rate r,
# w_obs = 1/sqrt(r(1-r)) - 2, which is zero for a coin flip (r = 0.5). Ties go to the mechanism
# with the largest denominator, the one that keeps the readings farthest from its planes.
#
# The search is coarse (20 deg in strike, dip and rake) then fine (5, 5, 10 deg) around each
# distinct coarse minimum; "distinct" is decided by growing connected clusters of coarse
# mechanisms whose misfit is within the 90 % bound 1.282 sigma_F (Appendix A). The 90 % ranges of
# strike, dip and rake about the fine minimum are the uncertainties the catalogue publishes.
# Checked against the NCSN's published 2016 Geysers mechanisms by `_check_fpfit.py`.

R2D = 180 / np.pi
# the NCSN control file (fpfit_build/example_hinv.inp): hand-pick error rates for qualities 0-3,
# machine picks 0.2 for quality 0 and excluded otherwise
ERATE_HAND = (0.0229, 0.0757, 0.2285, 0.3395)
ERATE_MACHINE = (0.2, 1.0, 1.0, 1.0)


def obs_weight(quality, machine=None):
    """w_obs from the pick quality (0-4) through its error rate (input.f): 1/sqrt(r(1-r)) - 2."""
    q = np.asarray(quality)
    machine = np.zeros(len(q), bool) if machine is None else np.asarray(machine, bool)
    r = np.where(machine, np.take(ERATE_MACHINE, np.minimum(q, 3)), np.take(ERATE_HAND, np.minimum(q, 3)))
    with np.errstate(divide="ignore"):
        w = np.where(r < 0.001, 29.6386, 1 / np.sqrt(r - r * r) - 2)
    return np.where((r >= 0.5) | (q >= 4), 0.0, w)


def ray_coef(azimuth, takeoff):
    """PEXCF: the six products of the ray direction, in FPFIT's (up, south, east) frame."""
    a, i = np.radians(azimuth), np.radians(takeoff)
    u = np.stack([-np.cos(i), -np.sin(i) * np.cos(a), np.sin(i) * np.sin(a)], -1)
    return np.stack([u[:, 0] ** 2, 2 * u[:, 0] * u[:, 1], u[:, 1] ** 2,
                     2 * u[:, 2] * u[:, 0], 2 * u[:, 1] * u[:, 2], u[:, 2] ** 2], -1)


def shrflt(strike, dip, rake):
    """SHRFLT: the moment tensor of a shear fault, in the same frame, for arrays of angles (deg)."""
    s, d, l = (np.radians(np.asarray(x, float)) for x in (strike, dip, rake))
    a11 = -np.cos(s) * np.cos(l) - np.cos(d) * np.sin(l) * np.sin(s)
    a21 = np.sin(s) * np.cos(l) - np.cos(d) * np.sin(l) * np.cos(s)
    a31 = np.sin(d) * np.sin(l)
    a13, a23, a33 = np.sin(s) * np.sin(d), np.cos(s) * np.sin(d), np.cos(d)
    return np.stack([2 * a31 * a33, a11 * a33 + a31 * a13, 2 * a11 * a13,
                     a21 * a33 + a31 * a23, a11 * a23 + a21 * a13, 2 * a21 * a23], -1)


def fpfit_misfit(coef, pobs, wobs, strike, dip, rake):
    """F and its denominator for every trial mechanism (SEARCH's inner loop), readings x trials."""
    prad = coef @ shrflt(strike, dip, rake).T
    wth = np.sqrt(np.abs(prad)) * wobs[:, None]
    bot = wth.sum(0)
    return (np.abs(pobs[:, None] - np.sign(prad) * 0.5) * wth).sum(0) / bot, bot


def sigma_f(wobs):
    """Appendix A: the standard deviation of F from the binomial error rates, sqrt(n)/sum(w)."""
    w = wobs[wobs > 0]
    return np.sqrt(len(w)) / w.sum()


def _best(F, bot, eps=1.2e-7):
    """Index of the smallest F, ties broken by the largest denominator."""
    tie = np.flatnonzero(F - F.min() < eps)
    return tie[np.argmax(bot[tie])]


def refrmt(strike, dip, rake):
    """REFRMT: any (strike, dip, rake) as dip direction, dip in [0, 90], rake in (-180, 180]."""
    s, d, l = int(strike), int(dip), int(rake)
    if d > 90:
        d, s, l = 180 - d, s + 180, -l
    elif d < 0:
        d, s, l = -d, s + 180, l + 180
    dd = (s + 90) % 360
    l = (l % 360) - 360 if (l % 360) > 180 else l % 360
    if d == 90 and dd >= 180:
        l, dd = -l, dd - 180
    return dd, d, l


def auxpln(dd1, da1, sa1):
    """AUXPLN: the auxiliary plane (dip direction, dip, rake) of a plane given the same way."""
    phi1, del1, lam1 = np.radians((dd1 - 90) % 360), np.radians(da1), np.radians(sa1)
    dd2 = np.degrees(np.arctan2(np.cos(lam1) * np.sin(phi1) - np.cos(del1) * np.sin(lam1) * np.cos(phi1),
                                np.cos(lam1) * np.cos(phi1) + np.cos(del1) * np.sin(lam1) * np.sin(phi1)))
    phi2 = np.radians(dd2 - 90)
    if sa1 < 0:
        dd2 -= 180
    dd2 = dd2 + 360 if dd2 < 0 else dd2 - 360 if dd2 > 360 else dd2
    da2 = np.degrees(np.arccos(np.sin(abs(lam1)) * np.sin(del1)))
    x = np.clip(-np.cos(phi2) * np.sin(del1) * np.sin(phi1) + np.sin(phi2) * np.sin(del1) * np.cos(phi1), -1, 1)
    return dd2, da2, np.copysign(np.degrees(np.arccos(x)), sa1)


def _rdiff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def compl(sol, solns, aerr):
    """COMPL: index of an earlier solution that `sol` repeats, directly or as an auxiliary plane."""
    aux1 = auxpln(*sol)
    for i, t in enumerate(solns):
        aux2 = auxpln(*t)
        for a, b in ((sol, t), (t, aux1), (sol, aux2)):
            if abs(a[0] - b[0]) <= aerr and abs(a[1] - b[1]) <= aerr and _rdiff(a[2], b[2]) <= aerr:
                return i
    return None


def hedgehogs(good, start):
    """HHOG: label connected clusters of good coarse cells (26 neighbours, rake wraps round).

    The first cluster is grown from `start`, the coarse best; the rest in the order FPFIT stores
    its good cells, rake slowest and dip fastest. Label 0 is "not good".
    """
    lab = np.zeros(good.shape, int)
    order = [start] + [(j, k, m) for m in range(good.shape[2]) for k in range(good.shape[1])
                       for j in range(good.shape[0]) if good[j, k, m]]
    n = 0
    for seed in order:
        if lab[seed]:
            continue
        n += 1
        stack, lab[seed] = [seed], n
        while stack:
            j, k, m = stack.pop()
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for dm in (-1, 0, 1):
                        jj, kk, mm = j + dj, k + dk, (m + dm) % good.shape[2]
                        if 0 <= jj < good.shape[0] and 0 <= kk < good.shape[1] and good[jj, kk, mm] and not lab[jj, kk, mm]:
                            lab[jj, kk, mm] = n
                            stack.append((jj, kk, mm))
    return lab, n


def _extent(ok):
    """First and last index where `ok` holds, along one axis."""
    i = np.flatnonzero(ok)
    return i.min(), i.max()


def _fine_axis(c, i1, lo, hi, step_c, step_f, half):
    """Fine-search start and count for dip or strike: `half` fine steps either side of the coarse
    best, but no further than one coarse step beyond the cluster's 90 % extent (lo..hi)."""
    x0 = max(c[lo - 1], c[i1] - half * step_f) if lo > 0 else c[i1] - half * step_f
    x1 = min(c[hi + 1], c[i1] + half * step_f) if hi < len(c) - 1 else c[i1] + half * step_f
    return x0 + step_f * np.arange(min(2 * half + 1, int((x1 - x0) / step_f) + 1))


def fpfit(azimuth, takeoff, polarity, quality, machine=None):
    """One event through FPFIT. Returns its solutions in FPFIT's order (the first is the one the
    catalogue prints; the rest are its multiple solutions), each a dict with the dip direction,
    dip and rake as printed, the strike, F, and the 90 % half-ranges of strike, dip and rake."""
    coef = ray_coef(azimuth, takeoff)
    wobs = obs_weight(quality, machine)
    pobs = 0.5 * np.sign(polarity)
    fit90 = 1.282 * sigma_f(wobs)

    # coarse: strike 0-160, dip 10-90, rake -180-160, all by 20 degrees; arrays are (dip, strike, rake)
    cs, cd, cr = np.arange(0, 161, 20.0), np.arange(10, 91, 20.0), np.arange(-180, 161, 20.0)
    D, S, R = np.meshgrid(cd, cs, cr, indexing="ij")
    Fc, Bc = (v.reshape(D.shape) for v in fpfit_misfit(coef, pobs, wobs, S.ravel(), D.ravel(), R.ravel()))
    good = Fc <= Fc.min() + fit90
    k0 = _best(Fc.transpose(2, 1, 0).ravel(), Bc.transpose(2, 1, 0).ravel())   # scan order: rake, strike, dip
    m0, n0, j0 = np.unravel_index(k0, (D.shape[2], D.shape[1], D.shape[0]))
    lab, n = hedgehogs(good, (j0, n0, m0))

    # each cluster's best coarse cell; a cluster that repeats an earlier one (the same double
    # couple found through its other plane) is searched only once, preferring one off the grid edge
    bests, coarse, keep = [], [], []
    for h in range(1, n + 1):
        cells = np.argwhere((lab == h).transpose(2, 1, 0))[:, ::-1]        # this cluster, in scan order
        c = cells[_best(Fc[tuple(cells.T)], Bc[tuple(cells.T)])]
        edge = c[0] in (0, len(cd) - 1) or c[1] in (0, len(cs) - 1) or c[2] in (0, len(cr) - 1)
        sol = refrmt(cs[c[1]], cd[c[0]], cr[c[2]])
        i = compl(sol, coarse, 40.0) if coarse else None
        keep.append(True)
        if i is not None:
            if bests[i][1] and not edge:
                keep[i] = False
            else:
                keep[-1] = False
        bests.append((c, edge)); coarse.append(sol)

    sols, nl = [], len(cr)
    for h in range(1, n + 1):
        if not keep[h - 1]:
            continue
        in_h = lab == h
        j1, n1, m1 = bests[h - 1][0]
        jlo, jhi = _extent(in_h[:, n1, m1])
        nlo, nhi = _extent(in_h[j1, :, m1])
        mlo, mhi = _extent(in_h[j1, n1, :])
        fd = _fine_axis(cd, j1, jlo, jhi, 20, 5, 9)
        fs = _fine_axis(cs, n1, nlo, nhi, 20, 5, 9)
        if mlo == 0 and mhi == nl - 1:                 # the rake extent may wrap round +-180
            ring = in_h[j1, n1, :]
            a = 0
            while a + 1 < nl and ring[a + 1] and a + 1 <= nl - 1: a += 1
            z = nl - 1
            while z - 1 >= 0 and ring[z - 1]: z -= 1
            mlo, mhi = z, a                             # wrapped: mlo > mhi
        if mlo <= mhi:
            r0 = max(cr[m1] - 30, cr[mlo] - 20); r1 = min(cr[m1] + 30, cr[mhi] + 20)
        elif m1 >= mlo:
            r0 = max(cr[m1] - 30, cr[mlo] - 20); r1 = min(cr[m1] + 30, cr[mhi] + 360 + 20)
        else:
            r0 = max(cr[m1] - 30, cr[mlo] - 360 - 20); r1 = min(cr[m1] + 30, cr[mhi] + 20)
        fr = r0 + 10 * np.arange(min(7, int((r1 - r0) / 10) + 1))
        Df, Sf, Rf = np.meshgrid(fd, fs, fr, indexing="ij")
        Ff, Bf = (v.reshape(Df.shape) for v in fpfit_misfit(coef, pobs, wobs, Sf.ravel(), Df.ravel(), Rf.ravel()))
        k = _best(Ff.transpose(2, 1, 0).ravel(), Bf.transpose(2, 1, 0).ravel())
        mf, nf, jf = np.unravel_index(k, (Df.shape[2], Df.shape[1], Df.shape[0]))
        lim = Ff[jf, nf, mf] + fit90
        # FPFIT's reported ranges: half the union of the coarse and fine 90 % extents on each axis
        a, z = _extent(Ff[:, nf, mf] <= lim); dip_rng = (max(cd[jhi], fd[z]) - min(cd[jlo], fd[a])) / 2
        a, z = _extent(Ff[jf, :, mf] <= lim); str_rng = (max(cs[nhi], fs[z]) - min(cs[nlo], fs[a])) / 2
        a, z = _extent(Ff[jf, nf, :] <= lim)
        if mlo <= mhi:
            rk_rng = (max(cr[mhi], fr[z]) - min(cr[mlo], fr[a])) / 2
        elif m1 >= mlo:
            rk_rng = (max(cr[mhi] + 360, fr[z]) - min(cr[mlo], fr[a])) / 2
        else:
            rk_rng = (max(cr[mhi], fr[z]) - min(cr[mlo] - 360, fr[a])) / 2
        dd, dip, rake = refrmt(Sf[jf, nf, mf], Df[jf, nf, mf], Rf[jf, nf, mf])
        if sols and compl((dd, dip, rake), [(t["dipdir"], t["dip"], t["rake"]) for t in sols], 20.0) is not None:
            continue
        sols.append(dict(dipdir=dd, dip=dip, rake=rake, strike=(dd - 90) % 360, misfit=float(Ff[jf, nf, mf]),
                         d_strike=int(np.rint(str_rng)), d_dip=int(np.rint(dip_rng)), d_rake=int(np.rint(rk_rng)),
                         coarse=(cs[n1], cd[j1], cr[m1])))
    return sols


# ================================================================================================
# HASH
# ================================================================================================
# HASH in numpy: a reading of Hardebeck & Shearer (2002), HASH v1.2 (`fmech_subs.f`, `uncert_subs.f`,
# `pol_subs.f`, `hash_driver1.f`).
#
# What HASH adds to FPFIT is one idea, in three steps:
#   1. the take-off angles are not known, so draw `nmc` versions of them from their uncertainty;
#   2. for every draw, keep every mechanism whose polarity misfit is within a tolerance of the best --
#      the tolerance is a guess at how many readings are wrong (`badfrac`) -- and pool the sets;
#   3. report the pooled set's average, the fraction of it within `cangle` of that average (the
#      probability) and its rms spread, and look again at what was thrown out for a second solution.
#
# The grid is not strike/dip/rake. It is a set of rotations close to uniform over orientation: the
# fault normal on a sphere with `360/dang*sin(theta)` points per ring, then the slip rotated in the
# plane. Coordinates are north-east-down, and a take-off angle is HASH's own -- measured from the
# UPWARD vertical after `hash_driver1.f` flips the catalogue's (`qthe = 180 - ith`).
# Checked against the Fortran by `_check_hash.py`.

D2R = np.pi / 180


def to_car(the, phi):
    """HASH's TO_CAR: unit ray from take-off `the` (deg from up) and azimuth `phi` (deg E of N)."""
    the, phi = np.radians(the), np.radians(phi)
    return np.stack([np.sin(the) * np.cos(phi), np.sin(the) * np.sin(phi), -np.cos(the)], -1)


def hash_grid(dang=5.0):
    """The rotations FOCALMC searches: slip b1 and normal b3, one row per trial mechanism."""
    b1, b3 = [], []
    for ithe in range(int(90.1 / dang) + 1):
        th = ithe * dang * D2R
        numphi = int(np.rint(360.0 / dang * np.sin(th)))
        dphi = 360.0 / numphi if numphi else 10000.0
        for iphi in range(int(359.9 / dphi) + 1):
            ph = iphi * dphi * D2R
            n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            a = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
            c = np.cross(n, a)
            for iz in range(int(179.9 / dang) + 1):
                z = iz * dang * D2R
                b3.append(n)
                b1.append(a * np.cos(z) + c * np.sin(z))
    return np.array(b1), np.array(b3)


def focalmc(azi_mc, the_mc, pol, qual, b1, b3, nextra, ntotal):
    """Indices of every grid mechanism acceptable for at least one trial (FOCALMC).

    `azi_mc`, `the_mc` are (npol, nmc); `pol` is +-1; `qual` is 0 impulsive, 1 emergent.
    A mechanism predicts compression where the ray is on the same side of both of its planes.
    """
    good = np.zeros(len(b1), bool)
    for im in range(azi_mc.shape[1]):
        a = to_car(the_mc[:, im], azi_mc[:, im])
        pred = np.where((a @ b1.T) * (a @ b3.T) > 0, 1, -1)          # ray x mechanism
        miss = pred != pol[:, None]
        nmiss, nmiss0 = miss.sum(0), (miss & (qual[:, None] == 0)).sum(0)
        n0min = nmiss0.min()
        n01min = nmiss[nmiss0 == n0min].min()                          # best total at the best impulsive count
        max0 = max(ntotal, n0min + nextra)
        maxall = max(ntotal if n0min == 0 else len(pol), n01min + nextra)
        good |= (nmiss0 <= max0) & (nmiss <= maxall)
    return np.flatnonzero(good)


def fpcoor(n, s):
    """Strike, dip, rake of the plane with normal `n` and slip `s` (FPCOOR, idir=2)."""
    if 1 - abs(n[2]) <= 1e-7:
        phi = np.arctan2(-s[0], s[1])
        return np.degrees(phi) % 360, 0.0, np.degrees(np.arctan2(np.sin(phi) * s[0] - np.cos(phi) * s[1],
                                                                  np.cos(phi) * s[0] + np.sin(phi) * s[1]))
    phi = np.arctan2(-n[0], n[1])
    dl = np.arctan2(np.hypot(n[0], n[1]), -n[2])
    lam = np.arctan2(-s[2] / np.sin(dl), np.cos(phi) * s[0] + np.sin(phi) * s[1])
    if dl > np.pi / 2:
        dl, phi, lam = np.pi - dl, phi + np.pi, -lam
    rake = np.degrees(lam)
    rake = rake + 360 if rake <= -180 else rake - 360 if rake > 180 else rake
    return np.degrees(phi) % 360, np.degrees(dl), rake


def sdr_vectors(strike, dip, rake):
    """Normal and slip of a plane (FPCOOR, idir=1)."""
    f, d, l = np.radians([strike, dip, rake])
    n = np.array([-np.sin(d) * np.sin(f), np.sin(d) * np.cos(f), -np.cos(d)])
    s = np.array([np.cos(l) * np.cos(f) + np.cos(d) * np.sin(l) * np.sin(f),
                  np.cos(l) * np.sin(f) - np.cos(d) * np.sin(l) * np.cos(f), -np.sin(l) * np.sin(d)])
    return n, s


def mech_rot(n1, s1, N, S, B=None):
    """Smallest rotation from (n1, s1) to each row of (N, S), and the rows swapped/flipped to match
    (MECH_ROT, for many mechanisms at once).

    The four candidates are the symmetries of a double couple: the same pair (N, S), both negated,
    or the planes exchanged, (S, N) and (-S, -N). The rotation R between two frames has angle
    arccos((trace R - 1)/2), and trace R is the sum of the dot products of matching axes. With
    b = n x s the null axis, the four traces need only five dot products:
        (N, S): a + b + c    (-N, -S): -a - b + c    (S, N): p + q - c    (-S, -N): -p - q - c
    where a = N.n1, b = S.s1, c = B.b1, p = S.n1, q = N.s1 (the null axis of (S, N) is -B).
    HASH gets the same angle from a difference-vector construction.
    """
    N, S = np.atleast_2d(N), np.atleast_2d(S)
    B = np.cross(N, S) if B is None else np.atleast_2d(B)
    a, b_, c, p, q = N @ n1, S @ s1, B @ np.cross(n1, s1), S @ n1, N @ s1
    tr = np.stack([a + b_ + c, -a - b_ + c, p + q - c, -p - q - c])
    k = np.argmin(-tr, 0)                                   # largest trace = smallest angle
    ang = np.degrees(np.arccos(np.clip((tr[k, np.arange(len(N))] - 1) / 2, -1, 1)))
    swap, flip = (k >= 2)[:, None], np.where(k % 2 == 1, -1.0, 1.0)[:, None]
    return ang, flip * np.where(swap, S, N), flip * np.where(swap, N, S)


def mech_avg(N, S, B=None):
    """Average of a set of mechanisms (MECH_AVG): align each to the first, sum, re-orthogonalise."""
    if len(N) == 1:
        return N[0], S[0]
    _, A, C = mech_rot(N[0], S[0], N, S, B)
    na, sa = A.sum(0), C.sum(0)
    na, sa = na / np.linalg.norm(na), sa / np.linalg.norm(sa)
    _, A, C = mech_rot(na, sa, N, S, B)
    av1 = np.sqrt(np.mean(np.arccos(np.clip(A @ na, -1, 1)) ** 2))
    av2 = np.sqrt(np.mean(np.arccos(np.clip(C @ sa, -1, 1)) ** 2))
    if av1 + av2 < 1e-4:
        return na, sa
    fr = av1 / (av1 + av2)
    for _ in range(100):                               # nudge the two vectors back to 90 degrees
        misf = 90 - np.degrees(np.arccos(np.clip(na @ sa, -1, 1)))
        if abs(misf) <= 0.01:
            break
        t1, t2 = np.radians(misf * fr), np.radians(misf * (1 - fr))
        na, sa = na - sa * np.sin(t1), sa - na * np.sin(t2)
        na, sa = na / np.linalg.norm(na), sa / np.linalg.norm(sa)
    return na, sa


def mech_prob(N, S, cangle=45.0, prob_max=0.1):
    """Preferred mechanism(s), each with its probability and rms spread (MECH_PROB).

    Average the set; drop the member farthest from the average; repeat until every member is
    within `cangle`. The survivors' share is the probability. The dropped members are then
    searched the same way for a second solution, kept if its share reaches `prob_max`.
    Dropping one at a time is HASH's algorithm and is sequential; each pass is two O(N) array
    operations on null axes computed once.
    """
    out, nf = [], len(N)
    B_all = np.cross(N, S)
    rest = np.arange(nf)
    for imult in range(5):
        if len(rest) < 1:
            break
        keep = rest.copy()
        while True:
            na, sa = mech_avg(N[keep], S[keep], B_all[keep])
            rot = mech_rot(na, sa, N[keep], S[keep], B_all[keep])[0]
            if rot.max() <= cangle:
                break
            keep = np.delete(keep, int(np.argmax(rot)))
        prob = len(keep) / nf
        if imult > 0 and prob < prob_max:
            break
        _, A, C = mech_rot(na, sa, N, S, B_all)                          # rms over the WHOLE set
        rms = [np.degrees(np.sqrt(np.mean(np.arccos(np.clip(V @ v, -1, 1)) ** 2))) for V, v in ((A, na), (C, sa))]
        out.append(dict(sdr=fpcoor(na, sa), prob=prob, rms=rms, normal=na, slip=sa))
        rest = np.setdiff1d(rest, keep, assume_unique=True)
    return out


def get_misf(azi, the, pol, qual, strike, dip, rake):
    """FPFIT's weighted misfit and station-distribution ratio of one mechanism (GET_MISF).

    Each reading is weighted by sqrt(|P radiation|) at that mechanism and by 1 or 0.5 for its
    quality; `mfrac` is the weighted fraction on the wrong side, `stdr` the mean radiation weight.
    """
    n, s = sdr_vectors(strike, dip, rake)
    a = to_car(the, azi)
    amp = 2 * (a @ n) * (a @ s)                        # = sin(2 theta) cos(phi) about the fault normal
    wt, wo = np.sqrt(np.abs(amp)), np.where(qual == 0, 1.0, 0.5)
    wrong = np.sign(amp) != pol
    return (wt * wo * wrong).sum() / (wt * wo).sum(), (wt * wo).sum() / wo.sum()


def hash_quality(prob, rms, mfrac, stdr):
    """HASH's A-D grade: probability, mean rms spread, misfit and station distribution."""
    v = np.mean(rms)
    for q, (p, r, m, s) in zip("ABC", ((0.8, 25, 0.15, 0.5), (0.6, 35, 0.2, 0.4), (0.5, 45, 0.3, 0.3))):
        if prob > p and v <= r and mfrac <= m and stdr >= s:
            return q
    return "D"


def hash_mech(azi_mc, the_mc, pol, qual, dang=5.0, badfrac=0.1, cangle=45.0, prob_max=0.1, _grid=None,
              maxout=500, seed=0):
    """One event through HASH: the acceptable set, then the preferred solution(s) and their grades.

    As in HASH, at most `maxout` acceptable mechanisms (a random subset when there are more; HASH's
    own examples use 500) go into the averaging, whose drop-one-at-a-time loop costs the square of
    the set size. `maxout=None` keeps them all, which is how the check against the Fortran runs.
    Returns the solutions and the indices of the full acceptable set."""
    b1, b3 = hash_grid(dang) if _grid is None else _grid
    npol = len(pol)
    ntotal, nextra = max(int(np.rint(npol * badfrac)), 2), max(int(np.rint(npol * badfrac * 0.5)), 2)
    idx = focalmc(azi_mc, the_mc, pol, qual, b1, b3, nextra, ntotal)
    use = idx if maxout is None or len(idx) <= maxout else np.random.default_rng(seed).choice(idx, maxout, replace=False)
    sols = mech_prob(b3[use], b1[use], cangle, prob_max)
    for s in sols:
        s["mfrac"], s["stdr"] = get_misf(azi_mc[:, 0], the_mc[:, 0], pol, qual, *s["sdr"])
        s["quality"] = hash_quality(s["prob"], s["rms"], s["mfrac"], s["stdr"])
    return sols, idx


# ================================================================================================
# CAP
# ================================================================================================
# Cut-and-paste (CAP) in numpy: a reading of gCAP's `cap.c`/`cap_sub.c`/`fft.c` (Zhu & Helmberger
# 1996; Zhu & Ben-Zion 2013), restricted to a double couple. Checked against the compiled gCAP by
# `_check_cap.py`.
#
# What CAP changes relative to a whole-trace least-squares inversion (TDMT) is four things:
#   1. each record is CUT into five windows -- Pnl on R and Z, surface waves on R and Z (P-SV), and
#      SH on T -- tapered, and weighted: Pnl by `w_pnl`, and each by (distance/100 km)^p with p = 1
#      for body waves and 0.5 for surface waves, so neither the big surface waves nor the near
#      stations decide the answer alone;
#   2. each window pair gets its OWN time shift: the lag, within +-`max_shift`, that maximises the
#      cross-correlation of data and synthetic (Pnl R+Z share one; with tie = 0.5 the three surface
#      windows share one), because a wrong velocity model delays body and surface waves differently;
#   3. the source is found by a GRID SEARCH over strike, dip and rake at a fixed Mw, and Mw by a
#      parabolic line search around it, so the misfit surface itself is the uncertainty;
#   4. the misfit is sum over windows of |d - s(lag)|^2 = |d|^2 + |s|^2 - 2 max_lag c(lag).
#
# gCAP band-passes each cut window with SAC's causal Butterworth. Here whole traces may be
# band-passed zero-phase before cutting instead (`band=None` turns it off, which is how the check
# against gCAP runs, with gCAP's filters off too).

GF_KEYS = ("TSS", "TDS", "RSS", "RDS", "RDD", "ZSS", "ZDS", "ZDD", "REX", "ZEX")
COMPS = ("T", "R", "Z", "R", "Z")                 # gCAP's order: SH, SV-r, SV-z, Pnl-r, Pnl-z


def wave_design(gf, azimuth):
    """Per-component design matrices {Z, R, T} (samples x 6), columns [Mxx Myy Mxy Mxz Myz Mzz].
    The same trigonometry as `mt_design` in B.7 (Dreger's tdmt_invc_iso.c)."""
    a = np.radians(azimuth)
    s1, c1, s2, c2 = np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)
    g = {k: np.asarray(gf[k], float) for k in GF_KEYS}
    zero = np.zeros_like(g["TSS"])
    T = np.stack([0.5 * s2 * g["TSS"], -0.5 * s2 * g["TSS"], -c2 * g["TSS"], -s1 * g["TDS"], c1 * g["TDS"], zero], 1)
    R = np.stack([g["RDD"] / 6 - 0.5 * c2 * g["RSS"] + g["REX"] / 3, g["RDD"] / 6 + 0.5 * c2 * g["RSS"] + g["REX"] / 3,
                  -s2 * g["RSS"], c1 * g["RDS"], s1 * g["RDS"], g["REX"] / 3 - g["RDD"] / 3], 1)
    Zc = np.stack([g["ZDD"] / 6 - 0.5 * c2 * g["ZSS"] + g["ZEX"] / 3, g["ZDD"] / 6 + 0.5 * c2 * g["ZSS"] + g["ZEX"] / 3,
                   -s2 * g["ZSS"], c1 * g["ZDS"], s1 * g["ZDS"], g["ZEX"] / 3 - g["ZDD"] / 3], 1)
    return {"Z": Zc, "R": R, "T": T}


def bandpass(x, dt, f1, f2, axis=0):
    """Zero-phase 4-pole Butterworth, applied identically to data and Green's functions."""
    return sosfiltfilt(butter(4, [f1, f2], "bandpass", fs=1 / dt, output="sos"), x, axis=axis)


def travel_times(distance, depth, model=None):
    """First-arrival P and S times (s) through a flat-layered model, gil7 by default, from
    session 2's `_tt` (checked against HypoDD's Fortran to 0.03 ms). Windows are placed on
    these, as gCAP places them on the P and S times written into the Green's functions' headers;
    reading them off the Green's functions with an amplitude threshold fails at short distance,
    where early numerical energy crosses any small threshold seconds before P."""
    m = gl.GIL7 if model is None else model
    return tuple(float(np.atleast_1d(gl.first(*gl.layers(m, ph), depth, float(distance)))[0]) for ph in "PS")


def taper(x):
    """cap_sub.c taper(): a cosine ramp over 30 % of the window at each end, along axis 0."""
    y = np.array(x, float)
    n = len(y); m = int(np.rint(0.3 * n))
    if m:
        t = 0.5 * (1 - np.cos(np.arange(m) * np.pi / m))
        t = t.reshape((m,) + (1,) * (y.ndim - 1))
        y[:m] *= t; y[n - m:] *= t[::-1]
    return y


def cut(x, i0, n):
    """cutTrace(): n samples from index i0, zero outside the trace, then tapered."""
    out = np.zeros((n,) + x.shape[1:])
    a, b = max(i0, 0), min(i0 + n, len(x))
    if b > a:
        out[a - i0:b - i0] = x[a:b]
    return taper(out)


def xcorr(rec, syn, m):
    """crscrl(): c(lag) = sum_t rec(t + lag) syn(t, :) for lag = -m/2 .. m/2, per column of syn."""
    n = len(rec)
    return np.stack([rec[max(0, l):n + min(0, l)] @ syn[max(0, -l):n - max(0, l)]
                     for l in range(-(m // 2), m // 2 + 1)])


def cap_station(gf, data, azimuth, distance, dt, pnl_len=35.0, surf_len=70.0, pnl_shift=2.0, surf_shift=4.0,
            w_pnl=2.0, p_body=1.0, p_surf=0.5, band=None, tp=None, ts=None):
    """One station's five windows, as cap.c builds them with vp = vs = 0 (windows from the Green's
    functions' own P and S times, the data's P pick at the synthetic's). Returns, per window, the
    data energy, the synthetic basis's Gram matrix, and the correlations for every lag and column."""
    if tp is None or ts is None:
        raise ValueError("give the P and S times, e.g. travel_times(distance, depth)")
    G0 = wave_design(gf, azimuth)
    # `band` is one (f1, f2) for all windows, or ((Pnl f1, f2), (surface f1, f2)) as gCAP takes them
    bands = (None, None) if band is None else (band, band) if np.ndim(band[0]) == 0 else tuple(band)
    filt = lambda x, b: x if b is None else bandpass(np.asarray(x, float), dt, *b)
    mm0, mm1 = int(np.rint(pnl_len / dt)), int(np.rint(surf_len / dt))
    t1, t2 = tp - 0.2 * mm0 * dt, ts                      # Pnl
    t3 = ts - 0.3 * mm1 * dt; t4 = t3 + mm1 * dt          # surface waves
    n1, n2 = min(int(np.rint((t2 - t1) / dt)), mm0), min(int(np.rint((t4 - t3) / dt)), mm1)
    t0 = (t3, t4 - n2 * dt, t4 - n2 * dt, t1, t1)
    n = (n2, n2, n2, n1, n1)
    m = (2 * int(np.rint(surf_shift / dt)),) * 3 + (2 * int(np.rint(pnl_shift / dt)),) * 2
    wins = []
    for j, c in enumerate(COMPS):
        w = w_pnl * (distance / 100.0) ** p_body if j >= 3 else (distance / 100.0) ** p_surf
        i0 = int(np.rint(t0[j] / dt))
        b = bands[0] if j >= 3 else bands[1]
        rec = w * cut(filt(data[c], b), i0, n[j])
        syn = w * cut(filt(G0[c], b), i0, n[j])                          # n x 6
        wins.append(dict(rec2=float(rec @ rec), SS=syn.T @ syn, crl=xcorr(rec, syn, m[j]), m=m[j]))
    return wins


def cap_misfit(stations, B, tie=0.5):
    """gCAP's error() for trial tensors B (trials x 6, the design's convention and moment units).
    Returns the total misfit per trial, and the per-window errors and lags (samples) per trial."""
    total, errs, lags = 0.0, [], []
    rows = np.arange(len(B))
    for w in stations:
        c = [B @ x["crl"].T for x in w]                                    # trials x lags
        syn2 = [np.einsum("tk,kj,tj->t", B, x["SS"], B) for x in w]
        # SH and P-SV lags from tie-weighted sums (the same lag when tie = 0.5); Pnl R+Z share one
        l_sh = ((1 - tie) * c[0] + tie * (c[1] + c[2])).argmax(1)
        l_sv = (tie * c[0] + (1 - tie) * (c[1] + c[2])).argmax(1)
        l_pnl = (c[3] + c[4]).argmax(1)
        pick = (l_sh, l_sv, l_sv, l_pnl, l_pnl)
        e = [x["rec2"] + s - 2 * cc[rows, l] for x, s, cc, l in zip(w, syn2, c, pick)]
        total = total + sum(e)
        errs.append(np.stack(e, 1)); lags.append(np.stack([l - x["m"] // 2 for l, x in zip(pick, w)], 1))
    return total, np.stack(errs, 1), np.stack(lags, 1)


def dc_tensors(sdr):
    """Unit double couples in the design's order [Mxx Myy Mxy Mxz Myz Mzz], north-east-down
    (Aki & Richards 4.88; the same numbers gCAP's nmtensor gives with iso = clvd = 0)."""
    s, d, r = np.radians(np.atleast_2d(np.asarray(sdr, float))).T
    mxx = -(np.sin(d) * np.cos(r) * np.sin(2 * s) + np.sin(2 * d) * np.sin(r) * np.sin(s) ** 2)
    myy = np.sin(d) * np.cos(r) * np.sin(2 * s) - np.sin(2 * d) * np.sin(r) * np.cos(s) ** 2
    mxy = np.sin(d) * np.cos(r) * np.cos(2 * s) + 0.5 * np.sin(2 * d) * np.sin(r) * np.sin(2 * s)
    mxz = -(np.cos(d) * np.cos(r) * np.cos(s) + np.cos(2 * d) * np.sin(r) * np.sin(s))
    myz = -(np.cos(d) * np.cos(r) * np.sin(s) - np.cos(2 * d) * np.sin(r) * np.cos(s))
    mzz = np.sin(2 * d) * np.sin(r)
    return np.stack([mxx, myy, mxy, mxz, myz, mzz], 1)


def dc_grid(strike=(0, 350, 10), dip=(10, 90, 10), rake=(-180, 170, 10)):
    """The trial mechanisms, in gCAP's loop order (rake slowest, strike fastest)."""
    ax = [np.arange(a, b + 0.01, c) for a, b, c in (strike, dip, rake)]
    r, d, s = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
    return np.stack([s.ravel(), d.ravel(), r.ravel()], 1)


def moment(mw):
    """gCAP's amp: Mw to moment in units of the Green's functions' 1e20 dyne-cm."""
    return 10 ** (1.5 * mw + 16.1 - 20)


def cap_search(stations, mw0=5.0, dmw=0.1, sdr=None, sign=-1.0, tie=0.5):
    """Grid over strike/dip/rake at each Mw, and error()'s line search in Mw: step by `dmw`
    downhill from `mw0` until the misfit rises, then take the parabola through the last three.
    `sign` maps north-east-down onto the Green's functions' convention (-1 for ours)."""
    sdr = dc_grid() if sdr is None else np.atleast_2d(np.asarray(sdr, float))
    U = sign * dc_tensors(sdr)

    def at(mw):
        tot, e, l = cap_misfit(stations, moment(mw) * U, tie)
        return float(tot.min()), tot, e, l

    e0 = at(mw0)[0]
    mw = mw0
    if dmw > 0:
        dx = dmw
        mw = mw0 + dx
        e2 = at(mw)[0]
        if e2 > e0:                                         # wrong way: turn round, and stand at mw0 again
            dx = -dx
            e0, e2 = e2, e0
            mw = mw0
        e1 = e0
        while e2 < e0:                                      # walk until past the minimum
            e1, e0 = e0, e2
            mw += dx
            e2 = at(mw)[0]
        mw = mw - dx - 0.5 * dx * (e2 - e1) / (e2 + e1 - 2 * e0)
    _, tot, e, l = at(mw)
    k = int(np.argmin(tot))
    return dict(sdr=sdr[k], mw=mw, misfit=float(tot[k]), window_err=e[k], lags=l[k], all=(sdr, tot))

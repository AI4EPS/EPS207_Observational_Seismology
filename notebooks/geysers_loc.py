"""Travel times in a flat-layered model, and the hypocentre solver, for EPS 207 session 2.

This is the plumbing the notebook does not need to show: building a travel-time table and
interpolating in it.  What the session *does* need to show -- the Jacobian, Geiger's
iteration, the loss functions, RANSAC, the bootstrap and the coverage test -- is written out
in notebook cells instead, and `tools/topic02/_check.py` checks the cell versions against
the ones here so the two cannot drift apart.

Verified against the codes that made the catalogue:

  * travel times against HypoDD's Fortran `ttime`/`direct1`/`refract`, 500 rays x 2 models
    x P and S: max 0.03 ms;
  * `locate` against ADLoc's L-BFGS-B on an identical Huber objective, 400 events:
    median 0.32 m;
  * end to end against HYPOINVERSE 1.40 on noise-free synthetic times through its own
    `gey.crh`: the planted hypocentre recovered exactly, HYPOINVERSE 50 m deep;
  * against closed forms where machine accuracy is meaningful: a two-layer head wave to
    0.3 ulp, a homogeneous halfspace to 4e-12 s;
  * `rays` against the VELEST 3.1 Fortran (Kissling et al. 1994) on 16,199 Geysers picks:
    median 0.4 ms, which falls to 18 us once the comparison feeds this code the same rounded
    station coordinates VELEST's own input format carries (4 decimal degrees, integer metres);
    tightening VELEST's 0.02 km ray-tracing tolerance a thousandfold changes nothing, so the
    remaining 18 us is not the ray tracing.  Its layer path lengths reproduce VELEST's separate
    direct-ray and head-wave dT/dv expressions algebraically, and match a finite difference of
    `rays` itself to a part in 10^4.

Coordinates are local Cartesian km: x east, y north, z **down** from sea level, so a station
1.2 km up the ridge sits at z = -1.2 and can be above the earthquake.
"""

import numpy as np

# thickness, Vp, Vs (km, km/s, km/s).  gil7 is the northern California model the Berkeley
# moment tensors are computed in -- the *wrong* model in this session's synthetic world.
GIL7 = np.array([[1.0, 3.20, 1.50], [2.0, 4.50, 2.40], [1.0, 4.80, 2.78],
                 [1.0, 5.51, 3.18], [3.0, 6.21, 3.40], [9.0, 6.21, 3.40],
                 [8.0, 6.89, 3.98], [60.0, 7.83, 4.52]])


def layers(model, phase="P"):
    """Interface depths and the velocity of each layer, for one phase."""
    thick = np.asarray(model, float)[:, 0]
    v = np.asarray(model, float)[:, 1 if phase.upper() == "P" else 2]
    top = np.concatenate([[0.0], np.cumsum(thick)])
    return top, v


def _split(top, v, z0, z1):
    """The layer pieces a vertical path from z0 up to z1 crosses: (thickness, velocity).

    Refuses a path that leaves the model rather than clipping it.  Clipping is silent and
    returns a travel time that is simply too small -- a station 1 km above the model top
    loses that kilometre of path -- which is the same class of mistake as the take-off sign
    bug this module already had.  `_loc.extend` is how a station on topography is handled.
    """
    lo, hi = min(z0, z1), max(z0, z1)
    if lo < top[0] - 1e-9 or hi > top[-1] + 1e-9:
        raise ValueError(f"path {lo:.4f}..{hi:.4f} km leaves the model "
                         f"({top[0]:.4f}..{top[-1]:.4f}); extend it first")
    dz, vv = [], []
    for k in range(len(v)):
        a, b = max(top[k], lo), min(top[k + 1], hi)
        if b - a > 1e-9:            # a zero-thickness sliver at an interface is not a layer:
            dz.append(b - a)        # it contributes no time but would cap the ray parameter
            vv.append(v[k])
    return np.array(dz), np.array(vv)


def direct(top, v, zs, r, zr=0.0, iters=60):
    """The up-going ray from (0, zs) to (r, zr).  Returns time and ray parameter."""
    dz, vv = _split(top, v, zr, zs)
    if r < 0:
        return np.inf, 0.0
    if len(dz) == 0:                    # source and receiver side by side: a horizontal ray
        k = min(int(np.searchsorted(top, zs, side="right")) - 1, len(v) - 1)
        return float(r / v[k]), float(1.0 / v[k])
    pmax = 1.0 / vv.max()

    def X(p):
        eta = np.sqrt(np.maximum(1.0 - (p * vv) ** 2, 1e-12))
        return np.sum(dz * p * vv / eta)

    if r == 0.0:
        p = 0.0
    else:
        lo, hi = 0.0, pmax * (1 - 1e-12)
        for _ in range(iters):                      # X(p) increases from 0 to infinity
            mid = 0.5 * (lo + hi)
            if X(mid) < r:
                lo = mid
            else:
                hi = mid
        p = 0.5 * (lo + hi)
    eta = np.sqrt(np.maximum(1.0 - (p * vv) ** 2, 1e-12))
    return float(np.sum(dz / (vv * eta))), float(p)


def head(top, v, zs, r, zr=0.0):
    """Every head wave that can reach (r, zr) from (0, zs).  Returns (time, p) pairs."""
    out = []
    for n in range(len(v)):
        if top[n] < max(zs, zr):                    # the refractor must lie below both
            continue
        vn = v[n]
        dz_s, vv_s = _split(top, v, zs, top[n])
        dz_r, vv_r = _split(top, v, zr, top[n])
        if (vv_s.size and vv_s.max() >= vn) or (vv_r.size and vv_r.max() >= vn):
            continue                                 # not a refractor for this path
        q_s = np.sqrt(np.maximum(1.0 / vv_s ** 2 - 1.0 / vn ** 2, 0.0))
        q_r = np.sqrt(np.maximum(1.0 / vv_r ** 2 - 1.0 / vn ** 2, 0.0))
        # horizontal distance used up by the two slanted legs
        x_s = np.sum(dz_s * (1.0 / vn) / q_s) if q_s.size else 0.0
        x_r = np.sum(dz_r * (1.0 / vn) / q_r) if q_r.size else 0.0
        if r < x_s + x_r:                            # inside the critical distance
            continue
        t = np.sum(dz_s * q_s) + np.sum(dz_r * q_r) + r / vn
        out.append((float(t), 1.0 / vn))
    return out


def first(top, v, zs, r, zr=0.0):
    """First arrival: time, ray parameter, and whether the ray left the source upward."""
    td, pd = direct(top, v, zs, r, zr)
    best, p, up = td, pd, True
    for t, ph in head(top, v, zs, r, zr):
        if t < best:
            best, p, up = t, ph, False
    return best, p, up


def rays(top, v, zs, r, zr=0.0, iters=70):
    """First-arrival time and the length of the ray inside every layer, for many picks at once.

    `zs`, `r` and `zr` are arrays of source depth, epicentral distance and receiver depth.
    Returns `t` (n,), `L` (n, nlayer) path lengths, `p` (n,) ray parameters and `head` (n,)
    marking the picks whose first arrival is a head wave.

    The lengths are what a velocity inversion needs and a travel-time table cannot give:
    dT/dv_k = -L_k / v_k**2.  Everything is vectorised over picks because the model changes
    at every iteration, so the table would have to be rebuilt each time.
    """
    zs, r = np.atleast_1d(zs).astype(float), np.atleast_1d(r).astype(float)
    zr = np.broadcast_to(np.atleast_1d(np.asarray(zr, float)), zs.shape).copy()
    n, nl = len(r), len(v)
    lo, hi = np.minimum(zs, zr), np.maximum(zs, zr)
    dz = np.maximum(np.minimum(top[None, 1:], hi[:, None])
                    - np.maximum(top[None, :-1], lo[:, None]), 0.0)

    # --- the direct ray: find the ray parameter that covers exactly the distance r
    vmax = np.where(dz > 0, v[None, :], 0.0).max(1)
    plo, phi = np.zeros(n), (1.0 / np.maximum(vmax, 1e-9)) * (1 - 1e-9)

    def reach(p):
        eta = np.sqrt(np.maximum(1.0 - (p[:, None] * v[None, :]) ** 2, 1e-12))
        return (dz * p[:, None] * v[None, :] / eta).sum(1)

    for _ in range(iters):
        pm = 0.5 * (plo + phi)
        near = reach(pm) < r
        plo, phi = np.where(near, pm, plo), np.where(near, phi, pm)
    p = 0.5 * (plo + phi)
    L = dz / np.sqrt(np.maximum(1.0 - (p[:, None] * v[None, :]) ** 2, 1e-12))
    t = (L / v[None, :]).sum(1)
    head = np.zeros(n, bool)

    # --- head waves: one candidate refractor per layer below both ends of the path
    for k in range(1, nl):
        above = np.arange(nl) < k
        if (v[above] >= v[k]).any():
            continue                                  # not a refractor for any path
        ok = (top[k] >= zs) & (top[k] >= zr)          # refractor must lie below both ends
        if not ok.any():
            continue
        cos_i = np.sqrt(np.maximum(1.0 - (v[None, :] / v[k]) ** 2, 1e-12)) * above
        legs = []
        for z0 in (zs, zr):
            d0 = np.maximum(np.minimum(top[None, 1:], top[k]) - np.maximum(top[None, :-1],
                                                                          z0[:, None]), 0.0) * above
            legs.append(np.divide(d0, cos_i, where=cos_i > 0, out=np.zeros_like(d0)))
        slant = legs[0] + legs[1]
        run = r - (slant * v[None, :] / v[k]).sum(1)  # what is left for the refractor itself
        tk = (slant / v[None, :]).sum(1) + run / v[k]
        better = ok & (run >= 0) & (tk < t)
        if better.any():
            Lk = slant.copy()
            Lk[:, k] = run
            t = np.where(better, tk, t)
            L = np.where(better[:, None], Lk, L)
            p = np.where(better, 1.0 / v[k], p)
            head |= better
    return t, L, p, head


# ---------------------------------------------------------------------------------------
# Travel-time curves and the table the locator actually uses.
#
# Bisecting for the ray parameter one distance at a time is wasteful: sweep p instead, and
# every p gives both the distance it reaches and the time it takes.  That IS the travel-time
# curve, and the table is the curve resampled onto a regular distance grid.

def curve(top, v, zs, zr=0.0, n=6001):
    """The direct branch as (distance, time, ray parameter), swept over p.

    The distance a ray covers runs away as p approaches 1/v of its fastest layer, so the
    samples are packed towards that end -- a uniform sweep in p would leave the far
    distances unsampled and the curve would stop short."""
    dz, vv = _split(top, v, zr, zs)
    if len(dz) == 0 or abs(zs - zr) < 1e-6:   # source and receiver side by side
        k = min(int(np.searchsorted(top, zs, side="right")) - 1, len(v) - 1)
        x = np.linspace(0.0, 1.0e4, n)
        return x, x / v[k], np.full(n, 1.0 / v[k])
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    p = (1.0 - (1.0 - u) ** 4) / vv.max()
    eta = np.sqrt(np.maximum(1.0 - np.outer(p, vv) ** 2, 1e-14))
    x = (dz * p[:, None] * vv / eta).sum(1)
    t = (dz / (vv * eta)).sum(1)
    return x, t, p


def table(model, phase, zs_grid, r_grid, zr=0.0):
    """T(zs, r) and the ray parameter of the first arrival, plus whether it leaves upward."""
    top, v = layers(model, phase)
    T = np.full((len(zs_grid), len(r_grid)), np.inf)
    P = np.zeros_like(T)
    # Which arrivals are head waves.  A head wave always leaves the source downward; a
    # direct ray leaves upward only when the source is below the receiver, and with a
    # kilometre of topography it sometimes is not.  The caller signs dT/dz from this.
    HEAD = np.zeros(T.shape, bool)
    for i, zs in enumerate(zs_grid):
        x, t, p = curve(top, v, zs, zr)
        keep = np.isfinite(x) & (np.diff(x, prepend=-1) > 0)      # x must increase to interpolate
        T[i] = np.interp(r_grid, x[keep], t[keep], left=t[keep][0], right=np.inf)
        P[i] = np.interp(r_grid, x[keep], p[keep], left=0.0, right=p[keep][-1])
        # head waves: t = intercept + r / vn, valid beyond the critical distance
        for n_ in range(len(v)):
            if top[n_] < max(zs, zr):
                continue
            vn = v[n_]
            dz_s, vv_s = _split(top, v, zs, top[n_])
            dz_r, vv_r = _split(top, v, zr, top[n_])
            if (vv_s.size and vv_s.max() >= vn) or (vv_r.size and vv_r.max() >= vn):
                continue
            q_s = np.sqrt(np.maximum(1 / vv_s ** 2 - 1 / vn ** 2, 0.0)) if dz_s.size else np.zeros(0)
            q_r = np.sqrt(np.maximum(1 / vv_r ** 2 - 1 / vn ** 2, 0.0)) if dz_r.size else np.zeros(0)
            x_c = ((dz_s / vn / q_s).sum() if q_s.size else 0.0) + ((dz_r / vn / q_r).sum() if q_r.size else 0.0)
            t_h = (dz_s * q_s).sum() + (dz_r * q_r).sum() + r_grid / vn
            better = (r_grid >= x_c) & (t_h < T[i])
            T[i, better], P[i, better], HEAD[i, better] = t_h[better], 1.0 / vn, True
    return T, P, HEAD


R_EARTH = 6371.0


def extend(model, up=2.0):
    """Add `up` km of the surface layer above sea level, so stations can sit on topography."""
    m = np.asarray(model, float).copy()
    return np.vstack([[up, m[0, 1], m[0, 2]], m])


def build(model, phase, zr_grid, zs_grid, r_grid, up=2.0, datum=0.0):
    """T, ray parameter and take-off direction on a (station z, source z, distance) grid.

    `datum` is the elevation, in km above sea level, of the model's own zero depth.  A
    crustal model written from the land surface needs it; one written from sea level does
    not.  Everything outside this function stays in sea-level coordinates.
    """
    zr_grid = np.asarray(zr_grid, float) + datum
    zs_grid = np.asarray(zs_grid, float) + datum
    # The extension must cover the shallowest node asked for, or _split now refuses the
    # path.  It used to be clipped silently, which made the top rows of the table too fast.
    up = max(up, -min(zs_grid.min(), zr_grid.min()) + 0.1)
    m = extend(model, up)
    T = np.empty((len(zr_grid), len(zs_grid), len(r_grid)))
    P = np.empty_like(T)
    HEAD = np.empty(T.shape, bool)
    for k, zr in enumerate(zr_grid):
        T[k], P[k], HEAD[k] = table(m, phase, zs_grid + up, r_grid, zr + up)
    top, v = layers(m, phase)
    return dict(T=T, P=P, HEAD=HEAD, zr=np.asarray(zr_grid, float) - datum,
                zs=np.asarray(zs_grid, float) - datum, r=np.asarray(r_grid, float),
                top=top - up - datum, v=v, datum=float(datum))


def _bracket(grid, x):
    if len(grid) < 2:                   # a one-node axis has no cell to interpolate across
        raise ValueError("travel-time table axes need at least two nodes")
    i = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2)
    w = (x - grid[i]) / (grid[i + 1] - grid[i])
    return i, np.clip(w, 0.0, 1.0)


def predict(tab, zr, zs, r):
    """Travel time and its derivatives (dT/dr, dT/dz_source), trilinearly interpolated."""
    k, wk = _bracket(tab["zr"], zr)
    i, wi = _bracket(tab["zs"], zs)
    j, wj = _bracket(tab["r"], r)

    def lerp(A):
        a = (1 - wi) * A[k, i, j] + wi * A[k, i + 1, j]
        b = (1 - wi) * A[k, i, j + 1] + wi * A[k, i + 1, j + 1]
        c = (1 - wi) * A[k + 1, i, j] + wi * A[k + 1, i + 1, j]
        d = (1 - wi) * A[k + 1, i, j + 1] + wi * A[k + 1, i + 1, j + 1]
        return (1 - wk) * ((1 - wj) * a + wj * b) + wk * ((1 - wj) * c + wj * d)

    t = lerp(tab["T"])
    p = lerp(tab["P"])
    iz = np.clip(np.searchsorted(tab["top"], zs) - 1, 0, len(tab["v"]) - 1)
    eta = np.sqrt(np.maximum(1.0 / tab["v"][iz] ** 2 - p ** 2, 0.0))
    # a head wave leaves the source downward; a direct ray leaves upward only if the
    # source is deeper than the station, which topography does not guarantee
    sgn = np.where(tab["HEAD"][k, i, j], -1.0, np.sign(zs - zr))
    return t, p, sgn * eta


# --------------------------------------------------------------------------- the losses

def rho(res, kind="l2", sigma=0.10):
    """The penalty a residual pays.  This is the function being minimised."""
    a = np.abs(res)
    if kind == "l2":
        return 0.5 * res ** 2
    if kind == "l1":
        return a
    if kind == "huber":                       # quadratic inside sigma, linear outside
        return np.where(a <= sigma, 0.5 * res ** 2, sigma * a - 0.5 * sigma ** 2)
    raise ValueError(kind)


def weights(res, kind="l2", sigma=0.10):
    """IRLS weights.  Minimising sum rho(r) is weighted least squares with w = rho'(r)/r."""
    a = np.maximum(np.abs(res), 1e-6)
    if kind == "l2":
        return np.ones_like(res)
    if kind == "l1":
        return 1.0 / a
    if kind == "huber":
        return np.where(a <= sigma, 1.0, sigma / a)
    raise ValueError(kind)


def grid_search(sta_xyz, t_obs, phase, tabs, loss="l2", sigma=0.10, pick_w=None,
                terms=None, half=12.0, step=1.0, zstep=0.5):
    """The derivative-free reference: score a coarse grid and keep the best node.

    The origin time is not searched -- at every trial position the best t0 is just the
    middle of the leftover residuals, so three unknowns are searched and the fourth solved.
    The misfit surface has more than one low spot, which is why a locator that starts from
    a guess and only ever goes downhill lands in the wrong one on about one event in ten.
    """
    xs, ys, zs_ = sta_xyz[:, 0], sta_xyz[:, 1], sta_xyz[:, 2]
    w0 = np.ones(len(t_obs)) if pick_w is None else np.asarray(pick_w, float)
    dt = np.zeros(len(t_obs)) if terms is None else np.asarray(terms, float)
    zs_grid = tabs[0]["zs"]
    gx = np.arange(xs.mean() - half, xs.mean() + half + 1e-9, step)
    gy = np.arange(ys.mean() - half, ys.mean() + half + 1e-9, step)
    gz = np.arange(max(zs_grid.min(), -1.0), min(zs_grid.max(), 10.0) + 1e-9, zstep)
    best, best_m = np.inf, None
    for z in gz:
        for x in gx:
            r = np.maximum(np.hypot(x - xs, np.subtract.outer(gy, ys)), 1e-3)   # (ny, npick)
            t = np.empty_like(r)
            for ph in (0, 1):
                s = phase == ph
                if s.any():
                    t[:, s] = predict(tabs[ph], np.broadcast_to(zs_[s], (len(gy), s.sum())),
                                      z, r[:, s])[0]
            d = t_obs - (t + dt)
            t0 = np.median(d, axis=1, keepdims=True)                 # the middle of what is left
            f = np.sum(w0 * rho(d - t0, loss, sigma), axis=1)
            k = int(np.argmin(f))
            if f[k] < best:
                best, best_m = float(f[k]), np.array([x, gy[k], z, float(t0[k, 0])])
    return best_m


def locate(sta_xyz, t_obs, phase, tabs, x0, loss="l2", sigma=0.10,
           pick_w=None, terms=None, iters=60, zmin=None, zmax=None):
    """Damped Gauss-Newton (Levenberg-Marquardt) for (x, y, z, t0).

    Each step is accepted only if it lowers the penalty above; if it does not, the damping
    is raised and the step retried.  Testing the damping against the weighted least-squares
    surrogate instead of the real penalty leaves the search at a worse point on about one
    event in eight here, which is how this was found.
    """
    # The search may not leave the table: outside it the interpolation is flat, the
    # gradient vanishes, and the iteration parks against the wall it walked into.
    zmin = float(tabs[0]["zs"].min()) if zmin is None else max(zmin, float(tabs[0]["zs"].min()))
    zmax = float(tabs[0]["zs"].max()) if zmax is None else min(zmax, float(tabs[0]["zs"].max()))
    xs, ys, zs_ = sta_xyz[:, 0], sta_xyz[:, 1], sta_xyz[:, 2]
    w0 = np.ones(len(t_obs)) if pick_w is None else np.asarray(pick_w, float)
    dt_sta = np.zeros(len(t_obs)) if terms is None else np.asarray(terms, float)

    def forward(m):
        dx, dy = m[0] - xs, m[1] - ys
        r = np.maximum(np.hypot(dx, dy), 1e-3)
        t = np.empty_like(r); p = np.empty_like(r); dz = np.empty_like(r)
        for ph in (0, 1):
            s = phase == ph
            if s.any():
                t[s], p[s], dz[s] = predict(tabs[ph], zs_[s], m[2], r[s])
        res = t_obs - (t + dt_sta + m[3])
        J = np.column_stack([p * dx / r, p * dy / r, dz, np.ones_like(r)])
        return res, J

    m = np.array(x0, float)
    m[2] = np.clip(m[2], zmin, zmax)
    res, J = forward(m)
    f = float(np.sum(w0 * rho(res, loss, sigma)))
    lam = 1e-3
    for _ in range(iters):
        w = w0 * weights(res, loss, sigma)
        A = J.T @ (w[:, None] * J)
        b = J.T @ (w * res)
        d = np.diag(A).copy() + 1e-12
        stepped = False
        for _ in range(12):                    # raise the damping until the step helps
            try:
                step = np.linalg.solve(A + lam * np.diag(d), b)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            m_try = m + step
            m_try[2] = np.clip(m_try[2], zmin, zmax)
            res_try, J_try = forward(m_try)
            f_try = float(np.sum(w0 * rho(res_try, loss, sigma)))
            if f_try < f:
                m, res, J, f = m_try, res_try, J_try, f_try
                lam = max(lam * 0.3, 1e-12)
                stepped = True
                break
            lam *= 10.0
        if not stepped or np.max(np.abs(step[:3])) < 1e-5:
            break
    return dict(m=m, res=res, J=J, loss=f, rms=float(np.sqrt(np.mean(res ** 2))), n=len(res))


def residuals(sta_xyz, t_obs, phase, tabs, m, terms=None):
    """Observed minus predicted at the model m = (x, y, z, t0), for every pick."""
    xs, ys, zs_ = sta_xyz[:, 0], sta_xyz[:, 1], sta_xyz[:, 2]
    dt = np.zeros(len(t_obs)) if terms is None else np.asarray(terms, float)
    r = np.maximum(np.hypot(m[0] - xs, m[1] - ys), 1e-3)
    t = np.empty_like(r)
    for ph in (0, 1):
        s = phase == ph
        if s.any():
            t[s] = predict(tabs[ph], zs_[s], m[2], r[s])[0]
    return t_obs - (t + dt + m[3])


def ransac(sta_xyz, t_obs, phase, tabs, x0, thresh=0.15, trials=60, minimal=6,
           score=None, loss="l2", sigma=0.10, seed=0, **kw):
    """Fit a minimal subset, count who agrees with it, keep the largest agreeing set, refit.

    RANSAC does not down-weight a bad pick, it *excludes* it: an explicit inlier/outlier
    model, where L1 and Huber are heavy-tailed models of one population.  ADLoc's change is
    one line -- draw the subsets with probability proportional to the picker's confidence,
    so a pick the network was unsure of is less likely to be the one defining the trial.
    """
    rng = np.random.default_rng(seed)
    n = len(t_obs)
    prob = None
    if score is not None:
        s = np.maximum(np.asarray(score, float), 1e-6)
        prob = s / s.sum()
    best_in, best = np.zeros(n, bool), None
    for _ in range(trials):
        k = rng.choice(n, size=min(minimal, n), replace=False, p=prob)
        try:
            trial = locate(sta_xyz[k], t_obs[k], phase[k], tabs, x0,
                           loss=loss, sigma=sigma, **kw)
        except np.linalg.LinAlgError:
            continue
        r = residuals(sta_xyz, t_obs, phase, tabs, trial["m"])
        inl = np.abs(r - np.median(r)) <= thresh
        if inl.sum() > best_in.sum():
            best_in = inl
    if best_in.sum() < 4:                       # nothing agreed; fall back to all the picks
        best_in = np.ones(n, bool)
    best = locate(sta_xyz[best_in], t_obs[best_in], phase[best_in], tabs, x0,
                  loss=loss, sigma=sigma, **kw)
    best["inliers"] = best_in
    return best


def ellipse(out, sigma=None):
    """The formal covariance sigma^2 (J^T J)^-1, with sigma from the residuals if not given."""
    J, res, n = out["J"], out["res"], out["n"]
    if sigma is None:
        sigma = np.sqrt(np.sum(res ** 2) / max(n - 4, 1))
    try:
        C = sigma ** 2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        C = np.full((4, 4), np.nan)
    return C


# ------------------------------------------------------------------- the Geysers, 2016
# The NCSN's own Geysers model, as HYPOINVERSE 1.40 ships it in `gey.crh`.  This is the model
# the routine catalogue was made with, and it is Eberhart-Phillips & Oppenheimer (1984).
VPVS = 1.78
_v = np.array([4.43, 5.12, 5.47, 5.58, 5.62, 5.86, 7.90])
_top = np.array([0.00, 1.50, 3.00, 4.25, 6.00, 8.00, 21.00])
GEY = np.column_stack([np.append(np.diff(_top), 60.0), _v, _v / VPVS])

LAT0, LON0 = 38.80, -122.80          # the frame's origin: x east, y north, in km

# The stations that recorded the 2016 picks, from NCEDC FDSN.  Embedded rather than fetched:
# a lecture should not depend on a web service answering.
_STATIONS = [
    ("BG", "ACR", 38.83670, -122.76028, 803.0),
    ("BG", "AL1", 38.83822, -122.88345, 704.0),
    ("BG", "AL2", 38.81601, -122.89799, 657.0),
    ("BG", "AL3", 38.82755, -122.85687, 781.0),
    ("BG", "AL4", 38.83859, -122.83549, 661.0),
    ("BG", "AL5", 38.84000, -122.86862, 593.0),
    ("BG", "AL6", 38.79995, -122.86134, 749.0),
    ("BG", "BRP", 38.85486, -122.79714, 905.0),
    ("BG", "BUC", 38.82312, -122.83503, 888.0),
    ("BG", "CLV", 38.83841, -122.79034, 989.0),
    ("BG", "DEB", 38.76395, -122.68070, 533.0),
    ("BG", "DES", 38.76818, -122.69928, 650.0),
    ("BG", "DRH", 38.82360, -122.95270, 311.0),
    ("BG", "DRK", 38.78833, -122.80307, 757.0),
    ("BG", "DVB", 38.76410, -122.68104, 682.0),
    ("BG", "DXR", 38.82304, -122.77203, 1021.0),
    ("BG", "EPR", 38.74711, -122.69336, 890.0),
    ("BG", "ESM", 38.77588, -122.70258, 454.0),
    ("BG", "FFA", 38.79221, -122.71646, 789.0),
    ("BG", "FNF", 38.77072, -122.76575, 870.0),
    ("BG", "FUM", 38.79318, -122.78792, 673.0),
    ("BG", "HBW", 38.85853, -122.87569, 985.0),
    ("BG", "HER", 38.84476, -122.91380, 668.0),
    ("BG", "HVC", 38.84260, -122.77640, 779.0),
    ("BG", "INJ", 38.80501, -122.79362, 947.0),
    ("BG", "JKB", 38.80063, -122.75972, 1030.0),
    ("BG", "JKR", 38.80062, -122.75971, 1067.0),
    ("BG", "LCK", 38.81954, -122.74174, 1166.0),
    ("BG", "MCL", 38.85466, -122.82335, 961.0),
    ("BG", "MNS", 38.79221, -122.71646, 789.0),
    ("BG", "NEG", 38.83047, -122.76991, 922.0),
    ("BG", "PFR", 38.75272, -122.74453, 1020.0),
    ("BG", "PSB", 38.75059, -122.69115, 697.0),
    ("BG", "PSR", 38.75068, -122.69106, 855.0),
    ("BG", "RGP", 38.87833, -122.81127, 799.0),
    ("BG", "RTB", 38.77890, -122.68780, 276.0),
    ("BG", "SB4", 38.80887, -122.82918, 327.0),
    ("BG", "SQK", 38.82339, -122.81007, 639.0),
    ("BG", "SRB", 38.74073, -122.71156, 912.0),
    ("BG", "SSB", 38.76159, -122.72486, 820.0),
    ("BG", "SSR", 38.74019, -122.71122, 1076.0),
    ("BG", "STY", 38.81326, -122.78069, 1112.0),
    ("BG", "TCH", 38.78324, -122.73673, 951.0),
    ("BG", "U14", 38.78507, -122.77234, 662.0),
    ("BG", "US1", 38.79334, -122.83503, 761.0),
    ("BG", "US3", 38.80489, -122.83984, 502.0),
    ("BG", "WRK", 38.76316, -122.72390, 1008.0),
    ("BK", "PWOD", 38.58075, -122.70187, 438.1),
    ("BK", "SKGS", 38.68900, -123.07842, 640.4),
    ("CE", "68035", 38.91220, -122.76570, 710.0),
    ("NC", "GAC", 38.87274, -122.86292, 968.0),
    ("NC", "GAX", 38.71068, -122.75724, 367.0),
    ("NC", "GAXB", 38.71073, -122.75733, 386.0),
    ("NC", "GBG", 38.81436, -122.68263, 1103.0),
    ("NC", "GBO", 38.82433, -122.84283, 879.0),
    ("NC", "GCM", 38.80583, -122.75517, 1286.0),
    ("NC", "GCPN", 38.77712, -122.74473, 832.0),
    ("NC", "GCR", 38.77383, -122.71650, 693.0),
    ("NC", "GCV", 38.76900, -123.01483, 125.0),
    ("NC", "GCVB", 38.76927, -123.01487, 151.0),
    ("NC", "GDX", 38.80777, -122.79474, 910.0),
    ("NC", "GDXB", 38.80797, -122.79530, 939.0),
    ("NC", "GFT", 38.79300, -122.83400, 755.0),
    ("NC", "GGL", 38.89667, -122.77634, 893.0),
    ("NC", "GGP", 38.76427, -122.84528, 1023.0),
    ("NC", "GGPB", 38.76440, -122.84518, 1047.0),
    ("NC", "GGPC", 38.76877, -122.84650, 918.0),
    ("NC", "GMC", 38.79287, -123.12791, 421.0),
    ("NC", "GMK", 38.96955, -122.78773, 904.0),
    ("NC", "GMM", 38.83816, -122.79884, 963.0),
    ("NC", "GPM", 38.84654, -122.94710, 736.0),
    ("NC", "GRT", 38.93867, -122.67110, 590.0),
    ("NC", "GSG", 38.86634, -122.71088, 1057.0),
    ("NC", "GSM", 38.76917, -122.78117, 1017.0),
    ("NC", "GSS", 38.70205, -123.01428, 274.0),
    ("NC", "GSX", 38.84980, -122.52254, 492.0),
    ("NC", "GTK", 38.97486, -122.76575, 1308.7),
    ("NC", "IRGGP", 38.76470, -122.84519, 1052.0),
    ("NC", "N005", 38.53406, -122.78836, 33.0),
    ("NC", "NHB", 38.58933, -122.90900, 165.0),
    ("NC", "NHE", 38.67033, -122.63384, 1200.0),
    ("NC", "NHS", 38.65652, -122.61442, 1213.0),
    ("NC", "NLB", 38.74270, -122.55306, 371.0),
    ("NC", "NMC", 38.59095, -122.91286, 119.0),
    ("NC", "NMH", 38.66937, -122.63321, 1288.0),
    ("NC", "NMW", 38.55035, -122.72324, 119.0),
    ("NP", "1216", 38.71616, -123.00227, 82.0),
    ("NP", "1744", 38.61618, -122.87231, 33.0),
    ("NP", "ADS2", 38.77446, -122.69974, 430.0),
    ("NP", "ADSP", 38.77421, -122.70561, 470.0),
    ("NP", "COB", 38.83870, -122.75280, 777.0),
]


def xy(lon, lat):
    """Local Cartesian km, x east and y north, about (LAT0, LON0)."""
    return ((np.asarray(lon) - LON0) * 111.195 * np.cos(np.radians(LAT0)),
            (np.asarray(lat) - LAT0) * 111.195)


def stations_2016(n=None, within=12.0, seed=0):
    """The real 2016 Geysers stations, in the local frame.  `z` is down from sea level, so a
    station 1.2 km up the ridge sits at z = -1.2 and can be ABOVE the earthquake."""
    import pandas as pd
    st = pd.DataFrame(_STATIONS, columns=["net", "sta", "lat", "lon", "elev"])
    st["x"], st["y"] = xy(st.lon, st.lat)
    st["z"] = -st.elev / 1000.0
    st = st[np.hypot(st.x, st.y) <= within].reset_index(drop=True)
    if n is not None and n < len(st):
        st = st.sample(n, random_state=seed)
    return st.sort_values("sta").reset_index(drop=True)


def observe(truth, st, tabs, noise=0.02, bad=0.0, bad_scale=0.35, s_frac=0.0, drop=None, seed=0):
    """The arrival times a planted hypocentre would produce at these stations.

    `noise` is the analyst's reading error, Gaussian.  `bad` is the fraction of picks replaced
    by a gross error -- a pick on the wrong phase, the wrong event, or on noise.  Returning
    `is_bad` lets a robust method be scored on what it actually recovered.
    """
    rng = np.random.default_rng(seed)
    x, y, z, t0 = truth
    keep = np.ones(len(st), bool)
    if drop is not None:
        keep[np.asarray(drop)] = False
    S = st.loc[keep, ["x", "y", "z"]].to_numpy()
    sta, phase = S, np.zeros(len(S), int)
    if s_frac > 0:
        k = rng.random(len(S)) < s_frac
        if k.any():
            sta = np.vstack([S, S[k]])
            phase = np.concatenate([phase, np.ones(int(k.sum()), int)])
    r = np.maximum(np.hypot(x - sta[:, 0], y - sta[:, 1]), 1e-3)
    t = np.empty(len(sta))
    for ph in (0, 1):
        s = phase == ph
        if s.any():
            t[s] = predict(tabs[ph], sta[s, 2], z, r[s])[0]
    t_obs = t + t0 + rng.normal(0.0, noise, len(t))
    is_bad = rng.random(len(t)) < bad
    if is_bad.any():
        # A gross error, either sign, at least 0.2 s and with an exponential tail beyond it.
        # `bad` and the 0.2 s threshold are the measured Geysers numbers: 10.5 % of real picks
        # sit beyond 0.2 s. An exponential tail rather than a uniform one because most bad
        # picks are only somewhat bad and a few are very bad, which is what the data show.
        n = int(is_bad.sum())
        mag = 0.2 + rng.exponential(bad_scale, n)
        t_obs[is_bad] += np.where(rng.random(n) < 0.5, -1.0, 1.0) * mag
    return sta, t_obs, phase, is_bad


def events(n, st, tabs, zrange=(0.5, 3.5), half=4.0, seed=0, **kw):
    """A population of planted events over the field, each with its own observations.

    The depth range is the producing reservoir, 0.5-3 km below sea level, so a coverage test
    is run where the field's own scientific claim lives.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n):
        truth = np.array([rng.uniform(-half, half), rng.uniform(-half, half),
                          rng.uniform(*zrange), rng.uniform(-0.5, 0.5)])
        yield (truth, *observe(truth, st, tabs, seed=int(rng.integers(1 << 31)), **kw))


# ------------------------------------------- three ways to put an error bar on a hypocentre

CHI2_95_2D = 5.9915                             # chi-square, 95 %, 2 degrees of freedom
CHI2_95_3D = 7.8147                             # and 3
def bootstrap(sta_xyz, t_obs, phase, tabs, m0, n=200, seed=0, **kw):
    """Resample the picks with replacement and relocate each time; the spread is the answer.
    This asks "what if the network had recorded a slightly different set of arrivals?" and
    answers it by experiment rather than by a derivative.  It still cannot see an error the
    velocity model makes at every station at once, which is the whole point of `coverage`.
    """
    rng = np.random.default_rng(seed)
    n_pick = len(t_obs)
    out = []
    for _ in range(n):
        k = rng.integers(0, n_pick, n_pick)
        if len(np.unique(k)) < 5:
            continue
        try:
            r = locate(sta_xyz[k], t_obs[k], phase[k], tabs, m0, **kw)
        except np.linalg.LinAlgError:
            continue
        out.append(r["m"])
    return np.array(out)
def posterior(sta_xyz, t_obs, phase, tabs, centre, half=1.2, step=0.04,
              sigma=0.03, loss="l2", loss_sigma=0.10):
    """The exact posterior on a grid: no Gaussian assumed, no derivative taken.
    The origin time is profiled out rather than searched -- at every trial position the best
    t0 is the middle of what is left over -- so a 4-D problem is drawn as a 3-D volume.
    `sigma` is the arrival-time uncertainty the likelihood assumes, in seconds.
    """
    gx = np.arange(centre[0] - half, centre[0] + half + 1e-9, step)
    gy = np.arange(centre[1] - half, centre[1] + half + 1e-9, step)
    gz = np.arange(max(centre[2] - half, tabs[0]["zs"].min() + 1e-6),
                   min(centre[2] + half, tabs[0]["zs"].max() - 1e-6) + 1e-9, step)
    xs, ys, zs_ = sta_xyz[:, 0], sta_xyz[:, 1], sta_xyz[:, 2]
    logp = np.empty((len(gz), len(gy), len(gx)))
    for i, z in enumerate(gz):
        for j, y in enumerate(gy):
            r = np.maximum(np.hypot(np.subtract.outer(gx, xs), y - ys), 1e-3)  # (nx, npick)
            t = np.empty_like(r)
            for ph in (0, 1):
                s = phase == ph
                if s.any():
                    t[:, s] = predict(tabs[ph],
                                           np.broadcast_to(zs_[s], (len(gx), int(s.sum()))),
                                           z, r[:, s])[0]
            d = t_obs - t
            t0 = (d.mean(axis=1) if loss == "l2" else np.median(d, axis=1))[:, None]
            logp[i, j] = -np.sum(rho(d - t0, loss, loss_sigma), axis=1) / sigma ** 2
    p = np.exp(logp - logp.max())
    return dict(x=gx, y=gy, z=gz, logp=logp, p=p / p.sum())
def laplace(out, sigma=None):
    """The Gaussian that matches the posterior's peak and its curvature there.
    This IS the formal ellipse: sigma^2 (J^T J)^-1 is the inverse curvature of the L2
    log-likelihood at the optimum.  Naming it Laplace says what the ellipse assumes -- that
    the posterior is a paraboloid in log space, which the grid above can be asked about.
    """
    return ellipse(out, sigma)
def inside(C, delta, dof=2):
    """Is `delta` inside the 95 % ellipse with covariance C?  Mahalanobis against chi-square."""
    C = np.asarray(C)[:dof, :dof]
    d = np.asarray(delta)[:dof]
    if not np.all(np.isfinite(C)):
        return False, np.inf
    try:
        d2 = float(d @ np.linalg.solve(C, d))
    except np.linalg.LinAlgError:
        return False, np.inf
    return d2 <= (CHI2_95_2D if dof == 2 else CHI2_95_3D), d2

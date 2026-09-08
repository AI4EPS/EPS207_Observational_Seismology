#!/usr/bin/env python
"""2 · Functions — the calculations, with no data loading and no drawing in any of them.

Two groups. The first is geometry on the map: distance, the shape of a Mercator panel, and the
depth section that §§ 4 and 8 both cut through the field. The second is focal-mechanism arithmetic:
strike, dip and rake give a fault normal and a slip direction (Aki & Richards); those give the P, T
and B axes; the axis nearest vertical gives Frohlich's (1992) faulting style; summing many unit
double couples and taking the best double couple of the sum gives one representative mechanism for
a cell.
"""
import math

import numpy as np
import pandas as pd

from geysers_data import CENTER, KM_LAT, KM_LON      # the region, declared in the data cell


# ──────────────────────────────────────────────────────────────────────── geometry on the map
def distance_km(lat, lon, lat0=CENTER[0], lon0=CENTER[1]):
    """Kilometres from (lat0, lon0) on the local flat-earth approximation."""
    return np.hypot((np.asarray(lat) - lat0) * KM_LAT, (np.asarray(lon) - lon0) * KM_LON)


def mercator_ratio(region):
    """Width per unit height for a region drawn on a Mercator projection."""
    merc = lambda la: math.log(math.tan(math.pi / 4 + math.radians(la) / 2))
    return math.radians(region[1] - region[0]) / (merc(region[3]) - merc(region[2]))


def section_axis(xy, mags, floor):
    """The section line the project's Figure 4 uses: the principal axis of the epicentres.

    Returns the two ends, the unit vector along it and its length, all from the seismicity itself
    rather than from a chosen azimuth, so the section is reproducible from the catalogue alone.
    """
    c0 = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - c0, full_matrices=False)
    e = vt[0] if vt[0][1] > 0 else -vt[0]                   # unit vector, pointing NW
    t = (xy - c0) @ e
    # The ends are percentiles of the along-axis distance rather than its extremes, so a handful of
    # outlying events cannot stretch the section. The southeast end is taken a little further out
    # than the northwest: the seismicity thins there gradually rather than stopping, and the 1st
    # percentile clipped cells that the section panel then had to draw beyond the line.
    t1, t2 = np.percentile(t, [0.3, 99])
    end_a = (CENTER[1] + (c0[0] + t1 * e[0]) / KM_LON, CENTER[0] + (c0[1] + t1 * e[1]) / KM_LAT)
    end_b = (CENTER[1] + (c0[0] + t2 * e[0]) / KM_LON, CENTER[0] + (c0[1] + t2 * e[1]) / KM_LAT)
    return end_a, end_b, e, t2 - t1


def section_line(cat_field, floor=1.5):
    """The A-A' profile: the principal axis of the field's epicentres."""
    d = cat_field[cat_field.mag >= floor]
    xy = np.column_stack([(d.longitude - CENTER[1]) * KM_LON,
                          (d.latitude - CENTER[0]) * KM_LAT])
    return section_axis(xy, d.mag.values, floor)


def along_section(df, A, evec, LEN, half_width=6.0):
    """Distance along the profile from A', and across it, for a table of events."""
    dx = (df.longitude - A[0]) * KM_LON
    dy = (df.latitude - A[1]) * KM_LAT
    out = df.assign(along=LEN - (dx * evec[0] + dy * evec[1]),
                    across=-dx * evec[1] + dy * evec[0])
    return out[out.along.between(0, LEN) & (out.across.abs() <= half_width)]


def ground_profile(L, lon0, lat0, ex, ey, length_km, step_km=0.1, half_width_km=1.0):
    """Mean elevation (km) along a line, averaged across a corridor, from the elevation model."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((L["lat"], L["lon"]), L["z"], bounds_error=False, fill_value=np.nan)
    t = np.arange(0, length_km + step_km / 2, step_km)
    q = np.linspace(-half_width_km, half_width_km, 9)
    lon = lon0 + (t[:, None] * ex + q[None, :] * (-ey)) / KM_LON
    lat = lat0 + (t[:, None] * ey + q[None, :] * ex) / KM_LAT
    z = interp(np.stack([lat.ravel(), lon.ravel()], -1)).reshape(lon.shape)
    return t, np.nanmean(z, axis=1) / 1000.0


def merge_intervals(intervals, slack):
    """Merge (start, end) pairs that touch or lie within `slack` of one another.

    Used to turn a list of months in which a station has files into the small number of continuous
    spans that should be drawn as bars.
    """
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + slack:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out



# ───────────────────────────────────────────────────────────────── focal-mechanism arithmetic
QUALITY = dict(nfm=10, misfit=0.2)          # first motions and FPFIT misfit accepted (fact 4)


def fault_vectors(strike, dip, rake):
    """Fault normal n and slip s in (north, east, down), from degrees."""
    st, dp, rk = np.radians(strike), np.radians(dip), np.radians(rake)
    n = np.column_stack([-np.sin(dp) * np.sin(st), np.sin(dp) * np.cos(st), -np.cos(dp)])
    s = np.column_stack([np.cos(rk) * np.cos(st) + np.cos(dp) * np.sin(rk) * np.sin(st),
                         np.cos(rk) * np.sin(st) - np.cos(dp) * np.sin(rk) * np.cos(st),
                         -np.sin(dp) * np.sin(rk)])
    return n, s


def az_plunge(v):
    """Azimuth and plunge in degrees of an axis, taken into the lower hemisphere."""
    v = np.where(v[:, [2]] < 0, -v, v)
    az = (np.degrees(np.arctan2(v[:, 1], v[:, 0])) + 360) % 360
    pl = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
    return az, pl


def add_axes(m):
    """Add the P, T and B axes, the Frohlich style and the continuous faulting type to a table."""
    n, s = fault_vectors(m.strike.values, m.dip.values, m.rake.values)
    P, T, B = (n - s) / np.sqrt(2), (n + s) / np.sqrt(2), np.cross(n, s)
    for name, v in (("p", P), ("t", T), ("b", B)):
        m[f"{name}_az"], m[f"{name}_pl"] = az_plunge(v)
    # the axis nearest vertical: P vertical is normal faulting, T vertical reverse, B vertical strike-slip
    # named `faulting`, not `style`: a column called style is shadowed by pandas' Styler accessor
    m["faulting"] = np.array(["normal", "reverse", "strike-slip"])[
        np.argmax(np.column_stack([m.p_pl, m.t_pl, m.b_pl]), axis=1)]
    m["faulting_index"] = np.sin(np.radians(m.rake))    # -1 normal, 0 strike-slip, +1 reverse
    return m


def double_couple(P, T):
    """Strike, dip and rake of the double couple with these P and T unit vectors."""
    import math
    n = (T + P) / np.linalg.norm(T + P)
    s = (T - P) / np.linalg.norm(T - P)
    if n[2] > 0:                                        # fault normal pointing up
        n, s = -n, -s
    dip = math.degrees(math.acos(-n[2]))
    strike = math.degrees(math.atan2(-n[0], n[1])) % 360
    st = math.radians(strike)
    e_strike = np.array([math.cos(st), math.sin(st), 0.0])
    e_updip = np.cross(n, e_strike)
    rake = math.degrees(math.atan2(np.dot(s, e_updip), np.dot(s, e_strike)))
    return strike, dip, rake


def mean_mechanism(m):
    """The summed unit double couple of a group of mechanisms.

    Returns its strike, dip and rake, and how much of the summed tensor the double couple explains —
    1 where the group agrees, lower where it is a mixture.
    """
    n, s = fault_vectors(m.strike.values, m.dip.values, m.rake.values)
    M = np.einsum("ki,kj->ij", n, s) + np.einsum("ki,kj->ij", s, n)
    w, v = np.linalg.eigh(M / len(m))                   # ascending: P, B, T
    return (*double_couple(v[:, 0], v[:, 2]), (w[2] - w[0]) / 2)


def quality(mech, one_per_event=True):
    """One solution per event at the lowest misfit, then the first-motion and misfit cut."""
    m = mech.sort_values("misfit").drop_duplicates("id", keep="first") if one_per_event else mech
    m = m[(m.nfm >= QUALITY["nfm"]) & (m.misfit <= QUALITY["misfit"])].copy()
    return add_axes(m)


def cell_mechanisms(m, x, y, cell_x, cell_y, nmin):
    """Sum the mechanisms in each cell of a grid, keeping cells with at least `nmin` of them."""
    import pandas as pd
    ix = np.floor(x / cell_x).astype(int)
    iy = np.floor(y / cell_y).astype(int)
    rows = []
    for (i, j), g in m.groupby([ix, iy]):
        if len(g) < nmin:
            continue
        strike, dip, rake, coherence = mean_mechanism(g)
        rows.append(dict(x=(i + 0.5) * cell_x, y=(j + 0.5) * cell_y, n=len(g), strike=strike,
                         dip=dip, rake=rake, coherence=coherence,
                         faulting_index=float(np.sin(np.radians(rake)))))
    return pd.DataFrame(rows)


def radiation_p(strike, dip, rake, azimuth, takeoff):
    """Far-field P radiation amplitude for rays leaving the source at these angles.

    `azimuth` is degrees east of north and `takeoff` is degrees from the downward vertical, which is
    how the NCSN phase archive reports them: a take-off angle above 90 deg is an up-going ray. A
    positive amplitude is compressional, so the first motion is up.
    """
    n, s = fault_vectors(strike, dip, rake)
    M = np.outer(n, s) + np.outer(s, n)                 # the unit double couple
    az, ih = np.radians(np.asarray(azimuth, float)), np.radians(np.asarray(takeoff, float))
    g = np.column_stack([np.sin(ih) * np.cos(az), np.sin(ih) * np.sin(az), np.cos(ih)])
    return np.einsum("ij,jk,ik->i", g, M, g)


def fit_first_motions(azimuth, takeoff, up, coarse=10, fine=2):
    """Grid-search the double couple that explains the most observed first motions.

    This is what FPFIT does, without its weighting scheme or its error analysis: score every
    orientation on a coarse grid by the fraction of polarities it gets wrong, then refine around the
    best one. Returns ((strike, dip, rake), misfit_fraction, predicted_up).
    """
    up = np.asarray(up, bool)

    def score(sv, dv, rv):
        best = (2.0, 0, 45, 0)
        for st in sv:
            for dp in dv:
                for rk in rv:
                    wrong = float(np.mean((radiation_p(st, dp, rk, azimuth, takeoff) > 0) != up))
                    if wrong < best[0]:
                        best = (wrong, st, dp, rk)
        return best

    w, st, dp, rk = score(np.arange(0, 360, coarse),
                          np.arange(coarse, 91, coarse),
                          np.arange(-180, 180, coarse))
    w, st, dp, rk = score(np.arange(st - coarse, st + coarse + 1, fine) % 360,
                          np.clip(np.arange(dp - coarse, dp + coarse + 1, fine), 1, 90),
                          np.arange(rk - coarse, rk + coarse + 1, fine))
    return (float(st), float(dp), float(rk)), w, radiation_p(st, dp, rk, azimuth, takeoff) > 0


def detect_first_motion(waveform, phase_index, pre_window=3, post_window=10):
    """Detect polarity from waveform first motion on Z component.

    Finds the first local extremum after phase arrival relative to
    pre-arrival baseline. Trough -> "D", peak -> "U".
    """
    if phase_index is None or (isinstance(phase_index, float) and np.isnan(phase_index)):
        return None
    phase_index = int(phase_index)
    if phase_index < pre_window or phase_index + post_window >= waveform.shape[1]:
        return None

    z_post = waveform[2, phase_index : phase_index + post_window]
    z_pre_mean = np.mean(waveform[2, phase_index - pre_window : phase_index])

    pol = ""
    for ii, s in enumerate(z_post):
        if s <= z_pre_mean:
            if (s < z_post[ii + 1]) | (ii == post_window - 2):
                if ii == 0:
                    break
                pol = "D"
                break
        else:
            if (s > z_post[ii + 1]) | (ii == post_window - 2):
                if ii == 0:
                    break
                pol = "U"
                break

    result = {}
    for i, comp in enumerate(["e", "n", "z"]):
        before = np.mean(waveform[i, phase_index - pre_window : phase_index])
        after = np.mean(waveform[i, phase_index : phase_index + post_window])
        result[comp] = float(after - before)
    result["label"] = pol
    return result

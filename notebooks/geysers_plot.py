#!/usr/bin/env python
"""3 · Plotting — the house style, the colour tables, and maps in matplotlib alone.

The project's map figures are drawn with PyGMT, which needs the GMT C library and cannot be
installed on Colab or DataHub without root. Everything GMT does for those figures has a matplotlib
equivalent, and this is that equivalent:

    grdgradient + grdimage   ->  LightSource.hillshade + imshow   (in the data cell)
    plot(data=multi-segment) ->  segments(): those files are plain ASCII
    makecpt cmap=batlow      ->  BATLOW below, sampled from Crameri's table
    basemap, Mercator        ->  set_aspect(1 / cos(latitude)), exact enough over 90 km
    meca                     ->  obspy.imaging.beachball.beach
"""
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Patch, Rectangle
from matplotlib.patches import Rectangle as plt_rect

from geysers_data import CENTER, KM_LAT, KM_LON, segments   # declared in the data cell
from geysers_func import mercator_ratio                     # in the functions cell

CM = 1 / 2.54          # matplotlib works in inches; every size in these figures is in cm

# The house style, set once: every figure in the notebook should look like it belongs to the same
# paper. 17 cm is the full text width these figures were designed for.
# Arial before Helvetica: this machine's Helvetica cannot rasterise digits below about
# 6 pt ("failed to load glyph" on a station code like 68035), and several of these figures
# label at 4-5 pt. Arial is metrically compatible, so nothing about the layout changes.
# A family name that is not installed costs one `findfont: Font family 'X' not found` warning for
# every piece of text drawn, and on Linux -- DataHub, Colab -- neither Arial nor Helvetica is
# present, so the notebook's own output disappears under thousands of them. Hence the preference
# list is filtered to what this machine actually has. DejaVu Sans ships with matplotlib.
_FAMILY = [f for f in ("Arial", "Helvetica", "DejaVu Sans")
           if f in {m.name for m in font_manager.fontManager.ttflist}] or ["DejaVu Sans"]
plt.rcParams.update({"font.family": _FAMILY, "font.size": 7,
                     "axes.linewidth": .6, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.labelsize": 7.5, "xtick.major.width": .5, "ytick.major.width": .5,
                     "legend.fontsize": 6.5, "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
                     "figure.dpi": 110, "savefig.dpi": 300})

LAKE_FILL = "#b9d9ea"


LAKE_EDGE = "#7fb2d5"


# Crameri's batlow, sampled at 32 stops — perceptually uniform, and readable in greyscale.
BATLOW = [(0.005, 0.098, 0.350), (0.032, 0.147, 0.358), (0.049, 0.191, 0.366), (0.059, 0.230, 0.372),
          (0.067, 0.263, 0.378), (0.076, 0.293, 0.382), (0.088, 0.322, 0.385), (0.107, 0.350, 0.385),
          (0.133, 0.375, 0.379), (0.168, 0.398, 0.368), (0.209, 0.417, 0.350), (0.254, 0.435, 0.326),
          (0.302, 0.450, 0.300), (0.352, 0.465, 0.272), (0.403, 0.480, 0.245), (0.456, 0.496, 0.218),
          (0.511, 0.511, 0.193), (0.570, 0.526, 0.175), (0.632, 0.541, 0.170), (0.694, 0.554, 0.183),
          (0.754, 0.565, 0.212), (0.812, 0.575, 0.253), (0.865, 0.586, 0.302), (0.913, 0.599, 0.361),
          (0.951, 0.617, 0.429), (0.976, 0.638, 0.502), (0.989, 0.661, 0.575), (0.993, 0.685, 0.645),
          (0.993, 0.708, 0.712), (0.991, 0.730, 0.779), (0.989, 0.754, 0.847), (0.986, 0.778, 0.917)]


DEPTH_CMAP = LinearSegmentedColormap.from_list("batlow_r", BATLOW[::-1])


# romaO, cyclic: 0 and 180 degrees are the same direction, so an azimuth scale must wrap
ROMAO = [(0.451, 0.223, 0.342), (0.474, 0.220, 0.298), (0.496, 0.225, 0.260), (0.519, 0.241, 0.228),
         (0.544, 0.266, 0.202), (0.571, 0.302, 0.182), (0.601, 0.346, 0.172), (0.634, 0.399, 0.171),
         (0.668, 0.458, 0.184), (0.704, 0.524, 0.213), (0.741, 0.595, 0.258), (0.778, 0.667, 0.322),
         (0.810, 0.736, 0.399), (0.832, 0.796, 0.483), (0.838, 0.842, 0.565), (0.826, 0.871, 0.639),
         (0.796, 0.883, 0.701), (0.749, 0.881, 0.749), (0.688, 0.864, 0.783), (0.617, 0.835, 0.803),
         (0.543, 0.795, 0.811), (0.472, 0.747, 0.808), (0.409, 0.693, 0.796), (0.360, 0.636, 0.776),
         (0.326, 0.577, 0.750), (0.309, 0.517, 0.717), (0.310, 0.457, 0.675), (0.325, 0.399, 0.625),
         (0.349, 0.344, 0.567), (0.377, 0.298, 0.506), (0.403, 0.262, 0.446), (0.428, 0.237, 0.391)]


AZ_CMAP = LinearSegmentedColormap.from_list("romaO", ROMAO)


# lajolla, sequential: the plunge of an axis, 0 horizontal to 90 vertical
LAJOLLA = [(0.099, 0.100, 0.000), (0.126, 0.109, 0.016), (0.153, 0.119, 0.032), (0.183, 0.129, 0.050),
           (0.217, 0.141, 0.068), (0.256, 0.154, 0.087), (0.300, 0.169, 0.109), (0.349, 0.186, 0.135),
           (0.403, 0.204, 0.163), (0.461, 0.223, 0.193), (0.523, 0.241, 0.222), (0.587, 0.259, 0.247),
           (0.652, 0.274, 0.267), (0.716, 0.289, 0.281), (0.774, 0.310, 0.291), (0.820, 0.339, 0.299),
           (0.852, 0.377, 0.304), (0.870, 0.418, 0.308), (0.882, 0.458, 0.312), (0.890, 0.497, 0.315),
           (0.898, 0.534, 0.318), (0.906, 0.571, 0.320), (0.913, 0.607, 0.323), (0.921, 0.644, 0.326),
           (0.928, 0.681, 0.329), (0.936, 0.720, 0.336), (0.946, 0.762, 0.351), (0.957, 0.808, 0.388),
           (0.969, 0.857, 0.455), (0.980, 0.901, 0.542), (0.989, 0.940, 0.635), (0.996, 0.973, 0.723)]


PLUNGE_CMAP = LinearSegmentedColormap.from_list("lajolla", LAJOLLA)


FAULT_CMAP = LinearSegmentedColormap.from_list("faulting", ["#d7191c", "#ffffbf", "#2c7bb6"])


# The nine geology classes of the state map, young to old, in the order the staged files carry them.
GEOLOGY = [("Quaternary alluvium and landslides", "#fbf5dc"), ("Clear Lake Volcanics", "#eb9a86"),
           ("Plio-Pleistocene sediments", "#f0dea6"), ("Sonoma Volcanics", "#f4b47c"),
           ("Paleogene and Neogene marine rocks", "#dde8b8"), ("Great Valley Sequence", "#a9d3a0"),
           ("Franciscan Complex", "#bccbe3"), ("Coast Range Ophiolite", "#6fa8c9"),
           ("Serpentinite and ultramafic rocks", "#b38bcb")]


# Quaternary faults by the age of the most recent rupture. The staged file order is the order the
# ages first appear in the national shapefile, not the age order. Verified 2026-09-06 by re-reading
# Qfaults_US_Database.shp: faults_0 historic, faults_1 undifferentiated, faults_2 late, faults_3
# latest Quaternary. Do not re-derive this from the trace counts -- 1 and 3 are close enough to
# swap by accident, and swapping them mis-colours every fault on the map.
FAULT_AGES = [("faults_0.txt", "Historic (< 150 yr)", "#7f0000", 1.0),
              ("faults_3.txt", "Latest Quaternary (< 15 ka)", "#b22222", 0.7),
              ("faults_2.txt", "Late Quaternary (< 130 ka)", "#d9772e", 0.55),
              ("faults_1.txt", "Undifferentiated Quaternary (< 1.6 Ma)", "#606060", 0.35)]


KGRA_COLOR = "#006d2c"


# unit vectors for "put this label on the NE side of that trace"
SIDE = {"NE": (1, 1), "SW": (-1, -1), "NW": (-1, 1), "SE": (1, -1)}


STYLE_COL = {"normal": "#1f78b4", "strike-slip": "#33a02c", "reverse": "#e31a1c"}


# ───────────────────────────────────────────────────────────────────── panels and map furniture
def two_panels(win, gap, region_a, region_b):
    """Widths and the common height for two map panels that share a row.

    Both panels get the same height and each its own width, so neither is letterboxed inside a box
    that does not match its aspect — which is what happens if you fix both widths and heights.
    """
    ra, rb = mercator_ratio(region_a), mercator_ratio(region_b)
    h = (win - gap) / (ra + rb)
    return ra * h, rb * h, h


def map_figure(region, width=17.0, left=1.15, right=1.45, bottom=1.0, top=0.3):
    """One map panel, in centimetres, sized so the region has its true shape.

    The panel *height* is computed rather than chosen: a Mercator map of a region 1.00 degrees wide
    and 0.76 degrees tall is not square, and a map drawn to the wrong aspect is a wrong map. The
    margins are the space left for tick labels, the title, and whatever legend goes underneath.
    """
    panel_w = width - left - right
    panel_h = panel_w / mercator_ratio(region)
    height = top + panel_h + bottom
    fig = plt.figure(figsize=(width * CM, height * CM))
    ax = fig.add_axes([left / width, bottom / height, panel_w / width, panel_h / height])
    return fig, ax


def frame(ax, region, xtick, ytick, scale_km, right_labels=False, scale_loc=(.06, .06)):
    """Finish a map: crop it to the region, set the projection, and add ticks and a scale bar.

    Latitude and longitude are not interchangeable units -- one degree of longitude is only
    cos(latitude) as long as one degree of latitude. Rather than reproject the coordinates, the
    axes are told to draw a degree of longitude shorter by exactly that factor, which over a region
    90 km across is indistinguishable from a true Mercator projection.
    """
    ax.set(xlim=region[:2], ylim=region[2:])
    ax.set_aspect(1 / math.cos(math.radians(CENTER[0])))
    degree_minute_ticks(ax, xtick, ytick, right=right_labels)
    scale_bar_gmt(ax, scale_km, loc=scale_loc)


def geology(ax, geo, alpha=0.55):
    """The nine state-map geology classes as filled polygons, under everything else."""
    import pathlib as _pl
    geo = _pl.Path(geo)
    for i, (_label, colour) in enumerate(GEOLOGY):
        f = geo / f"geol_{i}.txt"
        if not f.exists():
            continue
        for poly in segments(f):
            ax.fill(poly[:, 0], poly[:, 1], facecolor=colour, edgecolor="none", alpha=alpha, zorder=1)


def basemap(ax, L, region, fault_lw=0.3, shade_lo=-0.9, shade_hi=1.35, shade_alpha=0.55):
    """Shaded relief, lakes and Quaternary faults, on a locally Mercator-equivalent aspect.

    The relief is a background, not the subject, so it is drawn light: the published figures use
    GMT's `transparency=55` over white, and `shade_alpha` is the same thing.
    """
    ax.imshow(L["shade"], cmap="gray", extent=L["extent"], origin="lower", alpha=shade_alpha,
              vmin=shade_lo, vmax=shade_hi, zorder=0, interpolation="bilinear")
    for w in L["water"]:
        ax.fill(w[:, 0], w[:, 1], facecolor=LAKE_FILL, edgecolor=LAKE_EDGE, lw=0.2, zorder=1)
    for f in L["faults"]:
        ax.plot(f[:, 0], f[:, 1], color="#8c8c8c", lw=fault_lw, zorder=2)
    ax.set(xlim=region[:2], ylim=region[2:])
    ax.set_aspect(1 / math.cos(math.radians(CENTER[0])))


def scale_bar_gmt(ax, km, segments_n=4, loc=(0.06, 0.06)):
    """A segmented scale bar, alternating black and white, as GMT draws one."""
    x0, x1 = ax.get_xlim()
    dx = km / KM_LON / (x1 - x0) / segments_n
    x, y = loc
    for k in range(segments_n):
        ax.add_patch(plt_rect((x + k * dx, y), dx, 0.011, transform=ax.transAxes,
                              facecolor="black" if k % 2 == 0 else "white",
                              edgecolor="black", lw=.4, zorder=9, clip_on=False))
    for k in (0, segments_n):
        ax.text(x + k * dx, y - 0.004, f"{int(km * k / segments_n)}", transform=ax.transAxes,
                ha="center", va="top", fontsize=6, zorder=9)
    ax.text(x + segments_n * dx / 2, y + 0.018, "km", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=6, zorder=9)


def scale_bar(ax, km, loc=(0.06, 0.06), color="black"):
    """A plain kilometre scale bar, in axes coordinates."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    dx = km / KM_LON / (x1 - x0)
    x, y = loc
    ax.plot([x, x + dx], [y, y], transform=ax.transAxes, color=color, lw=1.6, solid_capstyle="butt")
    ax.text(x + dx / 2, y + 0.012, f"{km:g} km", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.5, color=color)


def thin_tick_labels(ax, axis="x", pad=3.0):
    """Blank any tick label that would touch its neighbour, measured on the rendered figure.

    The tick step is chosen for the map, not for how wide the panel happens to be drawn, so a
    narrow panel ends up with labels running into each other -- "123.25°W123.00°W". Estimating from
    character counts is unreliable with degree signs and minus signs, so the boxes matplotlib
    actually draws are measured and every second label dropped until they clear.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    arts = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    out, last = [], None
    for art in arts:
        b = art.get_window_extent(r)
        clash = last is not None and ((b.x0 < last.x1 + pad) if axis == "x"
                                      else (b.y0 < last.y1 + pad))
        out.append("" if clash else art.get_text())
        if not clash:
            last = b
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)(out)


def degree_ticks(ax, xstep, ystep, right=False, thin=True):
    """Longitude and latitude ticks labelled the way the project's figures label them."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.set_xticks(np.arange(np.ceil(x0 / xstep) * xstep, x1, xstep))
    ax.set_yticks(np.arange(np.ceil(y0 / ystep) * ystep, y1, ystep))
    ax.set_xticklabels([f"{abs(v):.2f}°W" for v in ax.get_xticks()])
    ax.set_yticklabels([f"{v:.2f}°N" for v in ax.get_yticks()])
    if right:
        ax.yaxis.tick_right()
    if thin:
        thin_tick_labels(ax, "x")
        thin_tick_labels(ax, "y")


def degree_minute_ticks(ax, xstep, ystep, right=False):
    """Ticks written the way a map is written: 123 degrees 00 minutes W, not 123.25 W."""
    def dm(v, hemi):
        a = abs(v)
        d = int(a)
        m = round((a - d) * 60)
        if m == 60:
            d, m = d + 1, 0
        return f"{d}°{m:02d}'{hemi}"
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.set_xticks(np.arange(np.ceil(x0 / xstep) * xstep, x1, xstep))
    ax.set_yticks(np.arange(np.ceil(y0 / ystep) * ystep, y1, ystep))
    ax.set_xticklabels([dm(v, "W") for v in ax.get_xticks()])
    ax.set_yticklabels([dm(v, "N") for v in ax.get_yticks()])
    if right:
        ax.yaxis.tick_right()


def mag_size(m):
    """Symbol area in points², from the project's diameter rule of 0.022 x 1.6^M centimetres.

    matplotlib's `s` is the marker AREA, so a circle of diameter d points has s = pi d^2 / 4. The
    factor was 0.30, which drew every symbol at 62 % of its intended diameter.
    """
    return (0.022 * 1.6 ** np.clip(m, 0, 6) * 28.35) ** 2 * (np.pi / 4)


# ────────────────────────────────────────────────────────────────────────────── placing labels
def label_along_traces(ax, traces, anchors, fontsize=7.0, colour="#7f0000", offset_km=0.9):
    """Write each name along the trace it belongs to, on the requested side of it.

    `anchors` is {name: (lon, lat, side)}. The text angle is not stored with the anchor: it is
    fitted to the stretch of trace the text will actually cover, so that moving an anchor along a
    curving fault keeps the name parallel to it. `side` is a compass direction, and the name is put
    on that side of the trace.
    """
    pts, out = np.vstack(traces), []
    for name, (alon, alat, side) in anchors.items():
        d = np.hypot((pts[:, 0] - alon) * KM_LON, (pts[:, 1] - alat) * KM_LAT)
        near = pts[d < max(2.5, 0.16 * len(name))]           # the stretch the text will cover
        if len(near) < 3:
            continue
        c = near.mean(axis=0)
        xy = np.column_stack([(near[:, 0] - c[0]) * KM_LON, (near[:, 1] - c[1]) * KM_LAT])
        _, _, vt = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)   # the local trend
        ang = np.degrees(np.arctan2(vt[0][1], vt[0][0]))
        ang = ang - 180 if ang > 90 else (ang + 180 if ang < -90 else ang)    # keep text upright
        n = np.array([-np.sin(np.radians(ang)), np.cos(np.radians(ang))])     # normal to the trace
        if np.dot(n, SIDE[side]) < 0:
            n = -n
        off = n * offset_km
        out.append(ax.text(alon + off[0] / KM_LON, alat + off[1] / KM_LAT, name, rotation=ang,
                           rotation_mode="anchor", ha="center", va="center", fontsize=fontsize,
                           color=colour, zorder=8,
                           path_effects=[pe.withStroke(linewidth=1.8, foreground="white")]))
    return out


def label_water_body(ax, polygons, inside, text, **kw):
    """Write a name across the water body that contains `inside`, on its widest open stretch.

    Two things go wrong if this is done casually. The lake is stored as several polygons and the
    largest is not always the one you mean, so the one containing a known interior point is chosen;
    and a name placed at the centroid of a bent lake can land on the shore, so it goes on the
    widest horizontal chord instead.
    """
    def contains(poly, pt):                                   # ray casting: odd crossings = inside
        x, y = poly[:, 0], poly[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        c = ((y > pt[1]) != (y2 > pt[1])) & (pt[0] < (x2 - x) * (pt[1] - y) / (y2 - y + 1e-12) + x)
        return c.sum() % 2 == 1

    hits = [w for w in polygons if len(w) > 20 and contains(w, inside)]
    if not hits:
        return None
    big = max(hits, key=len)
    y = np.median(big[:, 1])
    xs = big[np.abs(big[:, 1] - y) < .01][:, 0]
    cx = (xs.min() + xs.max()) / 2 if len(xs) > 1 else big[:, 0].mean()
    return ax.text(cx, y, text, ha="center", va="center", zorder=8,
                   path_effects=[pe.withStroke(linewidth=2.4, foreground="white")], **kw)


def _overlap(a, b, pad=1.0):
    return not (a.x1 + pad < b.x0 or b.x1 + pad < a.x0 or a.y1 + pad < b.y0 or b.y1 + pad < a.y0)


def place_labels(fig, ax, movable, fixed):
    """Move each label to the first offset around its anchor whose drawn box is clear.

    Boxes are the ones matplotlib actually renders, measured with the renderer, rather than an
    estimate from the character count — which is what made the first attempt collide.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in fixed]
    # beside first, then above and below, then further out
    cands = [(9, 0), (-9, 0), (0, 8), (0, -8), (7, 7), (-7, 7), (7, -7), (-7, -7),
             (18, 0), (-18, 0), (0, 15), (0, -15), (15, 10), (-15, 10), (15, -10), (-15, -10),
             (28, 0), (-28, 0), (0, 22), (0, -22)]
    from matplotlib.transforms import offset_copy

    def area(a, b):
        return (max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
                * max(0, min(a.y1, b.y1) - max(a.y0, b.y0)))

    moved = 0
    for t, (x, y) in movable:
        t.set_position((x, y))
        best = None
        for k, (dx, dy) in enumerate(cands):
            t.set_ha("left" if dx > 0 else ("right" if dx < 0 else "center"))
            t.set_va("center" if abs(dy) < 4 else ("bottom" if dy > 0 else "top"))
            t.set_transform(offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points"))
            box = t.get_window_extent(r)
            clash = sum(area(box, b) for b in boxes)
            if clash == 0:
                best = (0, k, dx, dy, box)
                break
            # keep the least-bad candidate: falling through to whichever was tried last is how
            # a label ends up sitting on another one
            if best is None or clash < best[0]:
                best = (clash, k, dx, dy, box)
        _, k, dx, dy, box = best
        t.set_ha("left" if dx > 0 else ("right" if dx < 0 else "center"))
        t.set_va("center" if abs(dy) < 4 else ("bottom" if dy > 0 else "top"))
        t.set_transform(offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points"))
        boxes.append(box)
        moved += k > 0
    return moved


def place_codes(ax, lon, lat, labels, fontsize=4.2, merge_km=0.30, colour="#111111"):
    """Label point symbols on a map, merging any that are too close to label separately.

    Symbols within `merge_km` of one another share a single label, so that a cluster of three
    stations is annotated "ESM/ADS2/ADSP" rather than with three codes that would overprint. The
    merged labels are then passed to `place_labels`, which measures the drawn boxes and moves each
    one to the first offset that is clear. Returns the number of labels and how many had to move.
    """
    x = (np.asarray(lon) - CENTER[1]) * KM_LON          # kilometres, so `merge_km` means what it says
    y = (np.asarray(lat) - CENTER[0]) * KM_LAT
    labels = np.asarray(labels)

    groups, used = [], set()
    for i in range(len(labels)):
        if i in used:
            continue
        near = [j for j in range(len(labels))
                if j not in used and np.hypot(x[j] - x[i], y[j] - y[i]) <= merge_km]
        used.update(near)
        groups.append(("/".join(labels[near]), float(np.mean(x[near])), float(np.mean(y[near]))))

    texts = []
    for name, gx, gy in sorted(groups, key=lambda g: -len(g[0])):   # longest first: hardest to place
        glon, glat = CENTER[1] + gx / KM_LON, CENTER[0] + gy / KM_LAT
        texts.append((ax.text(glon, glat, name, fontsize=fontsize, ha="center", va="center",
                              zorder=8, color=colour,
                              path_effects=[pe.withStroke(linewidth=1.3, foreground="white")]),
                      (glon, glat)))
    return len(groups), place_labels(ax.figure, ax, texts, [])



PLUNGE_MAX = 60.0            # a steeply plunging axis has no meaningful azimuth to draw


def axis_bars(ax, m, az, pl, lon_span_km, panel_cm):
    """Draw each principal axis as a short bar through its epicentre, along its own azimuth.

    An axis that plunges steeply has no meaningful azimuth to draw, so bars are drawn only for
    axes flatter than `PLUNGE_MAX`. Bar length follows the same magnitude rule as `mag_size`, and
    colour follows azimuth on a cyclic scale, because 0 and 180 degrees are the same direction.
    """
    keep = pl < PLUNGE_MAX
    d = m[keep]
    a = np.radians(np.asarray(az)[keep] % 180)
    half = 0.5 * 2.2 * (0.022 * 1.6 ** np.clip(d.mag.values, 0, 6)) * lon_span_km / panel_cm
    dx, dy = half * np.sin(a) / KM_LON, half * np.cos(a) / KM_LAT
    seg = np.stack([np.column_stack([d.lon - dx, d.lat - dy]),
                    np.column_stack([d.lon + dx, d.lat + dy])], axis=1)
    lc = LineCollection(seg, array=np.degrees(a), cmap=AZ_CMAP, norm=plt.Normalize(0, 180),
                        linewidths=.45, alpha=.75, zorder=3)
    ax.add_collection(lc)
    return lc, int(keep.sum())


def beachballs(ax, cells, xcol, ycol, base, in_km=False):
    """One beachball per cell, its width growing with the logarithm of the number summed.

    The nodal planes are not drawn; a thin rim is placed underneath instead, so that the white
    dilatational quadrants remain visible against the page.
    """
    from obspy.imaging.beachball import beach
    w = base * (0.55 + 0.45 * np.log10(cells.n) / np.log10(cells.n.max()))
    for r, wi in zip(cells.itertuples(), w):
        xy = (getattr(r, xcol), getattr(r, ycol))
        wd, ht = (wi, wi) if in_km else (wi / KM_LON, wi / KM_LAT)
        ax.add_patch(Ellipse(xy, wd, ht, facecolor="white", edgecolor="#444444", lw=.25, zorder=4))
        ax.add_collection(beach([r.strike, r.dip, r.rake], xy=xy,
                                width=wi if in_km else (wd, ht),
                                facecolor=FAULT_CMAP((r.faulting_index + 1) / 2),
                                linewidth=0, zorder=5))



def save(fig, name=None):
    """In the notebook a figure is shown, not written to disk."""
    plt.show()
    return None

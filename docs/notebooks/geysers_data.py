#!/usr/bin/env python
"""1 · Data — where every dataset in this course lives, and how each one is read.

Four facts are declared here and nowhere else. They drifted once already, and the third is what
that cost.

1. **Region.** Download box -123.30..-122.30 E, 38.44..39.20 N; study region the 40 km circle about
   (38.820 N, -122.803 E), the median NCSN epicentre for 2012-2026; field window -122.99..-122.65 E,
   38.70..38.899 N.
2. **Placeholders.** Catalogue rows with magnitude 0.0 and magnitude type `Unk` or `MU` are not
   measurements. They are excluded from every magnitude statistic.
3. **Strike.** In the NCSN `.mech` files the orientation columns are DIP DIRECTION, dip, rake
   (ncedc.org/pub/doc/cat5/ncsn.mech.txt). `strike = dip_direction - 90`. Reading that column as the
   strike rotates every mechanism by 90 degrees.
4. **One solution per event.** FPFIT can report several minima for the same earthquake; keep the
   lowest misfit. Quality means at least 10 first motions and a misfit of 0.2 or less -- applied by
   `quality()` in the functions cell, at the point of use rather than hidden inside a loader.

Two kinds of function live here: the ones that fetch from an archive, used to build the course's
release assets, and the ones that read those assets back. Nothing here draws anything.
"""
import io
import json
import pathlib
import re
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LightSource

# ── where things are
BUCKET = "https://ncedc-pds.s3.amazonaws.com"


RELEASE = "https://github.com/AI4EPS/EPS207_Observational_Seismology/releases/download/data-2026fall"


CEC = "https://ncedc.org/pub/assembled/geysersCEC"

# ── the region, in one place (fact 1). Longitudes first, then latitudes, in the order matplotlib
# wants for an axis: [west, east, south, north].
REGION = [-123.30, -122.30, 38.44, 39.20]        # the download box, and the map region
FIELD = [-122.99, -122.65, 38.70, 38.899]        # the producing field
CENTER = (38.820, -122.803)


RADIUS_KM = 40.0


KM_LAT = 111.19


KM_LON = 111.19 * np.cos(np.radians(CENTER[0]))


CATALOG_COLUMNS = ["id", "time", "latitude", "longitude", "depth", "mag", "magType", "nst", "gap",
                   "dmin", "rms", "horizontalError", "depthError", "magError", "magNst", "status", "type"]


# ───────────────────────────────────────────────────────────── fetching from the public archives
def fetch(url, tries=4, timeout=120, headers=None, ok=(200,)):
    """GET with retries. Returns the response, or None on a 404 (a month with no file)."""
    for k in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code in ok or r.status_code == 206:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"unreachable after {tries} tries: {url}")


def in_box(df, lat="latitude", lon="longitude"):
    """Rows inside the download box."""
    return df[df[lat].between(REGION[2], REGION[3]) & df[lon].between(REGION[0], REGION[1])]


def read_ehpcsv(content):
    """One NCEDC yearly catalogue file, cut to the box.

    The files carry a stray control byte in the current year and mixed time formats across years,
    so the encoding is forced and the timestamps parsed permissively.
    """
    d = pd.read_csv(io.BytesIO(content), low_memory=False, encoding="latin-1", encoding_errors="replace")
    d = d[d.type != "\x1a"]
    d["time"] = pd.to_datetime(d.time, format="mixed", utc=True)
    return in_box(d)


# fact 3: the three orientation fields are dip direction, dip, rake — not strike, dip, rake.
MECH_RE = re.compile(
    r"^(\d{8}) ?(\d{4})\s*([\d.]+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    r"\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*([A-Za-z]?)"
    r"\s*(\d{1,3})\s+(\d{1,2})\s*(-?\d{1,3})\s+([\d.]+)\s+(\d+)"
    r"(?:\s+(\d+)\s+(\d+)\s+(\d+))?\s*(\*?)\s*(\S+)\s*$")


def parse_mech(text):
    """NCSN first-motion (FPFIT) solutions from one monthly `.mech` file.

    A Y2K HYPO71 summary card, then dip direction, dip, rake, the misfit and the number of first
    motions; then the 90% ranges of strike, dip and rake, a `*` if the event has more than one
    solution, and the event id.
    """
    rows = []
    for line in text.splitlines():
        m = MECH_RE.match(line)
        if not m:
            continue
        (d, hm, s, la, lam, lo, lom, dep, mag, nph, gap, dmin, rms, erh, erz, mt,
         ddr, dip, rake, fit, nfm, dst, ddp, drk, multi, eid) = m.groups()
        lat, lon = int(la) + float(lam) / 60, -(int(lo) + float(lom) / 60)
        if not (REGION[2] <= lat <= REGION[3] and REGION[0] <= lon <= REGION[1]):
            continue
        rows.append(dict(id=eid, date=d, hhmm=hm, sec=float(s), lat=lat, lon=lon,
                         depth=float(dep), mag=float(mag), magtype=mt, nph=int(nph), gap=int(gap),
                         dip_direction=int(ddr), strike=(int(ddr) - 90) % 360,          # fact 3
                         dip=int(dip), rake=int(rake), misfit=float(fit), nfm=int(nfm),
                         d_strike=int(dst) if dst else np.nan, d_dip=int(ddp) if ddp else np.nan,
                         d_rake=int(drk) if drk else np.nan, multiple=bool(multi)))
    return rows


def best_solution(mech):
    """One row per event: the lowest-misfit FPFIT solution (fact 4)."""
    return mech.sort_values("misfit").drop_duplicates("id", keep="first").sort_values("time")


def s3_list(prefix, delimiter=None):
    """List a public bucket prefix over plain HTTPS. Returns (key, bytes) pairs, or folder names."""
    import xml.etree.ElementTree as ET
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    out, token = [], None
    while True:
        u = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000"
        if delimiter:
            u += f"&delimiter={delimiter}"
        if token:
            u += f"&continuation-token={requests.utils.quote(token)}"
        root = ET.fromstring(fetch(u).text)
        if delimiter:
            out += [p.find("s3:Prefix", ns).text for p in root.findall("s3:CommonPrefixes", ns)]
        else:
            out += [(c.find("s3:Key", ns).text, int(c.find("s3:Size", ns).text))
                    for c in root.findall("s3:Contents", ns)]
        t = root.find("s3:NextContinuationToken", ns)
        if t is None:
            return out
        token = t.text


RECORD = 4096          # miniSEED record length in the NCEDC continuous archive


def read_window(key, t0, seconds, opener=None):
    """A window out of one channel-day file without downloading the file.

    miniSEED records are fixed length and every header carries the start time of its first sample,
    so a time is found by bisection on the record index and only the records that overlap the window
    are read. `opener` returns bytes for a byte interval; the default uses HTTP Range requests, and
    the notebook passes an fsspec file object instead to show that the same code works on `s3://`.
    """
    import obspy
    from obspy.io.mseed.util import get_record_information
    url = f"{BUCKET}/{key}"
    calls = [0]

    def grab(a, b):
        calls[0] += 1
        if opener is not None:
            return opener(a, b)
        return fetch(url, headers={"Range": f"bytes={a}-{b}"}).content

    if opener is not None:
        size = opener(None, None)
    else:
        size = int(fetch(url, headers={"Range": "bytes=0-0"}).headers["Content-Range"].split("/")[-1])
    lo, hi = 0, size // RECORD - 1
    rec_time = lambda k: get_record_information(io.BytesIO(grab(k * RECORD, k * RECORD + 511)))["starttime"]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rec_time(mid) <= t0:
            lo = mid
        else:
            hi = mid
    n = 8
    while True:                                   # read forward until the window is covered
        raw = grab(lo * RECORD, (lo + n) * RECORD - 1)
        st = obspy.read(io.BytesIO(raw))
        if st[-1].stats.endtime >= t0 + seconds or n > 128:
            break
        n *= 2
    st.merge()
    st.trim(t0, t0 + seconds)
    return st, len(raw), calls[0]


def fsspec_opener(key):
    """An `opener` for read_window that reads the same object through fsspec on s3://."""
    import fsspec
    f = fsspec.open(f"s3://ncedc-pds/{key}", "rb", anon=True, default_block_size=8192).open()

    def opener(a, b):
        if a is None:                              # the size probe
            f.seek(0, 2)
            return f.tell()
        f.seek(a)
        return f.read(b - a + 1)
    return opener


# ─────────────────────────────────────────────────────────────────── reading the map-layer files
def segments(path):
    """A GMT multi-segment ASCII file as a list of (n, 2) arrays. No GMT needed to read one."""
    out, cur = [], []
    for line in open(path):
        if line.startswith(">"):
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = []
        else:
            v = line.split()
            if len(v) >= 2:
                cur.append((float(v[0]), float(v[1])))
    if len(cur) > 1:
        out.append(np.array(cur))
    return out


def load_layers(root):
    """The unpacked map layers: the elevation model, its hillshade, and the traces to draw."""
    root = pathlib.Path(root)
    geo = root / "geo"
    d = np.load(geo / "dem.npz")
    z, lon, lat = d["z"].astype(float), d["lon"], d["lat"]
    if lat[0] > lat[-1]:                                   # LightSource wants north up in array order
        z, lat = z[::-1], lat[::-1]
    dx = (lon[1] - lon[0]) * KM_LON * 1000
    dy = (lat[1] - lat[0]) * KM_LAT * 1000
    shade = LightSource(azdeg=315, altdeg=40).hillshade(z, dx=dx, dy=dy, vert_exag=1.6)
    osm = json.load(open(geo / "osm_geothermal_volcanic.json"))["elements"]
    plants = [(e.get("lon") or e["center"]["lon"], e.get("lat") or e["center"]["lat"])
              for e in osm if e.get("tags", {}).get("power") == "plant"]
    return dict(z=z, lon=lon, lat=lat, shade=shade,
                extent=[lon[0], lon[-1], lat[0], lat[-1]],
                faults=[s for f in sorted(geo.glob("faults_*.txt")) for s in segments(f)],
                water=segments(geo / "water.txt") if (geo / "water.txt").exists() else [],
                plants=plants)


# ──────────────────────────────────────────────── reading the course's own datasets, over HTTPS
# Seven of the eight hosted datasets are gzipped CSV, so most of what follows is one `pd.read_csv`
# of a URL plus whatever tidying that dataset needs. The eighth, the map layers, is a tar archive
# that `layers()` unpacks into ./layers on first use.
LAYER_DIR = pathlib.Path("layers")


def catalog(earthquakes_only=True):
    """The routine catalogue for the download box, placeholder magnitudes removed (fact 2)."""
    c = pd.read_csv(f"{RELEASE}/geysers_catalog_1969-2026.csv.gz", low_memory=False)
    c["time"] = pd.to_datetime(c.time, format="mixed", utc=True)
    c["year"] = c.time.dt.year
    if earthquakes_only:
        c = c[c.type == "eq"]
    return c[~(c.magType.isin(["Unk", "MU"]) & (c.mag == 0))].copy()


def mechanisms():
    """Every first-motion solution in the box. Apply `quality()` before using them (fact 4)."""
    m = pd.read_csv(f"{RELEASE}/geysers_mechanisms_1975-2026.csv.gz", dtype={"id": str})
    m["time"] = pd.to_datetime(m.time)
    return m


def production():
    """CalGEM monthly field totals, converted from tonnes to megatonnes."""
    p = pd.read_csv(f"{RELEASE}/geysers_injection_production_1969-2026.csv")
    p.index = pd.to_datetime(dict(year=p.year, month=p.month, day=1))
    p["production"] = p.production_t / 1e6
    p["injection"] = p.injection_t / 1e6
    return p


def coverage():
    """Months in which each station has continuous waveform files on the archive."""
    c = pd.read_csv(f"{RELEASE}/geysers_archive_coverage.csv.gz")
    c["t"] = pd.to_datetime(dict(year=c.year, month=c.month, day=1))
    return c


def phases(year=2016):
    """Every P and S arrival for events in the box, one release asset per year."""
    p = pd.read_csv(f"{RELEASE}/geysers_phases_{year}.csv.gz", low_memory=False)
    p["phase_time"] = pd.to_datetime(p.phase_time)
    return p


def station_polarity():
    """Per-station, per-epoch check of whether the archive's first motions follow the metadata.

    `agree_fraction` is how often the analyst picks at that station agree with the sign the channel
    metadata implies. A value near zero means the station's polarity is reversed with respect to the
    convention the published mechanisms use, and its readings have to be flipped before they can be
    compared with one. Built from every polarity pick in the NCEDC archive.
    """
    p = pd.read_csv(f"{RELEASE}/geysers_station_polarity.csv.gz")
    p["epoch_begin"] = pd.to_datetime(p.epoch_begin)
    p["epoch_end"] = pd.to_datetime(p.epoch_end.str.replace("3000", "2100", regex=False))
    return p


# The reference event used throughout §§ 5-7: one earthquake, one station list, so the waveforms,
# the picks and the magnitude are all talking about the same recordings. Chosen because it is large
# enough to be well recorded and still has S arrivals -- the field's largest events have none.
REFERENCE_EVENT = dict(id=72615575, time="2016-03-31T23:49:02.880000Z",
                       lat=38.82333, lon=-122.76283, depth=2.07, mag=2.91)


def event_stations(event_id=None, year=2016, max_km=40.0):
    """The stations that recorded one event, inside `max_km`, ordered by distance.

    Every section from § 5 to § 7 builds its station list this way, so all three show the same
    recordings without having to pass anything between them.
    """
    e = phases(year)
    e = e[e.event_id == (event_id or REFERENCE_EVENT["id"])]
    s = (e[e.distance_km <= max_km].sort_values("distance_km")
         .drop_duplicates("station").reset_index(drop=True))
    s["cha"] = s.instrument + "Z"
    s["loc"] = s.location.fillna("").astype(str).str.replace("nan", "", regex=False)
    s["d"] = s.distance_km
    s["klass"] = [("BG" if n == "BG" else
                   "broadband" if i in ("HH", "BH") else
                   "short-period" if i in ("EH", "SH", "EP") else "strong-motion")
                  for n, i in zip(s.network, s.instrument)]
    return s.rename(columns={"network": "net", "station": "sta"})


def _unpack():
    """Download and unpack the map-layer archive once, into ./layers."""
    if not (LAYER_DIR / "geo" / "dem.npz").exists():
        with urllib.request.urlopen(f"{RELEASE}/geysers_map_layers.tar.gz") as r:
            tar = tarfile.open(fileobj=io.BytesIO(r.read()))
            try:
                # `filter="data"` refuses members that would write outside the destination. It
                # arrived in Python 3.12 and was backported to later 3.9-3.11 patch releases, so it
                # is not available everywhere a student may run this -- DataHub, at the time of
                # writing, is one such place.
                tar.extractall(LAYER_DIR, filter="data")
            except TypeError:
                tar.extractall(LAYER_DIR)
    return LAYER_DIR


def sites():
    """The station table the station figures draw."""
    return pd.read_csv(_unpack() / "survey" / "stations_fig2_sites.csv")


def cec():
    """The 91 nodes of the California Energy Commission array."""
    return (pd.read_csv(_unpack() / "cec_stations_Sheet1.csv")
            .rename(columns={"Latitude": "lat", "Longitude": "lon"}))


def towns():
    """Named settlements from the OpenStreetMap query that built the map layers."""
    els = json.load(open(_unpack() / "geo" / "osm_places.json"))["elements"]
    return sorted({(e["lon"], e["lat"], e["tags"]["name"])
                   for e in els if "name" in e.get("tags", {})}, key=lambda t: t[2])


def power_plants():
    """The geothermal power plants: longitude, latitude and OpenStreetMap name."""
    els = json.load(open(_unpack() / "geo" / "osm_geothermal_volcanic.json"))["elements"]
    return [(e.get("lon") or e["center"]["lon"], e.get("lat") or e["center"]["lat"],
             e["tags"].get("name", "")) for e in els if e.get("tags", {}).get("power") == "plant"]


def layers():
    """Elevation model, geology, faults, water, the KGRA outline and the OSM places."""
    return load_layers(_unpack())          # from the drawing cell of § 0


DATA = LAYER_DIR                            # figure code that builds its own paths

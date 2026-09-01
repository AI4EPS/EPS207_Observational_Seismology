#!/usr/bin/env python
"""Topic 1 — Regression & uncertainty, taught through magnitude calibration."""
from topickit import md, run, ckpt, yrs, write

SLUG = "01_regression_uncertainty"

md(r"""# How do you build a magnitude scale?

**EPS 207 · Topic 1 · Regression and uncertainty**

Two papers, 88 years apart, solving the same problem.

**Richter (1935)** — a new instrument, the Wood–Anderson seismometer, and no way to turn its
amplitudes into a number meaning the same thing at every distance. He had to *measure* the distance
correction from data. That paper invented magnitude, and it is a regression.
[`10.1785/bssa0250010001`]

**Hutton & Boore (1987)** — fifty years later, the same correction refitted for southern California
with far more data. They published
$-\log_{10}A_0 = 1.110\log_{10}(R/100) + 0.00189(R-100) + 3.0$, with standard errors of **±0.017**
and **±0.0005** on those two coefficients. That is the curve you will compare your own fit against — **but not on equal terms.** They fitted
7,355 amplitudes spread across the whole of southern California, *with* per-station corrections,
over 10-700 km, discarding readings below 0.3 mm. We will fit one sequence in one box, with no
station terms, over 1-400 km, keeping readings 300x smaller. On p. 2077 they say explicitly that
they had enough data to break the region into smaller areas but chose not to. We are about to do the
sub-regional fit they declined. Any gap between the two numbers is at least as much about that as
about the Earth. [`10.1785/bssa0770062074`]

**The data.** Every reading in this notebook comes from the **2019 Ridgecrest sequence** — one
aftershock sequence inside a single 0.9-degree box of the eastern California desert, recorded over
one year. That matters more than anything else you will read here: a correction curve fitted to one
sequence in one small region is not the same object as one fitted to a whole province, and the
comparison you are about to make turns on that difference.

Code cells reading `# your code here` are yours to write. Everything else is written already — run
it and read it.""")

md("""## 1 · What is an amplitude measurement?""")
run("""# Berkeley DataHub does not ship obspy. Install it once per session if it is missing;
# on Colab or a local environment that already has it, this cell does nothing.
import sys, subprocess
try:
    import obspy
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "obspy"], check=True)
    import obspy
print(f"obspy {obspy.__version__}")""")

run("""import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings, time
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 9})

URL = ("https://github.com/AI4EPS/EPS207_Observational_Seismology/releases/download/"
       "data-2026fall/ridgecrest_amplitudes.csv.gz")
for attempt in range(3):
    try:
        d = pd.read_csv(URL, compression="gzip"); break
    except Exception as e:
        if attempt == 2:
            raise RuntimeError(f"fetch failed 3x ({e}); download {URL} by hand and read it locally")
        time.sleep(3)
# SCEDC files some readings under two amplitude IDs, so the same measurement appears twice.
# Collapse to one reading per channel per earthquake before anything else touches the table.
d = d.drop_duplicates(subset=["event_id", "station", "channel"]).reset_index(drop=True)
print(f"{len(d):,} readings, {d.event_id.nunique():,} events, {d.station.nunique()} stations")
d.columns.tolist()""")

md("""Each row is **one station's reading of one earthquake**. `amp_mm` is the amplitude a
Wood–Anderson seismograph would have recorded, `hyp_km` the hypocentral distance, `magnitude` the
catalogue magnitude of the event, and `station_magnitude` the magnitude the network derived from
*this single reading*.

one earthquake, seen by every station that recorded it.""")
run("""one = d[d.event_id == d.event_id.value_counts().idxmax()].sort_values("hyp_km")
print(f"event {one.event_id.iloc[0]}  M{one.magnitude.iloc[0]}  seen by {len(one)} channels "
      f"on {one.station.nunique()} stations")
one[["station", "channel", "hyp_km", "amp_mm", "station_magnitude"]].head(8)""")

run("""fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.loglog(one.hyp_km, one.amp_mm, "o", ms=4, alpha=.6, color="#2b6cb0")
ax.set(xlabel="hypocentral distance R (km)", ylabel="amplitude (mm)",
       title=f"One earthquake (M{one.magnitude.iloc[0]}), {len(one)} readings")
plt.tight_layout(); plt.show()
print(f"amplitude falls by a factor of {one.amp_mm.max()/one.amp_mm.min():,.0f} across "
      f"{one.hyp_km.min():.0f}–{one.hyp_km.max():.0f} km")""")

md("""## 2 · Do the amplitudes mean what you think?

You are about to regress hundreds of thousands of numbers somebody else measured. Before trusting a column,
**measure one of them yourself.** This is the only cell in the notebook that touches a raw
seismogram, and it exists so the rest of the session rests on something you checked.

A Wood–Anderson seismograph has not been built in decades. "Wood–Anderson amplitude" means: take
the real instrument's record, remove its response, and *simulate* what a Wood–Anderson would have
drawn — a damped pendulum with T₀ = 0.8 s, damping 0.8, static gain 2080.

one earthquake, one station, straight from the SCEDC archive.""")

run("""from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import numpy as np
import time

ROW = d[(d.event_id == "ci38457487") & (d.station == "CCC") & (d.channel == "HHN")].iloc[0]
print(f"catalogue row: {ROW.network}.{ROW.station}.{ROW.channel}  "
      f"M{ROW.magnitude}  R={ROW.hyp_km:.1f} km  amp_mm={ROW.amp_mm:.1f}")

SCEDC = Client("SCEDC", timeout=90)

def fetch(net, sta, cha, t, pre=20, post=80, tries=3):
    # One FDSN endpoint, everyone in the room at once. Retry, then say plainly what to do.
    for k in range(tries):
        try:
            return SCEDC.get_waveforms(net, sta, "*", cha, t-pre, t+post,
                                       attach_response=True)[0]
        except Exception as e:
            if k == tries - 1:
                raise RuntimeError(
                    f"SCEDC failed {tries}x for {net}.{sta}.{cha} ({type(e).__name__}). "
                    f"SCEDC may be throttling the room. Section 2 is self-contained - nothing after it "
                    f"depends on these waveforms - so re-run in a minute, or skip ahead."
                ) from e
            time.sleep(2 * (k + 1))

t0 = UTCDateTime(str(ROW.time))
tr = fetch(ROW.network, ROW.station, ROW.channel, t0)
print(f"downloaded {tr.stats.npts:,} samples at {tr.stats.sampling_rate:.0f} Hz")""")

run("""WOOD_ANDERSON = {"poles": [-6.2832 - 4.7124j, -6.2832 + 4.7124j],
                 "zeros": [0j], "gain": 1.0, "sensitivity": 2080.0}

def to_wood_anderson(trace):
    w = trace.copy()
    w.detrend("linear"); w.taper(0.05)
    w.remove_response(output="VEL", pre_filt=(0.05, 0.1, 20, 25), water_level=10)
    w.simulate(paz_simulate=WOOD_ANDERSON, water_level=10)
    return w.data * 1000.0                       # mm on the simulated pendulum

def largest_swing(a, with_index=False):
    # half the peak-to-peak of the largest SINGLE swing: adjacent turning points
    sgn = np.sign(np.diff(a)); sgn[sgn == 0] = 1
    turn = np.where(np.diff(sgn) != 0)[0] + 1
    k = np.argmax(np.abs(np.diff(a[turn])))
    half = abs(a[turn[k+1]] - a[turn[k]]) / 2
    return (half, turn[k], turn[k+1]) if with_index else half

wa_mm = to_wood_anderson(tr)
CONV = {"zero-to-peak":       np.abs(wa_mm).max(),
        "half peak-to-peak":  (wa_mm.max() - wa_mm.min()) / 2,
        "half largest swing": largest_swing(wa_mm)}
print(f"catalogue AML        {ROW.amp_mm:8.1f} mm")
for name, v in CONV.items():
    print(f"{name:20s} {v:8.1f} mm    ratio {v/ROW.amp_mm:.3f}")
measured_mm = CONV["zero-to-peak"]""")

md("""One station cannot settle this: for a roughly symmetric swing the three definitions differ
by less than a factor of two, so any of them would look like agreement. Repeat it across the
nearest broadband channels that recorded this earthquake.""")

run("""ev = d[(d.event_id == ROW.event_id) & (d.channel.str.startswith("HH"))]
rows = []
for _, r in ev.nsmallest(4, "hyp_km").iterrows():   # 4, not 8: eight students share one endpoint
    try:
        a = to_wood_anderson(fetch(r.network, r.station, r.channel, UTCDateTime(str(r.time))))
        rows.append((r.station, r.channel, r.amp_mm, np.abs(a).max(),
                     (a.max()-a.min())/2, largest_swing(a)))
    except Exception as e:
        print(f"  skipped {r.station}.{r.channel} ({type(e).__name__})")

cmp = pd.DataFrame(rows, columns=["station", "channel", "catalogue",
                                  "zero_pk", "half_p2p", "half_swing"])
assert len(cmp) >= 3, (f"only {len(cmp)} channel(s) returned - too few to separate the "
                       f"conventions. Re-run, or skip ahead; nothing later needs this.")
print(f"measured {len(cmp)} broadband channels")
for c in ["zero_pk", "half_p2p", "half_swing"]:
    ratio = cmp[c] / cmp.catalogue
    print(f"  {c:11s} median ratio {ratio.median():.3f}"
          f"   offset {abs(np.log10(ratio.median())):.3f} magnitude units")""")

run("""t = np.arange(tr.stats.npts) / tr.stats.sampling_rate - 20
z = CONV["zero-to-peak"]
_, i0, _ = largest_swing(wa_mm, with_index=True)

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(t, wa_mm, lw=.5, color="#2b6cb0")
for sign in (+1, -1):
    ax.axhline(sign * z, color="#c53030", lw=1, ls="--")
ax.set(xlim=(t[i0]-7, t[i0]+22), xlabel="seconds from origin time",
       ylabel="Wood-Anderson (mm)",
       title=f"{ROW.station}.{ROW.channel}  M{ROW.magnitude} at {ROW.hyp_km:.0f} km")
plt.tight_layout(); plt.show()""")

md(r"""**Three definitions, three numbers.** Now compare them with what the people who made this catalogue
say they did.

Hutton & Boore (1987) state their measurement exactly: *"The amplitudes are read as one-half the
peak-to-peak distance on the largest single swing of the S wave."* And the SCSN catalogue paper says
the same of its own practice — $M_L = \log_{10}A - \log A_0 + C_s$, where *"the amplitude that was
actually read was half the peak-to-peak distance on the largest single swing of the trace"*, and
*"for most records (≥ 95%), the peak occurred on the S wave"* (Hutton, Woessner & Hauksson 2010,
`10.1785/0120090130`).

Our simulation matches zero-to-peak more closely than either of those. We cannot resolve that from
here: both papers describe reading photographic records, neither states the procedure used on modern
digital data, and the network recalibrated again in 2008.

**The amplitude also falls steeply with distance** — the same earthquake looks a thousand times
smaller at 300 km than at 5 km. Before it can mean anything you have to remove that, and nobody
hands you the correction. You measure it.""")

md("""The catalogue also contains values that cannot be true. Look before you fit.""")

run("""print(f"amplitude spans {d.amp_mm.min():.2e} to {d.amp_mm.max():.2e} mm")
tiny = d[d.amp_mm < 1e-3]          # below a micron
print(f"\\nreadings below 1 micron: {len(tiny):,} ({100*len(tiny)/len(d):.2f}%)")
print(f"  their distances: {tiny.hyp_km.min():.0f}–{tiny.hyp_km.max():.0f} km, "
      f"median {tiny.hyp_km.median():.0f} km")
print(f"  their magnitudes: {tiny.magnitude.min():.1f}–{tiny.magnitude.max():.1f}")
print()
print("These are trace amplitudes on a simulated pendulum with a gain of 2080, so a")
print("micron of trace is about half a nanometre of ground - at or under the noise floor.")
print("These are measurements of noise, recorded as if they were signal.")""")

yrs("""# solution
before = len(d)
d = d[(d.amp_mm > 1e-3) & (d.hyp_km > 1) & (d.hyp_km < 400)].copy()
d["logA"] = np.log10(d.amp_mm)
print(f"kept {len(d):,} of {before:,} readings ({100*len(d)/before:.1f}%)")
print(f"logA now spans {d.logA.min():.2f} to {d.logA.max():.2f}")
print(f"distance {d.hyp_km.min():.1f}–{d.hyp_km.max():.0f} km, "
      f"magnitude {d.magnitude.min():.1f}–{d.magnitude.max():.1f}")""",
# planned: 4 min
"""
Decide what to exclude and do it. Drop readings below 1 µm, and cut the distance range to where you
actually have data. Recompute `d["logA"] = log10(amp_mm)` afterwards. Report how many readings you
kept.

Write down the cut you chose — **you will test at the end how much the answer depends on it.**""")

md(r"""## 3 · Why moment magnitude had to be invented

An earthquake is not a point. It is a rupture of finite size, and that size controls the *shape* of
the radiated spectrum, not just its amplitude. The standard description is Brune's (1970): the
far-field displacement spectrum is **flat at low frequency** and **falls as $f^{-2}$ above a corner
frequency** $f_c$,

$$\Omega(f) = \frac{\Omega_0}{1 + (f/f_c)^2}, \qquad \Omega_0 \propto M_0$$

Two facts do all the work.

**1. The flat level measures the earthquake.** $\Omega_0$ is proportional to the seismic moment
$M_0 = \mu A D$ — rigidity × rupture area × slip. That is the physical size of the event.

**2. The corner moves *down* as the earthquake gets bigger.** A larger rupture takes longer to
happen, so its radiation is shifted to longer periods. Under constant stress drop $\Delta\sigma$,
$M_0 \propto r^3$ and $f_c \propto \beta/r$, giving

$$f_c \propto \Delta\sigma^{1/3} M_0^{-1/3}$$

Now put an instrument on it. A Wood–Anderson responds around **1.25 Hz** ($T_0$ = 0.8 s), and it
measures amplitude *in that band* — it cannot see $\Omega_0$ directly.

- **Small earthquake**: $f_c$ is well above 1.25 Hz, so the instrument sits on the flat part.
  Measured amplitude $\propto \Omega_0 \propto M_0$. One magnitude unit per decade of moment. The
  scale works.
- **Large earthquake**: $f_c$ has dropped *below* 1.25 Hz, so the instrument is sampling the
  $f^{-2}$ tail. There the amplitude is $\Omega_0 (f_c/f)^2 \propto M_0 \cdot M_0^{-2/3} =
  M_0^{1/3}$. **Growth compresses by a factor of three**, and the scale saturates.

This is why every amplitude-at-a-fixed-period scale saturates, each at its own size: $m_b$ near 6.5
(1 s), $M_S$ near 8 (20 s), $M_L$ around 6 (SCSN's own catalogue paper puts it near 6.3). Note the cartoon in the next
figure bends earlier than that, because it treats the instrument as a single frequency rather than
a band. And it is why $M_w$ does not — it is defined from
$M_0$ itself, $M_w = \tfrac{2}{3}\log_{10}M_0 - 6.07$ (Hanks & Kanamori 1979), so it measures the
flat level rather than a band.

the mechanism, drawn.""")

run(r"""BETA, DSIGMA = 3500.0, 3e6            # shear velocity m/s, stress drop 3 MPa
f = np.logspace(-2, 1.5, 400)
F_WA = 1/0.8                            # Wood-Anderson centre frequency, Hz

def brune(Mw):
    M0 = 10**(1.5*(Mw + 6.07))          # N m, from Hanks & Kanamori
    r  = (7*M0/(16*DSIGMA))**(1/3)      # source radius, m
    fc = 0.37*BETA/r
    return M0/(1 + (f/fc)**2), fc, M0

fig, ax = plt.subplots(figsize=(5.6, 3.8))
for Mw, col in zip([3, 5, 7], ["#90cdf4", "#4299e1", "#1a365d"]):
    sp, fc, _ = brune(Mw)
    ax.loglog(f, sp, color=col, lw=1.8, label=f"$M_w$ {Mw}   $f_c$={fc:.2f} Hz")
    ax.plot(fc, np.interp(fc, f, sp), "o", color=col, ms=5)
ax.axvline(F_WA, color="#c53030", lw=1.4, ls="--")
ax.annotate("Wood-Anderson band", (F_WA*1.15, 3e14), color="#c53030", fontsize=8)
ax.set(xlabel="frequency (Hz)", ylabel=r"displacement spectrum $\Omega$ (N m)",
       title="Bigger earthquakes move their corner LEFT")
ax.legend(fontsize=7.5, loc="lower left")
plt.tight_layout(); plt.show()""")

md("""Now ask what a **fixed-period instrument** sees as the corner sweeps past it.""")

run("""MW = np.linspace(2, 8, 200)
amp = np.array([np.interp(F_WA, f, brune(m)[0]) for m in MW])
M0s = np.array([brune(m)[2] for m in MW])
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(MW, np.log10(amp) - np.log10(amp[0]), lw=2, color="#2b6cb0",
        label="what a Wood-Anderson measures")
ax.plot(MW, np.log10(M0s) - np.log10(M0s[0]), lw=1.6, ls="--", color="#38a169",
        label=r"what $M_0$ does (slope 1.5)")
ax.set(xlabel="$M_w$", ylabel="log$_{10}$ amplitude (relative)",
       title="The measured amplitude bends over; the moment does not")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

lo = np.polyfit(MW[MW < 3.5], np.log10(amp)[MW < 3.5], 1)[0]
hi = np.polyfit(MW[MW > 6.5], np.log10(amp)[MW > 6.5], 1)[0]
print(f"slope of measured amplitude vs Mw:  {lo:.2f} at small M,  {hi:.2f} at large M")
print(f"the ratio is {lo/hi:.1f} — the factor of 3 the f^-2 falloff predicts")
print()
print("Careful: that 2.9 is a fit to OUR OWN synthetic curve against its own asymptotes.")
print("It checks the code, not the Earth. The catalogue test comes next.")""")

md("""**The figure above is the mechanism**: three earthquakes, three corner frequencies, and the
instrument's fixed window. At M3 the corner is far to the right of the red line and the instrument
sees the flat level. At M7 the corner has moved to the *left* of it, and the instrument is measuring
the falling tail.

**The next figure is the consequence.** The dashed line is what the moment does — slope 1.5 per
magnitude unit, forever. The solid line is what the instrument records: the same slope while the
corner is above the band, then a bend to a third of it.

Now check the prediction against the catalogue. For large events we have both a catalogue
M<sub>w</sub> from moment-tensor inversion and the M<sub>L</sub> the network's own stations
reported, so the gap between them should open up exactly where the theory says.""")

run("""ev = (d.groupby("event_id")
        .agg(catalog=("magnitude", "first"), n_sta=("station_magnitude", "size"),
             median_ML=("station_magnitude", "median"))
        .query("n_sta >= 20"))
ev["offset"] = ev.median_ML - ev.catalog
print(ev.sort_values("catalog", ascending=False).head(5).round(2).to_string())""")

run("""fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.scatter(ev.catalog, ev.offset, s=8, alpha=.25, color="#4a5568")
b = pd.cut(ev.catalog, np.arange(2.0, 7.5, 0.25))
m = ev.groupby(b, observed=True).offset.median()
ctr = [i.mid for i in m.index]
ax.plot(ctr, m.values, "o-", color="#c53030", lw=1.6, ms=4, label="median in 0.25 bins")
ax.axhline(0, color="k", lw=.8)
ax.set(xlabel="catalogue magnitude", ylabel=r"median station $M_L$ − catalogue",
       title="Below M3.4 the catalogue IS the station average. Above it, the scales part.")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()""")

md("""Two things. Below ~M3.4 the offset is zero **by construction** — there the catalogue magnitude
*is* the average of those station M<sub>L</sub>s. Above it the catalogue switches to M<sub>w</sub>
and the two scales separate. And at the top, the M7.1: its own stations say 6.72 against a
catalogue 7.10 — **$M_L$ under-reads by 0.38.**

That is saturation, and it is also a warning about the cartoon. The Brune model above, taken
literally, predicts a deficit of well over a magnitude unit at $M_w$ 7.1; the catalogue shows 0.38.
The mechanism is right in direction and wrong in size, because a Wood-Anderson is not a filter at
1.25 Hz — it is flat to *displacement* above it, so the measured peak integrates the whole band
rather than sampling one frequency. **Keep the mechanism; do not trust the cartoon's numbers.**

**The ground truth changes in the middle of your dataset.** Remember it when you read the fit.

## 4 · Building the model, one term at a time

Do not write the final model down. Build it, and let the residuals tell you what is missing — that
is what fitting actually looks like.

**Start with magnitude alone**: $\\log_{10}A = c_0 + c_1 M$.""")
run("""def ols(X, y):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    s2 = r @ r / (len(y) - X.shape[1])
    return b, np.sqrt(np.diag(XtX_inv) * s2), r, s2, XtX_inv

y = d.logA.values
X1 = np.column_stack([np.ones(len(d)), d.magnitude])
b1, se1, r1, s2_1, _ = ols(X1, y)
print(f"model 1  logA ~ M            residual sd = {np.sqrt(s2_1):.3f}")""")

run("""fig, ax = plt.subplots(figsize=(6.5, 3.4))
k = rng.choice(len(d), 30000, replace=False)          # positional, so any index works
ax.semilogx(d.hyp_km.values[k], r1[k], ".", ms=1.5, alpha=.2, color="#4a5568")
ax.axhline(0, color="#c53030", lw=1.2)
ax.set(xlabel="hypocentral distance R (km)", ylabel="residual",
       title="Model 1 residuals: a clear trend with distance — the model is missing R")
plt.tight_layout(); plt.show()""")

md("""The residuals slope with distance, so distance is missing. Geometric spreading predicts
amplitude falls as a power of $R$, which is a straight line in $\\log R$.

**Add $\\log_{10} R$**, then look again.""")
run("""X2 = np.column_stack([np.ones(len(d)), d.magnitude, np.log10(d.hyp_km)])
b2, se2, r2, s2_2, _ = ols(X2, y)
X3 = np.column_stack([np.ones(len(d)), d.magnitude, np.log10(d.hyp_km), d.hyp_km])
b3, se3, r3, s2_3, XtX_inv = ols(X3, y)
print(f"model 1  logA ~ M                    sd = {np.sqrt(s2_1):.3f}")
print(f"model 2  logA ~ M + log R            sd = {np.sqrt(s2_2):.3f}")
print(f"model 3  logA ~ M + log R + R        sd = {np.sqrt(s2_3):.3f}")""")



run("""fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
idx = rng.choice(len(d), 25000, replace=False)
for ax, r, t in zip(axs, [r1, r2, r3], ["M only", "+ log R", "+ log R + R"]):
    ax.semilogx(d.hyp_km.values[idx], r[idx], ".", ms=1.2, alpha=.15, color="#4a5568")
    bins = np.logspace(np.log10(d.hyp_km.min()), np.log10(d.hyp_km.max()), 25)
    who = np.digitize(d.hyp_km.values, bins)
    med = [np.median(r[who == k]) if (who == k).sum() > 30 else np.nan
           for k in range(1, len(bins))]
    ax.plot(bins[:-1], med, "-", color="#c53030", lw=1.8)
    ax.axhline(0, color="k", lw=.7); ax.set(title=t, xlabel="R (km)", ylim=(-2, 2))
axs[0].set_ylabel("residual")
plt.suptitle("Each term removes structure the previous model left behind", y=1.02)
plt.tight_layout(); plt.show()""")

md("""The red line is the median residual in distance bins — the structure the model has *not*
explained. It flattens as each term goes in. **That is how you decide a model is done**: not by
$R^2$, but by looking at what is left.

The final model is Richter's, with the two terms that carry physics:

$$\\log_{10}A = c_0 + c_1 M + \\underbrace{c_2 \\log_{10}R}_{\\text{geometric spreading}} +
\\underbrace{c_3 R}_{\\text{anelastic attenuation}}$$

Compare with the published values.""")
run("""TERMS = ["c0 constant", "c1 magnitude", "c2 log10 R", "c3 R"]
HB    = [None, 1.0,  -1.110,  -0.00189]      # Hutton & Boore 1987, eq. for -log A0
HB_SE = [None, None,  0.017,   0.0005]       # their published standard errors
print(f"{'term':16s}{'fitted':>10}{'std err':>10}{'H&B 1987':>12}{'their SE':>10}{'gap/their SE':>14}")
for t, b, s, h, hs in zip(TERMS, b3, se3, HB, HB_SE):
    if h is None:
        print(f"{t:16s}{b:10.4f}{s:10.4f}")
    elif hs is None:
        print(f"{t:16s}{b:10.4f}{s:10.4f}{h:12.4f}")
    else:
        print(f"{t:16s}{b:10.4f}{s:10.4f}{h:12.4f}{hs:10.4f}{abs(b-h)/hs:13.1f}")
print(f"\\nresidual sd = {np.sqrt(s2_3):.3f} log10 units")
print(f"condition number of X'X = {np.linalg.cond(X3.T @ X3):.3e}")""")

ckpt(1, """y = d.logA.values
X3 = np.column_stack([np.ones(len(d)), d.magnitude, np.log10(d.hyp_km), d.hyp_km])
b3, se3, r3, s2_3, XtX_inv = ols(X3, y)
print(f"model refitted: c2 = {b3[2]:.4f}")""")

md(r"""**Note that condition number.** $(X^\top X)$ is nearly singular, which is a warning that some
combination of the columns is barely constrained. Hold that thought.

## 5 · How well do we know the coefficients?

Everything so far has been `np.linalg` doing the work. Before trusting the numbers it returns, derive
them once.

**The estimator.** Least squares minimises the sum of squared residuals,

$$J(\beta) = \|y - X\beta\|^2 = (y - X\beta)^\top(y - X\beta)
          = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X\beta$$

Differentiate with respect to $\beta$ and set to zero:

$$\nabla J = -2X^\top y + 2X^\top X\beta = 0
\qquad\Longrightarrow\qquad
X^\top X\,\hat\beta = X^\top y$$

Those are the **normal equations**, and if $X^\top X$ can be inverted,

$$\hat\beta = (X^\top X)^{-1}X^\top y$$

**Its uncertainty.** Notice that $\hat\beta$ is *linear* in $y$. So if the data really were generated
by $y = X\beta + \varepsilon$ with $\mathbb{E}[\varepsilon] = 0$ and
$\mathrm{Var}(\varepsilon) = \sigma^2 I$, then $\hat\beta$ is unbiased, and its covariance follows
from $\mathrm{Var}(Ay) = A\,\mathrm{Var}(y)\,A^\top$ with $A = (X^\top X)^{-1}X^\top$:

$$\mathrm{Var}(\hat\beta) = A(\sigma^2 I)A^\top
 = \sigma^2 (X^\top X)^{-1}X^\top X (X^\top X)^{-1}
 = \boxed{\sigma^2 (X^\top X)^{-1}}$$

$\sigma^2$ is unknown, so estimate it from the residuals, $\hat\sigma^2 = \|r\|^2/(n-p)$, and the
standard error of a single coefficient is the square root of a diagonal entry,
$\mathrm{SE}(\hat\beta_j) = \hat\sigma\sqrt{[(X^\top X)^{-1}]_{jj}}$.

**Read what that says.** All the uncertainty comes through $(X^\top X)^{-1}$ — the geometry of where
you sampled — scaled by how badly the model fits. It says nothing about whether the model is right,
and that is the whole of sections 7 and 8.

$\mathrm{Cov}(\hat{\boldsymbol\beta}) = \sigma^2 (X^\top X)^{-1}$. The diagonal gave the
standard errors. The **off-diagonal** explains the condition number.""")
run("""cov = s2_3 * XtX_inv
corr = cov / np.outer(se3, se3)
print(pd.DataFrame(corr, index=[t.split()[0] for t in TERMS],
                   columns=[t.split()[0] for t in TERMS]).round(3).to_string())
print()
print(f"correlation of the PREDICTORS  log10(R) vs R : "
      f"{np.corrcoef(np.log10(d.hyp_km), d.hyp_km)[0,1]:+.3f}")
print(f"correlation of the ESTIMATES   c2-hat vs c3-hat: {corr[2,3]:+.3f}")
print("Both are monotone in R, so the predictors MUST correlate positively;")
print("the negative number is a property of the estimates, not of the columns.")""")

md("""See the trade-off directly, by refitting on resampled data.""")
run("""B, n = 300, len(d)

def boot_sd(unit=None, size=40000, reps=B):
    # Resample rows, or whole events/stations. Returns the sd of c2 across replicates.
    out = np.empty(reps)
    if unit is not None:
        groups = d.groupby(unit).indices          # label -> row positions
        labels = np.array(list(groups))
    for i in range(reps):
        if unit is None:
            k = rng.choice(n, size, replace=True)
        else:
            pick = rng.choice(labels, len(labels), replace=True)
            k = np.concatenate([groups[g] for g in pick])
        Xb = X3[k]
        out[i] = np.linalg.solve(Xb.T @ Xb, Xb.T @ y[k])[2]
    return out.std()

boot = np.empty((B, 4))
for i in range(B):
    k = rng.choice(n, 40000, replace=True)
    Xb = X3[k]
    boot[i] = np.linalg.solve(Xb.T @ Xb, Xb.T @ y[k])
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
axs[0].scatter(boot[:, 2], boot[:, 3], s=9, alpha=.5, color="#2b6cb0")
axs[0].plot(-1.110, -0.00189, "*", ms=14, color="#c53030", label="Hutton & Boore")
axs[0].set(xlabel="$c_2$  (geometric spreading)", ylabel="$c_3$  (anelastic)",
           title=f"Bootstrap: $c_2$ and $c_3$ trade off (r = {corr[2,3]:+.2f})")
axs[0].legend(fontsize=8)
axs[1].hist(boot[:, 1], bins=30, color="#90cdf4", edgecolor="white")
axs[1].axvline(1.0, color="#c53030", lw=1.5, label="Richter's definition, $c_1=1$")
axs[1].set(xlabel="$c_1$  (magnitude)", ylabel="bootstrap resamples", title="Is $c_1$ equal to 1?")
axs[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
NB = 40000
print(f"analytic sd of c2 (formula, assumes independent rows) : {se3[2]:.4f}")
print(f"bootstrap resampling ROWS    (n={NB:,})              : {boot[:,2].std():.4f}"
      f"   [formula at that n: {se3[2]*np.sqrt(n/NB):.4f}]")
print("  -> agrees with the formula. An iid row bootstrap CANNOT detect clustering;")
print("     it makes the same independence assumption the formula does.")
print()
sd_ev, sd_st = boot_sd("event_id", reps=60), boot_sd("station", reps=60)
print(f"bootstrap resampling whole EVENTS   : {sd_ev:.4f}   ({sd_ev/se3[2]:.0f}x the formula)")
print(f"bootstrap resampling whole STATIONS : {sd_st:.4f}   ({sd_st/se3[2]:.0f}x the formula)")
print("  -> readings are not independent. The effective sample size for anything")
print("     path-related is the number of stations, not the number of rows.")""")

md(r"""The left panel is **collinearity made visible**: the cloud is a diagonal ridge, not a blob.
Any $c_2$ can be compensated by a matching $c_3$. The fit constrains the *combination* tightly and
each coefficient loosely — so *geometric spreading* and *anelastic attenuation* are **not separately
measured by this experiment**, however small their standard errors look.

When two parameters trade off, the fit constrains their combination tightly and each one loosely.
More rows over the same distance range shrink both standard errors while leaving the correlation
where it is: what breaks a trade-off is a wider range of $R$, not more of the same. To get a unique,
stable answer you must add information from outside the data.

""")

md(r"""## 6 · Adding information the data does not contain

**You already know one way to do this, under a different name.** Damped least squares — the standard
fix for an ill-conditioned inverse problem — minimises $\|y - X\beta\|^2 + \lambda\|\beta\|^2$
instead of the misfit alone. Statistics calls the same estimator **ridge regression**. What follows
is why the damping term is not a numerical trick.

### 1 · Write the forward model probabilistically

$$y = X\beta + \varepsilon, \qquad \varepsilon \sim N(0,\ \sigma^2 I_n)$$

Gaussian noise of variance $\sigma^2$ on each of the $n$ observations, independent. That makes the
**likelihood** — the probability of the data you actually recorded, for a candidate $\beta$ —

$$p(y \mid \beta) = (2\pi\sigma^2)^{-n/2}\exp\!\left(-\frac{\|y - X\beta\|^2}{2\sigma^2}\right)$$

Maximising this alone *is* ordinary least squares: the exponent contains the misfit and nothing else.

Worth checking rather than believing. Maximise the likelihood numerically, from a deliberately bad
start, and see whether it lands on the normal-equations solution.""")

run("""from scipy.optimize import minimize

def neg_loglik(b, sigma):
    r = y - X3 @ b
    return 0.5*len(y)*np.log(2*np.pi*sigma**2) + (r @ r) / (2*sigma**2)

sigma_hat = np.sqrt(s2_3)
mle = minimize(neg_loglik, x0=np.zeros(4), args=(sigma_hat,), method="BFGS",
               options={"gtol": 1e-10})

print(f"{'':14s}{'c0':>10s}{'c1':>10s}{'c2':>10s}{'c3':>11s}")
print(f"{'normal eqns':14s}" + "".join(f"{v:10.4f}" for v in b3[:3]) + f"{b3[3]:11.5f}")
print(f"{'max likelihood':14s}" + "".join(f"{v:10.4f}" for v in mle.x[:3]) + f"{mle.x[3]:11.5f}")
print(f"largest difference: {np.abs(mle.x - b3).max():.2e}")""")

md(r"""### 2 · Write down the prior

The claim "the coefficients are probably small" becomes: each $\beta_j$ is drawn independently from
a Gaussian centred on zero with variance $s^2$.

$$p(\beta) = (2\pi s^2)^{-p/2}\exp\!\left(-\frac{\|\beta\|^2}{2s^2}\right)$$

This is stated *before* seeing $y$. $s$ has the units of $\beta$ and encodes how big you think a
coefficient plausibly is.

### 3 · Bayes' theorem

$$p(\beta \mid y) = \frac{p(y \mid \beta)\, p(\beta)}{p(y)} \;\propto\; p(y \mid \beta)\, p(\beta)$$

The denominator $p(y)$ does not involve $\beta$, so it is an irrelevant constant for finding the
maximum.

### 4 · Take the negative logarithm

Maximising the posterior is the same as minimising its negative log, and the log turns the product
into a sum:

$$-\log p(\beta \mid y) = \frac{\|y - X\beta\|^2}{2\sigma^2} + \frac{\|\beta\|^2}{2s^2} + \text{const}$$

Both exponents were quadratic, so the whole thing is quadratic in $\beta$.

### 5 · Rescale to recover the damped objective

Multiply through by $2\sigma^2$ — a positive constant, so it moves the value of the function but not
the location of its minimum:

$$\|y - X\beta\|^2 + \frac{\sigma^2}{s^2}\|\beta\|^2$$

Compare with the damped least-squares objective $\|y - X\beta\|^2 + \lambda\|\beta\|^2$. They are
identical, with

$$\boxed{\lambda = \sigma^2 / s^2}$$

That is the whole equivalence. **Damping is not a numerical trick; it is the log-prior term.** Read
the ratio: $\lambda$ is large when the data are noisy or when you are confident the coefficients are
small, and $\lambda \to 0$ returns ordinary least squares — the statement that you had no prior
opinion at all.

### 6 · Minimise, to get the estimator

$$J(\beta) = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X\beta + \lambda\beta^\top\beta$$

Differentiate and set to zero:

$$\nabla J = -2X^\top y + 2X^\top X\beta + 2\lambda\beta = 0$$

$$(X^\top X + \lambda I)\,\beta = X^\top y
\qquad\Longrightarrow\qquad
\hat\beta_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$$

The Hessian is $2(X^\top X + \lambda I)$, positive definite for any $\lambda > 0$ **even when
$X^\top X$ is singular** — so the minimum exists and is unique. And because the posterior is
Gaussian, this MAP point is also the posterior *mean*, with covariance
$\sigma^2(X^\top X + \lambda I)^{-1}$.

### 7 · Why $\lambda I$ actually cures the ill-conditioning

Use the SVD $X = UDV^\top$ with singular values $d_i$. Then $X^\top X = VD^2V^\top$, and

- **OLS:** $\hat\beta = VD^{-1}U^\top y$ — each component is divided by $d_i$, so a near-zero
  singular value amplifies noise without bound.
- **Ridge:** $\hat\beta_\lambda = V\,\mathrm{diag}\!\left(\dfrac{d_i}{d_i^2 + \lambda}\right)U^\top y$

Equivalently, each OLS component is multiplied by a **filter factor**
$f_i = d_i^2/(d_i^2 + \lambda)$:

- $d_i^2 \gg \lambda$: $f_i \approx 1$ — well-resolved directions pass through untouched.
- $d_i^2 \ll \lambda$: $f_i \approx 0$ — unresolved directions are shrunk toward the prior mean
  instead of exploding.

And as $d_i \to 0$ the ridge factor $d_i/(d_i^2 + \lambda) \to 0$ rather than $\infty$.

Our own design matrix has four singular values. Compute them, and the filter factors.""")
run("""# Is ridge really the MAP estimate? Pick a prior width, derive lambda from it, and compare
# the closed-form ridge solution against a direct maximisation of the log-posterior.
s_prior = 0.5                       # we believe each coefficient is within roughly +-0.5 of zero
lam_map = s2_3 / s_prior**2         # lambda = sigma^2 / s^2

b_ridge = np.linalg.solve(X3.T @ X3 + lam_map*np.eye(4), X3.T @ y)

def neg_logpost(b):
    r = y - X3 @ b
    return (r @ r) / (2*s2_3) + (b @ b) / (2*s_prior**2)

b_map = minimize(neg_logpost, x0=np.zeros(4), method="BFGS",
                 options={"gtol": 1e-12}).x

print(f"prior sd s = {s_prior},  sigma = {np.sqrt(s2_3):.3f}  ->  lambda = {lam_map:.4f}")
print()
print(f"{'':22s}{'c1':>10s}{'c2':>10s}{'c3':>11s}")
print(f"{'ridge, closed form':22s}" + "".join(f"{v:10.4f}" for v in b_ridge[1:3]) + f"{b_ridge[3]:11.5f}")
print(f"{'MAP, maximised':22s}" + "".join(f"{v:10.4f}" for v in b_map[1:3]) + f"{b_map[3]:11.5f}")
print(f"largest difference: {np.abs(b_ridge - b_map).max():.2e}")""")

run("""d_sv = np.linalg.svd(X3, compute_uv=False)
print("singular values of X:", "  ".join(f"{v:.3g}" for v in d_sv))
print(f"condition number d_max/d_min = {d_sv[0]/d_sv[-1]:.3g}")
print()
print(f"{'lambda':>10s}" + "".join(f"{'f'+str(i+1):>9s}" for i in range(len(d_sv))))
for lam in [0.0, 1e-2, 1.0, 1e2, 1e4]:
    f = d_sv**2 / (d_sv**2 + lam)
    print(f"{lam:10.2f}" + "".join(f"{v:9.4f}" for v in f))
print()
print("f ~ 1 means the direction survives; f ~ 0 means it is replaced by the prior.")""")

run("""lams = np.logspace(-1, 6, 40)
path = np.array([np.linalg.solve(X3.T @ X3 + l*np.eye(4), X3.T @ y) for l in lams])
rms  = [np.sqrt(np.mean((y - X3 @ p)**2)) for p in path]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for j, lab in [(2, "$c_2$ log R"), (3, r"$c_3\\times$100")]:
    axs[0].semilogx(lams, path[:, j] * (100 if j == 3 else 1), lw=1.6, label=lab)
axs[0].axhline(-1.110, ls="--", color="#c53030", lw=1, label="H&B $c_2$")
axs[0].set(xlabel=r"$\\lambda$", ylabel="coefficient", title="Ridge path")
axs[0].legend(fontsize=8)
axs[1].semilogx(lams, rms, lw=1.6, color="#2b6cb0")
axs[1].set(xlabel=r"$\\lambda$", ylabel="rms residual", title="What the prior costs you")
plt.tight_layout(); plt.show()""")

md(r"""$c_2$ travels a long way for almost no cost in rms — the data barely distinguishes it from
$c_3$, exactly as the bootstrap cloud showed.

## 7 · Two different questions, two different intervals

Both intervals come from the covariance you just derived. Only the second question has an extra term,
and it is worth seeing where it comes from.

**Where is the line?** At a point $x$ the fitted mean is $x^\top\hat\beta$, a linear function of
$\hat\beta$, so

$$\mathrm{Var}(x^\top\hat\beta) = x^\top \mathrm{Var}(\hat\beta)\, x
 = \sigma^2\, x^\top (X^\top X)^{-1} x$$

Every factor of that shrinks as you add data, because $(X^\top X)^{-1}$ does.

**What will one station read?** A future observation at the same $x$ is
$y_{\text{new}} = x^\top\beta + \varepsilon_{\text{new}}$, and you predict it with
$x^\top\hat\beta$. The prediction error is

$$x^\top\hat\beta - y_{\text{new}} = \underbrace{x^\top(\hat\beta - \beta)}_{\text{where the line is}}
 \;-\; \underbrace{\varepsilon_{\text{new}}}_{\text{that station's own noise}}$$

$\varepsilon_{\text{new}}$ belongs to an earthquake that has not happened yet, so it took no part in
the fit and is independent of $\hat\beta$. Variances of independent terms add:

$$\mathrm{Var} = \sigma^2\, x^\top (X^\top X)^{-1} x + \sigma^2
 = \boxed{\sigma^2\left(1 + x^\top (X^\top X)^{-1} x\right)}$$

**That $1$ is the whole difference.** It does not contain $(X^\top X)^{-1}$, so it does not shrink
with $n$: no amount of data tells you what the next station will do, because that station's noise has
not happened yet.

Now evaluate both at M3.0 and see how far apart they are.""")
run("""Rg = np.logspace(np.log10(5), np.log10(400), 200); M0 = 3.0
Xg = np.column_stack([np.ones_like(Rg), np.full_like(Rg, M0), np.log10(Rg), Rg])
vm = np.einsum("ij,jk,ik->i", Xg, cov, Xg)
ci, pi = 1.96*np.sqrt(vm), 1.96*np.sqrt(vm + s2_3)
k = np.argmin(abs(Rg - 50))
print(f"at M{M0}, R=50 km:  confidence ±{ci[k]:.4f}   prediction ±{pi[k]:.4f}"
      f"   ratio {pi[k]/ci[k]:.0f}x")""")

md(r"""**These two have standard names.** *Aleatoric* uncertainty is scatter more data cannot remove — one
station's spread about the model, which is what the prediction interval is made of. *Epistemic*
uncertainty is not knowing the right model; the covariance you just derived is only the part of it
that concerns the coefficients of the form you already chose, which is why a confidence interval can
be narrow around a model that is wrong.

Neither word is seismological — the split is standard across uncertainty quantification and routine
in machine learning. What seismology adds is a decomposition of the aleatoric part into between-event
and within-event terms, and the within-event part again into site-to-site and single-station.

**And the split is a property of your model, not of the Earth.** Whatever the model does not explain
gets called irreducible. Section 7 adds a station term and watches some of this "irreducible" scatter
become structure.""")

run("""mu = Xg @ b3
sub = d[(d.magnitude > 2.9) & (d.magnitude < 3.1)]

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.scatter(sub.hyp_km, sub.logA, s=1.5, alpha=.08, color="#718096")
ax.fill_between(Rg, mu-pi, mu+pi, color="#dd6b20", alpha=.30, lw=0,
                label="95% prediction (one station)")
ax.plot(Rg, mu-ci, color="#2b6cb0", lw=1)
ax.plot(Rg, mu+ci, color="#2b6cb0", lw=1, label="95% confidence (the mean)")
ax.set_xscale("log")
ax.set(xlabel="R (km)", ylabel=r"$\\log_{10}$A", ylim=(-2.2, 2.2),
       title=f"M 2.9-3.1  (n={len(sub):,})")
ax.legend(fontsize=7.5, loc="lower left")
plt.tight_layout(); plt.show()""")

md("""The confidence band is the hairline you can barely see; the prediction band is the wide one. The
printed numbers above give the gap: **two orders of magnitude and more**. No amount of extra
data narrows the orange one.

That is most of the answer to a question you will be asked as a seismologist: *why do two agencies
report different magnitudes for the same earthquake?*""")

yrs("""# solution
d["resid"] = y - X3 @ b3
g = d.groupby("station").resid
mu_s, n_s = g.transform("mean"), g.transform("size")
m = n_s >= 50
tot, between = d.resid[m].std(), mu_s[m].std()
within = (d.resid[m] - mu_s[m]).std()
print(f"{d.station[m].nunique()} stations, {m.sum():,} readings")
print(f"  total sd        {tot:.3f}")
print(f"  between-station {between:.3f}   {100*between**2/tot**2:.0f}% of the variance")
print(f"  within-station  {within:.3f}   {100*within**2/tot**2:.0f}%")
print(f"\\ncorrecting for site: ±{tot:.2f} -> ±{within:.2f} for one reading")
for k_ in (1, 10, 30):
    print(f"  {k_:2d} stations: ±{tot/np.sqrt(k_):.3f} raw, ±{within/np.sqrt(k_):.3f} corrected")
site = d[m].groupby("station").resid.agg(["size", "mean"]).rename(
    columns={"size": "n", "mean": "site"})
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.hist(site.site, bins=40, color="#90cdf4", edgecolor="white")
ax.axvline(0, color="#c53030", lw=1.2)
ax.set(xlabel="site term (log10 units)", ylabel="stations",
       title=f"Site terms span {site.site.max()-site.site.min():.2f} log10 units "
             f"= {10**(site.site.max()-site.site.min()):.0f}x in amplitude")
plt.tight_layout(); plt.show()""",
# planned: 10 min
"""
The model has no term for *which station* took the measurement, yet some sit on rock and some on
basin sediment, and sediment amplifies.

Compute each reading's residual, group by station, keep stations with ≥ 50 readings, and split the
total residual variance into a **between-station** part (the site term) and a **within-station**
part. Report both as a percentage of the variance, then plot the distribution of site terms.

Then answer: **how much better would one station's magnitude be if you corrected for site — and
would averaging 30 stations still help?**""")

ckpt(2, """y = d.logA.values
X3 = np.column_stack([np.ones(len(d)), d.magnitude, np.log10(d.hyp_km), d.hyp_km])
b3, se3, r3, s2_3, XtX_inv = ols(X3, y)
cov = s2_3 * XtX_inv
print("state rebuilt")""")

md(r"""**Now put the station terms into the model**, which is what Hutton & Boore were trying to achieve.
They could not afford the joint solve in 1987 — "computer resources proved insufficient to do this"
(p. 2079) — and iterated between station residuals and a two-parameter attenuation fit instead.
Doing it in one sparse solve is the thing that has changed.""")

run("""from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

# One extra column per station - the ordinary way to put station terms in a linear
# inverse problem. 405 of them, so the design matrix is sparse.
codes, stations = pd.factorize(d.station)
S = csr_matrix((np.ones(len(d)), (np.arange(len(d)), codes)), shape=(len(d), len(stations)))
G = hstack([csr_matrix(np.column_stack([d.magnitude, np.log10(d.hyp_km), d.hyp_km])), S])

m = lsqr(G, y, atol=1e-12, btol=1e-12, iter_lim=500)[0]
rfe = y - G @ m
print(f"G is {G.shape[0]:,} x {G.shape[1]} "
      f"({3} model terms + {len(stations)} station terms)")
print()
print(f"{'':22s}{'c1':>9s}{'c2':>10s}{'c3':>11s}{'resid sd':>11s}")
print(f"{'no station terms':22s}{b3[1]:9.4f}{b3[2]:10.4f}{b3[3]:11.5f}{r3.std():11.3f}")
print(f"{'+ station terms':22s}{m[0]:9.4f}{m[1]:10.4f}{m[2]:11.5f}{rfe.std():11.3f}")
print(f"{'Hutton & Boore 1987':22s}{1.0:9.4f}{-1.110:10.4f}{-0.00189:11.5f}{0.208:11.3f}")""")

md(r"""**Three things changed.** $c_1$ moved from 1.0254 to 1.0089 — most of its excess over
Richter's 1 was the missing station term. The residual scatter dropped to 0.233, near Hutton &
Boore's 0.208, and this is the first comparison in the notebook computed the way they computed
theirs: their 0.21 is the scatter *after* station corrections, so every earlier comparison against
it was against a number made a different way.

But $c_2$ is still nine of their standard errors from $-1.110$, and $c_3$ moved *further* from their
value, not closer. **Station terms fixed the scatter and did not fix the curve.**""")

md("""## 8 · What this calibration cannot tell you

Everything so far has treated the `magnitude` column as an independent measurement. It is not, and
the three tests in this section are the ones most worth carrying out of the room — each says
something about *any* published fit, not just this one.

**Where did the magnitude come from?** Below about M3.4 it is the network's median of the station
magnitudes in this table, after outlier rejection, and each of those was computed as
$M_{L,i} = \\log_{10}A_i + f(R_i)$ with $f$ the attenuation table the network already assumes.
Rearranged, $\\log_{10}A_i = M_{L,i} - f(R_i)$: the left side of our regression is the right side of
theirs. A fit against it partly **recovers an assumption** rather than measuring the Earth.

Test it by regressing on the *station* magnitude — the one derived from that exact reading, where
the circularity is complete.""")
run("""n_big = d[d.magnitude > 3.4].event_id.nunique()
print(f"events above M3.4 (the range where moment tensors are obtainable): {n_big}")
def fit_on(mag):
    Xa = np.column_stack([np.ones(len(d)), mag, np.log10(d.hyp_km), d.hyp_km])
    b = np.linalg.solve(Xa.T @ Xa, Xa.T @ y)
    return b, np.std(y - Xa @ b)
b_ev, s_ev = fit_on(d.magnitude.values)
b_st, s_st = fit_on(d.station_magnitude.values)
print(f"\\n{'regressed on':34s}{'c1':>9}{'c2':>10}{'c3':>11}{'sd':>8}")
print(f"{'event magnitude (net average)':34s}{b_ev[1]:9.4f}{b_ev[2]:10.4f}{b_ev[3]:11.5f}{s_ev:8.3f}")
print(f"{'station magnitude (this reading)':34s}{b_st[1]:9.4f}{b_st[2]:10.4f}{b_st[3]:11.5f}{s_st:8.3f}")
print(f"{'Hutton & Boore 1987':34s}{1.0:9.4f}{-1.110:10.4f}{-0.00189:11.5f}")""")

md("""Regressing on the station magnitude returns $c_1 \\approx 1.00$ and $c_2 \\approx -1.11$ —
Hutton & Boore's published values, nearly exactly. **The fit is inverting a definition.**

To break the circle you would regress against a magnitude *not* derived from these amplitudes:
M<sub>w</sub>, from moment-tensor inversion. But those events exist only above M3.4, so **magnitude
type and magnitude range are confounded** and the two fits cannot be cleanly compared.""")

yrs("""# solution
res = {}
for lo, hi, lab in [(1.9, 7.2, "everything"), (2.5, 7.2, "M>=2.5"), (3.4, 4.1, "M 3.4-4.1"),
                    (1.9, 3.4, "M<3.4 (ML only)")]:
    s_ = d[(d.magnitude >= lo) & (d.magnitude <= hi)]
    Xs = np.column_stack([np.ones(len(s_)), s_.magnitude, np.log10(s_.hyp_km), s_.hyp_km])
    res[lab] = (np.linalg.solve(Xs.T @ Xs, Xs.T @ s_.logA.values), len(s_))
print(f"{'sample':18s}{'n':>10}{'c1':>9}{'c2':>10}{'c3':>11}")
for k_, (b_, n_) in res.items():
    print(f"{k_:18s}{n_:10,}{b_[1]:9.4f}{b_[2]:10.4f}{b_[3]:11.5f}")
print(f"{'Hutton & Boore':18s}{'':>10}{1.0:9.4f}{-1.110:10.4f}{-0.00189:11.5f}")
sp = max(v[0][2] for v in res.values()) - min(v[0][2] for v in res.values())
print(f"\\nc2 moves {sp:.3f} across these samples; its standard error was {se3[2]:.4f}")
print(f"that is {sp/se3[2]:.0f} standard errors.")""",
# planned: 8 min
"""
Refit the model on four samples: everything; M ≥ 2.5; the narrow band M 3.4–4.1 where both scales
exist; and M < 3.4 where only M<sub>L</sub> exists. Tabulate `n`, `c1`, `c2`, `c3` beside Hutton &
Boore's values.

Then answer: **the standard error on $c_2$ was about 0.003. How far does $c_2$ move when you change
the sample?** What does that imply about quoting a standard error as the uncertainty on a
calibration?""")

md(r"""**Put the three tests together.** Each one is invisible in the regression output, and each one
applies to any published calibration you will ever read.

- **The target was built from the predictor.** Catalogue magnitude comes from these amplitudes
  through an assumed attenuation table, so part of what any such fit recovers is that assumption
  coming back.
- **The largest term in the model was the one we left out.** Adding a station intercept moved $c_1$
  most of the way to Richter's 1 and removed about a third of the scatter. That was an omitted
  variable, not a discovery about the magnitude scale — and nothing in the four-parameter fit hinted
  at it.
- **Choosing the sample moved the answer a hundred times further than the standard error did.**

So when you next read a calibration quoted as $c \pm \sigma$, the honest questions are not about
$\sigma$. They are: what was the target built from, what was left out, and which events were kept.""")

md(r"""## Seismology takeaways

- **M<sub>L</sub> is a regional calibration and does not transfer.** Hutton & Boore excluded the
  Mammoth Lakes events for attenuating differently, and re-anchored the scale at 17 km to make
  regions comparable (1987, p. 2091).
- **A refitted curve does not reproduce the catalogue.** Theirs ran above it for small earthquakes
  and below for large ones, by up to 0.6 units, mostly from uneven sampling in magnitude-distance
  space (1987, abstract).
- **Site is half the scatter** — 51% of the residual variance sits between stations. Averaging beats
  it down, but two networks averaging different stations differ systematically (1987, p. 2083).
- **M<sub>L</sub> saturates** because the corner frequency falls as $M_0^{-1/3}$ (Brune 1970) until
  it leaves the instrument's band; M<sub>w</sub> is defined from $M_0$ to avoid this (Hanks &
  Kanamori 1979).
- **A better calibration waits on its catalogue.** Hutton & Boore held theirs back rather than break
  the seismicity statistics; southern California recalibrated twenty-one years later (1987, p. 2091).

## Machine learning takeaways

- Least squares is maximum likelihood under independent Gaussian errors, and returns the parameter
  covariance $\sigma^2(X^\top X)^{-1}$ with the fit.
- A confidence interval covers the mean, a prediction interval one new observation; only the first
  shrinks with $n$. That is the *epistemic* / *aleatoric* split, and which scatter is which depends
  on your model.
- Collinear predictors are determined as a combination, not individually. A small standard error is
  not evidence that a coefficient is resolved.
- Ridge is damped least squares, and the MAP estimate under a Gaussian prior with
  $\lambda = \sigma^2/s^2$.
- Standard errors assume independent rows. Group the data and the effective sample size is the
  number of groups — resampling stations here inflates the uncertainty thirtyfold.""")

write(SLUG)

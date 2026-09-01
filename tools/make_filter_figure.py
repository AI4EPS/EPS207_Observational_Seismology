#!/usr/bin/env python
"""Generate docs/lectures/assets/filtering_response.png.

The 2023 Filtering slide illustrated the idea with an image from open.edu whose link
has since died, leaving a heading and one sentence. Rather than borrow another
figure, draw the point: what a Butterworth filter does to the spectrum, and what it
does to a trace. Regenerate with:

    uv run --with matplotlib --with scipy --with numpy python tools/make_filter_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

FS = 100.0
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))

for name, (lo, hi) in [("low-pass 5 Hz", (None, 5)),
                       ("band-pass 1-10 Hz", (1, 10)),
                       ("high-pass 2 Hz", (2, None))]:
    if lo is None:
        b, a = signal.butter(4, hi / (FS / 2), "low")
    elif hi is None:
        b, a = signal.butter(4, lo / (FS / 2), "high")
    else:
        b, a = signal.butter(4, [lo / (FS / 2), hi / (FS / 2)], "band")
    w, h = signal.freqz(b, a, worN=4096, fs=FS)
    ax[0].semilogx(w[1:], 20 * np.log10(abs(h[1:]) + 1e-12), label=name)

ax[0].set(xlim=(0.1, 50), ylim=(-60, 5), xlabel="Frequency (Hz)", ylabel="Gain (dB)",
          title="Butterworth filter response (order 4)")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

rng = np.random.default_rng(0)
t = np.arange(0, 20, 1 / FS)
event = np.exp(-((t - 8) ** 2) / 2.0) * np.sin(2 * np.pi * 4 * t)
noise = 0.6 * np.sin(2 * np.pi * 0.15 * t) + 0.25 * rng.standard_normal(t.size)
raw = event + noise
b, a = signal.butter(4, [1 / (FS / 2), 10 / (FS / 2)], "band")
ax[1].plot(t, raw + 3, lw=0.8, color="0.45", label="raw")
ax[1].plot(t, signal.filtfilt(b, a, raw), lw=0.8, color="tab:orange",
           label="band-pass 1-10 Hz")
ax[1].set(xlabel="Time (s)", yticks=[], title="The same trace, before and after")
ax[1].legend(fontsize=8, loc="upper right")

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "docs/lectures/assets/filtering_response.png"
fig.savefig(out, dpi=150)
print("wrote", out)

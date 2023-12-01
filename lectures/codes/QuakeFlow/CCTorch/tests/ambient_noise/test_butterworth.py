# %%
import numpy as np
import scipy.signal as signal
from scipy.signal import lfilter
import matplotlib.pyplot as plt

# Parameters
fs = 512  # Sampling frequency (Hz)
lowcut = 10  # Low cut-off frequency (Hz)
highcut = 30  # High cut-off frequency (Hz)
order = 4  # Filter order

# Generate a test signal
t = np.linspace(0, 1, fs, False)  # Time array
x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)  # Test signal

# Design the Butterworth bandpass filter
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = signal.butter(order, [low, high], btype='band')
b1, a1, = signal.butter(order//2, high, btype='low')
b2, a2, = signal.butter(order//2, low, btype='high')


# Perform the FFT of the input signal
X = np.fft.fft(x)
freq = np.fft.fftfreq(x.shape[-1])

# Compute the frequency response
w, h = signal.freqz(b, a, worN=freq)
w1, h1 = signal.freqz(b1, a1, worN=freq)
w2, h2 = signal.freqz(b2, a2, worN=freq)

# Apply the filter in the frequency domain
# H = np.fft.fft(h, len(x))  # Zero-pad the frequency response to the same length as the input signal
# H = np.fft.fftshift(h, len(x))
H = h
# H = h1 * h2
# H = h2
Y = X * H

# Perform the inverse FFT to obtain the filtered signal
y = np.fft.ifft(Y)

y = y[::-1]
y = np.fft.fft(y)
y = y * H
y = np.fft.ifft(y)
y = y[::-1]

# %%
# y_scipy = lfilter(b, a, x)
y_scipy = signal.filtfilt(b, a, x)

# %%
# Plot the results
plt.figure(figsize=(10, 6))

plt.subplot(1, 1, 1)
# plt.plot(t, x, label='Input signal')
plt.plot(t, np.sin(2 * np.pi * 20 * t), label='True signal')
plt.plot(t, y.real/np.max(y.real), label='Filtered signal', linestyle='dashed')
plt.plot(t, y_scipy/np.max(y_scipy), label='Filtered signal (scipy)', linestyle='dotted')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
# plt.xlim([0, 0.3])
plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(w * nyquist / np.pi, np.abs(h), label='Filter response')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Gain')
# plt.legend()

plt.tight_layout()
plt.show()

# %%

---
marp: true
paginate: true
theme: gaia
backgroundColor: #fff
---

# Seismic Signal Processing
Notebooks: 
[Signal Processing](codes/signal_processing.ipynb)
[Obspy](codes/obspy.ipynb)
[Deep Denoiser](https://ai4eps.github.io/DeepDenoiser/example_interactive/)

---

### Signal Processing 101

- Fourier Transform (FFT)
- Filtering
- Spectrogram
- Convolution and Cross-correlation
- Short-time Fourier Transform (STFT)
- Wavelet Transform
- Hilbert Transform
- ...

---

### Fourier Transform

Fourier Transform (FT) is a mathematical operation that decomposes a function into its constituent frequencies.

The Fourier Transform of a function $f(t)$ is given by:

$$F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt$$

The inverse Fourier Transform is given by:

$$f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i\omega t} d\omega$$

---

### Example: Fourier Transform of a Sine Function

$$
\begin{align}
f(t) &= \sin(\omega_0 t) \\
F(\omega) &= \int_{-\infty}^{\infty} \sin(\omega_0 t) e^{-i\omega t} dt \\
&= \int_{-\infty}^{\infty} \frac{e^{i\omega_0 t} - e^{-i\omega_0 t}}{2i} e^{-i\omega t} dt \\
&= \frac{1}{2i} \int_{-\infty}^{\infty} e^{i(\omega_0 - \omega) t} dt - \frac{1}{2i} \int_{-\infty}^{\infty} e^{i(\omega_0 + \omega) t} dt \\
&= \frac{1}{2i} \delta(\omega_0 - \omega) - \frac{1}{2i} \delta(\omega_0 + \omega) \\
\end{align}
$$

---

### Example: Fourier Transform of a Box Function

$$
\begin{align}
f(t) &= \begin{cases}
1 & \text{if } -\frac{1}{2} \leq t \leq \frac{1}{2} \\
0 & \text{otherwise}
\end{cases} \\
\end{align}
$$

[Wiki](https://en.wikipedia.org/wiki/Rectangular_function)

---

$$
\begin{align}
F(\omega) &= \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt \\
&= \int_{-\frac{1}{2}}^{\frac{1}{2}} e^{-i\omega t} dt \\
&= \frac{1}{-i\omega} \left[ e^{-i\omega t} \right]_{-\frac{1}{2}}^{\frac{1}{2}} \\
&= \frac{1}{-i\omega} \left[ \cos\left(\frac{\omega}{2}\right) - i\sin\left(\frac{\omega}{2}\right) - \cos\left(\frac{\omega}{2}\right) - i\sin\left(\frac{\omega}{2}\right) \right] \\
&= \frac{1}{-i\omega} \left[ -2i\sin\left(\frac{\omega}{2}\right) \right] \\
&= \frac{\sin\left(\frac{\omega}{2}\right)}{\frac{\omega}{2}}
\end{align}
$$

--- 

### Fourier Transform
![height:730px](https://mriquestions.com/uploads/3/4/5/7/34572113/9600204.gif?508)

--- 

### Fourier Analysis
![height:500px](https://www.nti-audio.com/portals/0/pic/news/FFT-Time-Frequency-View-540.png)


---

### Filtering

Filtering is a process of removing unwanted components or features from a signal.

![height:400px](https://www.open.edu/openlearn/pluginfile.php/1881285/mod_oucontent/oucontent/95937/1b694830/7ce1bfc5/t312_openlearn_fig22.tif.jpg)

---

### Example: Butterworth Low-pass Filter $H(s) = \frac{1}{1 + \left(\frac{s}{\omega_c}\right)^{2n}}$

![height:350px](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Butterworth_filter_bode_plot.svg/800px-Butterworth_filter_bode_plot.svg.png)

---

### Example: Chebyshev Filter

![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Filters_order5.svg/540px-Filters_order5.svg.png)

---

### Convolution

Convolution is a mathematical operation on two functions $f$ and $g$ that produces a third function that expresses how the shape of one is modified by the other.

$$
\begin{align}
(f * g)(t) &= \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau \\
&= \int_{-\infty}^{\infty} f(t - \tau) g(\tau) d\tau \\
\end{align}
$$

---

### Convolution in Frequency Domain

$$
\begin{align}
& \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau e^{-i\omega t}  dt \\
& = \int_{-\infty}^{\infty} f(\tau) \int_{-\infty}^{\infty} g(t - \tau) e^{-i\omega t} dt d\tau \\
& = \int_{-\infty}^{\infty} f(\tau) \int_{-\infty}^{\infty} g(\tau') e^{-i\omega (\tau' + \tau)} d\tau' d\tau, \tau'=t-\tau \\
& = \int_{-\infty}^{\infty} f(\tau) e^{-i\omega \tau} d\tau \int_{-\infty}^{\infty} g(\tau') e^{-i\omega \tau'} d\tau'  \\
& = F(\omega) G(\omega)
\end{align}
$$

---

### Cross-correlation

Cross-correlation is a measure of similarity of two series as a function of the displacement of one relative to the other.

$$
\begin{align}
(f \star g)(t) &= \int_{-\infty}^{\infty} f(\tau) g(t + \tau) d\tau \\
&= \int_{-\infty}^{\infty} f(t + \tau) g(\tau) d\tau \\
\end{align}
$$

---

### Cross-correlation in Frequency Domain

$$
\begin{align}
& \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f(\tau) g(t + \tau) d\tau e^{-i\omega t}  dt \\
& = \int_{-\infty}^{\infty} f(\tau) \int_{-\infty}^{\infty} g(t + \tau) e^{-i\omega t} dt d\tau \\
& = \int_{-\infty}^{\infty} f(\tau) \int_{-\infty}^{\infty} g(\tau') e^{-i\omega (\tau' - \tau)} d\tau' d\tau, \tau'=t+\tau \\
& = \int_{-\infty}^{\infty} f(\tau) e^{-i (-\omega) \tau} d\tau \int_{-\infty}^{\infty} g(\tau') e^{-i\omega \tau'} d\tau'  \\
& = F(-\omega) G(\omega)
\end{align}
$$

---

![](https://i.stack.imgur.com/JTnjz.jpg)

---

![width:800px](https://miro.medium.com/v2/resize:fit:1400/1*K500B9Jdwddeh3TTlViQLg.jpeg)

---

### Seismic Signal Processing using Obpsy

---

### Deep Denoiser

- STFT + Wiener Filter + Nerual Network


---

![](./assets/Screenshot%202023-09-17%20at%2018.16.57.png)

---
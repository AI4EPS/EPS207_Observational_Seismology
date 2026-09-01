---
marp: true
paginate: true
theme: gaia
backgroundColor: #fff
style: |
  section {
    font-size: 28px;
  }
  img + br + em {
    font-style: normal;
    display: inherit;
    text-align: right;
    font-size: 70%;
  }
---

# Observational Seismology

### EPS 207 · Fall 2026

Weiqiang Zhu · Tuesdays 9:00-10:59 · McCone 325

---

<!-- _class: lead -->

# Part 1
# Why we monitor

**What is the cost of not knowing?**

Classically: instrument the ground and wait.
With learning: the same instruments, read faster and more completely.

*Returns in every week*

---

### Large destructive earthquakes

| Year | Magnitude | MMI |  Deaths | Injuries |                Event               |
|:----:|:---------:|:---:|:-------:|:--------:|:----------------------------------:|
| 2023 | 7.8       | XII | 57,350+ | 130,000+ | 2023 Turkey–Syria earthquake       |
| 2011 | 9.1       | IX  | 19,747  | 6,000    | 2011 Tōhoku earthquake and tsunami |
| 2008 | 7.9       | XI  | 87,587  | 374,177  | 2008 Sichuan earthquake            |

---

### Hayward Fault

[History](https://seismo.berkeley.edu/hayward/hayward_history.html)

![height:500px](https://seismo.berkeley.edu/hayward/goole_earth_hayward_fault.jpg)

---

### [The California Memorial Stadium](https://pressbooks.pub/haywardfaultucberkeley/chapter/the-california-memorial-stadium/)

![height:500px](https://upload.wikimedia.org/wikipedia/commons/a/a3/Berkeley_stadium_fault_creep_P1320489.jpg)

---

### An estimated magnitude of 6.3 or greater. 
![height:500px](https://seismo.berkeley.edu/hayward/hf_history.jpg)

---

[Many more small earthquakes](https://earthquake.usgs.gov/earthquakes/map/?extent=-88.45674,-106.875&extent=88.43769,506.25&range=month&magnitude=all&showUSFaults=true&baseLayer=ocean&settings=true)
- [California](https://earthquake.usgs.gov/earthquakes/map/?extent=30.78904,-128.58398&extent=43.05283,-109.42383&range=month&magnitude=all&showUSFaults=true&baseLayer=ocean&settings=true)
- [Alaska](https://earthquake.usgs.gov/earthquakes/map/?extent=45.3367,-190.2832&extent=74.04372,-113.64258&range=month&magnitude=all&showUSFaults=true&baseLayer=ocean&settings=true)
- [Hawaii](https://earthquake.usgs.gov/earthquakes/map/?extent=16.35177,-161.78467&extent=23.58413,-152.20459&range=month&magnitude=all&showUSFaults=true&baseLayer=ocean&settings=true)
- [Oklahoma & Texas](https://earthquake.usgs.gov/earthquakes/map/?extent=26.78485,-109.81934&extent=39.62261,-90.65918&range=month&magnitude=all&showUSFaults=true&baseLayer=street&settings=true)

---

[Seismic Networks](http://ds.iris.edu/gmap/#network=*&starttime=2023-01-01&datacenter=IRISDMC&plates=on&planet=earth)
- [California](http://ds.iris.edu/gmap/#network=*&starttime=2023-01-01&maxlat=43.0799&maxlon=-113.3789&minlat=30.9776&minlon=-125.9234&datacenter=NCEDC,SCEDC&drawingmode=box&plates=on&planet=earth)
- [Alaska](http://ds.iris.edu/gmap/#network=AV,AK&starttime=2023-01-01&plates=on&planet=earth)
- [Hawaii](http://ds.iris.edu/gmap/#network=HV&maxlat=20.3285&maxlon=-154.6436&minlat=18.7711&minlon=-156.389&drawingmode=box&plates=on&planet=earth)
- [Oklahoma & Texas](http://ds.iris.edu/gmap/#network=*&starttime=2023-01-01&maxlat=38.2544&maxlon=-93.4717&minlat=27.2156&minlon=-105.608&drawingmode=box&plates=on&planet=earth)

[GPS Networks](https://www.unavco.org/instrumentation/networks/map/map.html#!/@45.65835440549003,-117.90988182323227,3.368z?network=nota,nota%20affiliated,polar,pi,igs,ggn,sgp,other&type=gps,gps%20realtime&view=map)

---

### Large-N and Large-T challenge

[IRIS dataset](https://ds.iris.edu/data/distribution/)
![height:500px](https://ds.iris.edu/files/stats/data/archive/Archive_Growth.jpg)

---

### Mining the IRIS dataset

![height:500px](https://ds.iris.edu/files/stats/data/shipments/GigabytesByYearAndType.jpg)

---

### What information can we get from seismic data?

- Take a look at seismic waveforms:
[ncedc.org/waveformDisplay/](https://ncedc.org/waveformDisplay/)

[Station Channel Codes](https://docs.fdsn.org/projects/source-identifiers/en/v1.0/channel-codes.html) 

[Station Channels Codes IRIS](https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/)

![height:500px](https://ncedc.org/gifs/annotatedQuakes.jpg)

---

- Can you find an earthquake?

[Raspberry Shake Network](https://stationview.raspberryshake.org/#/?lat=9.22307&lon=-0.57266&zoom=2.444)

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

---

### What information can we get from seismic data?

- Take a look at a recent earthquake: [M 5.1 - 7 km SE of Ojai, CA](https://earthquake.usgs.gov/earthquakes/eventpage/ci39645386/executive)
![height:500px](assets/M5.1.png)

---

### How are information extracted/determined?

* Detection of earthquakes
* Earthquake origin time and location
* Earthquake magnitude
* Earthquake focal mechanism/moment tensor
* Shake map/ground motion prediction
* Earthquake early warning
* "Did you feel it?"

---

### What additional information can we get from millions of earthquakes?

* Earthquake catalog
* Earthquake statistics
* Earthquake triggering
* Earthquake forecasting
* Fault zone structure
* Seismic tomography
* Volcano, glacier, and landslide monitoring

---

### How to use these information?

* Monitoring earthquakes and earthquake early warning
* Understand earthquake source physics
* Understanding the Earth's structure
* Applying seismology to environmental science, planetary science, climate science, etc.

---

### Earthquake monitoring and earthquake rick?

- Before an earthquake
- A few seconds after an earthquake
- Hours/days after an earthquake
- Years after an earthquake

---

### Before an earthquake

- [Eathquake Hazard Map](https://earthquake.usgs.gov/earthquakes/map/?extent=27.95559,-130.8252&extent=51.28941,-92.50488&range=month&magnitude=all&showPopulationDensity=true&showUSHazard=true&settings=true)
![height:500px](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/styles/side_image/public/thumbnails/image/2018nshm-longterm.jpg?itok=6tMRRjk3)

---

- Simulating earthquake scenarios
[Hayward Fault Scenarios](https://youtu.be/qZaKE4GuBXs?si=wI949Vnbk1EbO6xT)
![height:500px](https://earthquake.usgs.gov/education/shakingsimulations/hayward/images/tn-HaywardM72_SanPabloBayEp.jpg)

---

### A few seconds after an earthquake

![height:500px](./assets/ShakeAlert.webp)

---

- MyShake
[https://myshake.berkeley.edu/](https://myshake.berkeley.edu/)

- Mobile phones as seismometers
[Android EEW](https://www.youtube.com/watch?v=zFin2wZ56tM&ab_channel=Android)

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

---

### Hours/days after an earthquake

- Emergency response and damage assessment
[Fault Dimensions](https://www.src.com.au/earthquake-size/)

| Magnitude Mw | Fault Area km² | Typical rupture dimensions (km x km) |
|--------------|----------------|--------------------------------------|
| 4            | 1              | 1 x 1                                |
| 5            | 10             | 3 x 3                                |
| 6            | 100            | 10 x 10                              |
| 7            | 1,000          | 30 x 30                              |
| 8            | 10,000         | 50 x 200                             |

--- 

- [Aftershock prediction](https://earthquake.usgs.gov/data/oaf/overview.php)

![height:500px](https://earthquake.usgs.gov/data/oaf/images/fig4.gif)

---

### Years after an earthquake

- Understand earthquake rupture process
- Improve ground motion prediction models (GMPE)
- Improve hazard map and building codes
- Earthquake forecasting models

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

---

<!-- _class: lead -->

# Part 2
# What an earthquake is

**What quantity are we actually estimating?**

Classically: a double couple, six moment-tensor components, one magnitude.
With learning: nothing yet - this is the vocabulary the rest of the course uses.

*Returns in weeks 1 and 12*

---

### Earthquake faults

Earthquakes may be idealized as movement across a planar fault of arbitrary orientation

- strike: $\phi$, the azimuth of the fault from north where it intersects a horizontal surface
$0^\circ \leq \phi \leq 360^\circ$
- dip: $\delta$, the angle from the horizontal
$0^\circ \leq \delta \leq 90^\circ$
- rake: $\lambda$, the angle between the slip vector and the strike
$0^\circ \leq \lambda \leq 360^\circ$

![bg right:40% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250319214522.png)

---

### Earthquake faults

**Thrust faulting**: reverse faulting on faults with dip angles less than 45 
**Overthrust faults**: Nearly horizontal thrust faults
**Strike-slip faulting**: horizontal motion between the fault surfaces
**Dip-slip faulting**: vertical motion
**Right-lateral strike–slip motion**: standing on one side of a fault, sees the adjacent block move to the right
$\lambda = 0^\circ$: left-lateral faulting
$\lambda = 180^\circ$: right-lateral faulting
The San Andreas Fault: Right-lateral fault

![bg right:30% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250319215002.png)

---

### Earthquake double couple

- An earthquake is usually modeled as slip on a fault, a discontinuity in displacement across an internal surface in the elastic media.
- Internal forces resulting from an explosion or stress release on a fault must act in opposing directions so as to conserve momentum.
- A force couple is a pair of opposing point forces separated by a small distance 
- A double couple is a pair of complementary couples that produce no net torque

![bg right:40% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250318122046.png)

---

### Moment tensor

We define the force couple $M_{ij}$ as a pair of equal and opposite forces pointing in the $i$ direction and separated by a unit distance in the $j$ direction. 

The magnitude of $M_{ij}$ is the product of the force and the distance $f \times d$.

$$
M_{ij} = \begin{bmatrix}
    M_{11} & M_{12} & M_{13} \\
    M_{21} & M_{22} & M_{23} \\
    M_{31} & M_{32} & M_{33}
\end{bmatrix}
$$

The condition that angular momentum be conserved requires that is symmetric (e.g., $M_{ij} = M_{ji}$).

![bg right:30% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250318122515.png)

---

### Moment tensor

For example, right-lateral movement on a vertical fault oriented in the $x_1$ direction corresponds to the moment tensor representation

$$
M = \begin{bmatrix}
    M_{11} & M_{12} & M_{13} \\
    M_{21} & M_{22} & M_{23} \\
    M_{31} & M_{32} & M_{33}
\end{bmatrix}
= \begin{bmatrix}
    0 & M_0 & 0 \\
    M_0 & 0 & 0 \\
    0 & 0 & 0
\end{bmatrix}
$$

where $M_0$ is the scalar seismic moment: 

$$
M_0 = \mu d A
$$

where $\mu$ is the shear modulus, $d$ is the average fault displacement, and $A$ is the area of the fault.

The units for $M_0$ are N$\cdot$m (or dyne$\cdot$cm), the same as for force couples.

---

### Global CMT catalog

[Global Centroid Moment Tensor](https://www.globalcmt.org/)

![height:500px](https://raw.githubusercontent.com/zhuwq0/images/main/20250319214231.png)

---

### Beach balls

<div style="display: flex; justify-content: space-between;">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250319224343.png" width="48%">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250319224401.png" width="48%">
</div>

---

### Review: Basic types of faulting

![height:500px](https://raw.githubusercontent.com/zhuwq0/images/main/202503302341109.png)

---

### Basic types of faulting

![20250330233905 height:500px](https://raw.githubusercontent.com/zhuwq0/images/main/20250330233905.png)

---

### First-motion polarity

![20250330234223 height:500px](https://raw.githubusercontent.com/zhuwq0/images/main/20250330234223.png)

---

### Magnitude $M_0$

<div style="display: flex; justify-content: center;">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250331234322.png" width="44%">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250331234224.png" width="44%">
</div>


The magnitude of the equivalent body forces is $M_0$
The scalar seismic moment of the earthquake; units of dyn-cm, or N-m

---

### Earthquake Magnitude

How to quantify the size of an earthquake?

* For historical reasons the most well-known measure of earthquake size is the  earthquake magnitude.
* Derived from the largest amplitude that is recorded on  seismograms.
* There are now many different types of magnitude  scales, but all are connected in some way to the earliest definitions of  magnitude.

![bg right:40% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250403091807.png)

---

### Richter Magnitude (Local magnitude $M_L$)

The original magnitude scale is based on the maximum amplitude recorded on a standard Wood-Anderson torsion seismograph.

$$
M_L = \log_{10} A(X) - \log_{10} A_0(X)
$$
$A_0$: the amplitude of the reference event
$X$: the epicentral distance

![20250402232519 bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250402232519.png)

----

### Richter Magnitude (Local magnitude $M_L$)

An approximate empirical formula has been derived for $\log_{10} A_0(X)$ at different ranges. 
The local magnitude can be calculated by
$$
M_L = \log_{10} A(X) + 2.56 \log_{10} X - 1.67
$$
where $A(X)$ is the displacement amplitude in microns (10$^{-6}$ m) and X is in  kilometers.

* Events below about $M_L 3$ are generally not felt
* Significant damage to structures in California begins to  occur at about $M_L 5.5$
* A $M_L 6.0$ earthquake implies amplitude 100 times greater than a $M_L 4.0$ event.

---

### Global earthquakes: body wave magnitude $m_b$


$$
m_b = \log_{10} (A/T) + Q(h, \Delta)
$$

where A is the ground displacement in microns, T is the dominant period of  the measured waves, $\Delta$ is the epicentral distance in degrees, and Q is an  empirical function of range and event depth h.

* Why $A/T$?
* h?

---

### Global earthquakes: surface wave magnitude $M_s$

For Rayleigh waves on vertical instruments:
$$
M_s = \log_{10} (A/T) + 1.66 \log_{10} \Delta + 3.30
$$

Since the strongest Rayleigh wave arrivals are generally at a period of 20 s, this expression is  often written as
$$
M_s = \log_{10} A_{20} + 2.46 \log_{10} \Delta + 2.0
$$

* Note that this equation is applicable only to shallow events
* surface wave amplitudes are greatly reduced for deep events.

---

### Magnitude saturation

![bg right:70% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250402233819.png)

---

### Moment magnitude $M_w$

The saturation of the and scales for large events helped motivate  development of the moment magnitude $M_w$

$$
M_w = \frac{2}{3} (\log_{10} M_0 - 9.1)
$$
where is the moment measured in N-m.

* The advantage of the scale is that it is clearly related to a physical property of the source and it does not saturate for even the largest  earthquakes.
* One unit increase in $M_w$ corresponds to a $10^{3/2} \approx 32$ times increase in the moment.
* A $M_w 7$ earthquake releases about 1000 times more energy than a $M_w 5$ event.

---

### Magnitude as a function of moment

![20250402234333 height:450px](https://raw.githubusercontent.com/zhuwq0/images/main/20250402234333.png)

[USGS Magnitude Types](https://www.usgs.gov/programs/earthquake-hazards/magnitude-types); [Latest earthquake](https://earthquake.usgs.gov/earthquakes/eventpage/us7000pn9s/origin/magnitude)

---

### The intensity scale

The local strength of ground shaking as determined by damage to  structures and the perceptions of people who experienced the earthquake.

* One earthquake can have different intensities at different locations.

[USGS Latest Earthquakes](https://earthquake.usgs.gov/earthquakes/map/?extent=26.07652,-136.80176&extent=49.92294,-98.48145&range=month&listOnlyShown=true&settings=true&search=%7B%22name%22:%22Search%20Results%22,%22params%22:%7B%22starttime%22:%222020-12-02%2000:00:00%22,%22endtime%22:%222023-12-09%2023:59:59%22,%22maxlatitude%22:37.642,%22minlatitude%22:37.588,%22maxlongitude%22:-122.344,%22minlongitude%22:-122.412,%22orderby%22:%22time%22%7D%7D)

---

<!-- _class: lead -->

# Part 3
# The waveform

**Which part of this record is signal?**

Classically: a bandpass filter, chosen by eye.
With learning: a learned mask - but you must show it helps downstream, not that it looks better.

*Returns in week 11*

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

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

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

### Deep Denoiser

- Short-time Fourier Transform (STFT) + Wiener Filter + Neural Network

![width:1000px](./assets/Screenshot%202023-09-17%20at%2018.16.57.png)

---

### Deep Denoiser for Seismic Data

[Paper](https://drive.google.com/file/d/19g0nyCgAIUPOrQ6sPsU1PPA5B9zhQbBG/view?usp=drive_link)

---

<!-- _class: lead -->

# Part 4
# Detection

**Is there an earthquake in this hour of data?**

Classically: amplitude threshold, then STA/LTA, then template matching.
With learning: a detector trained on catalogues, finding events below every earlier threshold.

*Returns in weeks 6, 7 and 9*

---

### How to detect earthquakes?

* Amplitude threshold
* STA/LTA
* Template matching / Matched filter
* Deep learning

---

### Amplitude threshold

- PGA (Peak Ground Acceleration)
- PGV (Peak Ground Velocity)
- Displacement

[Recent M4.5 earthquake](https://earthquake.usgs.gov/earthquakes/eventpage/nc73938736/executive)

---

### Amplitude threshold

* Pros:
    - Simple and fast
    - Physical parameter
    - Directly related to shaking/damage
* Cons:
    - Limit to large earthquakes
    - Need backgroud noise level for small earthquakes
* Improvments:
    - How to make the threshold adaptive to the background noise level?

---

### STA/LTA

- STA/LTA = Short-Term Average / Long-Term Average

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

---

### STA/LTA


![height:500px](https://docs.obspy.org/_images/trigger_tutorial_classic_sta_lta.png)

---

### STA/LTA

* Pros:
    - Simple and fast
    - More sensitive than amplitude threshold
    - More robust for noisy data

* Cons:
    - More parameters for tuning
    - Prone to false detections

---

### Template matching / Matched filter

![](https://upload.wikimedia.org/wikipedia/commons/9/98/Cross_Correlation_Animation.gif)

Review of convolution and cross-correlation in last lecture: [cross-correlation](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/02_signal_processing.html#12)

Notebook: [cross-correlation](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/codes/signal_processing/#convolution)

---

### (QTM) Quake Template Matching

![width:1100px](https://www.science.org/cms/10.1126/science.aaw6888/asset/a82e0ba0-4c86-4db8-9d24-5b16db8361bc/assets/graphic/364_767_f1.jpeg)


--- 

### Template matching / Matched filter

- Pros:
    - Robust to noise
    - More sensitive to small earthquakes
- Cons:
    - High computational cost
    - Need existing catalog to build templates
    - Limited to waveform similarity with templates

---

### Similarity search



--- 

### FAST (Fingerprint And Similarity Thresholding)

![width:480px](https://www.its.caltech.edu/~cyoon/img/fastgraphic_final.png)

--- 

### Siminlarity search

- Pros:
    - Sensitive to small earthquakes
    - Computational efficient
- Cons:
    - Detect all repeating signals
    - Complex to implement

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

---

### Deep learning

- Generalized similarity search

![width:1100px](./assets/Screenshot%202023-09-24%20at%2023.39.39.png)

---

### Deep learning

- Pros:
    - Robust to noise
    - Sensitive to small earthquakes
    - Fast prediction
- Cons:
    - Need large amount of labeled data
    - Black box
    - Generalization ability

---

<!-- _class: lead -->

# Part 5
# Phase picking

**When did P and S arrive, on each station?**

Classically: an analyst, by hand, at a few tens of thousands of picks a year.
With learning: segmentation of the trace - the single change that grew catalogues by an order of magnitude.

*Returns in week 6*

---

<style scoped>
section {
  column-count: 2;
}
h3 {
  column-span: all;
}
p {
  margin: 0;
}
</style>

### Seismic waves

![height:300px](https://gpg.geosci.xyz/_images/pwave-animated-2.gif)
![height:350px](https://gpg.geosci.xyz/_images/s-wave-animated.gif)

---

### Seismic phases

![height:950px](http://ds.iris.edu/media/product/globalstacks/images/TraceProcessing2.png)

---

<style scoped>
section {
  column-count: 2;
}
h3 {
  column-span: all;
}
p {
  margin: 0;
}
</style>

### Information from seismic phases

* Earthquake source
* Earth's (Planetary) interior structure
* Subsurface exploration (reservior, geothermal, etc.)
* ...

![width:500px](https://www.science.org/cms/10.1126/science.abi7730/asset/50a260db-ccff-43b4-a8ca-b83c05832d16/assets/graphic/373_443_f3.jpeg)

---

### Picking P and S waves

![width:1100px](https://d3i71xaburhd42.cloudfront.net/5ae0f6a3b5fc882ce0b05ff1e8f333caf2e0549e/6-Figure4-1.png)

---

### Background: Semantic Segmentation vs. Classification

![width:950px](./assets/cv_tasks.png)

---

### Demo: Segment Anything Model (SAM)

Try the SAM model: [link](https://segment-anything.com/demo)

---

### How to apply deep learning to seismic phase picking?

---

### Generalized seismic phase detection with deep learning

![width:900px](https://d3i71xaburhd42.cloudfront.net/e178d94a0601f0f395cf6d81b884a238331fa869/3-Figure1-1.png)

---

### PhaseNet

![width:1200px](./assets/phasenet.png)

---

### EQTransformer for simultaneous earthquake detection and phase picking

![height:500px](./assets/eqtransformer.jpg)

---

### Next-Generation Seismic Monitoring with Neural Operators (PhaseNO)

![height:450px](./assets/phaseno.png)

---

<style scoped>
section {
  column-count: 2;
}
h3 {
  column-span: all;
}
p {
  margin: 0;
}
</style>

### Large training dataset + Clear objective function

*Figure: [Zhu & Beroza (2018), PhaseNet, GJI](https://doi.org/10.1093/gji/ggy423)*
![width:700px](./assets/dataset.png)

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

---

<!-- _class: lead -->

# Part 6
# Association

**Which picks belong to the same earthquake?**

Classically: grid search over candidate origins.
With learning: clustering with a physical forward model inside it.

*Returns in week 4*

---

### What is phase association?

![width:1200px](./assets/phase_picks.png)

---

### Grid-search / Back-projection, e.g. REAL

*Figure: [Zhang, Ellsworth & Beroza (2019), Rapid Earthquake Association and Location, SRL](https://doi.org/10.1785/0220190052)*

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

---

<style scoped>
section {
  column-count: 2;
}
h3 {
  column-span: all;
}
p {
  margin: 0;
}
</style>

### Graph-Neural-Network-based, e.g. GENIE


*Figure: [McBrearty & Beroza (2023), Earthquake Phase Association with Graph Neural Networks, BSSA](https://doi.org/10.1785/0120220182)*

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

---

### Clustering-based (Unsupervised), e.g. GaMMA

![height:500px](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/diagram_gamma_annotated.png)



--- 

### Clustering

![height:500px](https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png)

---

### K-means

![](https://sandipanweb.files.wordpress.com/2016/08/k3.gif?w=676)

---

<style scoped>
section {
  column-count: 2;
}
h3 {
  column-span: all;
}
p {
  margin: 0;
}
</style>

### Gaussian Mixture Model (GMM)


![height:450px](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/diagram_gamma_annotated.png)

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

---

### Gaussian Mixture Model Association (GaMMA)


![width:1200px](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/2019-07-04T18-02-01.074.png)

---

<!-- _class: lead -->

# Part 7
# Location, and how wrong it is

**Where was it, and how far could that be off?**

Classically: linearised least squares; a covariance matrix; a chi-square ellipse.
With learning: the same inverse problem, differentiated automatically, with a posterior instead of an ellipse.

*Returns in weeks 1, 3, 13 and 14*

---

### How to locate an earthquake?

![height:500px](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/styles/full_width/public/thumbnails/image/locating%20earthquakes%201.gif?itok=z60HGZwY)

---

### Optimization (Inverse) problem

- Minimize the difference between observed and predicted values

![height:400px](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1920px-Linear_regression.svg.png)

---

### How to solve an optimization/inversion problem?

- Forward function
- Objective/Loss function
- Gradient
- Optimizer

<!-- ---
<style scoped>
section {
  padding: 0px;
}
section::after {
  font-size: 0em;
}
</style>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTGPzx6m0wBOAta7qebCjW_n_lcXsay3Uqo7iKnVI5cxZNYWcmbTQNgOAwiuTx_ZwRuNRxOHCRBSFsq/embed?start=false&loop=true&delayms=60000" frameborder="0" width="100%" height="105%" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
 -->

---

### Locating earthquake using absolute arrival times

[notebook](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/codes/earthquake_location/#locating-earthquakes-using-both-absolute-travel-times-and-relative-travel-time-differences)

Earthquake location problem:

- Given:
  - Observed arrival times at multiple stations
  - Velocity model
- Goal:
  - Locate the hypocenter and origin time of the earthquake

---

### Forward function:

$$\hat{t}^i=f^i(\mathbf{m})$$

where $\hat{t}^i$ is the predicted arrival time at station $i$, $f^i$ is the forward non-linear function (e.g., ray tracing or eikonal equation), and $\mathbf{m}$ is the model parameter (e.g., source location, origin time, and velocity).

For a uniform velocity:
$$
\hat{t}^i=f^i(\mathbf{m})=\frac{\sqrt{(x^i-x_0)^2+(y^i-y_0)^2+(z^i-z_0)^2}}{v} + t^i_0
$$
where $(x^i,y^i,z^i)$ is the location of the $i$-th station, $(x_0,y_0,z_0)$ is the location of the source, $t_0$ is the origin time, and $v$ is the uniform velocity.

---

### Objective/Loss function:

The difference between the observed and predicted times is:
$$
r^i=t^i-\hat{t}^i=t^i-f^i(\mathbf{m})
$$

Loss functions:

- Mean squared error (MSE): $\mathcal{L}2= \sum_{i=1}^n\left\|r^i\right\|_2$
- Absolute error: $\mathcal{L}1= \sum_{i=1}^n\left|r^i\right|$
- Huber loss: $\mathcal{L}_{\text {huber }}=\sum_{i=1}^n \begin{cases}\left\|r^i\right\|_2^2 & \text { if } \left\|r^i\right\|_2 \leq \delta \\ 2 \delta\left(\left\|r^i\right\|_2-\frac{\delta}{2}\right) & \text { if } \left\|r^i\right\|_2>\delta\end{cases}$

![bg right:30% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250317211038.png)

---

### Iterative location methods

$$
\begin{aligned}
\hat{t}^i(m) &= \hat{t}^i(m_0) + \frac{\partial \hat{t}^i}{\partial m_j}\Delta m_j
\end{aligned}
$$
where $m_0$ is the initial model, $\Delta m_j$ is the perturbation of $(x, y, z, t0, v)$.

$$
\begin{aligned}
r^{m} &= t^i - \hat{t}^i(m) \\
&= t^i - \hat{t}^i(m_0) - \frac{\partial \hat{t}^i}{\partial m_j}\Delta m_j \\
&= r^i(m_0) - \frac{\partial \hat{t}^i}{\partial m_j}\Delta m_j
\end{aligned}
$$

---

### Iterative location methods


We seek to find the $\Delta m$ that 
$$
\begin{aligned}
r^i(m_0) &= \frac{\partial \hat{t}^i}{\partial m_j}\Delta m_j \\
r^i(m_0) &= G \Delta m
\end{aligned}
$$

$\Delta m$ can be obtained using standard least squares. Next, we set $m_0$ to $m_0 + \Delta m$ and repeat the process until the locatin converges.

---

### How to evaluate the results of earthquake location?

How do we define the "best" location?

The average least square residual:
$$
\epsilon = \frac{1}{n_df}\sum_{i=1}^n\left\|t^i-\hat{t}^i\right\|_2
$$
is called the *variance* of the residuals, where $n_{df}$ is the number of degrees of freedom.

A common term is *variance reduction* (VR), which is defined as:
$$
\text{VR} = \frac{\epsilon_{\text{old}}-\epsilon_{\text{new}}}{\epsilon_{\text{old}}} \times 100\%
$$

---

### How to define the uncertainty in the location?

![bg right:30% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250317213505.png)

Based on least squares and L2 norm, we define:

$$
\chi^2 = \sum_{i=1}^n\left(\frac{t^i-\hat{t}^i}{\sigma^i}\right)^2
$$

where $\sigma^i$ is the uncertainty of the $i$-th residual.

The $\chi^2$ distribution approximate the degree of freedom of the residuals $n_{df}$.

---

### The $\chi^2$ distribution

The $\chi^2$ distribution is a probability distribution that describes the sum of the squares of independent standard normal random variables.

The probability density function of the $\chi^2$ distribution is:
$$
f(x;k) = \frac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}e^{-x/2}
$$

![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250317214154.png)

---

### 90% confidence interval of $\chi^2$

The 90% confidence interval of the $\chi^2$ distribution is bounded by:
$$
\chi^2_{0.05;n_{df}} \leq \chi^2 \leq \chi^2_{0.95;n_{df}}
$$

Table for $n_{df}=5, 10, 20, 50, 100$:

| ndf | $\chi^2_{0.05}$ | $\chi^2_{0.50}$ | $\chi^2_{0.95}$ |
|-----|------------------------|-----------------------|------------------------|
| 5   | 0.412                  | 4.35                  | 11.1                  |
| 10  | 3.94                   | 9.34                  | 18.3                  |
| 20  | 10.9                   | 19.3                  | 31.4                  |
| 50  | 34.8                   | 49.3                  | 71.4                  |
| 100 | 77.9                   | 99.3                  | 129.6                 |

---

### How to apply to real data?

Note that the $\sigma^i$ are critical in the analysis, which is based on the assumption that the data misfit are random, uncorrelated, and have a Gaussian distribution.

The estimated **data uncertainty** $\sigma^i$ is often estimated from the residual of the best location:

$$
\sigma^i(m^*) = \frac{1}{n_{df}}\sum_{i=1}^n\left\|t^i-\hat{t}^i\right\|_2
$$
where $m^*$ is the best-fitting location. 

Then we can use the estimated $\sigma^i$ to calculate the $\chi^2$ value; then obtain an estimate of the 95% confidence ellipse for the solution.

---

### Challenges: unmodeled velocity heterogeneity

Case: Earthquakes located along a fault will often be mislocated if the seismic velocity changes across the fault.

![20250317220051](https://raw.githubusercontent.com/zhuwq0/images/main/20250317220051.png)

---

### Challenges: trade-off between event dpeth and origin time

Case: Earthquake locations for events outside of a network are often not well constrained.

<div style="display: flex; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250317220241.png" style="width: 48%;">
  <img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250317220317.png" style="width: 48%;">
</div>

Mitigations:

- $S-P$ time can be used to estimate the source-receiver range at each station
- Adding depth phase $pP$ (using the differential time $pP - P$) can help constrain the depth

---

### Locating earthquake using relative arrival times

[notebook](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/codes/earthquake_location/#locating-earthquakes-using-both-absolute-travel-times-and-relative-travel-time-differences)

In the common situation where the location error is dominated by the biasing effects of unmodeled 3-D velocity structure, the relative location among events within a localized region can be determined with much greater accuracy than the absolute location of any of the events.

![bg right fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250317222555.png)

---

### [HypoDD: Double-difference earthquake location](https://www.ldeo.columbia.edu/~felixw/hypoDD.html)

$$
\Delta r_k^{i j}=\left(t_k^i-t_k^j\right)-\left(\hat{t}_k^i-\hat{t}_k^j\right)
$$
where $t_k^i$ and $\hat{t}_k^i$ are the observed and predicted arrival times at the $k$-th station for the $i$-th earthquake, respectively.


![bg right height:550px fit](./assets/Screenshot%202023-10-08%20at%2021.57.01.png)

---

### [GrowClust: A Hierarchical Clustering Algorithm for Relative Earthquake Relocation](https://github.com/dttrugman/GrowClust)

![height:400px](./assets/Screenshot%202023-10-08%20at%2022.01.21.png)

Review: [clusering](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/05_phase_association.html#6)

---

### More on: Uncertainty

- Aleatoric uncertainty
  - The irreducible part of the uncertainty
  - Uncertainty due to inherent randomness, e.g., the outcome of flipping a coin
- Epistemic uncertainty
  - The reducible part of the uncertainty
  - Uncertainty due to lack of knowledge, e.g., lack of data

---

### Uncertainty Quantification


<!-- - [Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference)
- [Monte Carlo simulation](https://en.wikipedia.org/wiki/Monte_Carlo_method) -->
- Standard deviation of slope and intercept of linear regression
- [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [Markov Chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
- [Hamiltonian Monte Carlo (HMC)](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
- [Stein Variational Gradient Descent (SVGD)](https://arxiv.org/abs/1608.04471)
- [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142)

---

### [HypoSVI: Hypocentre inversion with Stein variational inference](https://arxiv.org/abs/2101.03271)

![width:1200px](./assets/Screenshot%202023-10-08%20at%2021.43.21.png)

---

<!-- _class: lead -->

# Part 8
# What a catalogue is for

**What do a million earthquakes say that one does not?**

Classically: Gutenberg-Richter, Omori, ETAS - three laws and a few parameters.
With learning: the same laws, but now the catalogue is large enough that the parameters move.

*Returns in weeks 1 and 4*

---

## The Earthquake Cycle

- Elastic rebound

![w:1000px](https://raw.githubusercontent.com/zhuwq0/images/main/20250407142312.png)

---

## Spring-block model

When the force exerted by the spring  exceeds the static friction $\mu_s$, the block will slide until the dynamic friction $\mu_d$ balances the reduced level of stress.
If $\mu_s$, $\mu_d$, and $v$ are all constant, then the “earthquakes” will repeat at regular recurrence intervals.

![20250407155433 h:300px](https://raw.githubusercontent.com/zhuwq0/images/main/20250407155433.png)

---

## Parkfield earthquake

Significant earthquakes at Parkfield, California, have repeated at  fairly regular intervals since 1850, leading to predictions of another event  before 1993. However the earthquake did not occur until 2004.

![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250407172642.png)

---

## Aftershocks

Earthquakes are thought to trigger aftershocks either from the dynamic effects of their radiated seismic waves or the resulting permanent static  stress changes

- The seismicity rate decays with time, following a power law  relationship, called Omori’s law after Omori (1894)

- Coulomb failure function (CFF)
$$CFF = |\tau_s| + \mu (\tau_n + P)$$

where $\tau_s$ is the shear traction on the fault, $\tau_n$ is the normal traction (positive for tension), $P$ is the pore fluid pressure, and $\mu$ is the coefficient of static friction.

<!-- --- -->

<!-- ![bg fit](./assets/deep_learning_earthquake_monitoring.png) -->

---

### Earthquake Source Parameters

- Magnitude
- Origin time
- Location
- Focal mechanism
- Stress drop
- Energy
- Frequency
- ...

---

### Statistical relationship between source parameters

[wiki](https://en.wikipedia.org/wiki/Aftershock)
- Gutenberg-Richter Law (1944)
- Omori Law (1894)
- Båth's Law (1965)
- The Epidemic Type Aftershock Sequence (ETAS) model (1988)
- ...

---

### The Gutenberg-Richter Law

$$
N=10^{a-b M}
$$
Where:
- $N$ is the number of events greater or equal to $M$
- $M$ is magnitude
- $a$ and $b$ are constants

---

### The Gutenberg-Richter Law

![bg right:60% h:600](./assets/Hutton2010.png)
<!-- footer: (Hutton et al. 2010) -->

---

### The Gutenberg-Richter Law
![h:500](./assets/Ross2019.png)
<!-- footer: (Ross et al. 2019) -->

---

### What controls the slop $b$?

![h:500](./assets/Scholz1968.png)
<!-- footer: (Scholz 1968) -->

---

### Temporal variation of $b$

![w:1200](./assets/Gulia_Wiemer_2019.png)
<!-- footer: (Gulia and Wiemer 2019) -->

---

### The magnitude completeness ($M_c$)

What affects the magnitude completeness?

* Station coverage
* Background noise
* Detection algorithms
* ...

![bg right h:500](./assets/Hutton2010.png)

<!-- footer: (Hutton et al. 2010) -->

---

### Omori Law

$$
n(t) = \frac{K}{c+t}
$$
The number of events $n(t)$ in time $t$ after the mainshock

![bg right:60% h:500](https://static.temblor.net/wp-content/uploads/2019/10/fig_OmoriPlot.jpg)

<!-- footer: (Omori 1894) -->

---

### A modified Omori Law

$$
n(t) = \frac{K}{(c+t)^p}
$$
𝐾: productivity of aftershocks
𝑝: decay rate
c: delay time


![bg right:50% w:500](./assets/Ogata1983.png)
<!-- footer: (Ogata 1983) -->

---

### The decay rate $p$

- $p \sim 1.1$
- valid for a long time range
- independent of magnitude

<!-- ![bg left:10% w:420](./assets/Utsu1995.png) -->
![bg right:50% w:450](./assets/Utsu2002b.png)

<!-- footer: (Utsu 2002) -->

---

### The aftershock productivity $K$

- Combined with the Gutenberg-Richter law

$$
K = K_0 10^{b (M_{main} - M)}
$$

$$
n(t, M) = \frac{10^{a + b (M_{main} - M)}}{(c+t)^p}
$$

![bg right:50% w:500](./assets/Ogata1983.png)

<!-- footer: (Reasenberg and Jones 1989) -->

---

### The Epidemic Type Aftershock Sequence (ETAS) model

![h:500px](https://raw.githubusercontent.com/zhuwq0/images/main/20250407180110.png)


<!-- footer: "" -->

---

### The Epidemic Type Aftershock Sequence (ETAS) model

$$
% g\left(t-t_i, M ; \theta\right)=\frac{K \cdot \exp \left(\beta\left(M-M_c\right)\right)}{\left(t-t_i+c\right)^p}
\lambda(t)=\mu+\sum_{t_i<t} K \cdot \exp \left(\beta\left(M_i-M_c\right)\right) \cdot\left(t-t_i+c\right)^{-p}
$$
- $\mu$ is the background rate
- $K$ is the productivity
- $M_c$ is the magnitude completeness
- $p$ is the decay rate
- $c$ is the delay time
- $\beta$ is the magnitude scaling
- $t_i$ is the occurrence times of previous earthquakes.


<!-- footer: (Ogata 1988) -->

---

### The ETAS model

- Modeling earthquake activity of a Poissonian background and a cluster process
- Analyzing “background” or “clustered” events
- Most widely used model for earthquake forecasting

![bg right:50% h:700](./assets/Utsu1995c.png)

<!-- footer: (Utsu et al. 1995) -->

---

### Coulomb failure stress (CFS) (Static triggering)

$$
\Delta \sigma_f=\Delta \tau+\mu\left(\Delta \sigma_n+\Delta p\right)
$$
$\Delta \tau$ : change in shear stress
$\Delta \sigma_n$ : change in normal stress (positive for tension)
$\Delta p$ : change in pore pressure
$\mu$ : friction coefficient

<!-- footer: (Stein and Lisowski 1983) -->

---

### Earthquake swarms

“[a sequence] where the number and the magnitude of earthquakes gradually increase with time, and then decreases after a certain period. There is no single predominant principal earthquake” - Mogi (1963)

![bg right:50% h:700](./assets/Mogi_1963.png)

<!-- footer: "" -->

<!-- ### Clustering analysis of earthquakes

![w:1100](./assets/Zaliapin_BenZion_2013.png) -->

---

### Deep learning for earthquake statistics

![h:450](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-018-0438-y/MediaObjects/41586_2018_438_Fig1_HTML.png?as=webp)

<!-- footer: "Deep learning of aftershock patterns following large earthquakes, Devries et al. 2018" -->

---

<!-- _class: lead -->

# Part 9
# Mechanism

**How did the fault move?**

Classically: first motions on a stereonet, or a waveform inversion.
With learning: machine polarities at a scale that makes mechanisms routine.

*Returns in week 12*

---

### Fault plane

![w:500](./assets/Screenshot%202023-10-29%20at%2014.36.58.png)

![bg right:55% 80%](./assets/Screenshot%202023-10-29%20at%2014.55.02.png)

---

### Focal Mechanism Beachball

<!-- ![](./assets/Screenshot%202023-10-29%20at%2014.49.33.png) -->

![w:400](./assets/Screenshot%202023-10-29%20at%2014.58.03.png)

![bg right:50% 80%](./assets/Screenshot%202023-10-29%20at%2014.53.33.png)

--- 

### Radiation pattern

![h:500](./assets/Screenshot%202023-10-29%20at%2014.38.56.png)

![bg right:50% 80%](./assets/Screenshot%202023-10-29%20at%2015.04.01.png)

<!-- footer: "Kumar et al. (2016)" -->

---

### How to determine focal mechanism?

**Review: [Inverse Problems in Geophysics](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/06_location_and_relocation.html#6)**

- Forward function: [last lecture](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/08_focal_mechanism_and_momemt_tensor/#33-more-specific-example-of-a-fault-described-by-a-double-couple-source)
- Objective/Loss function
- Gradient
- Optimizer

---

<style scoped>
section {
  font-size: 28px;
}
</style>

### Focal mechanism from first motion polarity

- FPFIT

Objective/Loss function: 
$$
F^{i, j}=\frac{\sum_k \left\{| p_0^{j, k}-p_t^{i, k} \mid \cdot w_0^{j, k} \cdot w_t^{i, k}\right\}}{\sum_k\left\{w_0^{j, k} \cdot w_t^{i, k}\right\}}
$$

$P_0^{j, k} are P_t^{i, k}$ are the observed and theoretical first-motion polarity (0.5 for compression, -0.5 for dilatation).
$w_t^{i, k}=[A(i, k)]^{1 / 2}$ is the square root of the normalized theoretical P-wave radiation amplitude $A(i, k)$ of earthquake $E^j$ recorded at the $k^{\text {th }}$ station for source model $M^i$.

<!-- footer: "Reasenberg (1985)" -->

---

### Focal mechanism from first motion polarity

- HASH

*Figure: [Hardebeck & Shearer (2002), A New Method for Determining First-Motion Focal Mechanisms, BSSA](https://doi.org/10.1785/0120010200)*

<!-- footer: "Hardebeck and Shearer (2002)" -->

<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->

---

<!-- _class: lead -->

# Part 10
# This course

---

### How can we better monitor earthquakes?

**Instrument side**
(How to collect more and better data?)

- Dense seismic networks
- New sensors: broadband seismometer, nodal array, and DAS (Distributed Acoustic Sensing)
- Remote sensing, LiDAR, etc.

--- 

### How can we better monitor earthquakes?

**Algorithm side**
(New techniques for processing data and extracting information?)

- Many signal processing algorithms, such as, STA/LTA, template matching, filtering, etc.

- Machine learning & deep learning

- Numerical simulation

- Inverse theory

- Statistical analysis

---

### Things to learn in this course

- Faimilar with seismic data
- Learn the state-of-the-art machine learning methods for seismic data processing
- Process seismic data, build seismic catalogs, and analyzing seismicity
- Learn basic inverse theory for earthquake location, focal mechanism, seismic tomography, etc.

---

### The advantages of machine learning

Deep Learning (Deep Neural Networks) is a new paradigm of software development

- [Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35)

- [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)

---

### Applications of deep learning in seismology

- Neural Networks
- Automatic Differentiation
- Optimization/Inversion

---

<span style="color:Violet;">Machine Learning</span> and <span style="color:LimeGreen;">Inversion</span>


|||
| --- | --- | 
| 09/18 | <span style="color:Violet;">Seismic Data Processing</span>  |
| 09/25 | <span style="color:Violet;">Earthquake Detection</span> |
| 10/02 | <span style="color:Violet;">Phase Picking & Association</span> |
| 10/09 | <span style="color:LimeGreen;">Earthquake Location & Relative Location |
| 10/16 | <span style="color:LimeGreen;">Focal Mechanism & Moment Tensor |
| 10/23 | <span style="color:Violet;">Earthquake Statistics</span> |
| 10/30 | <span style="color:black;">Ambient Noise</span> |
| 11/06 | <span style="color:LimeGreen;">Seismic Tomography</span> |
| 11/13 | <span style="color:LimeGreen;">Full-waveform Inversion</span> |

---

### Schedule

| Date | Seismology | Machine learning |
| --- | --- | --- |
| 09/01 | Magnitude calibration | Regression & uncertainty |
| 09/08 | Where aftershocks occur | Classification |
| 09/15 | Where aftershocks occur | Bias-variance, boosting, CV |
| 09/22 | Fault structure from seismicity | Clustering, mixture models, EM |
| 09/29 | Earthquake / quarry-blast discrimination | NN: classification |
| 10/06 | Phase picking | NN: segmentation |
| 10/13 | Event detection on DAS | NN: object detection |
| 10/20 | Ground-motion prediction | Transformers |
| 10/27 | Template matching | Similarity & embeddings |
| 11/03 | Waveform generation | VAE |
| 11/10 | Denoising | Denoising autoencoders |
| 11/17 | Focal mechanism & moment tensor | Inversion I - linear |
| 11/24 | Location & relocation | Inversion II - non-linear |
| 12/01 | Tomography | Inversion III - fields |

---

### How each week runs

One notebook, worked in the room, not watched.

1. A published claim, and the paper it comes from.
2. The data, and what is wrong with it.
3. **A baseline** - the simplest thing that could work.
4. The method, against that baseline, on the same data and the same metric.
5. What would have to be true for the result to be wrong.

A method that does not beat its baseline has not earned the week.

---

### Final project

The Geysers: the most seismically active field in California, and the shaking is
a side effect of an industrial process.

- One field, one catalogue, your own question.
- Proposal in week 5, presentations in week 15.
- `docs/project.md` lists nine questions - or bring your own.

---

### Grading

- Attendance and participation (50%)
- Final project (50%)
    - Project proposal (10%)
    - Project presentation (20%)
    - Project report (20%)
- Extra credit (up to 10%)

---

### Questions?

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

# Earthquake Statistics

<!-- Class project datasets:

- [Nodal Seismic Experiment at the Berkeley Section of the Hayward Fault](https://pubs.geoscienceworld.org/ssa/srl/article/93/4/2377/613344/Nodal-Seismic-Experiment-at-the-Berkeley-Section) (Taka'aki et al. 2022)

- An island on Mid-Atlantic Ridge: [Networks](http://ds.iris.edu/gmap/#maxlat=73.3732&maxlon=-1.582&minlat=68.7841&minlon=-15.1596&network=*&drawingmode=box&planet=earth), [Seismicity](https://nnsn.geo.uib.no/nnsn/#/)

- [California](https://earthquake.usgs.gov/earthquakes/map/?extent=30.25907,-128.67188&extent=42.65012,-109.51172&range=month&magnitude=all&listOnlyShown=true&settings=true) -->


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

## Earthquake recurrence model

![h:500px](https://raw.githubusercontent.com/zhuwq0/images/main/20250407155751.png)

---

## Parkfield earthquake

Significant earthquakes at Parkfield, California, have repeated at  fairly regular intervals since 1850, leading to predictions of another event  before 1993. However the earthquake did not occur until 2004.

![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250407172642.png)

---

## The block-slider model

A problem with the characteristic earthquake hypothesis is that it ignores the interactions with adjacent segments on the same fault, as well as interactions with other faults

![w:1000px](https://raw.githubusercontent.com/zhuwq0/images/main/20250407172905.png)

---

## Self-similar and fractial scaling relationship

* Power-law  distribution of seismicity rates (the b-value relationship)
* Nearly constant value of stress drop over a wide range of earthquake sizes
* Fractal dimension D approximately twice the b-value (Turcotte, 1997)

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

### Temporal variation of $b$

![w:1200](./assets/Gulia_Wiemer_2019_c.png)

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

![h:600](./assets/Hutton2010b.png)
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


### How about for foreshocks?

* Inverse Omori law $n(t) \propto t^p$

* but individual sequences rarely display this behavior


![bg right:60% w:700](./assets/Jones_Molnar_1979.png)

<!-- footer: (Jones and Molnar 1979) -->

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

### Incorporate spatial triggering into ETAS

$$
\lambda(x, y, t \mid \mathcal{H})=\mu+\sum_{t_i<t} \frac{K \cdot \exp \left(\beta\left(M_i-M_c\right)\right)}{\left(t-t_i+c\right)^p} \cdot \frac{1}{\left(x^2+y^2+d\right)^q}
$$

- $q$: the spatial decay rate of intensity following an event

<!-- footer: (Ogata 1998) -->

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


### Physical models on aftershocks spatial distribution

<div style="display: flex; justify-content: space-between;">
    <img src="./assets/King1994b.png" height="500">
    <img src="./assets/King1994c.png" height="500">
</div>

<!-- footer: (King et al 1994) -->

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

### Coulomb failure stress (CFS)

![bg right:58% h:700](./assets/King1994.png)

<!-- footer: (King et al 1994) -->

---

### Dynamic triggering

![h:500](./assets/Hill1993.png)

<!-- footer: (Hill et al. 1993) -->

---

### Dynamic triggering

![h:500](./assets/Hill1993b.png)

<!-- footer: (Hill et al. 1993) -->

---

### Earthquake swarms

“[a sequence] where the number and the magnitude of earthquakes gradually increase with time, and then decreases after a certain period. There is no single predominant principal earthquake” - Mogi (1963)

![bg right:50% h:700](./assets/Mogi_1963.png)

<!-- footer: "" -->

<!-- ### Clustering analysis of earthquakes

![w:1100](./assets/Zaliapin_BenZion_2013.png) -->

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

**2012 Brawley,CA swarm**
![h:500](./assets/Hauksson2013.png)

![h:500](./assets/Hauksson2013b.png)

<!-- footer: (Hauksson et al. 2013) -->

---

**2016 Cahuilla,CA swarm**

![bg right:70% h:700](./assets/ross2020.png)

<!-- footer: (Ross et al. 2020) -->

---

**2018 Pahala,Hawaii swarm**

![width:900px](./assets/Screenshot%202023-10-15%20at%2023.54.40.png)

<!-- ![](https://caltech-prod.s3.amazonaws.com/main/images/image001_oV7aIUL.max-500x500.gif) -->

<!-- footer: (Wilding et al. 2020) -->

---

**2018 Pahala,Hawaii swarm**

![width:600px](./assets/Screenshot%202023-10-15%20at%2023.57.09.png)
![bg right:45% w:500](https://caltech-prod.s3.amazonaws.com/main/images/image001_oV7aIUL.max-500x500.gif)

<!-- footer: (Wilding et al. 2020) -->

---

### Spatial-temporal evolution patterns of swarms

- Migration distance vs. time
    - $R \propto t^{\alpha}$, $\alpha \sim 0.5, 1$
    - $R \propto \log(t)$

- Migration speeds
    - m/day to km/hour

- Similarity to induced seismicity
<!-- **Aggregating 18 swarms in southen California**

![](./assets/chen2012b.png) -->

<!-- footer: "" -->

---

### Deep learning for earthquake statistics

![h:450](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-018-0438-y/MediaObjects/41586_2018_438_Fig1_HTML.png?as=webp)

<!-- footer: "Deep learning of aftershock patterns following large earthquakes, Devries et al. 2018" -->

---

### Deep learning for earthquake statistics


![w:1000px](./assets/grl66186-fig-0001-m.jpeg)

<!-- footer: "Using Deep Learning for Flexible and Scalable Earthquake Forecasting, Kelian et al. 2023" -->

---

### Deep learning for earthquake statistics

![w:1000px](./assets/grl66186-fig-0002-m.jpg)


---

### Deep learning for earthquake statistics

![w:1000px](./assets/41598_2023_38033_Fig1_HTML.webp)

<!-- footer: "A neural encoder for earthquake rate forecasting, Zlydenko et al. 2023" -->



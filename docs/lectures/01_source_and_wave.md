---
marp: true
paginate: true
theme: gaia
backgroundColor: #fff
---

# Earthquake Source and Seismic Wave

Textbooks: 
Shearer, P. M. (2019). Introduction to seismology. Cambridge University Press.
GEOPHYS 210: Earthquake Seismology by Eric Dunham
Segall, P. (2010). Earthquake and volcano deformation. Princeton University Press.

---

### Earthquake rupture

<video src="assets/rough_vy.mp4" controls width="80%"></video>
[Eric Dunham](https://pangea.stanford.edu/~edunham/publications.html)

---

### 1. How does an earthquake start?" ([notebook](codes/spring_slider/))

<!-- ![height:600px](./assets/ElasticRebound.jpeg) -->
![spring slider](./assets/Screenshot%202023-08-23%20at%2022.53.46.png)

---

<video src="./assets/GIF_1-BlockEQMachine_Graph.mp4" controls width="80%"></video>

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

### 2. How does seismic wave propagate? ([notebook](codes/wave_propagation/))

![height:300px](https://gpg.geosci.xyz/_images/pwave-animated-2.gif)
![height:350px](https://gpg.geosci.xyz/_images/s-wave-animated.gif)

<!-- ---

![height:500px](./assets/Screenshot%202023-08-24%20at%2010.46.37.png)

---

![height:600px](./assets/Screenshot%202023-08-24%20at%2010.48.17.png) -->

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

## [Strain](https://en.wikipedia.org/wiki/Deformation_%28physics%29)

$$
\gamma_{y z}=\gamma_{z y}=\frac{\partial u_y}{\partial z}+\frac{\partial u_z}{\partial y} \quad, \quad \gamma_{z x}=\gamma_{x z}=\frac{\partial u_z}{\partial x}+\frac{\partial u_x}{\partial z}
$$

$$
\underline{\underline{\varepsilon}}=\left[\begin{array}{lll}
\varepsilon_{x x} & \varepsilon_{x y} & \varepsilon_{x z} \\
\varepsilon_{y x} & \varepsilon_{y y} & \varepsilon_{y z} \\
\varepsilon_{z x} & \varepsilon_{z y} & \varepsilon_{z z}
\end{array}\right]=\left[\begin{array}{ccc}
\varepsilon_{x x} & \frac{1}{2} \gamma_{x y} & \frac{1}{2} \gamma_{x z} \\
\frac{1}{2} \gamma_{y x} & \varepsilon_{y y} & \frac{1}{2} \gamma_{y z} \\
\frac{1}{2} \gamma_{z x} & \frac{1}{2} \gamma_{z y} & \varepsilon_{z z}
\end{array}\right]
$$

![height:500px](https://upload.wikimedia.org/wikipedia/commons/2/23/2D_geometric_strain.svg)

---

## Earthquake recurrence model

![width:900px](./assets/Screenshot%202023-08-24%20at%2016.07.47.png)
( Shimazaki and Nakata, 1980)


---

### The Haskell source model

![height:500px](./assets/Screenshot%202023-08-24%20at%2010.49.47.png)

---

![height:250px](./assets/Screenshot%202023-08-24%20at%2010.52.41.png)

![height:250px](./assets/Screenshot%202023-08-24%20at%2010.56.22.png)

---


$$
|A(\omega)|=g M_0\left|\operatorname{sinc}\left(\omega \tau_r / 2\right)\right|\left|\operatorname{sinc}\left(\omega \tau_d / 2\right)\right|,
$$

![width:1100px](./assets/Screenshot%202023-08-27%20at%2023.29.24.png)

---

$$
\log |A(\omega)|=G+\log \left(M_0\right)+\log \left|\operatorname{sinc}\left(\omega \tau_r / 2\right)\right|+\log \left|\operatorname{sinc}\left(\omega \tau_d / 2\right)\right|
$$
where $G=\log g$

$\text { Approximate }|\operatorname{sinc} x| \text { as } 1 \text { for } x<1 \text { and } 1 / x \text { for } x>1:$
$$
\begin{aligned}
\log |A(\omega)|-G & =\log M_0, & & \omega<2 / \tau_d \\
& =\log M_0-\log \frac{\tau_d}{2}-\log \omega, & & 2 / \tau_d<\omega<2 / \tau_r \\
& =\log M_0-\log \frac{\tau_d \tau_r}{4}-2 \log \omega, & & 2 / \tau_r<\omega
\end{aligned}
$$

---

![width:700px](./assets/Screenshot%202023-08-24%20at%2016.22.14.png)

---

![width:800px](./assets/Screenshot%202023-08-24%20at%2016.20.06.png)


<!-- ---

![width:1100px](./assets/Screenshot%202023-08-24%20at%2010.54.16.png)

---

![width:1100px](./assets/Screenshot%202023-08-24%20at%2010.55.13.png) -->


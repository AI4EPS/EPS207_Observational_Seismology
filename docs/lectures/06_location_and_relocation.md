---
marp: true
paginate: true
theme: gaia
backgroundColor: #fff
style: |
  section {
    font-size: 30px;
  }
---

# Earthquake Location and Relocation

Notebooks: [codes/earthquake_location.ipynb](codes/earthquake_location/)

---

### How to locate an earthquake?

![height:500px](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/styles/full_width/public/thumbnails/image/locating%20earthquakes%201.gif?itok=z60HGZwY)

---

### How to locate an earthquake?

![height:500px](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/thumbnails/image/eq-ed-triangulation.gif)

---

### Optimization (Inverse) problem

- Minimize the difference between observed and predicted values

![height:400px](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1920px-Linear_regression.svg.png)
<!-- ![height:450px](https://www.jmp.com/en_no/statistics-knowledge-portal/what-is-multiple-regression/fitting-multiple-regression-model/_jcr_content/par/styledcontainer_2069/par/lightbox_4130/lightboxImage.img.png/1548351208631.png) -->

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

### [HypoDD: Double-difference earthquake location](https://www.ldeo.columbia.edu/~felixw/hypoDD.html)

![width:1200px](./assets/Screenshot%202023-10-08%20at%2021.58.46.png)

---


![bg fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250317222318.png)

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

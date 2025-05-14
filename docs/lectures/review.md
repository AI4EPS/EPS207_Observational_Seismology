---
marp: true
theme: default
paginate: true
style: |
  section {
    font-size: 20px;
    line-height: 1.2;
  }
---

# Seismology Review Session

---

# Review Session Overview

- **Part 1:** Pre-midterm topics review (Chapters 1-7)
  - Stress and strain
  - Seismic wave equation
  - Ray theory and travel times
  - Tomography
  - Surface waves and normal modes

- **Part 2:** Post-midterm topics (Chapters 8-13)
  - Earthquake sources
  - Earthquake prediction
  - Seismometers
  - Earth noise and ambient seismology
  - Anisotropy

<!-- ---

![bg fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430223415.png) -->

---

# Chapter 2: Stress and Strain

- **Stress Tensor $\sigma_{ij}$**: Force per unit area (units: Pa)
  - Symmetric: $\sigma_{ij} = \sigma_{ji}$
  - Traction: $T_i = \sigma_{ij}n_j$

![h:200px](https://raw.githubusercontent.com/zhuwq0/images/main/20250430224735.png)


- **Strain Tensor $\varepsilon_{ij}$**: Displacement gradients
  - $\varepsilon_{ij} = \frac{1}{2}(u_{i,j} + u_{j,i})$

![bg right:50% fit vertical](https://raw.githubusercontent.com/zhuwq0/images/main/20250430223837.png)
<!-- ![bg right:60% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430224441.png) -->

![bg right:50% fit vertical](https://raw.githubusercontent.com/zhuwq0/images/main/20250430224242.png)

---

# Chapter 2: Stress and Strain

- **Stress Tensor $\sigma_{ij}$**: Force per unit area (units: Pa)
  - Symmetric: $\sigma_{ij} = \sigma_{ji}$
  - Traction: $T_i = \sigma_{ij}n_j$

- **Strain Tensor $\varepsilon_{ij}$**: Displacement gradients
  - $\varepsilon_{ij} = \frac{1}{2}(u_{i,j} + u_{j,i})$

- **Linear Stress-Strain Relationship**: $\sigma_{ij} = C_{ijkl}\varepsilon_{kl}$
  - For isotropic media: $\sigma_{ij} = \lambda\delta_{ij}\varepsilon_{kk} + 2\mu\varepsilon_{ij}$
  - $\lambda$, $\mu$ (Lamé parameters)

---

# Chapter 3: The Seismic Wave Equation

- **Equation of motion**: $\rho\ddot{u}_i = \sigma_{ij,j} + f_i$

- **Wave equation in homogeneous media**:
  $\rho\ddot{\mathbf{u}} = (\lambda + \mu)\nabla(\nabla \cdot \mathbf{u}) + \mu\nabla^2\mathbf{u}$

- **Potentials**
  - $\mathbf{u} = \nabla\phi + \nabla\times\mathbf{\psi}$
  - $\nabla^2\phi - \frac{1}{\alpha^2}\ddot{\phi} = 0$
  - $\nabla^2\mathbf{\psi} - \frac{1}{\beta^2}\ddot{\mathbf{\psi}} = 0$

  - P-waves (compressional): $\alpha = \sqrt{\frac{\lambda + 2\mu}{\rho}}$
  - S-waves (shear): $\beta = \sqrt{\frac{\mu}{\rho}}$

- **Wave polarization**:
  - P-waves: parallel to propagation direction
  - S-waves: perpendicular to propagation direction

![bg right:45% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430224841.png)

---

# Chapter 4: Ray Theory - Travel Times

- **Snell's Law**: $\frac{\sin\theta_1}{v_1} = \frac{\sin\theta_2}{v_2} = p$ (ray parameter)

![h:150px](https://raw.githubusercontent.com/zhuwq0/images/main/20250430225157.png)


- **Ray paths**: Bend toward regions of lower velocity

![h:150px](https://raw.githubusercontent.com/zhuwq0/images/main/20250430225226.png)

- **Travel time curves**: Plot of arrival time vs. distance

<!-- ![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430225330.png) -->

<!-- ![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430225545.png) -->

![bg right:20% h:250px vertical](https://raw.githubusercontent.com/zhuwq0/images/main/20250430230307.png)
![bg right:50% h:350px](https://raw.githubusercontent.com/zhuwq0/images/main/20250430225839.png)

---

# Chapter 4: Ray Theory - Travel Times

- **Ray nomenclature**: 
  - P, S (mantle)
  - K (outer core)
  - I (inner core)
  - c (core-mantle boundary reflection)
  - i (inner core boundary reflection)

![bg right:40% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430225736.png)

![20250430225934](https://raw.githubusercontent.com/zhuwq0/images/main/20250430225934.png)

---

# Chapter 5: Inversion of Travel Time Data

- **Tomography**:
  - $Gm = d$ where $G$ is sensitivity matrix
  - $m$ is model parameters (velocity perturbations)
  - $d$ is data (travel time residuals)
  - Solutions: Damped least squares, minimum norm

![20250430230109 h:150px](https://raw.githubusercontent.com/zhuwq0/images/main/20250430230109.png)

- **Resolution tests**: Checkerboard, spike tests

![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430230141.png)
  
---

# Chapter 6: Ray Theory - Amplitude and Phase

- **Geometrical spreading**: Amplitude ~ 1/r for body waves
  
- **Reflection/Transmission coefficients**:
  - Functions of incident angle
  - Critical angles and post-critical reflections

$$
\begin{aligned}
\text{Reflection coefficient: } & \grave{S} \acute{S} =\frac{\rho_1 \beta_1 \cos \theta_1-\rho_2 \beta_2 \cos \theta_2}{\rho_1 \beta_1 \cos \theta_1+\rho_2 \beta_2 \cos \theta_2}, \\
\text{Transmission coefficient: } & \grave{S} \grave{S} =\frac{2 \rho_1 \beta_1 \cos \theta_1}{\rho_1 \beta_1 \cos \theta_1+\rho_2 \beta_2 \cos \theta_2} .
\end{aligned}
$$

- **Attenuation**:
  <!-- - Quality factor Q: $E/\Delta E = 2\pi Q$ -->
  - Quality factor Q: $\frac{1}{Q(\omega)} = - \frac{\Delta E}{2\pi E}$
  - $A(r, \omega) = A_0 e^{-\omega r/2cQ}$
  - Frequency dependent

![bg right:40% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430230643.png)  

---

# Chapter 7: Reflection Seismology

- **NMO correction**: $\Delta t_{NMO} = \frac{x^2}{2v^2t_0}$

![bg right:60% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430231405.png)
![bg right:60% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430231432.png)

- **CMP stacking**: Sum traces with common midpoint

- **Migration**: Moving reflections to true positions
  <!-- - Diffraction summation
  - Kirchhoff migration
  - F-K migration -->

<!-- - **Velocity analysis**: Semblance analysis -->

---

# Chapter 8: Surface Waves and Normal Modes

- **Love waves**: 
  - Transverse (SH) motion
  - Require velocity increasing with depth
  - Energy trapped near surface

- **Rayleigh waves**:
  - Elliptical particle motion (retrograde at surface)
  - Exist for any Earth model with free surface

- **Dispersion**: Phase velocity varies with frequency
  - Group velocity: $U = \frac{d\omega}{dk}$
  - Phase velocity: $c = \frac{\omega}{k}$

![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430231843.png)
![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430231917.png)

---

# Surface Wave Dispersion

<!-- - **Fundamental mode**: Lowest frequency for given wavenumber -->

<!-- - **Higher modes**: Additional solutions at higher frequencies -->

- **Dispersion curves**:
  - Love waves: Faster than Rayleigh waves
  - Sensitive primarily to shear velocity
  - Longer periods sample deeper structure

- **Applications**:
  - Upper mantle structure
  - Crustal thickness
  - Regional tomography

![h:200px](https://raw.githubusercontent.com/zhuwq0/images/main/20250430232152.png)  ![h:200px](https://raw.githubusercontent.com/zhuwq0/images/main/20250430232237.png) 

![bg right:40% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430232330.png)

---

# Normal Modes

- **Earth's free oscillations**: Standing waves

![h:150px](https://raw.githubusercontent.com/zhuwq0/images/main/20250430232453.png)

- **Types**:
  - Spheroidal modes (involve radial motion)
  - Toroidal modes (purely tangential motion)

- **Notation**: $_nS_l$ (spheroidal) and $_nT_l$ (toroidal)
$n$ is radial order; $l$ is angular order

- **Properties**: 
  - Complete orthogonal basis for Earth's response
  - Observable after large earthquakes
  - Used to constrain: Density structure, Q at long periods, Earth's rotation effects


![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250430232615.png)

---

# Chapter 9: Earthquakes and Source Theory

- Location
- Mechanism
- Magnitude
- Spectrum
- Energy

---

# Chapter 9.1: Earthquake location

Determing earthquake location by minimizing the average least square residual:
$$
\epsilon = \frac{1}{n_df}\sum_{i=1}^n\left\|t^i-\hat{t}^i\right\|_2
$$

The uncertainty in the location can be estimated by the $\chi^2$ distribution:

$$
\chi^2 = \sum_{i=1}^n\left(\frac{t^i-\hat{t}^i}{\sigma^i}\right)^2
$$

The $\sigma^i$ is often estimated from the residual of the best location:
$$
\sigma^i(m^*) = \frac{1}{n_{df}}\sum_{i=1}^n\left\|t^i-\hat{t}^i\right\|_2
$$

The 90% confidence interval of the $\chi^2$ distribution is bounded by: 
$$
\chi^2_{0.05;n_{df}} \leq \chi^2 \leq \chi^2_{0.95;n_{df}}
$$


![bg right:35% h:300px vertical](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/thumbnails/image/eq-ed-triangulation.gif)

![bg right:50% h:250px](https://raw.githubusercontent.com/zhuwq0/images/main/20250317213505.png)

---

# Challenges in earthquake location

- Unmodeled velocity heterogeneity
- Trade-off between event depth and origin time

<!-- ![20250317220051](https://raw.githubusercontent.com/zhuwq0/images/main/20250317220051.png) -->

<div style="display: flex; justify-content: center;">
  <img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250317220051.png" width="40%">
  <img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250317220317.png" width="50%">
</div>

---

# Constrain relative earthquake location using arrival time differences

![w:450px](https://raw.githubusercontent.com/zhuwq0/images/main/20250317222555.png) ![w:600px](https://raw.githubusercontent.com/zhuwq0/images/main/202505010932076.png)

---

# Chapter 9.2: Earthquake Mechanism

- **Fault parameters**: Strike, Dip, Rake

- **Fault types**: Strike-slip, Normal, Reverse/thrust

- **Double-couple source**: Representation of shear faulting

![20250330233905 height:350px](https://raw.githubusercontent.com/zhuwq0/images/main/20250331193229.png)

![bg right:50% w:250px vertical](https://raw.githubusercontent.com/zhuwq0/images/main/20250319214522.png)
![bg right:50% w:600px](https://raw.githubusercontent.com/zhuwq0/images/main/20250330233905.png)

---

# Moment Tensor

- **Moment tensor $M_{ij}$**: Representation of earthquake forces
  - 6 independent components
  - $M_0 = \mu A D$ (scalar moment)
    - $\mu$ = shear modulus
    - $A$ = fault area
    - $D$ = average displacement

![20250331000705 height:300px](https://raw.githubusercontent.com/zhuwq0/images/main/20250331000705.png)
![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250331000517.png)

---

# Radiation Patterns

![20250430235717 h:600px](https://raw.githubusercontent.com/zhuwq0/images/main/20250430235717.png)
![bg right:55% w:600px](https://raw.githubusercontent.com/zhuwq0/images/main/20250319231740.png)


---

# Focal Mechanism

<div style="display: flex; justify-content: center;">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250330235030.png" width="30%">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250331233446.png" width="35%">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250330235229.png" width="34%">
</div>

---

# Focal Mechanisms

- **Beach ball diagrams**:
  - Compressional quadrants vs dilatational quadrants
  - P and T axes
  - Two possible fault planes

<div style="display: flex; justify-content: center;">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250330235816.png" width="48%">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250330234223.png" width="48%">
</div>

--- 

# Moment tensor decomposition

![bg right:60% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250331003901.png)


<!-- ---

# Earthquake Spectra and Source Parameters

- **Stress drop**: $\Delta\sigma = \frac{7}{16} \frac{M_0}{r^3}$ (circular fault)

- **Corner frequency**: $f_c \propto \frac{\beta}{r}$

- **Source spectrum**: $\Omega(f) = \frac{\Omega_0}{1+(f/f_c)^2}$ (Brune model)

- **Energy partitioning**:
  - $E = E_R + E_F + E_G$ (radiated, frictional, fracture)
  - Radiation efficiency: $\eta_R = \frac{E_R}{E_R + E_G}$ -->

---

# Chapter 9.3: Earthquake Magnitude

- **Local magnitude $M_L$**: Richter's original scale

$$
M_L = \log_{10} A(X) + 2.56 \log_{10} X - 1.67
$$

![bg right:30% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250403091807.png)

- **Body wave magnitude $m_b$**: $m_b = \log_{10} (A/T) + Q(h, \Delta)$

- **Surface wave magnitude $M_S$**: $M_s = \log_{10} (A/T) + 1.66 \log_{10} \Delta + 3.30$

- **Moment magnitude $M_W$**: $M_W = \frac{2}{3}(\log_{10}M_0 - 9.1)$
  - Most physically meaningful
  - Does not saturate for large earthquakes

<!-- --- -->

<!-- # The Haskell fault model

![20250331235242 height:200px](https://raw.githubusercontent.com/zhuwq0/images/main/20250331235242.png)

![20250401002831 height:250px](https://raw.githubusercontent.com/zhuwq0/images/main/20250401002831.png) -->

---

# Magnitude saturation

The far-field amplitude spectrum for the Haskell fault model  may be expressed as

$$
|A(\omega)|=g M_0\left|\operatorname{sinc}\left(\omega \tau_r / 2\right)\right|\left|\operatorname{sinc}\left(\omega \tau_d / 2\right)\right|,
$$


$$
\log |A(\omega)|=G+\log \left(M_0\right)+\log \left|\operatorname{sinc}\left(\omega \tau_r / 2\right)\right|+\log \left|\operatorname{sinc}\left(\omega \tau_d / 2\right)\right|
$$

where $G=\log g$ is a scaling term that includes geometrical spreading, etc

![height:150px](https://raw.githubusercontent.com/zhuwq0/images/main/20250331235242.png)
![w:700px](https://raw.githubusercontent.com/zhuwq0/images/main/20250401092155.png)
![bg right:40% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250402233819.png)

<!-- ---

# Magnitude calibration

![20250402234333 height:450px](https://raw.githubusercontent.com/zhuwq0/images/main/20250402234333.png)

[USGS Magnitude Types](https://www.usgs.gov/programs/earthquake-hazards/magnitude-types); [Latest earthquake](https://earthquake.usgs.gov/earthquakes/eventpage/us7000pn9s/origin/magnitude) -->

---

# Intensity scale

The local strength of ground shaking as determined by damage to structures and the perceptions of people who experienced the earthquake.

One earthquake can have different intensities at different locations.

![bg right:60% fit](https://raw.githubusercontent.com/zhuwq0/images/main/202504030012755.png)

---

# Chapter 9.4: Earthquake spectra

- Stress drop

Moment $M_0 = \mu A D$

Stress drop $\Delta \sigma = \sigma_\text{final} - \sigma_\text{initial}$

large D $\times$ small A v.s. small D $\times$ large A? 

![20250409232116 width:900px](https://raw.githubusercontent.com/zhuwq0/images/main/20250409232116.png)


---

- Far-field seismic pulse shape

![20250409232944 height:500px](https://raw.githubusercontent.com/zhuwq0/images/main/20250409233012.png)

---

# Self-Similar Earthquake Scaling

<!-- <div style="display: flex; justify-content: space-between;">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250409234621.png" width="38%">
<img src="https://raw.githubusercontent.com/zhuwq0/images/main/20250409234815.png" width="38%">
</div> -->

![20250403002414 height:500px](https://raw.githubusercontent.com/zhuwq0/images/main/20250403002414.png)

Assuming dimensions are scaled proportionally, displacement D will increase by b

---

# Self-Similar Earthquake Scaling

<!-- ![h:300px](https://raw.githubusercontent.com/zhuwq0/images/main/20250409234921.png)  -->

- Brune model (1970)

$$u(f) = \frac{M_0}{1 + (f/f_c)^2}$$

- Corner frequency ($f_c$)
- Radiated energy ($E_s$)

![bg right:40% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250410000033.png)

---

# Chapter 9.5: Earthquake Energy

![20250409234107 height:450px](https://raw.githubusercontent.com/zhuwq0/images/main/20250409234107.png)

- Radiation efficiency: $\eta_R = \frac{E_R}{E_R + E_G}$


---

# Static and Dynamic

![20250410000604 height:550px](https://raw.githubusercontent.com/zhuwq0/images/main/20250410000604.png)

<!-- - **Gutenberg-Richter relation**: $\log_{10}N = a - bM$
  - $b$-value typically ~1 -->

---

# Chapter 10: Earthquake Statistics


- **Earthquake cycle**:
![h:200px](https://raw.githubusercontent.com/zhuwq0/images/main/20250501090955.png)

- **Earthquake recurrence models**:
![h:300px](https://raw.githubusercontent.com/zhuwq0/images/main/20250407155751.png)


---

# The Gutenberg-Richter Law

$$
N=10^{a-b M}
$$
Where:
- $N$ is the number of events greater or equal to $M$
- $M$ is magnitude
- $a$ and $b$ are constants

![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/202505010048776.png)

---

# Omori's law

$$
N(t) = \frac{K}{(t+c)^p}
$$
- $K$: productivity of aftershocks
- $p$: decay rate
- $c$: delay time

![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250501005452.png)

---

# The Epidemic-Type Aftershock Sequence (ETAS) model


<!-- ![h:500px](https://raw.githubusercontent.com/zhuwq0/images/main/20250407180110.png) -->
![h:500px](https://raw.githubusercontent.com/zhuwq0/images/main/20250501005527.png)

---

# Dynamic and static triggering

<div style="display: flex; justify-content: center;">
  <img src="https://raw.githubusercontent.com/zhuwq0/images/main/202505010050892.png" width="33%">
  <img src="https://raw.githubusercontent.com/zhuwq0/images/main/202505010050376.png" width="53%">
</div>

---

# Chapter 11: Seismometers and Seismographs

- **Inertial seismometer**: Mass-spring-damper system

- **Response types**: Displacement, Velocity, Acceleration

- **Govening equation**: 

$$
\begin{aligned}
-k z(t)-D \frac{d z(t)}{d t} &= m \frac{d^2}{d t^2}[u(t)+z(t)] \\
\ddot{z}+2 \epsilon \dot{z}+\omega_0^2 z &= -\ddot{u}
\end{aligned}
$$
$\omega_0 = \sqrt{k/m}, \epsilon = D/2m\omega_0$

![bg right:50% fit vertical](https://raw.githubusercontent.com/zhuwq0/images/main/20250501005625.png)
![bg right:50% fit h:200px](https://raw.githubusercontent.com/zhuwq0/images/main/20250501010132.png)


---

# Chapter 12: Earth Noise

- **Microseism peak**: 5-8 second period
  - Generated by ocean wave interactions

- **Sources of noise**:
  - Wind and cultural (high frequency)
  - Ocean waves (intermediate frequency)
  - Atmospheric pressure (long period)


![bg right:40% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250501010241.png)


---

# Cross-correlation of ambient noise
  - Reveals Green's function between stations
  - Used for tomography and monitoring

![20250501011112](https://raw.githubusercontent.com/zhuwq0/images/main/20250501011112.png)
![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250501010844.png)

---

# Chapter 13: Anisotropy

- **Definition**: Seismic velocity varies with direction 
- **Causes**:
  - Lattice-preferred orientation (LPO)
  - Shape-preferred orientation (SPO)
  - Aligned cracks or layering

![20250501011354 h:400px](https://raw.githubusercontent.com/zhuwq0/images/main/20250501011354.png)

---

# Shear-Wave Splitting

- **Definition**: Separation of S-waves into fast and slow components

- **Parameters**:
  - Fast polarization direction φ
  - Delay time δt

- **Measurement techniques**:
  - Eigenvalue minimization
  - Cross-correlation
  - Transverse energy minimization

- **Applications**:
  - Upper mantle flow
  - Crustal stress
  - Inner core structure

![bg right:50% fit](https://raw.githubusercontent.com/zhuwq0/images/main/20250501011513.png)

---

# Thank you!

Good luck on your final exam!
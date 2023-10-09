---
marp: true
paginate: true
theme: gaia
backgroundColor: #fff
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

<!-- ![height:400px](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1920px-Linear_regression.svg.png) -->
![height:450px](https://www.jmp.com/en_no/statistics-knowledge-portal/what-is-multiple-regression/fitting-multiple-regression-model/_jcr_content/par/styledcontainer_2069/par/lightbox_4130/lightboxImage.img.png/1548351208631.png)

---

### How to solve an optimization/inversion problem?

- Forward function
- Objective/Loss function
- Gradient
- Optimizer

---
<style scoped>
section {
  padding: 0px;
}
section::after {
  font-size: 0em;
}
</style>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTGPzx6m0wBOAta7qebCjW_n_lcXsay3Uqo7iKnVI5cxZNYWcmbTQNgOAwiuTx_ZwRuNRxOHCRBSFsq/embed?start=false&loop=true&delayms=60000" frameborder="0" width="100%" height="105%" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>


---
<style scoped>
section {
  font-size: 30px;
}
</style>

### Locating Earthquake using absolute arrival times

[notebook](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/codes/earthquake_location/#locating-earthquakes-using-absolute-travel-times)

$$
r^i=t^i-\hat{t}^i=t^i-f^i(\mathbf{m})
$$
Loss function:

- Mean squared error (MSE), $\mathcal{L}2$ loss: $\mathcal{L}= \sum_{i=1}^n\left\|r^i\right\|_2$
- Absolute error, $\mathcal{L}1$ loss: $\mathcal{L}= \sum_{i=1}^n\left|r^i\right|$

---

<style scoped>
section {
  font-size: 30px;
}
</style>

### Locating Earthquake using relative arrival times

[notebook](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/codes/earthquake_location/#locating-earthquakes-using-both-absolute-travel-times-and-relative-travel-time-differences)

$$
\Delta r_k^{i j}=\left(t_k^i-t_k^j\right)-\left(\hat{t}_k^i-\hat{t}_k^j\right)
$$

Loss function:
$$
\mathcal{L}=\sum_{k=1}^m\left\|\Delta r_k^{i j}\right\|_2
$$

---
<style scoped>
section {
  font-size: 30px;
}
</style>

### [HypoDD: Double-difference earthquake location](https://www.ldeo.columbia.edu/~felixw/hypoDD.html)

![height:550px](./assets/Screenshot%202023-10-08%20at%2021.57.01.png)

---
<style scoped>
section {
  font-size: 30px;
}
</style>

### [HypoDD: Double-difference earthquake location](https://www.ldeo.columbia.edu/~felixw/hypoDD.html)

![width:1200px](./assets/Screenshot%202023-10-08%20at%2021.58.46.png)

---

<style scoped>
section {
  font-size: 30px;
}
</style>

### [GrowClust: A Hierarchical Clustering Algorithm for Relative Earthquake Relocation](https://github.com/dttrugman/GrowClust)

![height:400px](./assets/Screenshot%202023-10-08%20at%2022.01.21.png)

Review: [clusering](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/05_phase_association.html#6)

---

### Uncertainty

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

![width:1100px](./assets/Screenshot%202023-10-08%20at%2021.43.21.png)

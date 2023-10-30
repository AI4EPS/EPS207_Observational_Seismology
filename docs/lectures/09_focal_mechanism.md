---
marp: true
paginate: true
theme: gaia
backgroundColor: #fff
---

# Focal Mechanism Inversion

Notebooks: [codes/focal_mechanism.ipynb](codes/focal_mechanism/)

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

![bg w:800](https://gsw.silverchair-cdn.com/gsw/Content_public/Journal/bssa/92/6/10.1785_0120010200/3/2270_01.jpeg?Expires=1701569826&Signature=m1vSnJ8ZiRXGIo78DsY0JEkKJ-dSi6hHsMj2VtVrvdqvNQy9xS9NOZDSWUyLMx4Bs0ray9g45EARu6F1qP9B8cmNgNQF5qj30O7H4DHKxHuFohN9YxQt0A7rRqKSVbUmZcUu3D5CdPCN8dxVVt2Bl2SyYC5Le8oFakbS4tcOw-NSFbztboZUitlMn4BuOGaznJARW3QuKZixGmNwdVgBxMPFG7qzZilkx3RNpRJVX6qcoU23tcR~sD61gKq4v1gyA6EcCu19qHQ~8Y-PbIjyuXoKA9x1WrfxoP0s6sgMsTjcETGDvU1uQKxkxrKZJNqdZtX5VuN1wL85Z-Hl6zPunw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

<!-- footer: "Hardebeck and Shearer (2002)" -->

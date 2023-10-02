---
marp: true
paginate: true
theme: gaia
backgroundColor: #fff
---

# Phase Association

Notebooks: [codes/quakeflow_demo.ipynb](codes/quakeflow_demo/)

---

### What is phase association?

![width:1200px](./assets/phase_picks.png)


---

### Grid-search / Back-projection, e.g. REAL

![height:500px](https://gsw.silverchair-cdn.com/gsw/Content_public/Journal/srl/90/6/10.1785_0220190052/3/0220190052fig1.png?Expires=1698959766&Signature=ECYWMqcr2KHcIXVlU9xCa1UswOW5T7KejW7cjEkOf~JC3u7zTwtDOs2oaFS4~SUM1CIM0znZM2EOpf5bCeU7TxrccHc~owL7t9qWx9zuiD-aFYi09Qu6qJw56~pm373HMEYU8pTo7pcUKUjENY1UCanOdQE3j1nhfkodVF4-gwv3tylx3zqtzzE26UIBYaSqgL-CX5JCCjJzqn5anYK1QmOlwkppd26d9GoD1UsUsZQFfP00YaiheY9Fky-6qsQkolbKc6qCF959-Sl26Bo2I7Mnfk0Q5xHoP-TA0K0rUW3mr0IrmKG85b7ppDNCbLcEjWjFW3zOmcclHsa39ivTAg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)


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


![width:550px](https://gsw.silverchair-cdn.com/gsw/Content_public/Journal/bssa/113/2/10.1785_0120220182/2/0120220182fig4.png?Expires=1698766782&Signature=K6l53qkcwRnQEC4SQZ7PmcJ5A3w6JOq73ftas5~WKnkGOShh7s5BpX6WBjaT5G3LJDbYU2KFY1ftlpiwJmo3LyYAKnwZfXvyP6sb0T8JWC8nDMhbA77MUcdnwFx4OAKRzv3RjwDnrTDJ51UUyvOsjK-gljsYRwi59sVnUyxSgVXkM67iRxEOXAhJv6sXKW7H~m-64hNNBs0qSHeGnc2fYftycgEQMOuF~O-IyfK0aCT5f7E4wHBm4ITkxA-0uyIQVrm3ejSVgH9zB5mNQbCQxuO2Rd-vX4Q~osnyMQNDU4CHkGwUhx7ROyLCbLTY3eRZI0qSMkfwGZiRTY0Hz3bJYw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

![width:600px](https://gsw.silverchair-cdn.com/gsw/Content_public/Journal/bssa/113/2/10.1785_0120220182/2/0120220182fig6.png?Expires=1698766782&Signature=I6~YeqbxMahTgIlFJ1NMtKpMJ8syNg3dhABdIgpbOb--xVGQsb91vMCevmxeHvCHFi96CMV-HWZN2mjpErdTKEEcqo3JM70kae18KC9TTGzunKhW1TwrJ6NqdbqA8B6A5JtYKNhfVnfn1chcK5pFU-v8Wcn25H8YrXrpvT3-OENTP49V2ZgfNR6tjQfZXpDfGQD8ONwGTGtWbUAQJuDqxQsUCsaMovpjF~nKxlpmKiOQttpT34H-infKVrLS-H4SYoCpEuhuAQk2BrZyUfvGhoUkAPbHXuFHtW69Mvu2fvk6SXhVEmyYNe43qX0SPaWDvOC3M-BIjiPu8YXHsrCAVg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)


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

![](https://www.mlpack.org/gsocblog/images/5_clusters_QGMM.gif)

![height:450px](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/diagram_gamma_annotated.png)

---

### Gaussian Mixture Model Association (GaMMA)


![width:1200px](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/2019-07-04T18-02-01.074.png)

---

**How to apply other clustering methods to phase association?**
**How to solve phase association at the global scale?**
---
marp: true
paginate: true
theme: gaia
backgroundColor: #fff
---

# Phase Picking

Notebooks: [codes/phase_picking.ipynb](codes/phase_picking/)

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

![height:600px](https://ds.iris.edu/media/news/2014/09/16/global-stacks-poster-contest/Old_IRIS_Poster.png)
<!-- ![height:500px](http://ds.iris.edu/media/product/globalstacks/images/Composite360deg_180min_midsize_1.png) -->

---

![](https://sites.northwestern.edu/sethstein/files/2017/06/Travel-Time-Curve-1q4cplr.gif)

<!-- ![height:550px](http://ds.iris.edu/media/product/globalstacks/images/GlobalStack.BHZ.8.sec.180deg.TTcurves.screenshot.png) -->

---

![width:900px](./assets/lecture3-4_2013.jpg)
<!-- https://sites.ualberta.ca/~ygu/courses/geoph624/notes/lecture3-4_2013.pdf -->

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

![width:1000px](https://miro.medium.com/v2/resize:fit:1000/1*RZnBSB3QpkIwFUTRFaWDYg.gif)


---

![width:1000px](https://miro.medium.com/v2/resize:fit:1000/1*NLnnf_M4Nlm4p1GAWrWUCQ.gif)

---

![width:1000px](https://raw.githubusercontent.com/matterport/Mask_RCNN/master/assets/4k_video.gif)


---

![width:1100px](https://github.com/bowenc0221/panoptic-deeplab/raw/master/docs/panoptic_deeplab.png)

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

![width:1000px](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/gji/216/1/10.1093_gji_ggy423/1/ggy423fig5.jpeg?Expires=1699221234&Signature=3kvXLTJk4qw7-jZeQwHgEzX-ziSjbGsM4zDM2OShQz1r84gzMyCBC~ewAjWFiyimpar05ClEAQqByFvEV0WPbBc89~EpJWDyTMzDYs0BqG9V4lF6SPLD5AFPxn5giRty9xmNmZ95c-1MTVXTcPv6ItzvNGqVxJsXqzg~urgjtbsV~K2pklWT3f8Q1VPsvmE~o0e3UH~z2A4i~lknJxuBd9Gogav1bnRZr1fd74dqqYBCZkOoKmnCLJEuKbljUvWd8JOS666L5o5brYp0yV0jjNdDsjeSxiZcqD3xrCmaEkl4o1K~QQfwKR0JKUMGHYlThmalA8AlS1BrKpTWX04f5Q__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)


---

### EQTransformer for simultaneous earthquake detection and phase picking

![height:500px](./assets/eqtransformer.jpg)

---

### Next-Generation Seismic Monitoring with Neural Operators (PhaseNO)

![height:450px](./assets/phaseno.png)

---

### Large training dataset + Clear objective function

![width:500px](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/gji/216/1/10.1093_gji_ggy423/1/ggy423fig1.jpeg?Expires=1699222393&Signature=S9~mjTR4ss8rdk51URN-hudRbXqwHmpYDeUMUIq1~fhGgprl5J15gJ4KzWL-9I2phJCphL0EFBVwehp7P-wbagz3vgDm1isTCt-a0fdIfkjRCXvdIwi-vzbp27JrHNLf3W9ES8H8yi2PoGbotUYD8bU6SvlX5m6~-SqlvPE5ON63lqqg7bJZ7CoTNErmGx5y4LqNhEd7Pp-zs4F3QA5Zp70eUza15p26mEZk-BSXsfb7JxZy3NLJsaVlTOTH1oBJyo~DqHXgzyE~749Tngw6khmMaT-htsYKwufj9IqHFS8KuSVzzWZbAWELx463Oif8m0U7ooBG9micMnhnI8ygyA__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

---

**What is the next model for detecting earthquake / picking seismic phases?**
**What other seismic tasks can be solved with deep learning?**
**What are other potential applications in Earth and Planetary Science?**
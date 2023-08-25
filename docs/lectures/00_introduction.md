---
marp: true
paginate: true
theme: gaia
backgroundColor: #fff
size: 16:12
---

# Observational Seismology
# (Machine Learning Seismology)

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
[ncedc.org/waveformDisplay/](https://ncedc.org/waveformDisplay/)  [Station Channel Codes](https://docs.fdsn.org/projects/source-identifiers/en/v1.0/channel-codes.html)

![height:500px](https://ncedc.org/gifs/annotatedQuakes.jpg)

---

- Can you find an earthquake?

[Raspberry Shake Network](https://stationview.raspberryshake.org/#/?lat=9.22307&lon=-0.57266&zoom=2.444)
![height:500px](https://cdn-dfdfc.nitrocdn.com/ZlrgOCOQhsvSWxEiiDoICIrRKemFKvaV/assets/images/optimized/rev-0d3aaf8/raspberryshake.org/wp-content/uploads/StationView_General-min.gif)

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
![height:500px](http://earthquake.ca.gov/wp-content/uploads/sites/8/2020/09/android_alerts.gif)

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
![height:300px](https://www.jreast.co.jp/e/development/theme/safety/img/safety07.jpg)

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
| 09/15 | <span style="color:Violet;">Seismic Data Processing</span>  |
| 09/22 | <span style="color:Violet;">Earthquake Detection</span> |
| 09/29 | <span style="color:Violet;">Phase Picking & Association</span> |
| 10/06 | <span style="color:LimeGreen;">Earthquake Location & Relative Location |
| 10/13 | <span style="color:LimeGreen;">Focal Mechanism & Moment Tensor |
| 10/20 | <span style="color:Violet;">Earthquake Statistics</span> |
| 10/27 | <span style="color:black;">Ambient Noise</span> |
| 11/03 | <span style="color:LimeGreen;">Seismic Tomography</span> |
| 11/10 | <span style="color:LimeGreen;">Full-waveform Inversion</span> |

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
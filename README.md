# Change-Point Detection for Emotion Recognition  

This project investigates the application of **unsupervised change-point detection algorithms** to physiological data for **emotion recognition** in Human-Robot Interaction (HRI). We compare two recent state-of-the-art methods on real-world multimodal signals collected during an HRI sales simulation study.  

---

## Dataset  

We use the **AFFECT-HRI dataset**, which contains recordings from ~150 participants engaged in a simulated sales interaction with a robot. Participants wore a wristband that captured various **physiological signals** (e.g., heart rate, electrodermal activity).  

📂 Dataset link: [AFFECT-HRI on Zenodo](https://zenodo.org/records/10422259)  

---

## Algorithms  

We benchmark two cutting-edge unsupervised methods for change-point detection:  

1. **Change-Interval Detection via Isolation Distributional Kernel (CID-IDK, 2024)**  
   - Paper: [JAIR 2024](https://dl.acm.org/doi/10.1613/jair.1.15762)  
   - Interval-based kernel method for robust detection of distributional changes in time series.  

2. **ChangeForest: Random Forests for Change-Point Detection (2022)**  
   - Repository: [mlondschien/changeforest](https://github.com/mlondschien/changeforest)  
   - Leverages random forests to identify structural changes in complex time series.  

---

## Goal  

- Detect **emotion-related change-points** in physiological time series.  
- Compare performance across algorithms in the HRI scenario.  
- Explore potential for **emotion recognition** and adaptation in real-time human-robot interaction.  

---

## References  

- AFFECT-HRI dataset: [Zenodo](https://zenodo.org/records/10422259)  
- CID-IDK (2024): [JAIR Paper](https://dl.acm.org/doi/10.1613/jair.1.15762)  
- ChangeForest (2022): [GitHub](https://github.com/mlondschien/changeforest)  

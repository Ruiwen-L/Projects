# Arctic Cloud Classification with Spatial Feature Engineering and Transfer Learning

This project develops a robust cloud classification system for Arctic satellite imagery by combining expert-engineered radiance features, spatial patch statistics, and deep autoencoder-based representations.

The objective is to build a model that generalizes across spatial regions while remaining robust to limited labeled data and sensor noise.

---

## Project Motivation

Accurate Arctic cloud detection is critical for climate modeling. However, distinguishing clouds from snow- and ice-covered surfaces is challenging due to similar spectral signatures.

This project investigates:

- How multi-angle radiance features behave under cloud vs. non-cloud conditions  
- Whether spatial patch-based features improve classification  
- How transfer learning can enhance performance under limited labels  
- Which models generalize best across spatially disjoint regions  

---

## Data

Satellite imagery from NASA's MISR instrument.

Three expert-labeled Arctic images:
- Cloud pixels (label = 1)  
- Non-cloud pixels (label = -1)  
- Unlabeled pixels (excluded from training)  

Final feature space: **93 features per pixel**

1. **Original expert features (11)**  
   - NDAI, SD, CORR  
   - Multi-angle radiance channels (DF, CF, BF, AF, AN)

2. **Spatial patch features (32)**  
   - Mean, standard deviation, min, max over 9×9 local neighborhoods  
   - Captures local texture and variability  

3. **Autoencoder embeddings (50)**  
   - Learned latent representations from unsupervised training  
   - Encodes nonlinear spatial-spectral structure  

---

## Exploratory Data Analysis

Key observations:

- Radiance angles are highly correlated under no-cloud conditions (>0.94)  
- Cloud presence disrupts angular correlation structure  
- NDAI and SD provide strongest separation  
- No single feature is sufficient → multi-feature modeling required  

---

## Feature Engineering

### Spatial Context Modeling

- Extracted 9×9 patches around each labeled pixel  
- Computed summary statistics across 8 spectral channels  
- Generated 32 spatially-aware features  

Spatial statistics significantly improved classification accuracy.

---

### Transfer Learning via Autoencoder

- Fully connected autoencoder trained on image patches  
- 9×9×8 inputs compressed to 50-dimensional latent vectors  
- Optimized hyperparameters:
  - Learning rate: 0.0001  
  - Batch size: 1028  
  - Embedding size: 50  
- Minimum validation MSE: **0.0409**

Latent embeddings were incorporated into downstream classification.

---

## Data Splitting Strategy

To prevent spatial leakage:

### Image-Level Split
- Train on 2 images  
- Test on 1 unseen image  
- Rotated across all 3 combinations  

### Quadrant-Based Split
- Training images divided into spatial quadrants  
- GroupKFold used to preserve spatial disjointness  

This mimics real-world deployment on unseen geographic regions.

---

## Predictive Modeling

Three classifiers were developed and tuned:

### Random Forest (Best Model)
- 200 trees  
- Max depth = 10  
- ROC-AUC: **0.94**  
- Test accuracy: **83%**  
- Cross-image mean accuracy: **0.87**

Patch-based spatial features dominated feature importance rankings.

---

### LightGBM
- ROC-AUC: 0.93  
- Accuracy: 82%

---

### K-Nearest Neighbors
- ROC-AUC: 0.84  
- Accuracy: 82%

Ensemble tree methods performed best in high-dimensional spatial data.

---

## Model Diagnostics

- Strong confidence calibration (most correct predictions > 0.9 probability)  
- Slightly higher false negatives (thin clouds harder to detect)  
- Spatial accuracy variation across quadrants  
- ROC-AUC of 0.94 confirms strong separability  

---

## Robustness & Stability Analysis

Simulated sensor perturbations:

- Additive Gaussian noise (σ = 0.1)  
- Multiplicative noise (σ = 0.05)  

Results:
- Minimal prediction shifts  
- Stable classification boundaries  
- Spatially coherent predictions on unlabeled images  

Demonstrates strong generalization under small input perturbations.

---

## Key Results

- ROC-AUC: **0.94**
- Cross-image mean accuracy: **0.87**
- Spatial feature engineering outperformed latent embeddings
- Model generalized across diverse Arctic cloud morphologies
- Robust to simulated sensor noise

---

## Tools & Technologies

- Python  
- Scikit-learn  
- LightGBM  
- PyTorch  
- Optuna  
- GroupKFold Cross-Validation  
- t-SNE Visualization  
- Matplotlib & Seaborn  

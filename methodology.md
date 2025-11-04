#  Methodology: Intrusion Detection using Random Oversampling (RO) Stacked Feature Engineering (SFE) and Feature Extraction.

This document details the **methodological framework** and **data processing pipeline** used in reproducing the research paper on **Machine learning‑based network intrusion
detection for big and imbalanced data using
oversampling, stacking feature embedding
and feature extraction**.

---

## 1. Data Preprocessing

The first stage ensures dataset quality and consistency:

- **Handling Missing Values**: Rows containing `NaN`, `inf`, or `-inf` are removed.
- **Column Normalization**: Spaces and irregular characters in column names are removed.
- **Duplicate Removal**: Eliminates repeated entries to reduce redundancy.
- **Class Merging**: Rare classes are merged into broader categories to ensure balanced learning.
- **Type Optimization**: Data types are downcasted (`int64 → int32`, `float64 → float32`) to reduce memory footprint.

---

## 2. Feature Scaling

To ensure equal weightage across features:

- **Z-score Standardization**: Each feature is centered around zero mean (μ = 0) and scaled to unit variance (σ = 1).
- **Label Encoding**: Categorical target labels are encoded into numerical form for ML compatibility.

---

## 3. Feature Resampling

The **Random Oversampling (RO)** method is used to counter class imbalance by duplicating minority-class samples.  
This enhances the classifier’s ability to detect rare attack patterns.

---

## 4. Stacked Feature Embedded (SFE)

SFE introduces unsupervised learning features as meta-layers:

- **K-Means Clustering** groups samples by similarity.
- **Gaussian Mixture Models (GMM)** identify probabilistic data clusters.
- The **cluster labels and probabilities** are embedded as new features, enriching the dataset and aiding pattern discovery.

---

## 5. Feature Extraction

Dimensionality reduction is performed using **Principal Component Analysis (PCA)** to mitigate redundancy.  
PCA transforms correlated variables into uncorrelated components, retaining 95–99% variance.

---

## 6. Data Splitting & Cross-Validation

- The data is divided into **training (90%)** and **testing (10%)** subsets.
- **10-Fold Cross-Validation** is applied for model reliability and variance reduction.

---

## 7. Model Training

Four supervised ML models are trained:

| Algorithm | Description |
|------------|--------------|
| Decision Tree (DT) | Basic rule-based classifier |
| Random Forest (RF) | Ensemble of decision trees |
| Extra Trees (ET) | Randomized ensemble improving variance reduction |
| XGBoost (XGB) | Gradient boosting for high accuracy and speed |

---

## 8. Evaluation Metrics

Models are compared using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC Curve / AUC**
- **Confusion Matrix**

---

## 9. Experimental Setup

- **Datasets**: IoT-23, UNSW-NB15, and CIC-IDS2017.  
- **Environment**: Python 3.10+, scikit-learn, XGBoost, imbalanced-learn.  
- **Validation**: Repeated experiments with random seed control (`random_state=42`).

---

## 10. Summary

This methodology combines data quality improvement, feature enrichment, and dimensionality reduction to enhance intrusion detection accuracy.  
By stacking unsupervised cluster outputs as meta-features, SFE bridges the gap between feature engineering and representation learning.

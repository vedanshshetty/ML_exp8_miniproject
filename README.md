# ML_exp8_miniproject
This repo contains all files of **ml lab exp 8** which are necessary to understand , and run an independent recreation of the techniques described in the **research paper: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-00886-w**

On a high level: It implements a complete pipeline integrating **unsupervised clustering**, **feature augmentation**, and **supervised classification** for both binary and multiclass intrusion detection tasks.
---

## 📘 Features

* **Automated preprocessing**: Label encoding, scaling, downcasting, and NaN handling.
* **Stacked Feature Engineering (SFE)**: Combines K-Means and Gaussian Mixture clustering outputs as meta-features.
* **Class balancing**: Uses `RandomOverSampler` from `imblearn` to mitigate data imbalance.
* **Model ensemble**: Evaluates multiple classifiers:

  * Decision Tree
  * Random Forest
  * Extra Trees
  * XGBoost
* **Dimensionality reduction**: PCA applied on augmented features.
* **Evaluation**:

  * Cross-validation with stratified folds
  * ROC, Precision, Recall, F1-score, Confusion Matrix visualization
  * Final test evaluation with ROC curve plots

---

##🖇️ Pipeline Overview

The intrusion detection process follows **eight key stages**, from raw data cleaning to final evaluation:

### **Step 1: Data Preprocessing**

This initial phase focuses on improving data quality and consistency:

* Handling missing values by removing rows containing `null`, `-inf`, or `inf` values.
* Removing spaces from column names.
* Dropping duplicate rows.
* Merging similar classes with low instance counts in the output column.
* Reducing dataset size by converting data types (e.g., `int64 → int32`, `float64 → float32`) for efficiency.

---

### **Step 2: Feature Scaling**

Normalization ensures all input features share a comparable scale:

* **Standardization (Z-score normalization)** is applied so that each feature has a mean (μ) of 0 and a standard deviation (σ) of 1.
* **Label Encoding** transforms categorical outputs into numerical form for model compatibility.

---

### **Step 3: Feature Resampling**

To overcome **class imbalance**, **Random Oversampling (RO)** is used:

* RO replicates samples from minority classes to balance the dataset.
* This prevents model bias toward majority classes and ensures fair learning.

---

### **Step 4: Stacked Feature Embedded (SFE)**

This novel stage enriches the feature space with unsupervised meta-features:

* **Cluster Formation**: Employs **K-Means** and **Gaussian Mixture Models (GMM)** to group samples based on intrinsic similarity.
* **Feature Embedding**: Cluster labels and probabilities are embedded as new meta-features.
* This **augmented feature representation** improves anomaly detection and pattern recognition.

---

### **Step 5: Feature Extraction**

To reduce redundancy and enhance model efficiency:

* **Principal Component Analysis (PCA)** converts correlated variables into a set of uncorrelated components.
* Dimensionality is reduced (e.g., to 10 features for UNSW-NB15 and CIC-IDS2017 datasets) while retaining essential variance and discriminative power.

---

### **Step 6: Data Splitting & Cross-Validation**

* The dataset is split into **training and testing subsets**.
* **10-Fold Cross-Validation (k=10)** is used: 90% training and 10% testing per fold.
* This approach ensures model stability and robust generalization.

---

### **Steps 7 & 8: Model Training and Evaluation**

Four **supervised machine learning classifiers** are trained and assessed for both binary and multilabel intrusion detection:

* **Decision Tree (DT)**
* **Random Forest (RF)**
* **Extra Trees (ET)**
* **Extreme Gradient Boosting (XGB)**

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve and AUC

---

## 📁 Repository Structure

```
├── dataset_merge.py             # Script to merge multiple .labeled Zeek log files
├── intrusion_detection_pipeline.py  # Main ML pipeline implementation
├── iot23_cleaned.csv            # (Optional) Example dataset
├── iot23_train.csv / iot23_test.csv # Auto-generated splits
├── README.md                    # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/intrusion-detection-sfe.git
cd intrusion-detection-sfe
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imblearn
```

---

## 🧩 Data Preparation


###  Use IoT-23 or Similar Dataset

Place your dataset (e.g. `iot23_cleaned.csv`) in the project directory. (Our example required merging hence the merge script)

---

## 🚀 Running the Intrusion Detection Pipeline

```bash
python intrusion_detection_pipeline.py
```

---

## 📊 Example Results (IoT-23 Dataset)

| Model         | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------- | -------- | --------- | ------ | -------- | ------- |
| Decision Tree | 0.94     | 0.93      | 0.94   | 0.93     | 0.96    |
| Random Forest | 0.98     | 0.98      | 0.98   | 0.98     | 0.99    |
| Extra Trees   | 0.97     | 0.97      | 0.97   | 0.97     | 0.98    |
| XGBoost       | 0.99     | 0.99      | 0.99   | 0.99     | 0.99    |






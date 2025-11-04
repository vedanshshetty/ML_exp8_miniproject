# 🧪 Results

This document presents the experimental outcomes of the reproduced research work on two datasets — **UNSW-NB15 (original dataset)** and **IoT-23 (extended dataset)**.  
It includes confusion matrices, ROC curves, multiclass visualizations, and terminal outputs illustrating model performance.

---

## 📘 Dataset 1: UNSW-NB15 (Original)

### **1. Binary Classification Results**

#### 🔹 Confusion Matrices
| Model | Confusion Matrix |
|--------|------------------|
| Decision Tree | ![DT Confusion Matrix](images/unswb_results/unswb_confusion_dt.jpeg) |
| Random Forest | ![RF Confusion Matrix](images/unswb_results/unswb_confusion_rf.jpeg) |
| Extra Trees | ![ET Confusion Matrix](images/unswb_results/unswb_confusion_et.jpeg) |
| XGBoost | ![XGB Confusion Matrix](images/unswb_results/unswb_confusion_xgb.jpeg) |

#### 🔹 ROC Curves
| Model | ROC Curve |
|--------|------------|
| Decision Tree | ![DT ROC Curve](images/unswb_results/unswb_roc_dt.jpeg) |
| Random Forest | ![RF ROC Curve](images/unswb_results/unswb_roc_rf.jpeg) |
| Extra Trees | ![ET ROC Curve](images/unswb_results/unswb_roc_et.jpeg) |
| XGBoost | ![XGB ROC Curve](images/unswb_results/xgb%20roc.jpeg) |

#### 🔹 Terminal Outputs
- ![Terminal Output 1](images/unswb_results/unswb_terminal_1.jpeg)  
- ![Terminal Output 2](images/unswb_results/unswb_terminal_2.jpeg)

---

### **2. Multiclass Classification Results**

#### 🔹 Confusion Matrices
| Model | Confusion Matrix |
|--------|------------------|
| Decision Tree | ![DT Multi Confusion](images/unswb_results/unswb_multi_confusion_dt.jpeg) |
| Extra Trees | ![ET Multi Confusion](images/unswb_results/unswb_multi_confusion_et.jpeg) |
| XGBoost | ![XGB Multi Confusion](images/unswb_results/unswb_multi_confusion_xgb.jpeg) |

---

## 🌐 Dataset 2: IoT-23 (Extended)

### **1. Binary Classification Results**

#### 🔹 Confusion Matrices
| Model | Confusion Matrix |
|--------|------------------|
| Decision Tree | ![DT Confusion Matrix](images/iot23_results/iot23_confusion_dt.png) |
| Random Forest | ![RF Confusion Matrix](images/iot23_results/iot23_confusion_rf.png) |
| Extra Trees | ![ET Confusion Matrix](images/iot23_results/iot23_confusion_et.png) |
| XGBoost | ![XGB Confusion Matrix](images/iot23_results/iot23_confusion_xgb.png) |

#### 🔹 ROC Curves
| Model | ROC Curve |
|--------|------------|
| Decision Tree | ![DT ROC Curve](images/iot23_results/iot23_roc_dt.png) |
| Random Forest | ![RF ROC Curve](images/iot23_results/iot23_roc_rf.png) |
| Extra Trees | ![ET ROC Curve](images/iot23_results/iot23_roc_et.png) |
| XGBoost | ![XGB ROC Curve](images/iot23_results/iot23_roc_xgb.png) |

#### 🔹 Terminal Output
- ![IoT-23 Terminal Output](images/iot23_results/iot23_terminal_1.jpeg)

---

## 📊 Summary of Findings

- **Random Forest** and **XGBoost** consistently delivered the best results in both binary and multiclass experiments.  
- **Stacking Feature Embedding (SFE)** significantly enhanced detection accuracy by introducing meta-features derived from clustering.  
- The **IoT-23 dataset** demonstrated the generalizability of the proposed pipeline for intrusion detection in IoT environments.  
- The visual results validate the robustness and scalability of the reproduced research workflow.

---

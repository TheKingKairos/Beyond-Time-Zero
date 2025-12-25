# Model Training & Evaluation

This folder contains all machine-learning models, grid-search tools, and survival-analysis algorithms used across the project. It includes implementations for logistic regression, random forests, XGBoost, Cox proportional hazards models, neural networks (MLP + sequence models), and full grid-search pipelines for each.



## Core Model Definitions

### **`dnn.py`**
Implements a fully-configurable feed-forward PyTorch MLP used for:
- binary classification
- logistic hazard modeling
- survival predictions via discretized hazards

**Key features**
- arbitrary depth + width  
- dropout, batchnorm, activation control  
- logits or probability outputs  
- collate utilities for PyTorch DataLoader  

---

### **`cox_models.py`**
Provides Cox Proportional Hazards model wrappers for:
- static Cox PH  
- time-varying Cox PH (TVC)  
- landmark Cox models  

**Includes**
- preprocessing helpers  
- ML-compatible `.pkl` load/save wrappers  
- support for your stored models in `cox_models/`  

---

### **`datasplitter.py`**
Utility for generating:
- random splits  
- group-aware splits (stay-level)  
- stratified group splits  

Used by many grid search scripts to ensure leakage-free CV.

---

## Grid Search Scripts — Deep Learning

### **`dnn_grid_search.py`**
Runs hyperparameter sweeps for MLP classification models.

**Searches over**
- learning rate  
- hidden layer sizes  
- dropout patterns  
- L2 regularization  
- batch size  

Outputs:
- best config JSON  
- training/val metrics  
- saved PyTorch model checkpoint  

---

### **`dnn_grid_search_group.py`**
Same as above, but with **grouped train/val splits**, ensuring no stay-level leakage.  
Ideal for clinical/longitudinal modeling.

---

### **`dnn_survival_analysis_grid.py`**
Grid search wrapper for **neural discrete-time survival models**.  
Uses:
- DNN backbone  
- logistic hazard objective  
- person-period data (`discrete_time_30min_train.csv`)  

Outputs C-index, AP, AUROC, and survival-curve compatible hazard predictions.

---

## Tree-Based Models

### **`random_forest_mimic.ipynb`**
Random forest classification experiments on the MIMIC-ED event-level dataset.

### **`random_forest_physionet.ipynb`**
Parallel notebook for the PhysioNet dataset.  
Useful for benchmarking generalization across datasets.

---

## Logistic Regression

### **`logistic_regression.py`**
Runs baseline LR models using:
- regularization sweeps  
- AUROC / AUPRC scoring  
- grouped CV option  

Acts as a sanity-check baseline for nonlinear models.

---

## XGBoost Models

### **`new_xgboost_sa.py`**
Standalone XGBoost survival model (Cox / logistic hazard).

### **`survival_analysis_xgboost.py`**
End-to-end XGBoost survival modeling pipeline:
- loads Cox/landmark/discrete data  
- trains survival booster  
- evaluates via C-index  

### **`survival_analysis_xgboost_c_index.py`**
C-index–focused evaluation version of the above.

---

### **`xgb_survival_analysis.py`**
Traditional gradient-boosted hazard modeling script for discretized person-period tables.

---

### **`xgb_survival_analysis_grid.py`**
Grid search for XGBoost survival hazards using:
- GPU acceleration  
- monotonic CV splits  
- AP/AUROC scoring  

---

### **`xgboost_grid_search_group.py`**
Full GPU-accelerated grid search engine for XGBoost with:

**Features**
- group-aware CV  
- hyperparameter combination sweeps  
- early stopping  
- atomic JSON and model checkpoint saving  
- Top-5 trial ranking  

This is your most robust and generalizable XGBoost grid-search tool.

---

## Sequence Models

### **`sequencenn.py`**
Defines a sequence-based neural network (LSTM/GRU/Transformer-like) for:
- modeling patient trajectories  
- dynamic prediction  
- multi-step survival forecasting  

### **`sequencenn_grid_search.py`**
Search wrapper to optimize:
- recurrent layer type  
- hidden size  
- sequence length  
- dropout  
- learning rates  

---

## Survival Model Variants

### **`xgb_survival_analysis_grid.py`**
Grid search for discrete-time survival using XGBoost hazard models.

### **`xgboost_grid_search_group.py`**
Group-aware version (no leakage).

### **`dnn_survival_analysis_grid.py`**
Neural hazard model grid search.

---

## Precomputed Models, Scalers, and Grid-Search Outputs

You **do not** need to rerun feature engineering, scaling, or the full grid-search pipelines to reproduce results.  
All necessary artifacts are already included in the repository:

### Precomputed Results, Features, and Scaling
1. cox_models/ - Plk files for Cox models
2. grid_runs/ - Neural networks (MLP, sequence models)
3. grid_runs_xgb/ - XGBoost classification grid search
4. outputs_cox_with_changes/ - Cox survival models after feature adjustments
5. outputs_discrete_mlp/ - DNN models and results
6. outputs_discrete_xgb/ - XGBoost models and results
7. outputs_discrete_xgb_grid/ - XGB discrete-time survival grid search results


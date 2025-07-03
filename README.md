# Diabetes Risk Prediction

**Overview:**  
An end‑to‑end machine learning pipeline to predict diabetes onset using the Pima Indians Diabetes dataset.  
Demonstrates data cleaning, exploratory analysis, preprocessing, model training & tuning, uncertainty analysis, and final model serialization.

**Key Results:**  
- Final test accuracy: **78%**  
- Final ROC‑AUC: **0.87**  
- Model stability: ROC‑AUC = 0.87 ± 0.01 over 30 random splits  

**How It Works:**  
1. **Load & Clean Data**  
   - Replace invalid zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI with column medians.  
2. **Preprocess**  
   - Standard-scale all features (mean=0, σ=1).  
   - Stratified 80/20 train/test split.  
3. **Modeling**  
   - Pipeline: `SimpleImputer` → `StandardScaler` → `RandomForestClassifier`.  
   - Hyperparameter tuning via `GridSearchCV` (n_estimators, max_depth, min_samples_split).  
4. **Evaluation & Uncertainty**  
   - Measured Accuracy, ROC‑AUC, classification report on hold‑out test set.  
   - Assessed variability by repeating train/test split 30× and plotting AUC distribution.  
5. **Serialization**  
   - Saved final pipeline with `joblib` for future inference.

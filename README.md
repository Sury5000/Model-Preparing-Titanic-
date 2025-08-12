# Titanic Survival Prediction — Logistic Regression (Base Model)

## 📌 Overview
This project uses the **Titanic dataset** (from Seaborn) to build a **Logistic Regression model** that predicts whether a passenger survived or not.  
The workflow includes:
- Data cleaning and preprocessing
- Feature engineering
- Encoding categorical variables
- Feature scaling and transformation
- Building and training a classification model
- Evaluating model performance  

This script is designed as a **baseline model** — no advanced tuning or feature engineering has been done yet. It will serve as a reference point for future improvements.

---

## Steps in the Code

### **1. Load Dataset**
We load the Titanic dataset directly from Seaborn.



### **2. Data Preprocessing**
- **Missing values**:
  - `age` → filled with median age
  - `embarked` and `embark_town` → filled with most frequent (mode) value  
- **Dropped** `deck` column (too many missing values)
- **Removed duplicates**
- **Created** a new feature:  
  `family_size = sibsp + parch + 1`

### **3. Encoding Categorical Features**
- Used **One-Hot Encoding** with `pd.get_dummies`:
  - Encoded `sex` and `embarked` (drop first category to avoid dummy trap)
  - Encoded `class` similarly

### **4. Feature Scaling & Transformation**
- **StandardScaler**: Standardized `age` and `family_size`
- **PowerTransformer (Yeo-Johnson)**: Transformed `fare` to reduce skewness

### **5. Feature Selection**
- Dropped target variable and unrelated/multicollinear columns:



### **6. Train-Test Split**
- Split data into:
- **80%** training data
- **20%** testing data
- **Stratify** by target variable to maintain class balance

### **7. Model Training**
- Model: **Logistic Regression** (`max_iter=1000`)
- Fit model on training data

### **8. Predictions & Evaluation**
- Made predictions on the test set  
- Evaluation metrics:
- **Accuracy**: Overall percentage of correct predictions
- **Confusion Matrix**: Shows counts of True Positives, False Positives, True Negatives, False Negatives  
- **Classification Report**: Precision, Recall, and F1-score for each class

Example output:


---

## 📊 Interpretation of Results
- **Accuracy ~77%**: Model predicts survival correctly in about 3 out of 4 cases.
- Performs slightly better for **non-survivors** (class 0) than survivors (class 1).
- Main error is predicting survivors as non-survivors (false negatives).

---

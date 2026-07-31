## Loan Default Risk Analysis & Prediction

This project focuses on exploratory data analysis and machine learning techniques to predict loan defaults using borrower financial and demographic information. The objective is to identify high-risk applicants and build a predictive model that can assist financial institutions in making informed lending decisions.

### Files

* `01_Data_Understanding.ipynb` — Data loading and understanding.
* `02_EDA.ipynb` — Exploratory Data Analysis (EDA).
* `03_Data_Preprocessing.ipynb` — Data cleaning, preprocessing, feature engineering, and SMOTE.
* `04_Logistic_Regression.ipynb` — Logistic Regression model implementation.
* `05_KNN.ipynb` — K-Nearest Neighbors model implementation.
* `06_Decision_Tree.ipynb` — Decision Tree model implementation.
* `07_Random_Forest.ipynb` — Random Forest model implementation.
* `08_Model_Comparison.ipynb` — Performance comparison of all models.
* `app.py` — Streamlit web application for real-time loan default prediction.
* `Loan_default.csv` — Dataset used for analysis.
* `Loan_Default_Risk_Analysis_Report.pdf` — Detailed project report.

> **Note:** The project uses separate saved model files (pickle format) for deployment in the Streamlit application.

---

### Features

* **Exploratory Data Analysis (EDA)** using Pandas, Matplotlib, and Seaborn to understand data distributions, feature relationships, and class imbalance.
* **Data preprocessing** including missing value handling, feature scaling, and categorical variable encoding.
* **Class imbalance handling** using **SMOTE (Synthetic Minority Over-sampling Technique)**.
* **Machine Learning models** implemented and compared:
  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Decision Tree
  * Random Forest
* **Model evaluation** using:
  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * ROC-AUC Score
  * Confusion Matrix
* **Interactive Streamlit application** for predicting loan default risk.

---

### Results

Multiple machine learning models were trained and evaluated on the loan default dataset. After comparing their performance, **Logistic Regression** achieved the best overall balance between precision and recall, resulting in the highest F1-score. Therefore, it was selected as the final model for deployment in the Streamlit application.

---

### 📌 Author

**Anushree Ghosh**

Msc.Statistics, IIT Kanpur

GitHub: https://github.com/anushreeghosh07
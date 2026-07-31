# Loan Default Risk Analysis & Prediction

## Project Overview

This project develops a machine learning solution to predict whether a borrower is likely to default on a loan. The objective is to help financial institutions identify high-risk applicants, improve credit risk assessment, and support better lending decisions.

The project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), handling class imbalance using SMOTE, model building, performance comparison, and deployment through a Streamlit web application.

---

## Objectives

- Understand borrower characteristics affecting loan default.
- Perform exploratory data analysis and data preprocessing.
- Handle imbalanced data using SMOTE.
- Build and compare multiple classification models.
- Select the best-performing model based on evaluation metrics.
- Deploy the final model using Streamlit.

---

## Dataset

The dataset contains borrower demographic, financial, and loan-related information.

**Target Variable**

- **Default**
  - 1 = Loan Default
  - 0 = No Default

### Features include

- Age
- Income
- Loan Amount
- Credit Score
- Employment Type
- Years at Job
- Debt-to-Income Ratio
- Loan Term
- Interest Rate
- Number of Credit Lines
- Loan Purpose
- Education
- Marital Status
- Housing Status
- and other financial indicators.

---

## Project Workflow

1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Feature Engineering
5. Data Preprocessing
6. Handling Class Imbalance using SMOTE
7. Model Building
8. Model Evaluation
9. Model Comparison
10. Streamlit Application Development

---

## Models Implemented

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

---

## Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

After comparing all models, **Logistic Regression** achieved the best overall balance between precision and recall, resulting in the highest F1-score. Therefore, it was selected as the final prediction model for deployment.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Streamlit

---

## Repository Structure

```
Loan-Default-Risk-Analysis/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Data_Preprocessing.ipynb
│   ├── 04_Logistic_Regression.ipynb
│   ├── 05_KNN.ipynb
│   ├── 06_Decision_Tree.ipynb
│   ├── 07_Random_Forest.ipynb
│   └── 08_Model_Comparison.ipynb
│
├── models/
│
├── figures/
│
├── reports/
│
└── results/
```

---

## Streamlit Application

The Streamlit application allows users to enter borrower information and receive a prediction indicating whether the applicant is likely to default on the loan.

The application provides:

- User-friendly interface
- Real-time prediction
- Loan risk classification
- Probability-based output

---

## Results

Among all the implemented models, **Logistic Regression** demonstrated the best overall performance based on F1-score and generalization capability.

The project successfully demonstrates how machine learning can support financial institutions in identifying potential loan defaulters and improving credit risk management.

---

## Future Improvements

- Hyperparameter tuning
- XGBoost and LightGBM implementation
- Probability calibration
- Feature importance analysis
- Explainable AI using SHAP
- Cloud deployment

---

## Author

**Anushree Ghosh**

GitHub: https://github.com/anushreeghosh07
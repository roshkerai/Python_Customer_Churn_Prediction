# Customer Churn Prediction using Machine Learning

A machine learning project that predicts whether a telecom customer is likely to churn (leave the service), built in Python using Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshkerai/Python_Customer_Churn_Prediction/blob/main/Copy_of_Customer_Churn_Prediction_using_ML_for_GitHub.ipynb)

## Project Overview

Customer churn is costly for subscription businesses, as keeping an existing customer is usually cheaper than winning a new one. This project explores the Telco Customer Churn dataset, builds and compares three classification models, and saves the best one so it can be reused to make predictions on new customers.

## Dataset

- **Source:** IBM Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, in the `data/` folder)
- **Size:** 7,043 customers, 21 columns (20 after dropping `customerID`)
- **Target:** `Churn` (Yes / No). 5,174 customers stayed and 1,869 churned, so the classes are imbalanced
- **Features:** customer demographics (gender, senior citizen, partner, dependents), account details (tenure, contract type, payment method, paperless billing), services (phone, internet, online security, tech support, streaming) and charges (monthly and total)

## Workflow

1. **Data loading and understanding:** checked shape, data types, unique values and missing values
2. **Data cleaning:** dropped `customerID`; fixed `TotalCharges`, which was stored as text with 11 blank entries (replaced with 0 and converted to float)
3. **Exploratory data analysis:** histograms with mean/median lines, box plots, a correlation heatmap for numerical features, and count plots for categorical features
4. **Preprocessing:** label encoded the target and all categorical features (encoders saved to `encoders.pkl`), then split the data 80/20 into train and test sets
5. **Handling class imbalance:** applied **SMOTE** to the training set only, balancing it to 4,138 samples per class
6. **Model training:** compared Decision Tree, Random Forest and XGBoost using 5-fold cross-validation
7. **Evaluation:** tested the best model (Random Forest) on the held-out test set
8. **Saving the model:** exported the trained model and feature names to `customer_churn_model.pkl`, then reloaded it to demonstrate a simple prediction pipeline

## Results

5-fold cross-validation accuracy (default hyperparameters, on the SMOTE-balanced training data):

| Model | CV Accuracy |
|---|---|
| Decision Tree | 0.78 |
| Random Forest | 0.84 |
| XGBoost | 0.83 |

Random Forest performed best and was evaluated on the untouched test set (1,409 customers):

| Metric | Value |
|---|---|
| Test accuracy | 77.9% |
| Precision (churn) | 0.58 |
| Recall (churn) | 0.59 |
| F1-score (churn) | 0.58 |

The model is good at identifying customers who stay (precision and recall of 0.85) and finds around 59% of customers who actually churn. The test accuracy is lower than the cross-validation accuracy because cross-validation was run on synthetically oversampled data, so the test result is the more realistic estimate of real-world performance.

## Tech Stack

- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost

## Repository Structure

```
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Copy_of_Customer_Churn_Prediction_using_ML_for_GitHub.ipynb
└── README.md
```

## How to Run

**Option 1: Google Colab (easiest)**
Click the "Open In Colab" badge above and run all cells. The dataset is loaded directly from this repository, so no manual upload is needed.

**Option 2: Locally**

```bash
git clone https://github.com/roshkerai/Python_Customer_Churn_Prediction.git
cd Python_Customer_Churn_Prediction
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
jupyter notebook
```

Then open the notebook and run all cells. Running it generates `encoders.pkl` and `customer_churn_model.pkl`.


## Acknowledgements

Built following [this YouTube tutorial](https://www.youtube.com/watch?v=qNglJgNOb7A). Dataset: IBM Telco Customer Churn.

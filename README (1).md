
# Customer Churn Analysis Project

This project predicts customer churn using machine learning on the Telco Customer Churn dataset. It includes full data cleaning, EDA, feature engineering, model training, and evaluation.

## Dataset
Download from: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Steps
1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training (Random Forest, Logistic Regression)
5. Evaluation (confusion matrix, classification report)
6. Feature importance visualization
7. Model saving with `joblib`

## Requirements
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- joblib

## Usage
```bash
python churn_analysis.py
```

## Output
- Console metrics
- Feature importance chart saved as `feature_importance.png`
- Saved model: `churn_model.pkl`

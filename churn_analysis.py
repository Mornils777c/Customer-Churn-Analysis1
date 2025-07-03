
# Customer Churn Analysis using Python

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Step 2: Load Data
df = pd.read_csv('Telco-Customer-Churn.csv')

# Step 3: Data Cleaning
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Step 4: Feature Engineering
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in cols:
    df[col] = df[col].replace({'No internet service': 'No'})
df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 5: Train/Test Split and Scaling
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Models
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Step 7: Evaluate
y_pred_rf = rf.predict(X_test_scaled)
print("Random Forest Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Step 8: Feature Importance
importances = rf.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feat_df.head(10))
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.savefig("feature_importance.png")

# Step 9: Save Model
joblib.dump(rf, 'churn_model.pkl')

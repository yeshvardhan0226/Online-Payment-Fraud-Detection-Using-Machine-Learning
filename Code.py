import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("fraud_dataset_sample.csv")

# Convert transaction type to numeric (label encoding)
df['type'] = df['type'].astype('category').cat.codes

# ------------------------------
# 1. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Transaction Features')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()

# ------------------------------
# 2. Fraud vs Non-Fraud Count
plt.figure(figsize=(6, 4))
sns.countplot(x='isFraud', data=df, palette='Set2')
plt.title('Fraud vs Non-Fraud Transactions')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.savefig('fraud_distribution.png')
plt.show()

# ------------------------------
# 3. Transaction Type Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='type', data=df, palette='Set3')
plt.title('Distribution of Transaction Types')
plt.xlabel('Transaction Type (encoded)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('transaction_type_distribution.png')
plt.show()

# ------------------------------
# 4. Amount Distribution by Fraud Status
plt.figure(figsize=(8, 5))
sns.boxplot(x='isFraud', y='amount', data=df)
plt.title('Transaction Amount Distribution by Fraud Status')
plt.xlabel('Fraud (0 = No, 1 = Yes)')
plt.ylabel('Transaction Amount')
plt.tight_layout()
plt.savefig('amount_by_fraud.png')
plt.show()
# Import necessary libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.preprocessing import LabelEncoder



# Load dataset

df = pd.read_csv("fraud_dataset_sample.csv")



# -----------------------------

# Data Preprocessing

df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)  # Drop non-useful IDs

le = LabelEncoder()

df['type'] = le.fit_transform(df['type'])  # Encode transaction type



# -----------------------------

# Visualizations (EDA)



# 1. Correlation heatmap

plt.figure(figsize=(10, 6))

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')

plt.title('Correlation Heatmap of Transaction Features')

plt.tight_layout()

plt.savefig('heatmap.png')

plt.close()



# 2. Fraud vs Non-Fraud bar chart

plt.figure(figsize=(6, 4))

sns.countplot(x='isFraud', data=df, palette='Set2')

plt.title('Fraud vs Non-Fraud Transactions')

plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])

plt.tight_layout()

plt.savefig('fraud_distribution.png')

plt.close()



# 3. Transaction type distribution

plt.figure(figsize=(6, 4))

sns.countplot(x='type', data=df, palette='Set3')

plt.title('Distribution of Transaction Types')

plt.tight_layout()

plt.savefig('transaction_type_distribution.png')

plt.close()



# 4. Amount vs fraud status

plt.figure(figsize=(8, 5))

sns.boxplot(x='isFraud', y='amount', data=df)

plt.title('Transaction Amount Distribution by Fraud Status')

plt.tight_layout()

plt.savefig('amount_by_fraud.png')

plt.close()



# -----------------------------

# Model Training



X = df.drop('isFraud', axis=1)

y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)



# -----------------------------

# Model Evaluation



y_pred = model.predict(X_test)

print("\nClassification Report:")

print(classification_report(y_test, y_pred))



print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))



print("\nROC AUC Score:")

print(roc_auc_score(y_test, y_pred))



# -----------------------------

# Save evaluation plots (Confusion Matrix)

import numpy as np

import seaborn as sns



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.tight_layout()

plt.savefig('confusion_matrix.png')

plt.close()

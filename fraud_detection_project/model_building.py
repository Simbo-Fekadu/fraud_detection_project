import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load datasets
fraud_data = pd.read_csv('Fraud_Data_transformed.csv')
fraud_labels = pd.read_csv('Fraud_Data_labels.csv')
creditcard_data = pd.read_csv('creditcard_scaled.csv')
creditcard_labels = pd.read_csv('creditcard_labels.csv')

# Prepare Fraud_Data
X_fraud = fraud_data
y_fraud = fraud_labels.iloc[:, 0]
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)

# Prepare creditcard
X_credit = creditcard_data
y_credit = creditcard_labels.iloc[:, 0]
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit)

# Initialize models
lr = LogisticRegression(random_state=42, max_iter=1000)
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, dataset_name):
    # Train
    model.fit(X_train, y_train)
    
    # Predict probabilities and labels
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)
    
    # F1-Score
    f1 = f1_score(y_test, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{dataset_name} - {model.__class__.__name__}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{dataset_name} - {model.__class__.__name__} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{dataset_name}_{model.__class__.__name__}_cm.png')
    plt.close()
    
    return auc_pr, f1

# Evaluate models on Fraud_Data
lr_fraud_auc_pr, lr_fraud_f1 = evaluate_model(lr, X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, 'Fraud_Data')
xgb_fraud_auc_pr, xgb_fraud_f1 = evaluate_model(xgb_model, X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, 'Fraud_Data')

# Evaluate models on creditcard
lr_credit_auc_pr, lr_credit_f1 = evaluate_model(lr, X_credit_train, X_credit_test, y_credit_train, y_credit_test, 'creditcard')
xgb_credit_auc_pr, xgb_credit_f1 = evaluate_model(xgb_model, X_credit_train, X_credit_test, y_credit_train, y_credit_test, 'creditcard')

# Model selection justification
print("\nModel Selection Justification:")
print("Fraud_Data:")
print(f"Logistic Regression - AUC-PR: {lr_fraud_auc_pr:.4f}, F1-Score: {lr_fraud_f1:.4f}")
print(f"XGBoost - AUC-PR: {xgb_fraud_auc_pr:.4f}, F1-Score: {xgb_fraud_f1:.4f}")
print("creditcard:")
print(f"Logistic Regression - AUC-PR: {lr_credit_auc_pr:.4f}, F1-Score: {lr_credit_f1:.4f}")
print(f"XGBoost - AUC-PR: {xgb_credit_auc_pr:.4f}, F1-Score: {xgb_credit_f1:.4f}")

best_model_fraud = 'XGBoost' if xgb_fraud_auc_pr > lr_fraud_auc_pr else 'Logistic Regression'
best_model_credit = 'XGBoost' if xgb_credit_auc_pr > lr_credit_auc_pr else 'Logistic Regression'
print(f"\nBest Model for Fraud_Data: {best_model_fraud} (higher AUC-PR)")
print(f"Best Model for creditcard: {best_model_credit} (higher AUC-PR)")
print("XGBoost is typically preferred for its ability to handle complex patterns and imbalanced data, but Logistic Regression may be chosen for interpretability if performance is comparable.")
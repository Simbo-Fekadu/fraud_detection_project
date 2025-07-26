import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned datasets
fraud_data = pd.read_csv('Fraud_Data_cleaned.csv')
creditcard_data = pd.read_csv('creditcard_cleaned.csv')

# Univariate Analysis
# Fraud_Data.csv: Distribution of purchase_value and class
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(fraud_data['purchase_value'], bins=30, kde=True)
plt.title('Purchase Value Distribution')
plt.subplot(1, 2, 2)
sns.countplot(x='class', data=fraud_data)
plt.title('Class Distribution')
plt.savefig('fraud_data_univariate.png')
plt.close()

# Creditcard.csv: Distribution of Amount and Class
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(creditcard_data['Amount'], bins=30, kde=True)
plt.title('Transaction Amount Distribution')
plt.subplot(1, 2, 2)
sns.countplot(x='Class', data=creditcard_data)
plt.title('Class Distribution')
plt.savefig('creditcard_univariate.png')
plt.close()

# Bivariate Analysis
# Fraud_Data.csv: Purchase value vs Class
plt.figure(figsize=(8, 6))
sns.boxplot(x='class', y='purchase_value', data=fraud_data)
plt.title('Purchase Value by Class')
plt.savefig('fraud_data_bivariate.png')
plt.close()

# Creditcard.csv: Amount vs Class
plt.figure(figsize=(8, 6))
sns.boxplot(x='Class', y='Amount', data=creditcard_data)
plt.title('Transaction Amount by Class')
plt.savefig('creditcard_bivariate.png')
plt.close()

print("EDA plots saved.")
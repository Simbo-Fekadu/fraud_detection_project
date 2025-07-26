import pandas as pd

# Load cleaned datasets
fraud_data = pd.read_csv('Fraud_Data_cleaned.csv')
creditcard_data = pd.read_csv('creditcard_cleaned.csv')

# Remove duplicates
fraud_data.drop_duplicates(inplace=True)
creditcard_data.drop_duplicates(inplace=True)

# Correct data types for Fraud_Data.csv
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
fraud_data['user_id'] = fraud_data['user_id'].astype(str)
fraud_data['device_id'] = fraud_data['device_id'].astype(str)
fraud_data['class'] = fraud_data['class'].astype(int)

# Correct data types for creditcard.csv
creditcard_data['Time'] = creditcard_data['Time'].astype(float)
creditcard_data['Amount'] = creditcard_data['Amount'].astype(float)
creditcard_data['Class'] = creditcard_data['Class'].astype(int)

# Verify data types
print("Fraud Data Types:\n", fraud_data.dtypes)
print("\nCredit Card Data Types:\n", creditcard_data.dtypes)

# Save cleaned datasets
fraud_data.to_csv('Fraud_Data_cleaned.csv', index=False)
creditcard_data.to_csv('creditcard_cleaned.csv', index=False)
print("Duplicates removed and data types corrected. Datasets saved.")
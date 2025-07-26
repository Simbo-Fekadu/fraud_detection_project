import pandas as pd

# Load merged dataset
fraud_data = pd.read_csv('Fraud_Data_merged.csv')

# Transaction frequency: Count transactions per user
freq = fraud_data.groupby('user_id').size().reset_index(name='transaction_frequency')
fraud_data = fraud_data.merge(freq, on='user_id', how='left')

# Transaction velocity: Transactions per hour per user
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
velocity = fraud_data.groupby('user_id')['purchase_time'].agg(['min', 'max']).reset_index()
velocity['time_span'] = (velocity['max'] - velocity['min']).dt.total_seconds() / 3600  # Hours
velocity['transaction_velocity'] = fraud_data.groupby('user_id').size() / velocity['time_span'].replace(0, 1)  # Avoid division by zero
fraud_data = fraud_data.merge(velocity[['user_id', 'transaction_velocity']], on='user_id', how='left')

# Time-based features
fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - pd.to_datetime(fraud_data['signup_time'])).dt.total_seconds() / 3600  # Hours

# Save dataset
fraud_data.to_csv('Fraud_Data_features.csv', index=False)
print("Feature engineering complete. Saved as 'Fraud_Data_features.csv'.")
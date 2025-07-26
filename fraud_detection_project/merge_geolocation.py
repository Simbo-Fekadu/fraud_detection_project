from intervaltree import IntervalTree
import pandas as pd

# Load data
fraud_data = pd.read_csv('Fraud_Data_cleaned.csv')
ip_data = pd.read_csv('./Data/IpAddress_to_Country.csv')

# Convert IPs to int
fraud_data['ip_address'] = fraud_data['ip_address'].astype(float).astype(int)
ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(int)
ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(int)

# Build IntervalTree
tree = IntervalTree()
for _, row in ip_data.iterrows():
    tree[row['lower_bound_ip_address']:row['upper_bound_ip_address'] + 1] = row['country']

# Map IPs to countries
def lookup_country(ip):
    matches = tree[ip]
    return list(matches)[0].data if matches else 'Unknown'

fraud_data['country'] = fraud_data['ip_address'].apply(lookup_country)

# Save
fraud_data.to_csv('Fraud_Data_merged.csv', index=False)
print("Finished fast IP-to-country mapping.")

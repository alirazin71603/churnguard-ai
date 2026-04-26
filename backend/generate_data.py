"""
Layer 1 — Data Generator
Generates a synthetic IBM Telco-style dataset and simulated Kafka events.
"""
import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)
N = 500

def generate_telco_dataset(n=N):
    genders = np.random.choice(['Male', 'Female'], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n)
    dependents = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
    tenure = np.random.randint(0, 72, n)
    phone_service = np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1])
    multiple_lines = np.where(phone_service == 'Yes',
                              np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.42, 0.48, 0.1]),
                              'No phone service')
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.24, 0.21])
    paperless_billing = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n)
    monthly_charges = np.round(np.random.uniform(18, 118, n), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    total_charges = np.clip(total_charges, 0, None)

    # Churn probability influenced by tenure, contract, monthly charges
    churn_prob = (
        0.35 * (contract == 'Month-to-month').astype(float) +
        0.25 * (tenure < 12).astype(float) +
        0.15 * (monthly_charges > 80).astype(float) +
        0.10 * (internet_service == 'Fiber optic').astype(float) +
        np.random.uniform(0, 0.15, n)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.where(churn_prob > 0.5, 'Yes', 'No')

    customer_ids = [f'CUST-{str(i).zfill(5)}' for i in range(1, n+1)]

    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': genders,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    return df

def generate_kafka_events(df, n_events=50):
    """Simulate real-time Kafka-style events from customer behavior."""
    events = []
    event_types = [
        'call_drop', 'app_login', 'payment_failure', 'service_complaint',
        'plan_downgrade', 'data_usage_drop', 'support_ticket', 'plan_upgrade'
    ]
    churn_events = ['call_drop', 'payment_failure', 'service_complaint', 'plan_downgrade', 'data_usage_drop']
    
    for i in range(n_events):
        row = df.iloc[np.random.randint(0, len(df))]
        is_churn_signal = row['Churn'] == 'Yes' and np.random.random() > 0.3
        event_type = np.random.choice(churn_events if is_churn_signal else event_types)
        
        event = {
            'event_id': f'EVT-{str(i).zfill(6)}',
            'customer_id': row['customerID'],
            'event_type': event_type,
            'timestamp': pd.Timestamp.now().isoformat(),
            'metadata': {
                'tenure': int(row['tenure']),
                'monthly_charges': float(row['MonthlyCharges']),
                'contract': row['Contract'],
                'churn_label': row['Churn']
            }
        }
        events.append(event)
    return events

if __name__ == '__main__':
    os.makedirs('../data', exist_ok=True)
    df = generate_telco_dataset()
    df.to_csv('../data/telco_demo.csv', index=False)
    events = generate_kafka_events(df)
    with open('../data/kafka_events.json', 'w') as f:
        json.dump(events, f, indent=2)
    print(f"Generated {len(df)} customer records → data/telco_demo.csv")
    print(f"Generated {len(events)} Kafka events → data/kafka_events.json")

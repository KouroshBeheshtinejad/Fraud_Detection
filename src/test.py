# test.py
import requests
import random

API_URL = "http://127.0.0.1:8000/predict"

def random_transaction():
    return {
        "Time": random.randint(0, 50000),
        **{f"V{i}": random.uniform(-3, 3) for i in range(1, 29)},
        "Amount": random.uniform(1, 10000)
    }

transactions = [random_transaction() for _ in range(30)]

print("Checking fraud probabilities...\n")
high_risk = []

for tx in transactions:
    response = requests.post(API_URL, json=tx)
    if response.status_code == 200:
        prob = response.json()['fraud_prob']
        print(f"Transaction ${tx['Amount']:.2f} -> Fraud Probability: {prob:.4f}")
        if prob > 0.7:
            high_risk.append((tx, prob))
    else:
        print(f"Error: {response.status_code} - {response.text}")

if high_risk:
    print("\nHigh-risk transactions (prob > 0.7):")
    for tx, prob in high_risk:
        print(f"Amount: ${tx['Amount']:.2f}, Fraud Probability: {prob:.4f}")
else:
    print("\nNo high-risk transactions found.")

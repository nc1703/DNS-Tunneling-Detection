"""
DNS Tunneling Detection Tool
----------------------------

This script simulates a DNS tunneling detection system using statistical analysis
of DNS query logs. DNS tunneling is a covert channel that encodes data within
DNS queries. This tool uses a Random Forest Classifier trained on features like
query length, entropy, and character frequency to detect suspicious queries.

Output:
- Model evaluation printed to console
- Trained model saved to disk

Note: This is based on simulated data. In production, use logs from DNS servers
and extract features from real DNS query traffic.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import math

# ------------------------------
# Step 1: Simulate DNS query data
# ------------------------------
def calculate_entropy(string):
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy

# Create synthetic dataset
queries = [
    "google.com", "facebook.com", "weather.gov", "cnn.com", "microsoft.com",
    "asjdhqweu123123.baddomain.cn", "asldkjasldkasjd123123.malware.net",
    "login.dropbox.com", "vpn.company.internal", "verylongdomainnameencodedforsuspicion.xyz"
]
labels = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1]  # 1 = tunneling/suspicious

# Extract features
df = pd.DataFrame({'query': queries, 'label': labels})
df['length'] = df['query'].apply(len)
df['entropy'] = df['query'].apply(calculate_entropy)
df['num_subdomains'] = df['query'].apply(lambda x: x.count('.'))

# ------------------------------
# Step 2: Feature set and model training
# ------------------------------
features = df[['length', 'entropy', 'num_subdomains']]
X_train, X_test, y_train, y_test = train_test_split(features, df['label'], test_size=0.3, random_state=42, stratify=df['label'])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Step 3: Evaluate and save model
# ------------------------------
predictions = model.predict(X_test)
print("Model Evaluation Report:\n")
print(classification_report(y_test, predictions))

# Save the model
joblib.dump(model, 'dns_model.pkl')

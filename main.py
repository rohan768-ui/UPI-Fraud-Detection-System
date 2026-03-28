import pandas as pd
import numpy as np
import random

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -----------------------------
# STEP 1: CREATE DATA
# -----------------------------

data = []

for i in range(200):
    amount = random.randint(10, 100000)
    hour = random.randint(0, 23)

    fraud = 0

    # Strong fraud condition
    if amount > 70000 and hour < 5:
        fraud = 1

    # Add random fraud cases
    elif random.random() < 0.1:
        fraud = 1

    data.append([amount, hour, fraud])

df = pd.DataFrame(data, columns=["Amount", "Hour", "Fraud"])

# -----------------------------
# STEP 2: FEATURE ENGINEERING
# -----------------------------

df["High_Amount"] = (df["Amount"] > 50000).astype(int)
df["Night"] = (df["Hour"] < 5).astype(int)

df["Risk"] = df["High_Amount"] & df["Night"]

print("\nSample Data:\n")
print(df.head())

print("\nDetected Risk Transactions:\n")
print(df[df["Risk"] == 1])

# -----------------------------
# STEP 3: MACHINE LEARNING
# -----------------------------

X = df[["Amount", "High_Amount", "Night"]]
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# STEP 4: TEST NEW TRANSACTION
# -----------------------------

sample = pd.DataFrame([{
    "Amount": 80000,
    "High_Amount": 1,
    "Night": 1
}])

prediction = model.predict(sample)

print("\nNew Transaction Check:")
print("⚠️ Fraud" if prediction[0] == 1 else "✅ Safe")
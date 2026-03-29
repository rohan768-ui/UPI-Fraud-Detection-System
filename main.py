import pandas as pd
import numpy as np
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    classification_report,
    confusion_matrix
)

# -----------------------------
# REPRODUCIBILITY
# -----------------------------
random.seed(42)
np.random.seed(42)

# -----------------------------
# DATA GENERATION (REALISTIC)
# -----------------------------
data = []

for _ in range(1000):
    amount = random.randint(10, 100000)
    hour = random.randint(0, 23)

    if amount > 80000 and hour < 6:
        fraud = 1
    elif amount > 60000 and hour < 3:
        fraud = 1
    elif random.random() < 0.05:
        fraud = 1
    else:
        fraud = 0

    data.append([amount, hour, fraud])

df = pd.DataFrame(data, columns=["Amount", "Hour", "Fraud"])

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df["High_Amount"] = (df["Amount"] > 50000).astype(int)
df["Night"] = (df["Hour"] < 5).astype(int)
df["Very_High"] = (df["Amount"] > 80000).astype(int)
df["Early_Morning"] = (df["Hour"] < 3).astype(int)

X = df[["Amount", "High_Amount", "Night", "Very_High", "Early_Morning"]]
y = df["Fraud"]

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy)
print("Mean Squared Error:", mse)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE OUTPUTS
# -----------------------------
df.to_csv("data.csv", index=False)
joblib.dump(model, "model.pkl")

print("\nSaved: data.csv and model.pkl")
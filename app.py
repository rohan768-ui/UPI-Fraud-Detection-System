import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
df = pd.read_csv("data.csv")
model = joblib.load("model.pkl")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

st.title("🚨 AI-Based UPI Fraud Detection System")

st.markdown("""
This system uses Machine Learning to detect suspicious UPI transactions.
""")

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("🔍 Check Transaction")

amount = st.number_input("Enter Amount", min_value=0)
hour = st.slider("Select Hour", 0, 23)

high_amount = 1 if amount > 50000 else 0
night = 1 if hour < 5 else 0
very_high = 1 if amount > 80000 else 0
early_morning = 1 if hour < 3 else 0

if st.button("Check Transaction"):
    sample = pd.DataFrame([{
        "Amount": amount,
        "High_Amount": high_amount,
        "Night": night,
        "Very_High": very_high,
        "Early_Morning": early_morning
    }])

    pred = model.predict(sample)[0]

    if pred == 1:
        st.error("⚠️ Fraud Detected")
    else:
        st.success("✅ Safe Transaction")

# -----------------------------
# METRICS
# -----------------------------
st.subheader("📊 Summary")

X = df[["Amount", "High_Amount", "Night", "Very_High", "Early_Morning"]]
y_true = df["Fraud"]
y_pred = model.predict(X)

accuracy = accuracy_score(y_true, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("Total", len(df))
col2.metric("Fraud", df["Fraud"].sum())
col3.metric("Accuracy", round(accuracy, 3))

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# BAR GRAPH
# -----------------------------
st.subheader("📊 Fraud vs Safe")

fig1 = plt.figure()
df["Fraud"].value_counts().plot(kind="bar")
st.pyplot(fig1)

# -----------------------------
# HISTOGRAM
# -----------------------------
st.subheader("📊 Amount Distribution")

fig2 = plt.figure()
plt.hist(df["Amount"])
st.pyplot(fig2)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)

fig3 = plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha="center", va="center")

st.pyplot(fig3)

# -----------------------------
# ROC CURVE
# -----------------------------
st.subheader("📈 ROC Curve")

y_prob = model.predict_proba(X)[:, 1]

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

fig4 = plt.figure()
plt.plot(fpr, tpr)
plt.title(f"ROC Curve (AUC = {round(roc_auc,3)})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

st.pyplot(fig4)

# -----------------------------
# DOWNLOAD
# -----------------------------
st.download_button(
    label="📥 Download Dataset",
    data=df.to_csv(index=False),
    file_name="transactions.csv"
)
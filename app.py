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

st.title("🚨 UPI Fraud Detection Dashboard")
st.markdown("Data Analysis + Machine Learning based fraud detection system")

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("🔍 Check Transaction")

colA, colB = st.columns(2)

amount = colA.number_input("Enter Amount", min_value=0)
hour = colB.slider("Select Hour", 0, 23)

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
col1.metric("Total Transactions", len(df))
col2.metric("Fraud Cases", df["Fraud"].sum())
col3.metric("Accuracy", round(accuracy, 3))

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# FRAUD DISTRIBUTION
# -----------------------------
st.subheader("📊 Fraud vs Safe Distribution")

fig1 = plt.figure()
df["Fraud"].value_counts().plot(kind="bar")
plt.title("Fraud vs Safe")
st.pyplot(fig1)

# -----------------------------
# AMOUNT DISTRIBUTION
# -----------------------------
st.subheader("📊 Transaction Amount Distribution")

fig2 = plt.figure()
plt.hist(df["Amount"])
plt.xlabel("Amount")
plt.ylabel("Frequency")
st.pyplot(fig2)

# -----------------------------
# BOXPLOT (VERY IMPORTANT FOR ANALYSIS)
# -----------------------------
st.subheader("📊 Amount vs Fraud (Boxplot)")

fig_box = plt.figure()
df.boxplot(column="Amount", by="Fraud")
plt.title("Amount vs Fraud")
plt.suptitle("")
st.pyplot(fig_box)

# -----------------------------
# CORRELATION MATRIX
# -----------------------------
st.subheader("📊 Correlation Matrix")

fig_corr = plt.figure()
plt.imshow(df.corr())
plt.title("Correlation Matrix")
plt.colorbar()
st.pyplot(fig_corr)

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
# FEATURE IMPORTANCE (PRO LEVEL)
# -----------------------------
st.subheader("📊 Feature Importance")

importance = model.feature_importances_
features = X.columns

fig5 = plt.figure()
plt.barh(features, importance)
plt.xlabel("Importance")
plt.ylabel("Features")
st.pyplot(fig5)

# -----------------------------
# DOWNLOAD
# -----------------------------
st.download_button(
    label="📥 Download Dataset",
    data=df.to_csv(index=False),
    file_name="transactions.csv"
)
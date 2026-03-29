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
ax1 = fig1.add_subplot(111)
df["Fraud"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_title("Fraud vs Safe")
ax1.set_xlabel("Class")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# -----------------------------
# AMOUNT DISTRIBUTION
# -----------------------------
st.subheader("📊 Transaction Amount Distribution")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.hist(df["Amount"])
ax2.set_title("Amount Distribution")
ax2.set_xlabel("Amount")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

# -----------------------------
# BOXPLOT
# -----------------------------
st.subheader("📊 Amount vs Fraud (Boxplot)")

fig_box = plt.figure()
ax_box = fig_box.add_subplot(111)

df.boxplot(column="Amount", by="Fraud", ax=ax_box)

ax_box.set_title("Amount vs Fraud")
plt.suptitle("")
ax_box.set_xlabel("Fraud (0 = Safe, 1 = Fraud)")
ax_box.set_ylabel("Transaction Amount")
ax_box.grid(True)

st.pyplot(fig_box)

# -----------------------------
# CORRELATION MATRIX
# -----------------------------
st.subheader("📊 Correlation Matrix")

fig_corr = plt.figure()
ax_corr = fig_corr.add_subplot(111)

cax = ax_corr.imshow(df.corr())
fig_corr.colorbar(cax)

ax_corr.set_title("Correlation Matrix")
ax_corr.set_xticks(range(len(df.columns)))
ax_corr.set_yticks(range(len(df.columns)))
ax_corr.set_xticklabels(df.columns, rotation=45)
ax_corr.set_yticklabels(df.columns)

st.pyplot(fig_corr)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)

fig_cm = plt.figure()
ax_cm = fig_cm.add_subplot(111)

cax2 = ax_cm.imshow(cm)
fig_cm.colorbar(cax2)

ax_cm.set_title("Confusion Matrix")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax_cm.text(j, i, cm[i][j], ha="center", va="center")

st.pyplot(fig_cm)

# -----------------------------
# ROC CURVE
# -----------------------------
st.subheader("📈 ROC Curve")

y_prob = model.predict_proba(X)[:, 1]

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)

ax4.plot(fpr, tpr)
ax4.set_title(f"ROC Curve (AUC = {round(roc_auc,3)})")
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")

st.pyplot(fig4)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📊 Feature Importance")

importance = model.feature_importances_
features = X.columns

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)

ax5.barh(features, importance)
ax5.set_title("Feature Importance")
ax5.set_xlabel("Importance")
ax5.set_ylabel("Features")

st.pyplot(fig5)

# -----------------------------
# DOWNLOAD
# -----------------------------
st.download_button(
    label="📥 Download Dataset",
    data=df.to_csv(index=False),
    file_name="transactions.csv"
)
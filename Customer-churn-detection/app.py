# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

@st.cache_data
def load_default_dataset():
    return pd.read_csv("Customer-Churn.csv")

def safe_label_transform(le, val):
    try:
        return int(np.where(le.classes_ == val)[0][0])
    except Exception:
        return 0

st.sidebar.header("Dataset & Upload")
uploaded_file = st.sidebar.file_uploader("Upload training CSV (optional)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded training dataset")
else:
    st.sidebar.info("Using default Telco Churn dataset")
    df = load_default_dataset()

original_df = df.copy()

if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna('missing')
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "model": model
    }

tabs = st.tabs(["Overview", "Model Results", "Predict New Customer", "Batch Predict (CSV)", "High-Risk Customers"])

with tabs[0]:
    st.header("Dataset Preview")
    st.dataframe(original_df.head())

with tabs[1]:
    st.header("Model Comparison")
    comparison = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy (%)": [results[m]["accuracy"]*100 for m in results]
    })
    st.dataframe(comparison)
    selected_model = st.selectbox("Select model", list(results.keys()))
    y_pred = results[selected_model]["model"].predict(X_test)
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with tabs[2]:
    st.header("Predict New Customer")
    with st.form("predict_form"):
        predict_input = {}
        cols = st.columns(2)
        for i, col in enumerate(X.columns):
            target_col = cols[i % 2]
            if col in cat_cols:
                val = target_col.selectbox(col, list(le_dict[col].classes_))
                predict_input[col] = safe_label_transform(le_dict[col], val)
            else:
                val = target_col.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].median()))
                predict_input[col] = val
        submit = st.form_submit_button("Predict Churn")
    if submit:
        input_df = pd.DataFrame([predict_input])
        input_scaled = scaler.transform(input_df)
        model_choice = st.selectbox("Model for prediction", list(models.keys()))
        churn_prob = models[model_choice].predict_proba(input_scaled)[0][1]
        st.success(f"Churn Probability: {churn_prob*100:.2f}%")
        if churn_prob > 0.7:
            st.warning("High Risk: Recommend Personal Outreach or Discount")
        elif churn_prob > 0.4:
            st.info("Medium Risk: Recommend Promotional Email")
        else:
            st.success("Low Risk: Monitor Only")

with tabs[3]:
    st.header("Batch Predict from CSV")
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'])
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        for col in cat_cols:
            if col in batch_df.columns:
                batch_df[col] = batch_df[col].fillna('missing').astype(str)
                le = le_dict[col]
                batch_df[col] = batch_df[col].apply(lambda v: safe_label_transform(le, str(v)))
        batch_scaled = scaler.transform(batch_df[X.columns])
        preds = models['Random Forest'].predict_proba(batch_scaled)[:, 1]
        batch_df['Churn_Probability'] = preds
        st.dataframe(batch_df.head(20))

with tabs[4]:
    st.header("Top 20 High-Risk Customers")
    probs = models['Random Forest'].predict_proba(X_scaled)[:, 1]
    df['Churn_Probability'] = probs
    st.dataframe(df.sort_values(by='Churn_Probability', ascending=False).head(20))

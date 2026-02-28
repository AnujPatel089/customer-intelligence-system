from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# --- App config ---
st.set_page_config(page_title="Customer Intelligence System")
st.title("Customer Intelligence Dashboard")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "rfm_data.csv"   # NOTE: folder name is "Data" in your repo (case-sensitive on Linux)
MODELS_DIR = BASE_DIR / "models"

@st.cache_data
def load_rfm():
    return pd.read_csv(DATA_PATH)

rfm = load_rfm()

page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Segmentation", "Churn Prediction", "CLV Prediction"],
)

if page == "Overview":
    st.metric("Total Customers", int(rfm.shape[0]))
    st.metric("Avg Monetary Value", round(float(rfm["Monetary"].mean()), 2))

    fig, ax = plt.subplots()
    ax.hist(rfm["Monetary"].dropna())
    ax.set_title("Monetary Distribution")
    st.pyplot(fig)

elif page == "Segmentation":
    st.subheader("Customer Clusters")
    st.write(rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean())

elif page == "Churn Prediction":
    model = joblib.load(MODELS_DIR / "churn_model.pkl")

    recency = st.number_input("Recency", min_value=0.0, value=10.0)
    frequency = st.number_input("Frequency", min_value=0.0, value=1.0)
    monetary = st.number_input("Monetary", min_value=0.0, value=100.0)

    if st.button("Predict"):
        prob = model.predict_proba([[recency, frequency, monetary]])[0][1]
        st.write("Churn Probability:", round(float(prob), 2))

elif page == "CLV Prediction":
    model = joblib.load(MODELS_DIR / "clv_model.pkl")

    recency = st.number_input("Recency", min_value=0.0, value=10.0)
    frequency = st.number_input("Frequency", min_value=0.0, value=1.0)

    if st.button("Predict CLV"):
        clv = model.predict([[recency, frequency]])[0]
        st.write("Predicted CLV:", round(float(clv), 2))

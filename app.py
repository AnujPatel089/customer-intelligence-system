import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Intelligence System")

st.title("📊 Customer Intelligence Dashboard")

page = st.sidebar.selectbox("Select Page",
                            ["Overview","Segmentation","Churn Prediction","CLV Prediction"])

rfm = pd.read_csv(r"E:\Personal Project\customer-intelligence-system\Data\rfm_data.csv")

if page == "Overview":
    st.metric("Total Customers", rfm.shape[0])
    st.metric("Avg Monetary Value", round(rfm['Monetary'].mean(),2))

    fig = plt.figure()
    rfm['Monetary'].hist()
    st.pyplot(fig)

elif page == "Segmentation":
    st.subheader("Customer Clusters")
    st.write(rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean())

elif page == "Churn Prediction":
    model = joblib.load("models/churn_model.pkl")

    recency = st.number_input("Recency")
    frequency = st.number_input("Frequency")
    monetary = st.number_input("Monetary")

    if st.button("Predict"):
        prob = model.predict_proba([[recency,frequency,monetary]])[0][1]
        st.write("Churn Probability:", round(prob,2))

elif page == "CLV Prediction":
    model = joblib.load("models/clv_model.pkl")

    recency = st.number_input("Recency")
    frequency = st.number_input("Frequency")

    if st.button("Predict CLV"):
        clv = model.predict([[recency,frequency]])[0]
        st.write("Predicted CLV:", round(clv,2))
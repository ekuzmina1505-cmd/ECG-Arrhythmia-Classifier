
import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(
    page_title="ЭКГ Аритмия | AI Диагностика",
    page_icon="❤️",
    layout="wide"
)

st.title("ECG Clinical AI V9.7")

data = st.text_area("Paste ECG feature vector")

if data:
    x = np.array([float(i) for i in data.split(",")])
    x = scaler.transform([x])

    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]

    st.write("Prediction:", encoder.inverse_transform([pred])[0])
    st.write("Confidence:", float(max(proba)))

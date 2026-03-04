import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="Specialist Recommender Demo", layout="centered")
st.title("Specialist Recommender (Demo)")
st.caption("Local FastAPI backend running at 127.0.0.1:8000")

age = st.number_input("Age", min_value=0, max_value=120, value=34)
gender = st.selectbox("Gender", ["male", "female", "other"])
severity = st.slider("Severity (1-10)", min_value=1, max_value=10, value=6)
duration = st.text_input("Duration", value="2 hours")
symptoms = st.text_area("Symptoms", value="chest tightness and shortness of breath")

if st.button("Recommend Specialist"):
    payload = {
        "age": int(age),
        "gender": gender,
        "symptoms": symptoms.strip(),
        "severity": int(severity),
        "duration": duration.strip(),
    }
    r = requests.post(API_URL, json=payload, timeout=30)
    if r.ok:
        out = r.json()
        st.success(f"Recommended: {out['recommended_specialist']}")
        st.write("Confidence:", out["confidence"])
        st.write("Model label:", out["model_label"])
    else:
        st.error(f"Error {r.status_code}: {r.text}")

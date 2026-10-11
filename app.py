import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("Heart Disease Prediction")
st.write("Enter the four numeric features used by the trained model.")

@st.cache_resource
def load_model():
    with open("svc_trained_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

cp = st.selectbox("Chest pain type (cp)", [0, 1, 2, 3])
ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
sex = st.selectbox("Sex (sex)", [0, 1])

if st.button("Predict"):
    user_data = pd.DataFrame([[cp, ca, oldpeak, sex]],
                             columns=["cp", "ca", "oldpeak", "sex"])
    prediction = model.predict(user_data)[0]
    if prediction == 1:
        st.warning("Model prediction: Heart disease present.")
    else:
        st.success("Model prediction: Heart disease not present.")
    st.caption("Educational use only; this is not a medical diagnosis.")
from pickle import load
import streamlit as st
#from streamlit_lottie import st_lottie
import requests
import numpy as np


## Cargue de modelo
model = load(open("../models/random_forest_regressor_42.sav", "rb"))

## Configuración de página
st.set_page_config(page_title="Boston Housing Predictor", page_icon="\U0001F3E0", layout="wide")

with st.container():
    st.title("🏡 Boston Housing Price Prediction App 🏙️")
    st.write("Esta aplicación predice el precio de una casa en Boston en función de las características ingresadas.")

with st.container():
    st.subheader("🔍 Ingresa las características de la casa")
    col1 = st.columns(1)[0]
    with col1:
        CRIM = st.slider("🏚️ CRIM: Tasa de criminalidad per cápita por ciudad", 0.0, 100.0, 1.0, 0.1)
        RM = st.slider("🏚️ RM: Número promedio de habitaciones por vivienda", 3.0, 10.0, 6.0, 0.1)
        DIS = st.slider("🏚️ DIS: Distancias ponderadas a cinco centros de empleo de Boston", 1.0, 12.0, 4.0, 0.1)
        LSTAT = st.slider("🏚️ LSTAT: Porcentaje de estatus socioeconómico bajo por ciudad", 0.0, 40.0, 10.0, 0.1)
        AGE = st.slider("🏚️ AGE: Proporción de unidades ocupadas por sus propietarios construidas antes de 1940", 0.0, 100.0, 50.0, 1.0)

with st.container():
    st.markdown("---")
    if st.button("✨ Predecir Precio de la Casa"):
        input_data = np.array([[CRIM,RM,DIS,LSTAT,AGE]])
        prediction = model.predict(input_data)[0]

        st.success(f"🏡 El precio predicho de la casa es: **${prediction * 1000:.2f} USD**")


st.markdown("---")
                   


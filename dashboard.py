import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import numpy as np

# Configuração do Dashboard
st.title("Real Estate Pricing Dashboard")
st.write("""
Este dashboard fornece estimativas de preços de imóveis com base em suas características.
""")

# Entrada de Dados do Usuário
st.sidebar.header("Insira as características do imóvel:")
area = st.sidebar.number_input("Área (m²):", min_value=1, max_value=1000, value=100)
rooms = st.sidebar.number_input("Número de Quartos:", min_value=1, max_value=10, value=3)
location = st.sidebar.selectbox("Localização:", ["Downtown", "Suburb", "Rural"])

# Carregar as colunas de features
feature_columns = joblib.load("feature_columns.pkl")

# Carregar o modelo treinado
model = joblib.load("real_estate_model.pkl")   

# Preprocessamento de Dados de Entrada
def preprocess_input(area, rooms, location):
    data = pd.DataFrame({
        "area": [area],
        "rooms": [rooms],
        "price_per_sq_meter": [0],  # Placeholder; será calculado após o ajuste do preço
        "room_density": [rooms / area],
        "location_Downtown": [1 if location == "Downtown" else 0],
        "location_Rural": [1 if location == "Rural" else 0],
        "location_Suburb": [1 if location == "Suburb" else 0]
    })
    
    # Adicionar colunas ausentes com valor 0
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    # Ordenar as colunas para coincidir com o treinamento
    data = data[feature_columns]
   

    # Normalização das variáveis numéricas
    scaler = StandardScaler()
    numerical_features = ["area", "rooms", "price_per_sq_meter", "room_density"]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

# Previsão de Preço
input_data = preprocess_input(area, rooms, location)
predicted_price = model.predict(input_data)[0]
st.subheader("Resultado da Estimativa")
st.write(f"Preço Estimado: **R$ {predicted_price:,.2f}**")

# Verificar se o arquivo existe antes de carregá-lo
if os.path.exists("real_estate_data.csv"):
    data = pd.read_csv("real_estate_data.csv")
    st.sidebar.header("Estatísticas do Conjunto de Dados")
    st.sidebar.write(data.describe())
else:
    st.sidebar.warning("O arquivo 'real_estate_data.csv' não foi encontrado. Estatísticas não serão exibidas.")

# Gráficos Interativos
st.subheader("Distribuição dos Preços de Imóveis")
#st.bar_chart(data["price"].value_counts().sort_index())



bins = st.slider("Escolha o número de intervalos (bins):", min_value=5, max_value=50, value=10)
hist_values = np.histogram(data["price"], bins=bins)[0]
fig, ax = plt.subplots()
data["price"].plot(kind="hist", bins=bins, ax=ax)
ax.set_title("Distribuição dos Preços de Imóveis")
ax.set_xlabel("Preço")
ax.set_ylabel("Frequência")
st.pyplot(fig)

st.bar_chart(hist_values)



st.subheader("Relação Área x Preço")
st.scatter_chart(data, x="area", y="price", size="rooms", color="location")

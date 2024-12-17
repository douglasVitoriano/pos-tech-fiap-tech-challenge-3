import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import matplotlib.pyplot as plt
import numpy as np

# Carregar o conjunto de dados
df = pd.read_csv("real_estate_data.csv")

# Configuração do Dashboard
st.title("Real Estate Pricing Dashboard")
st.write("""
Este dashboard fornece estimativas de preços de imóveis com base em suas características.
""")

# Entrada de Dados do Usuário
st.sidebar.header("Insira as características do imóvel:")
house_size = st.sidebar.number_input(
    "Área (m²):", min_value=1, max_value=20000, value=100)
bed = st.sidebar.number_input(
    "Número de Quartos:", min_value=1, max_value=2000, value=1)
city = st.sidebar.selectbox("Cidade:", df["city"].unique())

# Carregar as colunas de features
feature_columns = joblib.load("feature_columns.pkl")

# Carregar o modelo treinado
model = joblib.load("real_estate_model.pkl")

# Preprocessamento de Dados de Entrada

# Ajustar o StandardScaler nos dados de treinamento
scaler = StandardScaler()
numerical_features = ["house_size", "bed",
                      "price_per_sq_meter", "room_density"]
scaler.fit(df[numerical_features])


def preprocess_input(house_size, bed, city):
    data = pd.DataFrame({
        "house_size": [house_size],
        "bed": [bed],
        # Placeholder; será calculado após o ajuste do preço
        "price_per_sq_meter": [0],
        "room_density": [bed / house_size],
        "city": [city]
    })

    # One-hot encoding para a cidade
    data = pd.get_dummies(data, columns=["city"], drop_first=False)

    # Adicionar colunas ausentes com valor 0
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    # Ordenar as colunas para coincidir com o treinamento
    data = data[feature_columns]

    # Normalização das variáveis numéricas
    data[numerical_features] = scaler.transform(data[numerical_features])
    return data


# Previsão de Preço
input_data = preprocess_input(house_size, bed, city)
predicted_price = model.predict(input_data)[0]
st.subheader("Resultado da Estimativa")
st.write(f"Preço Estimado: **$ {predicted_price:,.2f}**")

# Verificar se o arquivo existe antes de carregá-lo
if os.path.exists("real_estate_data.csv"):
    data = pd.read_csv("real_estate_data.csv")
    st.sidebar.header("Estatísticas do Conjunto de Dados")
    st.sidebar.write(data.describe())
else:
    st.sidebar.warning(
        "O arquivo 'real_estate_data.csv' não foi encontrado. Estatísticas não serão exibidas.")

# Gráficos Interativos
st.subheader("Distribuição dos Preços de Imóveis")

bins = st.slider("Escolha o número de intervalos (bins):",
                 min_value=5, max_value=50, value=10)
hist_values = np.histogram(data["price"], bins=bins)[0]
fig, ax = plt.subplots()
data["price"].plot(kind="hist", bins=bins, ax=ax)
ax.set_title("Distribuição dos Preços de Imóveis")
ax.set_xlabel("Preço")
ax.set_ylabel("Frequência")
st.pyplot(fig)

st.subheader("Relação Cidade x Preço")
st.bar_chart(data.groupby("city")["price"].mean())

st.subheader("Relação Área x Preço")
st.scatter_chart(data, x="house_size", y="price", size="bed", color="city")

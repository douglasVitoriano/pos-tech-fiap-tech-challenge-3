import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Função para verificar estacionaridade
def check_stationarity(series):
    result = adfuller(series)
    print("Teste de Dickey-Fuller Aumentado (ADF):")
    print(f"Estatística ADF: {result[0]}")
    print(f"Valor-p: {result[1]}")
    print(f"Número de lags utilizados: {result[2]}")
    print(f"Número de observações: {result[3]}")
    if result[1] <= 0.05:
        print("A série é estacionária (valor-p <= 0.05).")
    else:
        print("A série não é estacionária (valor-p > 0.05).")

# Função para encontrar os melhores parâmetros ARIMA
from statsmodels.tsa.arima.model import ARIMA
def optimize_arima(series, p_range, d_range, q_range):
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except:
                    continue
    return best_order, best_model

# Baixar dados do Yahoo Finance
ticker = "MGLU3.SA"  # Alterar o ticker conforme necessidade
data = yf.download(ticker, start="2015-01-01", end="2023-12-01")['Close']

# Visualizar os dados
print(data.describe())
plt.figure(figsize=(12, 6))
plt.plot(data, label="Série Histórica")
plt.title(f"Preço de Fechamento - {ticker}")
plt.legend()
plt.show()

# Verificar estacionaridade
check_stationarity(data)

# Separar dados em treino e teste
train = data[:'2022-01-01']
test = data['2022-01-02':]

# Encontrar melhores parâmetros ARIMA
p_range = range(0, 6)
d_range = range(0, 2)
q_range = range(0, 6)
best_order, best_model = optimize_arima(train, p_range, d_range, q_range)
print(f"Melhor ordem ARIMA: {best_order}")

# Ajustar o modelo com os melhores parâmetros
model = ARIMA(train, order=best_order)
model_fit = model.fit()

# Fazer previsões
forecast = model_fit.forecast(steps=len(test))
forecast_conf_int = model_fit.get_forecast(steps=len(test)).conf_int()

# Avaliar a previsão
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = mean_absolute_percentage_error(test, forecast) * 100
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plotar resultados
plt.figure(figsize=(14, 8))
plt.plot(train, label="Treinamento")
plt.plot(test, label="Teste", color='blue')
plt.plot(test.index, forecast, label="Previsão", color='red')
plt.fill_between(test.index, 
                 forecast_conf_int.iloc[:, 0], 
                 forecast_conf_int.iloc[:, 1], 
                 color='pink', alpha=0.3, label="Intervalo de Confiança")
plt.title(f"Modelo ARIMA - Previsões para {ticker}")
plt.legend()
plt.show()

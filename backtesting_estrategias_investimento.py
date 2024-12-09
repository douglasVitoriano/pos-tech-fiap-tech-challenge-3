import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

# Parâmetros da estratégia
ticker = "MGLU3.SA"
start_date = "2015-01-01"
end_date = "2024-10-01"
short_window = 50  # Período da SMA curta
long_window = 200  # Período da SMA longa

# Baixar dados do Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

# Garantir que não haja valores ausentes
data = data[['Close']].dropna()

# Calcular médias móveis
data['SMA50'] = data['Close'].rolling(window=short_window).mean()
data['SMA200'] = data['Close'].rolling(window=long_window).mean()

# Criar sinais de compra/venda
data['Signal'] = np.where(data['SMA50'] > data['SMA200'], 1, 0)  # 1 = compra, 0 = venda

# Calcular retorno diário e da estratégia
data['Daily Return'] = data['Close'].pct_change()
data['Strategy Return'] = data['Signal'].shift(1) * data['Daily Return']

# Performance acumulada
data['Cumulative Market Return'] = (1 + data['Daily Return']).cumprod()
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

# Retorno anualizado e volatilidade
annualized_market_return = data['Daily Return'].mean() * 252
annualized_strategy_return = data['Strategy Return'].mean() * 252
volatility_market = data['Daily Return'].std() * np.sqrt(252)
volatility_strategy = data['Strategy Return'].std() * np.sqrt(252)

print(f"Retorno Anualizado (Mercado): {annualized_market_return:.2%}")
print(f"Volatilidade (Mercado): {volatility_market:.2%}")
print(f"Retorno Anualizado (Estratégia): {annualized_strategy_return:.2%}")
print(f"Volatilidade (Estratégia): {volatility_strategy:.2%}")

# Converter split_date para datetime e atribuir fuso horário UTC
split_date = pd.to_datetime("2020-01-01").tz_localize('UTC')

# Divisão de treino e teste
train_data = data[:split_date]
test_data = data[split_date:]

# Plotar resultados de treino e teste
plt.figure(figsize=(14, 8))
plt.plot(train_data['Cumulative Market Return'], label="Mercado (Treino)", color="blue", linestyle="--")
plt.plot(train_data['Cumulative Strategy Return'], label="Estratégia (Treino)", color="red", linestyle="--")
plt.plot(test_data['Cumulative Market Return'], label="Mercado (Teste)", color="blue")
plt.plot(test_data['Cumulative Strategy Return'], label="Estratégia (Teste)", color="red")
plt.axvline(split_date, color="gray", linestyle="--", label="Divisão Treino/Teste")
plt.title("Backtesting de Estratégia: Média Móvel")
plt.xlabel("Data")
plt.ylabel("Retorno Acumulado")
plt.legend()
plt.grid()
plt.show()

# Plotar cruzamento de médias móveis
plt.figure(figsize=(14, 8))
plt.plot(data['Close'], label="Preço de Fechamento", color="black", alpha=0.7)
plt.plot(data['SMA50'], label=f"SMA {short_window} dias", color="blue")
plt.plot(data['SMA200'], label=f"SMA {long_window} dias", color="red")
plt.title("Cruzamento de Médias Móveis")
plt.xlabel("Data")
plt.ylabel("Preço")
plt.legend()
plt.grid()
plt.show()

# Linha vertical indicando o ponto de divisão
plt.axvline(split_date, color="gray", linestyle="--", label="Divisão Treino/Teste")
plt.title("Backtesting de Estratégia: Média Móvel")
plt.xlabel("Data")
plt.ylabel("Retorno Acumulado")
plt.legend()
plt.grid()
plt.show()

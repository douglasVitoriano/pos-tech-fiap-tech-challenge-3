import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Baixar dados do Yahoo Finance
ticker = "MGLU3.SA"  # Alterar o ticker conforme necessário
data = yf.download(ticker, start="2015-01-01", end="2024-10-01")

# Criar variáveis de lag e uma média móvel para prever o próximo preço
df = pd.DataFrame(data)
df['Lag1'] = df['Close'].shift(1)
df['Lag2'] = df['Close'].shift(2)
df['Lag3'] = df['Close'].shift(3)
df['Moving_Avg'] = df['Close'].rolling(window=3).mean()  # Média móvel de 3 dias
df = df.dropna()

# Separar dados em recursos (X) e variável-alvo (y)
X = df[['Lag1', 'Lag2', 'Lag3', 'Moving_Avg']]
y = df['Close']

# Divisão treino-teste baseada em tempo (validação temporal)
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Otimização de hiperparâmetros usando GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Melhores Hiperparâmetros: {grid_search.best_params_}")

# Treinar o modelo com os melhores parâmetros
best_model.fit(X_train, y_train)

# Fazer previsões
y_pred = best_model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
print(f"Erro Percentual Absoluto Médio (MAPE): {mape:.2f}%")

# Visualizar as previsões
plt.figure(figsize=(14, 8))
plt.plot(y_test.values, label="Real", color="blue")
plt.plot(y_pred, label="Previsão", color="red")
plt.title("Previsão de Preços com Random Forest")
plt.xlabel("Amostras")
plt.ylabel("Preço de Fechamento")
plt.legend()
plt.show()

# Importância das variáveis
importances = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.title("Importância das Variáveis")
plt.xlabel("Importância")
plt.ylabel("Recursos")
plt.show()

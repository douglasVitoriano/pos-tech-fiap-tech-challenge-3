# Corrigindo o script para previsão de Bitcoin, análise de notícias e sentimentos

# Imports necessários
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Função para buscar notícias sobre Bitcoin
def fetch_bitcoin_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "Bitcoin",
        "apiKey": "3be6fb74511c432bb6741415eb0419d7"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        return [article["title"] for article in articles]
    else:
        return []

# Função para análise de sentimentos
def analyze_sentiments(titles):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for title in titles:
        sentiment = analyzer.polarity_scores(title)
        sentiments.append(sentiment["compound"])
    return sentiments

# Coleta de notícias e análise de sentimentos
noticias = fetch_bitcoin_news()
# Verifica se há notícias disponíveis
if noticias:
    sentimentos = analyze_sentiments(noticias)
    noticias_df = pd.DataFrame({
        "noticia": noticias,
        "sentimento": sentimentos
    })
else:
    print("Nenhuma notícia encontrada. Usando valores padrão para sentimentos.")
    noticias_df = pd.DataFrame({
        "noticia": ["Sem notícias disponíveis"],
        "sentimento": [0]  # Sentimento neutro
    })

print(noticias_df.head())

# Simulação de dados para predição
# Nota: No código real, substitua isso por dados reais de mercado
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=100)
prices = np.random.rand(100) * 50000  # Preços aleatórios entre 0 e 50 mil

data = pd.DataFrame({
    "date": dates,
    "price": prices
})

correlation_matrix = data.corr()
print(correlation_matrix)

data["price_change"] = data["price"].pct_change() * 100

# Preenchendo valores ausentes
data.fillna(0, inplace=True)

# Adicionando sentimentos simulados ao DataFrame
if len(noticias_df) > 1:  # Caso existam sentimentos reais, usa eles
    sentimentos = noticias_df["sentimento"].values
else:
    sentimentos = [0]  # Fallback para valor neutro

data["sentiment"] = np.random.choice(sentimentos, size=len(data), replace=True)

# Dividindo em variáveis independentes e dependentes
X = data[["price_change", "sentiment"]].values
y = data["price"].values

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predições
y_pred = model.predict(X_test)

# Métrica de avaliação
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualização do resultado
plt.figure(figsize=(10, 6))
plt.plot(data["date"], data["price"], label="Preços reais")
plt.scatter(data.loc[data.index.isin(X_test[:, 0].argsort())]["date"], y_pred, color="red", label="Predições")
plt.legend()
plt.title("Previsão de Preços do Bitcoin com Sentimentos")
plt.xlabel("Data")
plt.ylabel("Preço")
plt.show()

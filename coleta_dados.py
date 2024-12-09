import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

# 1. Fazer a requisição ao site
url = "https://finance.yahoo.com/quote/VIVT3.SA/news/"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

# 2. Parsing do HTML
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    # Selecionar as manchetes (div com classes que podem variar)
    headlines = soup.select("h3, p")  # Exemplos genéricos de tags
else:
    print("Erro ao acessar o site:", response.status_code)
    headlines = []

# 3. Extração das manchetes
texts = [headline.get_text(strip=True) for headline in headlines]

# 4. Análise de Sentimento
results = []
for text in texts:
    sentiment = TextBlob(text).sentiment
    results.append({
        "headline": text,
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity,
    })

# 5. Exibição dos resultados
for result in results:
    print(f"Headline: {result['headline']}")
    print(f"Sentiment Polarity: {result['polarity']}")
    print(f"Subjectivity: {result['subjectivity']}\n")

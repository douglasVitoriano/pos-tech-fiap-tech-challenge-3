import requests
import pandas as pd

# Define NewsAPI configuration
API_KEY = '28c5a8f81c1c4865b620ecae007161c5'
BASE_URL = 'https://newsapi.org/v2/everything'

# Keywords for SCAR3 and related markets
keywords = [
    'São Carlos Empreendimentos', 
    'SCAR3', 
    'real estate news Brazil', 
    'retail industry Brazil'
]

# Fetch news articles with advanced queries
def fetch_news(keyword):
    params = {
        'q': keyword,
        'apiKey': API_KEY,
        'language': 'pt',  # Set to 'pt' for Brazilian news
        'pageSize': 10    # Number of articles to fetch
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        print(f"Error fetching news for {keyword}: {response.status_code}")
        return []

# Aggregate all articles
all_articles = []
for keyword in keywords:
    articles = fetch_news(keyword)
    all_articles.extend(articles)

# Format and save results
if all_articles:
    df = pd.DataFrame(all_articles)
    # Extract relevant columns for clarity
    df = df[['source', 'author', 'title', 'description', 'url', 'publishedAt']]
    df.to_csv('scar3_retail_news.csv', index=False)
    print("News saved to scar3_retail_news.csv")
else:
    print("No news found.")

# Additional tip: scrape specific financial sites if needed

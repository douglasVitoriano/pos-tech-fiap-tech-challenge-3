from newsapi import NewsApiClient

# Inicialize o cliente com sua chave de API
newsapi = NewsApiClient(api_key='28c5a8f81c1c4865b620ecae007161c5')

def get_stock_news(ticker, page_size=10, page=1):
    try:
        # Obtenha notícias relacionadas ao ticker
        response = newsapi.get_everything(
            q=ticker,
            language='pt',
            sort_by='relevancy',
            page_size=page_size,
            page=page
        )
        return response.get('articles', [])
    except Exception as e:
        print(f"Erro ao buscar notícias: {e}")
        return []

if __name__ == "__main__":
    ticker = 'São Carlos Empreendimentos'
    news = []

    # Buscar até 5 páginas de notícias
    for page in range(1, 6):
        articles = get_stock_news(ticker, page_size=10, page=page)
        if not articles:
            print(f"Sem notícias na página {page} ou erro na requisição.")
            break
        news.extend(articles)

    # Exibir os títulos das notícias coletadas
    for idx, article in enumerate(news, 1):
        print(f"{idx}. {article['title']}")

import yfinance as yf
import mysql.connector
from datetime import datetime

# Configuração do Banco de Dados MySQL
""" conn = mysql.connector.connect(
    host="localhost",
    user="seu_usuario",
    password="sua_senha",
    database="dados_financeiros"
)
cursor = conn.cursor() """

# Lista de Tickers para Coleta
tickers = ["HGLG11.SA"]  # Apple e Microsoft

# Função para Coletar e Inserir Dados
def coletar_dados(tickers):
    dados = yf.download(tickers, start="2023-01-01", end="2023-12-31")

    # print(dados.head())  # Visualiza os primeiros registros

    acao = yf.Ticker("AAPL")

    info = acao.info

    # print("Nome da empresa:", info['longName'])
    # print("Setor:", info['sector'])
    # print("Market Cap:", info['marketCap'])

    dividendos = acao.dividends
    splits = acao.splits

    print("Dividendos:")
    print(dividendos)
    print("\nSplits:")
    print(splits)

    for ticker in tickers:
        # Coleta de dados históricos (últimos 30 dias)
        dados = yf.Ticker(ticker).history(period="1mo")
        
        # Preparar os dados para inserção no banco
        for data, linha in dados.iterrows():
            sql = """
                INSERT INTO precos_acoes (ticker, data, abertura, alta, baixa, fechamento, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            valores = (
                ticker,
                data.date(),
                linha["Open"],
                linha["High"],
                linha["Low"],
                linha["Close"],
                linha["Volume"]
            )
            """ cursor.execute(sql, valores)
        conn.commit() """
        # print(f"Dados do ticker {ticker.info} inseridos com sucesso.")

# Executa a Coleta de Dados
try:
    coletar_dados(tickers)
except Exception as e:
    print(f"Erro ao coletar ou inserir dados: {e}")
""" finally:
    cursor.close()
    conn.close() """

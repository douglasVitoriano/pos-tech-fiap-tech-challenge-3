import requests
import mysql.connector
from datetime import datetime

# Substitua pelos seus dados do RDS
host = 'fiap-db-instance.cbmhdpqu8pui.us-east-1.rds.amazonaws.com'  # Endereço do RDS
database = 'fiap_db'               # Nome do banco de dados
user = 'fiap'                     # Nome do usuário
password = 'fiap12345'                   # Senha do usuário

    # Estabelecendo a conexão com o banco de dados
conn = mysql.connector.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
if conn.is_connected():
    print(f'Conectado ao banco de dados {database} no RDS.')

cursor = conn.cursor() 

# URL da API CoinGecko
url = "https://api.coingecko.com/api/v3/coins/markets"
params = {
    "vs_currency": "usd",  # Moeda de referência (USD, BRL, etc.)
    "order": "market_cap_desc",  # Ordenação por capitalização de mercado
    "per_page": 10,  # Número de moedas por página
    "page": 1,  # Página inicial
    "sparkline": False  # Sem gráfico de tendência
}

# Função para coletar dados
def coletar_dados_cripto():
    response = requests.get(url, params=params)
    response.raise_for_status()

    dados = response.json()

    criptos = []
    for moeda in dados:
        nome = moeda["id"]
        preco = moeda["current_price"]
        volume = moeda["total_volume"]
        timestamp = datetime.now()

        criptos.append((nome, preco, volume, timestamp))

    # Inserir dados no banco de dados
    """ cursor.executemany(
        "INSERT INTO precos_cripto (moeda, preco, volume, timestamp) VALUES (%s, %s, %s, %s)",
        criptos
    )
    conn.commit() """
    print(f"{criptos} registros criptos.")

# Executar coleta
try:
    coletar_dados_cripto()
except Exception as e:
    print(f"Erro ao coletar dados: {e}")
""" finally:
    cursor.close()
    conn.close() """

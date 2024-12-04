# Análise de Sentimentos e Previsão de Preços de Ações

Este projeto realiza a análise de sentimentos em notícias financeiras e prevê o preço das ações com base nessa análise. Utiliza a API do NewsAPI para obter notícias, o modelo FinBERT para classificar os sentimentos das notícias, e o modelo XGBoost para prever os preços das ações.

## Estrutura do Projeto

1. **Importar Bibliotecas**: Importar as bibliotecas necessárias, como `pandas`, `newsapi`, `transformers`, `datetime`, `yfinance`, `xgboost`, e `matplotlib`.
2. **Inicializar a API do NewsAPI**: Configurar a API do NewsAPI com a chave de API.
3. **Obter Notícias**: Definir uma função para obter notícias de um ticker específico em um intervalo de datas.
4. **Classificar Sentimentos**: Definir uma função para classificar o sentimento das notícias usando o modelo FinBERT.
5. **Obter Dados Históricos de Preços**: Usar a biblioteca `yfinance` para obter dados históricos de preços das ações.
6. **Combinar Dados**: Combinar os dados de preços das ações com os sentimentos das notícias.
7. **Treinar Modelo**: Treinar um modelo XGBoost para prever os preços das ações.
8. **Avaliar Modelo**: Avaliar a precisão do modelo e visualizar os resultados.

## Requisitos

- Python 3.7 ou superior
- Bibliotecas Python:
  - pandas
  - requests
  - newsapi-python
  - transformers
  - torch
  - yfinance
  - xgboost
  - scikit-learn
  - matplotlib

## Instalação

1. Clone o repositório:
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio

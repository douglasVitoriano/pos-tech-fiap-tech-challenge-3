# Real Estate Pricing System

## Descrição
Este projeto tem como objetivo estimar preços de aluguel e venda de imóveis com base em suas características, utilizando algoritmos de aprendizado de máquina. Ele inclui um modelo treinado, uma API para previsões e um dashboard interativo para visualização e análise de dados.

---

## Funcionalidades
- **Treinamento de Modelo**: Treina modelos de regressão para prever preços de imóveis com base em dados históricos.
- **API Flask**: Permite prever preços através de chamadas RESTful.
- **Dashboard Streamlit**: Exibe insights sobre os dados, estatísticas e permite previsões interativas.
- **Gráficos Interativos**: Inclui gráficos de distribuição de preços e relações entre características dos imóveis.

---

## Tecnologias Utilizadas
- **Linguagem**: Python 3.12
- **Bibliotecas**:
  - Machine Learning: `scikit-learn`
  - Manipulação de Dados: `pandas`, `numpy`
  - API: `Flask`
  - Dashboard: `Streamlit`
  - Visualização: `matplotlib`
  - Persistência de Modelos: `joblib`
- **Ambiente Virtual**: `venv`

---

## Como Configurar o Projeto

### Pré-requisitos
- Python 3.8 ou superior.
- Git instalado na máquina.

### Passo a Passo

1. Clone o repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd real-estate-pricing-system
   ```

2. Crie e ative o ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Treine o modelo e gere os arquivos necessários:
   ```bash
   python coleta_dados.py
   ```

5. Inicie o dashboard:
   ```bash
   streamlit run dashboard.py
   ```

---

## Estrutura do Projeto
```
pos-tech-fiap-tech-challenge-3/
├── coleta_dados_melhorado.py        # Script de treinamento do modelo
├── dashboard.py                     # Dashboard interativo em Streamlit
├── requirements.txt                 # Dependências do projeto
├── real_estate_model.pkl            # Modelo treinado
├── feature_columns.pkl              # Colunas usadas no treinamento
├── real_estate_data.csv             # Conjunto de dados para análise
├── venv/                            # Ambiente virtual
```

---

## Uso

### Usando a API Flask

1. Inicie o servidor da API:
   ```bash
   python coleta_dados.py
   ```

2. Faça uma requisição POST para o endpoint `/predict`:
   ```bash
   curl -X POST http://127.0.0.1:5000/predict \
   -H "Content-Type: application/json" \
   -d '{
       "area": 120,
       "rooms": 3,
       "location_Downtown": 1,
       "location_Rural": 0,
       "location_Suburb": 0,
       "price_per_sq_meter": 25.0,
       "room_density": 0.025
   }'
   ```

3. Resposta esperada:
   ```json
   {
       "predicted_price": 3200.0
   }
   ```

## Licença
Este projeto está licenciado sob a MIT License. Consulte o arquivo LICENSE para mais detalhes.


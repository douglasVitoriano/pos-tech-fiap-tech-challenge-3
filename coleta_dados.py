import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import joblib
import kaggle
import os

def download_and_process_dataset():
    dataset_name = 'ahmedshahriarsakib/usa-real-estate-dataset'  

    # path do download
    download_path = './kaggle_data'

    # Baixando e descompactando os arquivos do dataset
    print(f"Baixando o dataset {dataset_name}...")
    kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)

    # checar conteudo baixado
    print("Conteúdo do diretório de download:", os.listdir(download_path))

    # carregando csv
    csv_file_path = os.path.join(download_path, 'realtor-data.zip.csv')  # Verifique o nome correto do arquivo após descompactar

    # Carregar o CSV em um DataFrame
    print(f"Carregando os dados de {csv_file_path}...")
    df = pd.read_csv(csv_file_path, nrows=100)

    # Exibir as primeiras linhas do dataset
    print("Exibindo as primeiras linhas do dataset:")
    print(df.head())

    # Salve o DataFrame em um novo arquivo CSV
    output_csv_path = './output/real_estate_data_raw.csv'

    # Crie o diretório de saída caso não exista
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Salve o DataFrame em um arquivo CSV
    df.to_csv(output_csv_path, index=False)

def process_train():

    df = pd.read_csv("/home/ayres/Documents/estudos/pos-tech-fiap-tech-challenge-3/output/real_estate_data_raw.csv", sep=",")

    
    df = df.drop('brokered_by', axis=1)
    df = df.drop('status', axis=1)
    df = df.drop('bath', axis=1)
    df = df.drop('street', axis=1)
    df = df.drop('state', axis=1)
    df = df.drop('zip_code', axis=1)
    df = df.drop('acre_lot', axis=1)
    df = df.drop('prev_sold_date', axis=1)

    #tratamento city
    df['city'] = df['city'].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
    df = df.drop_duplicates(subset='city').reset_index(drop=True)
    df = df.dropna()
    
    print("Exibindo as primeiras linhas do dataset tratado:")
    print(df.head())
    return df

# Feature Engineering
def feature_engineering(df):
    df["price_per_sq_meter"] = df["price"] / df["house_size"]
    df["room_density"] = df["bed"] / df["house_size"]
    df["house_size"] = df["house_size"] * 0.0929
    return df

get_data = download_and_process_dataset()
processing = process_train()
df = feature_engineering(processing)

# Step 3: Data Preprocessing
def preprocess_data(df):
    df = pd.get_dummies(df, columns=["city"], drop_first=True)  # One-hot encoding for 'location'
    scaler = StandardScaler()
    numerical_features = ["house_size", "bed", "price_per_sq_meter", "room_density"]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

df_processed = preprocess_data(df)
X = df_processed.drop("price", axis=1)
y = df_processed["price"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Development with Hyperparameter Tuning and Comparison
# Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}

gb_model = GradientBoostingRegressor()
gb_grid_search = GridSearchCV(gb_model, param_grid_gb, scoring='neg_mean_absolute_error', cv=5)
gb_grid_search.fit(X_train, y_train)

# Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7]
}

rf_model = RandomForestRegressor()
rf_grid_search = GridSearchCV(rf_model, param_grid_rf, scoring='neg_mean_absolute_error', cv=5)
rf_grid_search.fit(X_train, y_train)

# Ridge Regression
param_grid_ridge = {
    'alpha': [0.1, 1.0, 10.0]
}

ridge_model = Ridge()
ridge_grid_search = GridSearchCV(ridge_model, param_grid_ridge, scoring='neg_mean_absolute_error', cv=5)
ridge_grid_search.fit(X_train, y_train)

# Lasso Regression
param_grid_lasso = {
    'alpha': [0.1, 1.0, 10.0]
}

lasso_model = Lasso(max_iter=10000)
lasso_grid_search = GridSearchCV(lasso_model, param_grid_lasso, scoring='neg_mean_absolute_error', cv=5)
lasso_grid_search.fit(X_train, y_train)

# Compare Models
best_models = {
    'Gradient Boosting': gb_grid_search.best_estimator_,
    'Random Forest': rf_grid_search.best_estimator_,
    'Ridge Regression': ridge_grid_search.best_estimator_,
    'Lasso Regression': lasso_grid_search.best_estimator_
}

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} Evaluation:\nMAE: {mae}\nR2 Score: {r2}\n")

# Save the Best Model
final_model = gb_grid_search.best_estimator_  # Select Gradient Boosting as the final model

# Salvar o modelo treinado
joblib.dump(final_model, "real_estate_model.pkl")

# Salvar as colunas usadas no treinamento
feature_columns = list(X.columns)
joblib.dump(feature_columns, "feature_columns.pkl")

# Salvar os dados do DataFrame principal no treinamento
df.to_csv("real_estate_data.csv", index=False)

# Step 6: API Implementation
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input data
        input_data = request.json
        input_df = pd.DataFrame([input_data])
        input_processed = preprocess_data(input_df)
        
        # Ensure the columns match
        missing_cols = set(X.columns) - set(input_processed.columns)
        for col in missing_cols:
            input_processed[col] = 0
        
        input_processed = input_processed[X.columns]  # Reorder columns

        # Load model and predict
        model = joblib.load("real_estate_model.pkl")
        prediction = model.predict(input_processed)
        
        return jsonify({"predicted_price": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)


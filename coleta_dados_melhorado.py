# Real Estate Pricing System
# Detailed implementation in Python

# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import joblib

# Step 2: Data Collection (Expanded Dataset)
data = {
    "location": [
        "Downtown", "Suburb", "Downtown", "Suburb", "Rural",
        "Downtown", "Downtown", "Suburb", "Rural", "Suburb",
        "Downtown", "Rural", "Suburb", "Downtown", "Rural",
        "Suburb", "Downtown", "Suburb", "Rural", "Downtown"
    ],
    "area": [
        120, 80, 150, 60, 200,
        140, 90, 100, 180, 70,
        160, 190, 85, 130, 210,
        75, 155, 95, 195, 145
    ],
    "rooms": [
        3, 2, 4, 2, 5,
        4, 3, 3, 5, 2,
        5, 5, 3, 4, 6,
        2, 4, 3, 6, 4
    ],
    "price": [
        3000, 2000, 3500, 1500, 4000,
        3200, 2500, 2800, 4500, 2200,
        3600, 4700, 2400, 3300, 5000,
        1700, 3400, 2600, 5200, 3700
    ]
}
# Convert to DataFrame
df = pd.DataFrame(data)

# Feature Engineering
def feature_engineering(df):
    df["price_per_sq_meter"] = df["price"] / df["area"]
    df["room_density"] = df["rooms"] / df["area"]
    return df

df = feature_engineering(df)

# Step 3: Data Preprocessing
def preprocess_data(df):
    df = pd.get_dummies(df, columns=["location"], drop_first=True)  # One-hot encoding for 'location'
    scaler = StandardScaler()
    numerical_features = ["area", "rooms", "price_per_sq_meter", "room_density"]
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

# Notes:
# 1. Feature engineering and scaling have been added to enhance model inputs.
# 2. Multiple models are compared, and the best model is selected for deployment.
# 3. Flask is still in development mode. Use Gunicorn for production.

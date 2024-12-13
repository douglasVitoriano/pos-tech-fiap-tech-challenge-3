# Real Estate Pricing System
# Detailed implementation in Python

# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
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

# Step 3: Data Preprocessing
def preprocess_data(df):
    df = pd.get_dummies(df, columns=["location"], drop_first=True)  # One-hot encoding for 'location'
    return df

df_processed = preprocess_data(df)
X = df_processed.drop("price", axis=1)
y = df_processed["price"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Development
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:\nMAE: {mae}\nR2 Score: {r2}")

# Perform Cross-Validation
cv_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
print(f"Cross-validated MAE: {-np.mean(cv_scores)}")

# Save the Model
joblib.dump(model, "real_estate_model.pkl")

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
# 1. The dataset has been expanded to ensure proper testing and evaluation.
# 2. Cross-validation has been added to improve performance evaluation.
# 3. Flask is still in development mode. Use Gunicorn for production.
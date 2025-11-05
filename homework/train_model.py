# train_model.py

"""Build, deploy and access a model using scikit-learn"""

import os
import pickle

import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore

# Cargar datos
df = pd.read_csv("files/input/house_data.csv", sep=",")

# Features corregidas (observa 'bedrooms' en lugar de 'bedroomss')
features_list = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]

# Eliminar filas con valores faltantes en las columnas usadas
df = df.dropna(subset=features_list + ["price"])

X = df[features_list]
y = df["price"]

# División train/test para evaluar el modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimator = LinearRegression()
estimator.fit(X_train, y_train)

# Evaluación simple
preds = estimator.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MSE: {mse:.2f}, R2: {r2:.3f}")

# Asegurar que exista el directorio de salida
os.makedirs("homework", exist_ok=True)

# Guardar modelo + lista de features para despliegue/futuros usos
with open("homework/house_predictor.pkl", "wb") as file:
    pickle.dump({"model": estimator, "features": features_list}, file)





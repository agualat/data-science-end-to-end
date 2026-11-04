"""Ejercicio 3: los primeros modelos del proyecto (tarifa de viajes en taxi).

Usa la preparacion y la division del pipeline de la Semana 9, Sesion 2, y
entrena una linea base, un modelo lineal (con y sin regularizacion) y un
ensamblaje. Escribe resultados_proyecto.csv.

    python3 modelos_proyecto.py
"""

import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

AQUI = Path(__file__).parent
datos = pd.read_csv(AQUI.parent / "starter" / "viajes_features.csv")

OBJETIVO = "tarifa"
NUMERICAS = ["distancia_km", "pasajeros"]
YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]
CATEGORICAS = ["zona_origen", "zona_destino", "metodo_pago"]

X = datos[NUMERICAS + YA_LISTAS + CATEGORICAS]
y = datos[OBJETIVO]
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)


def preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
    ])


resultados = []


def evaluar(nombre, modelo, grado_interacciones=None):
    pasos = [("preparar", preparacion())]
    if grado_interacciones is not None:
        pasos.append(("interacciones", PolynomialFeatures(degree=grado_interacciones, include_bias=False)))
    pasos.append(("modelo", modelo))
    pipeline = Pipeline(pasos)
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    mae_ent = mean_absolute_error(y_ent, pipeline.predict(X_ent))
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
    resultados.append({
        "modelo": nombre,
        "mae_entrenamiento": round(mae_ent, 4),
        "mae_prueba": round(mae_prueba, 4),
        "brecha": round(mae_prueba - mae_ent, 4),
        "segundos": round(segundos, 3),
    })
    return pipeline


# Linea base: la mediana de la tarifa. La metrica es el MAE, y la constante
# que minimiza el error absoluto es la mediana (no la media, que ademas se
# jala hacia arriba por las tarifas fijas de 70 dolares del aeropuerto).
evaluar("linea base (mediana)", DummyRegressor(strategy="median"))

# Lineal sin interacciones: pocas columnas, no hace falta regularizar mucho.
evaluar("ridge alpha=1", Ridge(alpha=1.0))

# Lineal con interacciones de grado 2: mas columnas que pueden capturar
# tarifa fija x distancia, sin y con regularizacion.
evaluar("lineal grado 2", LinearRegression(), grado_interacciones=2)
evaluar("ridge grado 2 alpha=10", Ridge(alpha=10.0), grado_interacciones=2)

# Ensamblaje
evaluar("gradient boosting", GradientBoostingRegressor(random_state=42))

tabla = pd.DataFrame(resultados)
tabla.to_csv(AQUI / "resultados_proyecto.csv", index=False)
print(tabla.to_string(index=False))

"""Ejercicio 3: ensamblaje, red neuronal y LLM sobre el proyecto (tarifa
de viajes en taxi).

Entrena con la preparacion y la division de siempre el mejor ensamblaje
de la sesion pasada (gradient boosting) y una red neuronal simple, y
registra su MAE de prueba, los segundos de entrenamiento y los parametros
de la red. Despues compara los dos modelos con el LLM sobre 10 viajes de
prueba, con las respuestas de ChatGPT de ../starter/respuestas_llm.txt.

    python3 comparar_proyecto.py
"""

import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AQUI = Path(__file__).parent
STARTER = AQUI.parent / "starter"
datos = pd.read_csv(STARTER / "viajes_features.csv")

NUMERICAS = ["distancia_km", "pasajeros"]
YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]
CATEGORICAS = ["zona_origen", "zona_destino", "metodo_pago"]

X = datos[NUMERICAS + YA_LISTAS + CATEGORICAS]
y = datos["tarifa"]
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Las 10 filas de prueba para el LLM: las 10 primeras de la muestra del
# Ejercicio 2, que ya tienen respuesta de ChatGPT con un prompt de 30
# viajes de entrenamiento como ejemplos.
muestra = X_prueba.sample(20, random_state=7)
diez = muestra.iloc[:10]
y_diez = y_prueba.loc[diez.index]
with open(STARTER / "respuestas_llm.txt") as archivo:
    llm_diez = [float(linea) for linea in archivo if linea.strip()][:10]
caracteres_prompt = len((STARTER / "prompt_llm.txt").read_text())


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


def evaluar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    resultados.append({
        "modelo": nombre,
        "mae_prueba": round(mean_absolute_error(y_prueba, pipeline.predict(X_prueba)), 4),
        "mae_10_viajes": round(mean_absolute_error(y_diez, pipeline.predict(diez)), 4),
        "segundos": round(segundos, 3),
        "parametros": None,
    })
    return pipeline


evaluar("gradient boosting", GradientBoostingRegressor(random_state=42))
red = evaluar("red neuronal", MLPRegressor(hidden_layer_sizes=(64,), max_iter=1000, random_state=42))
modelo_red = red.named_steps["modelo"]
resultados[-1]["parametros"] = (sum(matriz.size for matriz in modelo_red.coefs_)
                                + sum(vector.size for vector in modelo_red.intercepts_))

# ChatGPT (GPT-5.5): 5 dolares por millon de tokens de entrada, un token
# cada 4 caracteres, un prompt por cada 20 viajes.
PRECIO_POR_MILLON_DE_TOKENS = 5.00
costo_10000 = (10000 / 20) * (caracteres_prompt / 4) * PRECIO_POR_MILLON_DE_TOKENS / 1_000_000
resultados.append({
    "modelo": "LLM (ChatGPT)",
    "mae_prueba": None,
    "mae_10_viajes": round(mean_absolute_error(y_diez, llm_diez), 4),
    "segundos": None,
    "parametros": None,
})

tabla = pd.DataFrame(resultados)
tabla.to_csv(AQUI / "comparacion_proyecto.csv", index=False)
print(tabla.to_string(index=False))
print(f"\nCosto del LLM para 10 000 viajes al mes: {costo_10000:.2f} dolares")

"""Servicio de prediccion de la tarifa: el modelo detras de una API.

Cuando el pasajero elige su destino, la app manda el viaje como JSON a
/predecir y recibe la tarifa estimada en la misma respuesta. Para levantarlo
en el Codespace, desde esta carpeta y despues de python entrenar.py:

    uvicorn servicio:app --port 8000

La documentacion interactiva queda en el puerto 8000, ruta /docs.
"""

import os
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from features_taxis import agregar_features
from preparacion_taxis import CATEGORICAS, NUMERICAS, YA_LISTAS

RUTA_MODELO = os.environ.get("RUTA_MODELO", "modelo_tarifa.joblib")

# El modelo se carga una sola vez, al arrancar el servicio, y no en cada
# peticion: leer el archivo tarda mucho mas que predecir.
modelo = None
if os.path.exists(RUTA_MODELO):
    modelo = joblib.load(RUTA_MODELO)

app = FastAPI(title="Tarifa estimada de taxis")


class Viaje(BaseModel):
    """Lo que la app sabe de un viaje antes de que empiece."""
    fecha_hora_inicio = datetime
    zona_origen: str
    zona_destino: str
    distancia_km: float | None
    pasajeros: float | None
    metodo_pago:str | None


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None}


@app.post("/predecir")
def predecir(viaje: Viaje):
    # TODO 2: convierte el viaje en un DataFrame de una fila
    # (viaje.model_dump() lo da como diccionario), agrega las features y
    # estima la tarifa con el modelo.
    dto = pd.DataFrame([viaje.model_dump()])
    datos_transformados = agregar_features(dto[CATEGORICAS + YA_LISTAS + NUMERICAS])
    tarifa = modelo.predict(datos_transformados[CATEGORICAS + YA_LISTAS + NUMERICAS])

    return {"tarifa_estimada": round(float(tarifa), 2)}

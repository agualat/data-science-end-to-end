import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)

# TODO: reemplaza por la ruta del archivo que tu pipeline de la Semana 7
# deja listo para modelar (una tabla del warehouse exportada a CSV, o el
# CSV limpio de tu etapa de calidad).
datos = pd.read_csv("../starter/viajes_taxi.csv")

# TODO: reemplaza por el nombre de la columna que tu modelo va a predecir.
OBJETIVO = "tarifa"

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))

# Paso 2: distribucion de la variable objetivo.
plt.figure()
plt.hist(datos[OBJETIVO], bins=40)
plt.title(OBJETIVO)
plt.savefig("figuras/distribucion_objetivo.png")
plt.close()

# Paso 3: atipicos de la variable objetivo con la regla del IQR.
q1 = datos[OBJETIVO].quantile(0.25)
q3 = datos[OBJETIVO].quantile(0.75)
iqr = q3 - q1
fuera = datos[(datos[OBJETIVO] < q1 - 1.5 * iqr) | (datos[OBJETIVO] > q3 + 1.5 * iqr)]
print(f"Atipicos de {OBJETIVO} segun IQR: {len(fuera)}")
print(fuera.head(10))

# Paso 4: la variable que mas deberia influir en la tarifa es la distancia.
# Primero quito los errores confirmados (tarifa 999.99 y duracion negativa)
# para que no aplasten el grafico.
limpios = datos[(datos[OBJETIVO] != 999.99) & (datos["duracion_min"] >= 0)]
print("Filas sin errores confirmados:", len(limpios))

plt.figure()
plt.scatter(limpios["distancia_km"], limpios[OBJETIVO], s=5)
plt.xlabel("distancia_km")
plt.ylabel(OBJETIVO)
plt.title("Distancia contra tarifa")
plt.savefig("figuras/distancia_vs_objetivo.png")
plt.close()

# Paso 5: consultas que confirman los hallazgos.

# Hallazgo 1: los errores de captura distorsionan la correlacion.
print("Correlacion distancia-tarifa con errores:",
      round(datos["distancia_km"].corr(datos[OBJETIVO]), 3))
print("Correlacion distancia-tarifa sin errores:",
      round(limpios["distancia_km"].corr(limpios[OBJETIVO]), 3))

# Hallazgo 2: la tarifa fija de JFK no sigue la distancia.
es_jfk = (limpios["zona_origen"] == "JFK Airport") | (limpios["zona_destino"] == "JFK Airport")
print("Viajes JFK:", es_jfk.sum(), "| con tarifa 70.00:", (limpios.loc[es_jfk, OBJETIVO] == 70).sum())
print("Correlacion distancia-tarifa sin JFK:",
      round(limpios.loc[~es_jfk, "distancia_km"].corr(limpios.loc[~es_jfk, OBJETIVO]), 3))

# Hallazgo 3: distancias en 0 y faltantes en las entradas.
print("Viajes con distancia 0:", (limpios["distancia_km"] == 0).sum())
print(limpios.loc[limpios["distancia_km"] == 0, OBJETIVO].describe().round(2))
print("Pasajeros vacios:", limpios["pasajeros"].isna().sum(),
      "| pasajeros 0:", (limpios["pasajeros"] == 0).sum(),
      "| metodo_pago vacio:", limpios["metodo_pago"].isna().sum())

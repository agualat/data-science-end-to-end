import os

import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('figuras', exist_ok= True)

data = pd.read_csv('./viajes_taxi.csv')

print("Analisis exploratorio descriptivo")

data.info()

print(data.dtypes)
print(data.isna().sum())
print(data.describe().round(2))

plt.figure()
plt.hist(data['tarifa'], bins=50)
plt.title("Distribucion de la tarifa")
plt.xlabel('USD')
plt.ylabel('Numero de viajes')
plt.savefig('./figuras/hist_tarifa.png')
plt.close()

plt.figure()
plt.boxplot(data['tarifa'])
plt.title("Boxplot de la tarifa")
plt.savefig('./figuras/boxplot_tarifa.png')
plt.close()

q1 = data['tarifa'].quantile(0.25)
q3 = data['tarifa'].quantile(0.75)

iqr = q3 - q1

limite_superior = q3 + 1.5  * iqr
limite_inferior = q1 - 1.5  * iqr

atipicos = data.query(f'tarifa < {limite_inferior} or tarifa > {limite_superior}')
sin_atipicos = data.query(f'tarifa >= {limite_inferior} and tarifa <= {limite_superior}')

print(f"Atipicos de tarifa segun IQR: {len(atipicos)}")
print(atipicos['tarifa'].value_counts().head())

plt.figure()
plt.scatter(data['distancia_km'], data['tarifa'])
plt.title("Dispersion de distancia KM vs tarifa")
plt.xlabel('Distancia (km)')
plt.ylabel('Tarifa (USD)')
plt.savefig('./figuras/dispersion_distancia_vs_tarifa.png')
plt.close()

print(data.corr(numeric_only=True)['tarifa'].round(3))

plt.figure()
plt.scatter(sin_atipicos['distancia_km'], sin_atipicos['tarifa'])
plt.title("Dispersion de distancia KM vs tarifa (sin atipicos)")
plt.xlabel('Distancia (km)')
plt.ylabel('Tarifa (USD)')
plt.savefig('./figuras/dispersion_distancia_vs_tarifa_sin_atipicos.png')
plt.close()

print(sin_atipicos.corr(numeric_only=True)['tarifa'].round(3))

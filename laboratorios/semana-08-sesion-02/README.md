# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas ya lo tienes desde la Semana 4. Instala matplotlib:

```
pip install matplotlib
```

Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. También puedes correr el mismo código en un notebook de Jupyter, donde el gráfico aparece debajo de la celda.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

Qué contienen los datos (filas, columnas, tipos), cuántos faltantes hay y dónde, cómo se distribuye cada variable, qué valores atípicos o imposibles aparecen, y cómo se relacionan las variables entre sí y con el objetivo. Con eso se decide qué limpiar, qué transformar y qué supuestos del modelo no se cumplen, antes de gastar tiempo entrenando.

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

Análisis univariado: estadísticas descriptivas (`describe()`), histogramas y boxplots para ver la distribución y los atípicos de una variable. Análisis bivariado: gráficos de dispersión para dos variables numéricas, promedios por grupo (o boxplots por categoría) para una categórica contra una numérica, y la matriz de correlación.

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

El modelo se distorsiona: 15 valores de 999.99 frente a una tarifa típica de ~23 USD suben la media de 22.6 a 31.4 y la desviación estándar a 71. En una regresión lineal (que minimiza el error cuadrático) esos puntos jalan la recta hacia arriba y el modelo sobreestima las tarifas.

## Ejercicio 1: resumen, distribución y atípicos

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv) tiene 3000 viajes en taxi de Nueva York: fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Primera mirada: qué contiene, cómo se distribuye la tarifa y qué valores están fuera de lo esperable.

**Respuesta 1, antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

- El máximo de `tarifa` = 999.99, cuando el 75% de los viajes cuesta 32.55 o menos. Es un valor centinela típico de error de captura.
- El mínimo de `duracion_min` = -74.0: una duración negativa es físicamente imposible.

Confirmado: 15 viajes tienen tarifa 999.99 y 10 tienen duración negativa.

**Respuesta 2:** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

No. Con Q1 = 13.99 y Q3 = 32.55, el límite superior es 60.40 y el IQR marca 334 viajes. Solo 15 son errores (999.99); 302 tienen una tarifa de exactamente 70.00, y casi todos van o vienen de JFK Airport (la tarifa fija del aeropuerto). Son viajes reales y largos (~25 km), no errores. Lo averigüé mirando los atípicos en vez de borrarlos: atipicos['tarifa'].value_counts() muestra el valor repetido, y value_counts() sobre zona_origen/zona_destino muestra que se concentran en JFK.

**La formulación completa, tal como la resuelvo yo:**

Script: [starter/EDA.py]. Lee viajes_taxi.csv, imprime info(), tipos y faltantes, guarda el histograma y el boxplot de la tarifa, calcula los límites del IQR, imprime los atípicos y grafica distancia contra tarifa. Las figuras se guardan en figuras/.

Para correrlo con Docker (las figuras quedan en data-contenedor/):

```
docker compose up --build
```

Lo que encontró:

- 3000 filas, 10 columnas. Faltantes: 120 en pasajeros y 60 en metodo_pago.
- La tarifa tiene mediana 22.60 pero media 31.39 y desviación 70.99: el histograma queda aplastado por los 15 valores de 999.99.
- duracion_min tiene 10 valores negativos (mínimo -74.0).
- El IQR marca 334 atípicos de tarifa: 15 errores (999.99), 302 tarifas fijas de 70.00 (JFK) y el resto viajes largos reales.
- Errores confirmados para quitar: tarifa 999.99 y duración negativa (quedan 2975 viajes).

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/explorar_relaciones.py`](./plantilla-ejercicio2/explorar_relaciones.py) y córrela desde su carpeta (`cd plantilla-ejercicio2`, `python3 explorar_relaciones.py`): lee el mismo `starter/viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0.

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

La tendencia general es lineal y fuerte (correlación 0.96), pero hay una línea horizontal en 70.00: 300 viajes con la misma tarifa sin importar la distancia (de 0 a 45 km, mediana 25.6 km), todos con origen o destino en JFK Airport. Es la tarifa fija del aeropuerto, una regla de negocio que una recta no captura. También hay 25 viajes con distancia 0 y tarifas entre 7.23 y 70.00 (mediana 25.18): probablemente el GPS no registró la distancia.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

Con tarjeta la propina promedio es 4.73 y ningún viaje tiene propina 0 (0 de 2027). Con efectivo la propina es 0 en los 888 viajes. Esto describe al sistema, no a los pasajeros: la propina en efectivo se entrega en mano y el taxímetro no la registra. No significa que quien paga en efectivo no deje propina. Si el modelo usa propina, aprendería un artefacto del registro; además hay 60 viajes sin método de pago.

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. La correlación es alta (0.956, casi igual a la distancia con 0.960), pero la duración solo se conoce cuando el viaje termina. Usarla sería fuga de datos (data leakage): el modelo se vería muy bien en la evaluación y fallaría en producción, donde ese dato no existe al pedir el taxi. Lo mismo vale para propina. Sí se pueden usar la distancia estimada de la ruta, las zonas de origen y destino, y la hora de inicio.

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar, y la plantilla [`plantilla-eda/explorar_proyecto.py`](./plantilla-eda/explorar_proyecto.py).

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3.

Dataset usado: [starter/viajes_taxi.csv], con objetivo tarifa (script: [`plantilla-eda/explorar_proyecto.py`](./plantilla-eda/explorar_proyecto.py)).

- 3000 filas y 10 columnas.
- Faltantes: pasajeros 120 y metodo_pago 60. El resto de las columnas está completo.
- El IQR marcó 334 atípicos en tarifa (límite superior 60.40): 302 son la tarifa fija de 70.00 y 15 son el error 999.99.

### Paso 2: relaciones (10 minutos)

Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.

Dispersión de distancia_km contra tarifa (figuras/distancia_vs_objetivo.png), después de quitar los errores confirmados (quedan 2975 viajes). Muestra una relación lineal muy fuerte, más una línea horizontal en 70.00 (los viajes de JFK) y un grupo de puntos pegados a distancia 0. Las consultas del paso 5 miden la correlación con y sin errores, cuentan los viajes de JFK y revisan las distancias en 0 y los faltantes.

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

Para cada hallazgo, escribe qué encontraste, el número o el gráfico que lo confirma, y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar).

**Hallazgo 1:** los errores de captura esconden la relación principal. Con las 15 tarifas de 999.99 y las 10 duraciones negativas, la correlación distancia-tarifa es 0.229. Sin ellas sube a 0.960. En el modelo: quito esas 25 filas antes de entrenar. Si no, un modelo lineal casi no aprendería de la distancia.

**Hallazgo 2:** la tarifa de JFK no depende de la distancia. 300 de los 373 viajes de JFK cuestan exactamente 70.00. Fuera de JFK la correlación distancia-tarifa sube a 0.969. En el modelo: no quito esas filas, porque son viajes reales. Agrego una variable es_jfk (origen o destino en JFK) para que el modelo aprenda la tarifa fija, o modelo esos viajes con la regla de negocio.

**Hallazgo 3:** hay entradas vacías o imposibles. 25 viajes tienen distancia 0, con tarifas de 7.23 a 70.00 (mediana 25.18), así que la distancia no se registró. En pasajeros hay 119 vacíos y 19 en 0, y 60 viajes no tienen metodo_pago. En el modelo: marco la distancia 0 como faltante y la imputo (por ejemplo con la mediana de su par de zonas) o quito esas filas. Imputo pasajeros con 1, la moda. No uso duracion_min ni propina, porque no se conocen al pedir el taxi.

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Con feature engineering, creando columnas nuevas a partir de las que ya existen:

- Tarifa fija de JFK: una variable binaria es_jfk = 1 si zona_origen o zona_destino es "JFK Airport". Así el modelo puede aprender que esos viajes cuestan 70.00 sin importar la distancia (o se modela aparte con la regla de negocio).
- Hora del día: extraer la hora de fecha_hora_inicio (pd.to_datetime(...).dt.hour), y a partir de ella variables como hora_pico (por ejemplo 7-10 y 16-19 h), es_noche o el día de la semana / es_fin_de_semana, que capturan tráfico y recargos.

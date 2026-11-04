# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el pipeline de tu proyecto de la Semana 9, Sesión 2.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un ensamblaje combina las predicciones de varios modelos (normalmente árboles) en una sola. En **bagging** (p. ej. random forest) cada modelo se entrena en paralelo e independiente sobre una muestra bootstrap de los datos, y al final se promedian (o se vota). En **boosting** (p. ej. gradient boosting) los modelos se entrenan en secuencia: cada nuevo árbol se ajusta a los errores (residuos) que dejaron los anteriores, y la predicción es la suma ponderada de todos.

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Porque cada árbol se equivoca de forma distinta: tiene poco sesgo pero mucha varianza, y sus errores dependen de la muestra que le tocó. Si los errores no están muy correlacionados, al promediarlos se cancelan parcialmente y la varianza del promedio baja, mientras que el sesgo se mantiene. El mejor árbol solo sigue teniendo toda su varianza.

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está **sobreajustando**: memorizó el ruido y los casos particulares del entrenamiento en lugar de aprender el patrón general, así que no generaliza a datos que no vio. La brecha entre el error de prueba y el de entrenamiento es la señal.

## Ejercicio 1: línea base, un árbol y un bosque

Archivo: [`starter/ejercicio1_linea_base_y_ensamblaje.py`](./starter/ejercicio1_linea_base_y_ensamblaje.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. El script ya trae la preparación de la Semana 9 y una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-01/starter
python3 ejercicio1_linea_base_y_ensamblaje.py
```

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Esperaba un error grande, del orden de la dispersión de la tarifa alrededor de su mediana (unos 12–13 dólares). Sirve como piso de referencia: cualquier modelo que use columnas tiene que ganarle por mucho; si no, no está aprendiendo nada útil.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

En entrenamiento, prácticamente 0 (cada hoja es un viaje, lo reproduce exacto). En prueba, un error claramente mayor, porque memorizó el ruido: una brecha grande.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad, la brecha se reduce (sube el error de entrenamiento y baja el de prueba). Con 200 árboles promediados espero un error de prueba bajo y una brecha menor que la del árbol sin límite, porque el promedio reduce la varianza aunque cada árbol esté sobreajustado.

**Lo que confirmó la tabla, TODO por TODO:**

| Modelo | MAE entrenamiento | MAE prueba | Brecha |
|---|---|---|---|
| Línea base (mediana) | 13.1998 | 12.5308 | −0.6690 |
| Árbol sin límite | 0.0029 | 1.7185 | 1.7156 |
| Árbol `max_depth=8` | 1.0098 | 1.2766 | 0.2668 |
| Bosque aleatorio (200) | 0.4746 | 1.2999 | 0.8254 |

El árbol sin límite memorizó el entrenamiento (0.003) y tiene la peor prueba de los modelos que aprenden. Limitarlo a profundidad 8 fue lo que más ayudó: la brecha bajó de 1.72 a 0.27 y quedó como el mejor en prueba. El bosque redujo mucho el error de prueba del árbol sin límite, pero sigue con una brecha de 0.83 porque sus árboles no tienen límite; en este dataset, donde la distancia explica casi todo, el árbol podado le gana por poco.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(se completa en clase)_

**Tu razonamiento:**

_(se completa en clase)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(se completa en clase)_

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Archivo: [`starter/ejercicio2_regularizacion.py`](./starter/ejercicio2_regularizacion.py), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_regularizacion.py
```

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

Con interacciones de grado 3 la regresión recibe **3275 columnas** y hay solo **2380 filas** de entrenamiento: más parámetros que datos. Un modelo así puede ajustar el entrenamiento casi perfecto (MAE 0.89) pero el sistema está indeterminado y los coeficientes se disparan, por eso en prueba el error explota a 22.73, peor que la línea base (12.53).

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

`lineal grado 3`: entrenamiento 0.8892, prueba 22.7273 (brecha 21.84). `ridge grado 3`: entrenamiento 1.0272, prueba 1.3209 (brecha 0.29). Ridge penaliza el tamaño de los coeficientes (alpha=10), así que ya no puede usar coeficientes enormes para pasar exacto por cada punto de entrenamiento: empeora un poco ahí, a cambio de generalizar muchísimo mejor.

> Nota: en esta Mac la comprobación del Paso 2 dice `revisa` porque pide brecha > 100 y aquí salió 21.8. El modelo y las 3275 columnas son correctos; el tamaño exacto de la explosión depende de la librería de álgebra lineal (Accelerate en macOS vs. OpenBLAS en Codespaces).

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

El **gradient boosting**: MAE de prueba 1.2750, contra 12.5308 de la línea base, es decir, 11.26 dólares menos por viaje (≈ 90 % menos error).

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

La línea base no aprende de las columnas, solo toma la mediana de entrenamiento. Su error depende de qué tan dispersas estén las tarifas en cada conjunto: si por azar la división dejó en prueba menos viajes extremos (por ejemplo, menos tarifas fijas de 70 dólares al aeropuerto), las tarifas de prueba están más cerca de la mediana y el error es menor. Es pura variación de muestreo, no aprendizaje.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el script tú.

### Entregable 1: `modelos_proyecto.py`

Un script, en la carpeta de tu proyecto, que entrene con tu pipeline:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

Usé `DummyRegressor(strategy="median")`: la métrica del proyecto es el MAE y la constante que minimiza el error absoluto es la mediana, no la media (que además se jala hacia arriba por las tarifas fijas de 70 dólares). No es una serie de tiempo (cada viaje es independiente), así que una regla tipo "igual que ayer" no tiene sentido.

Script: [`proyecto/modelos_proyecto.py`](./proyecto/modelos_proyecto.py). Resultados: [`proyecto/resultados_proyecto.csv`](./proyecto/resultados_proyecto.csv).

| Modelo | MAE entrenamiento | MAE prueba | Brecha | Segundos |
|---|---|---|---|---|
| Línea base (mediana) | 13.1998 | 12.5308 | −0.6690 | 0.011 |
| Ridge alpha=1 | 2.4945 | 2.5618 | 0.0674 | 0.007 |
| Lineal grado 2 | 1.1195 | 1.2039 | 0.0844 | 0.025 |
| Ridge grado 2 alpha=10 | 1.2840 | 1.3781 | 0.0941 | 0.008 |
| Gradient boosting | 1.0705 | 1.2750 | 0.2044 | 0.120 |

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

Ninguno se sobreajusta de forma grave: la brecha más grande es la del gradient boosting (0.20 sobre un error de 1.07, ≈ 19 %); los lineales tienen brechas de 0.07–0.09. El sobreajuste fuerte solo apareció en los Ejercicios 1 y 2 (árbol sin límite, brecha 1.72; lineal de grado 3, brecha 21.84).

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

Ridge con alpha=1 en el modelo sin interacciones y alpha=10 en el de grado 2. Con grado 2 (≈ 350 columnas para 2380 filas) no hizo falta regularizar: la regresión sin penalizar ya tenía brecha 0.08, y Ridge alpha=10 subió el error en ambos lados (prueba 1.20 → 1.38) sin reducir la brecha, o sea, le agregó sesgo. La regularización solo vale la pena cuando hay muchas más columnas que señal, como en el grado 3 del Ejercicio 2.

**¿Cuánto mejora tu mejor modelo a la línea base?**

El mejor en prueba fue la **regresión lineal con interacciones de grado 2**: 1.2039 contra 12.5308 de la línea base, 11.33 dólares menos por viaje (≈ 90 % menos error). Le gana por poco al gradient boosting (1.2750), que es el mejor ensamblaje.

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error (MAE): el **costo** por predicción y de entrenar, la **latencia** (la app necesita la tarifa al instante), la **consistencia** (que el mismo viaje dé siempre la misma tarifa), la **explicabilidad** (poder justificarle al conductor/pasajero por qué cuesta eso), y la **robustez** en casos raros (aeropuerto, distancias faltantes).

# Semana 12, Sesión 1: Patrones de despliegue: ¿batch u online?

> Objetivo de la sesión: elegir entre servir el modelo de tu proyecto en modalidad batch o en modalidad online, justificando la decisión según la latencia que exige el caso de uso propuesto, y proteger con pruebas automáticas, que corren en cada cambio, el código que lo sirve.

**Lectura previa de esta sesión:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 7 (despliegue de modelos y servicio de predicción).

## Requisitos antes de esta sesión

Hoy se agregan tres herramientas: joblib (guardar un modelo entrenado en un archivo), pytest (pruebas automáticas) y GitHub Actions (integración continua). Las librerías nuevas ya están en `requirements.txt`, junto con FastAPI para la sesión siguiente. Sincroniza tu fork y reconstruye el contenedor ("Rebuild Container"), como en la Semana 11. Comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Debe mostrar `pytest 8.3.3` y `fastapi 0.115.4`.

Los laboratorios de hoy son scripts, no notebooks: lo que producen corre sin nadie mirando. Cada ejercicio entrena primero su modelo con un script que ya viene completo (`entrenar.py`), que lo guarda en un archivo `.joblib` dentro de la carpeta del ejercicio. Ese archivo no se sube al repositorio (`*.joblib` está en `.gitignore`).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué distingue un sistema de predicción batch de uno online? Da un ejemplo de cada uno.**

Un sistema batch calcula las predicciones de forma periódica (cada hora, cada noche) para un conjunto de casos conocido y las guarda en una tabla; quien las usa las lee después, sin esperar al modelo. Un sistema online calcula la predicción en el momento en que llega la petición y la devuelve en milisegundos. Ejemplo batch: recomendaciones de películas que Netflix precalcula cada noche para cada usuario. Ejemplo online: la estimación de tiempo de llegada que muestra Uber al pedir un viaje.

**¿Qué limitación tiene el sistema batch que el online resuelve, y qué cuesta resolverla?**

El batch solo puede responder sobre casos que conocía al correr el lote y con los datos que había en ese momento: una petición nueva o un cambio reciente (un usuario nuevo, un destino no reservado) no tiene fila que leer, y la predicción queda desactualizada hasta el siguiente lote. El online lo resuelve, pero cuesta: un servicio siempre encendido, con requisitos de latencia y disponibilidad, features que deben calcularse en tiempo real (y de la misma forma que al entrenar), y monitoreo y escalado de la infraestructura.

## Pregunta de apertura

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La tarifa de un viaje **programado** sí se puede calcular de noche, porque los viajes reservados se conocen de antemano. La tarifa de un viaje pedido en el momento no: hay infinitas combinaciones de origen, destino y hora, así que tiene que responder online, en unos pocos cientos de milisegundos (lo que el pasajero tolera mientras la app muestra el precio). El fraude, si se quiere bloquear el pago antes de aceptarlo, también es online y debe responder en milisegundos dentro del flujo de pago; si solo se usa para una cola de revisión manual, puede correr por lotes una vez al día.

## Ejercicio 1: la tarifa de los viajes programados, por lotes

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). La app permite reservar un viaje para el día siguiente. Cada noche, [`predecir_lote.py`](./starter/ejercicio1/predecir_lote.py) estima la tarifa de los 60 viajes reservados en [`viajes_programados.csv`](./starter/ejercicio1/viajes_programados.csv) con la red neuronal que elegiste en la Semana 11, y deja las estimaciones en una tabla que la app lee por la mañana. [`test_predecir_lote.py`](./starter/ejercicio1/test_predecir_lote.py) tiene sus pruebas. Los 5 `TODO`: cargar el modelo, estimar, armar la tabla de salida y dos pruebas.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python predecir_lote.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** sin resolver, `predecir_lote.py` escribe una tabla con una sola columna, y `pytest` dice `1 failed, 2 passed`. ¿Cómo pueden pasar dos pruebas si el script no estima ninguna tarifa?

Porque esas dos pruebas no miden la tarifa: una prueba sin ningún `assert` pasa siempre que no lance una excepción, y los `TODO 4` y `TODO 5` no tenían ningún `assert`. Solo `test_columnas_de_salida` comprobaba algo real (las tres columnas), y esa fallaba. Una prueba sin `assert` no protege nada.

**Respuesta 2 (TODO 1 y 2):** 4 de los 60 viajes no tienen pasajeros y 2 no tienen método de pago. ¿Esperas que el script falle en esos viajes, que los salte, o que les estime una tarifa? ¿Por qué?

Espero que les estime una tarifa. La preparación del pipeline imputa los faltantes (mediana para `pasajeros`, valor más frecuente para `metodo_pago`), igual que al entrenar, así que el modelo puede predecir. Fallar dejaría a toda la tabla sin tarifas por 6 viajes, y saltarlos dejaría a esos pasajeros sin precio por la mañana; un dato faltante en una feature secundaria no justifica ninguna de las dos.

**Respuesta 3 (TODO 3):** todas las filas de la tabla llevan la misma `version_modelo`. ¿Para qué guardarla en cada fila?

Para trazabilidad: si una tarifa resulta absurda o el modelo se reentrena, se puede saber exactamente qué versión del modelo (qué archivo `.joblib`) produjo cada estimación, comparar versiones y auditar o reprocesar solo las filas afectadas. La tabla se lee al día siguiente y puede mezclarse con lotes de otras noches, así que la versión tiene que viajar con cada fila.

**Respuesta 4 (TODO 4 y 5):** antes de escribir las dos pruebas, ¿qué cambio en `predecir_lote.py` haría fallar la del `TODO 4`? ¿Y la del `TODO 5`?

`TODO 4`: cambiar el orden o la cantidad de filas, por ejemplo ordenar la salida por tarifa, filtrar los viajes con datos faltantes (`viajes.dropna()`) o tomar `id_viaje` de otra tabla. `TODO 5`: dejar viajes sin estimación (por ejemplo, poner `NaN` a los viajes sin pasajeros en vez de imputarlos, o quitar el imputador) o no pasar la versión recibida (escribir una versión fija o `None` en `version_modelo`).

**Lo que mostró el lote:**

Versión del modelo: `c75c7248502c`. Tarifa más baja: 4.67 USD; más alta: 70.26 USD. Los 4 viajes sin pasajeros sí recibieron tarifa: 90023 → 33.94, 90024 → 29.88, 90025 → 25.80 y 90048 → 17.58 (coherentes con su distancia). Los 2 sin método de pago también: 90044 → 60.79 y 90050 → 29.59.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: la cola de revisión de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). Cada madrugada, [`puntuar_pagos.py`](./starter/ejercicio2/puntuar_pagos.py) calcula la probabilidad de fraude de los 300 pagos de ayer ([`pagos_ayer.csv`](./starter/ejercicio2/pagos_ayer.csv)) con el gradient boosting de la Semana 11, y arma la cola de revisión manual con los pagos que alcanzan el umbral de 0.10 que elegiste ese día. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO`; el paso 4 (la cola) y el paso 5 (las pruebas de [`test_puntuar_pagos.py`](./starter/ejercicio2/test_puntuar_pagos.py)) solo tienen un comentario que dice qué hacer. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** ¿cuántos pagos quedaron en la cola de revisión? ¿Cuántos fraudes esperarías encontrar en los 300 pagos, si la proporción es la de la Semana 11?

Quedaron 26 pagos en la cola (umbral 0.10). Con la proporción de la Semana 11 (295 fraudes en 6000 pagos, 4.9%) esperaría unos 15 fraudes en los 300 pagos; la cola es más grande porque el umbral bajo prioriza el recall a costa de algunos falsos positivos.

**Respuesta 2:** este modelo corre por lotes, una vez al día. ¿Qué pierde la empresa por no evaluar cada pago en el momento en que ocurre? ¿Qué tendría que pasar con la revisión para que eso importe?

Pierde la posibilidad de bloquear el fraude antes de que ocurra: el viaje ya se hizo y el pago ya se cobró (o se perdió) cuando el equipo lo revisa, horas después. Eso importa si la revisión tuviera que actuar antes de aceptar el pago (bloquearlo, pedir otra tarjeta o verificación); mientras la revisión sea manual y al día siguiente, evaluar en el momento no cambia nada porque nadie actúa antes.

**Respuesta 3:** la prueba de la cola de revisión arma su propia tabla de cuatro pagos con probabilidades escritas a mano, en vez de usar el modelo. ¿Qué gana la prueba con eso?

Determinismo y aislamiento: sabe exactamente qué resultado esperar (4, 2, 3), así que prueba solo la lógica de la cola (el filtro con `>=` y el orden) sin depender del modelo, de los datos ni de un reentrenamiento. Además cubre a propósito el caso límite 0.10, que con un modelo real sería casi imposible de forzar.

**Si terminas antes:** sube `UMBRAL` a 0.50 y vuelve a correr `puntuar_pagos.py`. ¿Cuántos pagos quedan en la cola? ¿Siguen pasando todas las pruebas? ¿Por qué?

Con `UMBRAL = 0.50` quedan 13 pagos en la cola. Las pruebas siguen pasando (4 passed): ninguna usa la constante `UMBRAL`, porque `test_cola_de_revision` llama a `a_revisar` con su propio umbral 0.10 y datos escritos a mano. Las pruebas comprueban el código, no el valor de negocio elegido.

## Ejercicio 3: el despliegue y las pruebas de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/despliegue.md`

Una página que diga si tu modelo se sirve por lotes o en línea, y por qué:

- Qué decisión habilita la predicción, quién la usa y cuánto tiempo puede esperar por ella (un número: milisegundos, horas, un día).
- Cuántos casos distintos hay que predecir, y si se conocen de antemano.
- Qué datos necesitan tus features en el momento de predecir, y si están disponibles en ese momento.
- Qué se pierde con el patrón que descartas.

### Entregable 2: las primeras pruebas de tu proyecto

Un archivo `psets/proyecto/modelado/test_pipeline_proyecto.py` con al menos tres pruebas sobre tu `pipeline_proyecto.py`, cada una con al menos un `assert`. Las pruebas arman sus propios datos (un `DataFrame` de pocas filas escrito a mano), no leen tu archivo de `data/`: en GitHub Actions ese archivo no existe. Ideas de qué probar: que la columna objetivo no está entre las que recibe el modelo; que una feature calculada da el valor que sabes calcular a mano; que una feature que mira hacia atrás en el tiempo no usa una fila del futuro; que la preparación acepta una categoría que no vio al entrenar.

Comprueba desde la raíz de tu fork que pasan, con el mismo comando que va a usar GitHub:

```
pytest psets/proyecto -v
```

### Entregable 3: la integración continua de tu fork

El archivo [`.github/workflows/pruebas.yml`](../../.github/workflows/pruebas.yml) le pide a GitHub que, en cada push, cree una máquina limpia, instale `requirements.txt` y corra `pytest psets/proyecto`. En tu fork viene apagado: en la página de tu fork, pestaña **Actions**, presiona "I understand my workflows, go ahead and enable them". Haz push de tus pruebas y espera la marca verde junto a tu commit.

**Pega el enlace a la ejecución de GitHub Actions donde pasan tus pruebas:**

_(tu respuesta)_

**¿Qué patrón elegiste para tu proyecto, y qué número de latencia lo decide?**

_(tu respuesta)_

**Rompe a propósito una de tus pruebas (cambia un valor esperado), haz push y mira qué pasa en GitHub. Después arréglala. ¿Qué te dijo GitHub, y dónde lo viste?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

Hoy la app lee las tarifas de una tabla calculada de noche. Pero cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Un servicio online: una API (por ejemplo con FastAPI) siempre encendida que cargue el modelo una vez y exponga un endpoint `POST /tarifa`. La app le mandaría en el momento los datos del viaje: zona de origen y destino, distancia, número de pasajeros, método de pago y la hora (de la que se derivan `hora_sin`, `hora_cos`, `es_fin_de_semana`, `es_hora_pico`, `es_tarifa_fija`), y recibiría la tarifa estimada y la versión del modelo en milisegundos.

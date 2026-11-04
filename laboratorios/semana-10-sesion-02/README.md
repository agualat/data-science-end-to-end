# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

El ML clásico conviene para **datos estructurados/tabulares** y problemas bien definidos (regresión, clasificación, predicción numérica) donde hay datos etiquetados del propio dominio: es barato de entrenar y correr, preciso y más explicable. Un LLM conviene para **datos no estructurados, sobre todo texto** (lenguaje natural, resumir, responder preguntas, generar contenido), donde el conocimiento general preentrenado vale más que un modelo específico.

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

El **costo**: la app estima la tarifa en cada solicitud de viaje, así que el costo por predicción se multiplica por miles de viajes al mes. Muy cerca está la explicabilidad, porque la tarifa se le cobra a una persona y hay que poder justificarla.

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error: el **costo** (de entrenar y por predicción, a 10 000+ viajes al mes), la **latencia**, si da **la misma respuesta** para el mismo viaje (reproducibilidad), si se puede **explicar** la tarifa, y cómo se comporta en **casos raros** (aeropuerto, datos faltantes).

## Ejercicio 1: una red neuronal frente al ensamblaje

Archivo: [`starter/ejercicio1_red_neuronal.py`](./starter/ejercicio1_red_neuronal.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). El script ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división. Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-02/starter
python3 ejercicio1_red_neuronal.py
```

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

Significa que el optimizador llegó al límite de iteraciones (200 por defecto) sin que la pérdida dejara de bajar: la red **no terminó de entrenar**, por eso sale peor (MAE prueba 1.6854 vs 1.2750 del ensamblaje). Con 1000 iteraciones espero que converja, desaparezca la advertencia y el error baje hasta quedar parecido al del ensamblaje.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

Capa oculta: 25 × 64 pesos + 64 sesgos = 1664. Capa de salida: 64 × 1 pesos + 1 sesgo = 65. Total: **1729 números**.

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

Espero que use sobre todo `distancia_km` (la tarifa es básicamente banderazo + precio por km), y después `es_tarifa_fija` (los 70 dólares fijos del aeropuerto). Para explicarle al conductor por qué son 70 dólares, el **ensamblaje** permite responder mejor: tiene importancias por columna y sus árboles se pueden inspeccionar ("es tarifa fija de aeropuerto"); la red son 1729 números sin interpretación directa.

**Lo que confirmó la salida, TODO por TODO:**

| Modelo | MAE entrenamiento | MAE prueba | Segundos |
|---|---|---|---|
| Gradient boosting | 1.0705 | 1.2750 | 0.13 |
| Red neuronal (200 iter, sin TODO 1) | 1.6009 | 1.6854 | 0.27 |
| Red neuronal (1000 iter) | 1.0882 | **1.1695** | 0.91 |

- TODO 1: con 1000 iteraciones la red converge y le gana al ensamblaje en prueba (1.17 vs 1.28), pero tarda ~7 veces más en entrenar.
- TODO 2: la red aprendió **1729** parámetros.
- TODO 3: las 3 columnas más importantes del ensamblaje son `distancia_km` (0.9824), `es_tarifa_fija` (0.0120) y `es_hora_pico` (0.0028). La distancia explica el 98 % de las decisiones.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(se completa en clase)_

**Tu razonamiento:**

_(se completa en clase)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(se completa en clase)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Archivo: [`starter/ejercicio2_tres_soluciones.py`](./starter/ejercicio2_tres_soluciones.py), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre el script. Además de entrenar, escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y vuelve a correr el script hasta que todas las comprobaciones digan `OK`. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

```
python3 ejercicio2_tres_soluciones.py
```

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

Usé **ChatGPT**. Para el costo usé el precio de entrada de GPT-5.5 en la API: 5 dólares por millón de tokens. El prompt tiene 2902 caracteres ≈ 726 tokens, y hacen falta 500 prompts para 10 000 viajes: 500 × 726 × 5 / 1 000 000 ≈ **1.81 dólares al mes**.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0.8957 | 0.13 s de CPU (≈ 0) | ≈ 0 (milisegundos en CPU propia) | Sí: importancias por columna, árboles inspeccionables |
| Red neuronal | 0.9637 | 0.91 s de CPU (≈ 0) | ≈ 0 (milisegundos en CPU propia) | Difícil: 1729 pesos sin interpretación directa |
| LLM (ChatGPT) | 1.1715 | 0 (no se entrena, solo 30 ejemplos en el prompt) | ≈ 1.81 dólares en API (+ latencia de red) | No de forma confiable: puede dar una justificación en texto, pero no es cómo calculó el número |

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

No: en una conversación nueva los números cambian (el LLM muestrea su respuesta, no es determinista). Para una app que cobra, eso significa que **el mismo viaje podría costar distinto según la llamada**, lo cual es injusto para el pasajero, imposible de auditar y abre la puerta a reclamos. Una tarifa tiene que ser reproducible.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

La peor es el viaje **JFK Airport → JFK Airport, 1.01 km, efectivo**: real 11.46, ChatGPT respondió 7.00 (error 4.46). Es un viaje raro (origen y destino el mismo aeropuerto, muy corto) y **no había ningún ejemplo así entre los 30 del prompt**, así que el LLM lo trató como un viaje corto cualquiera y no supo que dentro del aeropuerto la tarifa es más alta. El ensamblaje sí tenía **7 viajes JFK→JFK** entre las 2380 filas de entrenamiento (tarifa media 15.24), por eso predijo 12.74 (error 1.28). El segundo peor fue Midtown → Upper East Side con `pasajeros` vacío (17.40 vs 13.81).

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

No se mide sobre los 595 porque habría que mandar 30 prompts (o uno enorme) y copiar a mano 595 respuestas: lleva tiempo, cuesta tokens y se vuelve poco práctico en un chat. Con solo 20 viajes la comparación es **poco confiable**: un solo viaje raro (el JFK→JFK) mueve el MAE del LLM en ≈ 0.22, y el orden entre modelos cambia con la muestra; de hecho, en los 595 la red le gana al ensamblaje (1.17 vs 1.28) y en los 20 pasa lo contrario (0.96 vs 0.90). Sirve como indicio, no como conclusión.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.py`

Un script que entrene, con tu pipeline y tu división, el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

El proyecto es la estimación de la tarifa de taxi (mismos datos y pipeline de las sesiones anteriores).

- Entregable 1: [`proyecto/comparar_proyecto.py`](./proyecto/comparar_proyecto.py) (salida en `proyecto/comparacion_proyecto.csv`).
- Entregable 2: las 10 filas son las 10 primeras de la muestra del Ejercicio 2 (todas del conjunto de prueba), con las respuestas de ChatGPT a un prompt con 30 viajes de entrenamiento como ejemplos.

| Opción | MAE prueba (595) | MAE 10 viajes | Entrenamiento | Costo 10 000 viajes/mes | Explicabilidad |
|---|---|---|---|---|---|
| Gradient boosting | 1.2750 | 0.7794 | 0.11 s | ≈ 0 | Alta: importancias, árboles |
| Red neuronal (64, 1729 parámetros) | **1.1695** | 0.7251 | 0.91 s | ≈ 0 | Baja |
| LLM (ChatGPT) | no medido | **0.6280** | 0 | ≈ 1.81 USD + latencia | Muy baja, y no reproducible |

Usaría el **gradient boosting**. En las 10 filas el LLM salió mejor (0.63 vs 0.78), pero son 10 viajes y en los 20 del Ejercicio 2 salió peor (1.17 vs 0.90); en cuanto aparece un caso raro sin ejemplo en el prompt (JFK→JFK) falla por 4.46 dólares, cuesta dinero por cada viaje, tarda más y no devuelve el mismo número dos veces, algo inaceptable para cobrar. La red neuronal es la más precisa en los 595 viajes de prueba (1.17 vs 1.28, 0.11 dólares por viaje), pero esa ganancia es pequeña frente a lo que se pierde en explicabilidad: con el ensamblaje puedo decirle al conductor que la tarifa sale casi toda de la distancia (importancia 0.98) o de la tarifa fija de aeropuerto, y entrena 8 veces más rápido. Si más adelante la precisión pesara más que la explicación, la red sería la segunda opción; el LLM no.

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Tendría que saber de cada corrida: la **versión exacta del código** (commit), la **versión de los datos** (qué archivo/snapshot y qué limpieza), la **división** de entrenamiento/prueba (`random_state`, `test_size`, por fecha o al azar), los **hiperparámetros** y la semilla del modelo, la **preparación** (imputación, codificación, features), y las **versiones de las librerías** (scikit-learn, numpy) y del entorno. Con todo versionado y registrado, se puede reproducir cada corrida y encontrar la diferencia.

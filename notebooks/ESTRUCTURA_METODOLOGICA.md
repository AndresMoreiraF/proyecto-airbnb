# Estructura metodológica del proyecto

Qué se calcula en cada sección, con qué método y por qué ese y no otro. La columna de
justificación recoge la decisión frente a la alternativa que se descartó.

Ruta: la del curso, en seis pasos. Las secciones marcadas con **(nuevo)** no existían en la
versión anterior del proyecto y responden a defectos de medición detectados en los datos.

---

## Paso 0 — Preparación de la tabla

| Sección | Qué se calcula | Cómo | Por qué así |
|---|---|---|---|
| Apertura de la cotización **(nuevo)** | `es_mensual`, `descuento_cotizacion`, `noches_cotizadas`, `mes_cotizacion` | Se interpreta el JSON de `price_quote_raw`, campo a campo | La fuente no publica una tarifa de catálogo sino el resultado de una consulta de disponibilidad. Sin abrir ese objeto no se sabe qué significa la cifra de `price`, y el 16,2 % resulta ser una tarifa mensual prorrateada |
| Definición de población **(nuevo)** | `dias_expuesto`, `exposicion_completa`, `actividad_detectada`, `ingreso_por_dia` | Diferencia entre la fecha de corte y el primer comentario, con tope de 365 días | El ingreso es un agregado de 365 días y el 29 % de los anuncios no estuvo publicado todo ese tiempo. Comparar un total anual entre exposiciones distintas es un error de unidad |

---

## Paso 1 — ¿Machine learning?

| Sección | Qué se calcula | Cómo | Por qué así |
|---|---|---|---|
| Identidad contable del objetivo | Proporción en que `ingreso ÷ noches` reproduce la tarifa | Comparación numérica con tolerancia de una milésima | Define la lista de fugas. Si el objetivo es el producto de dos factores, los dos factores y todo lo que los reconstruya quedan prohibidos |
| Regla a mano | Mediana del ingreso anual por localidad y su error | Se ajusta en entrenamiento y se mide en prueba | Es el piso que el modelo debe superar para justificar su costo. Sin ella, un R² cualquiera no dice si el aprendizaje automático aporta |
| Línea base | Mediana de toda la ciudad | `DummyRegressor` con estrategia de mediana | Separa lo que aporta la localidad de lo que aporta no tener ningún modelo |
| Partición **(revisado)** | Entrenamiento y prueba, 80 / 20 | `GroupShuffleSplit` agrupando por `host_id`, semilla fija | El 66,5 % de los anuncios de prueba compartía anfitrión con entrenamiento y el 16,5 % tenía un gemelo casi idéntico. Con partición aleatoria el modelo recita el precio del hermano en lugar de predecirlo, y el R² sube 0,098 sin que mejore la predicción |

---

## Paso 2 — Datos

| Sección | Qué se calcula | Cómo | Por qué así |
|---|---|---|---|
| Fuente y licencia | Origen, fecha de compilación, condiciones de uso | Declaración | Requisito de atribución y condición para que el resultado sea reproducible |
| Papel de cada tabla | Resumen por alojamiento de calendario y comentarios | Agregación por `listing_id` | El calendario y los comentarios construyen y verifican el objetivo, y no aportan variables explicativas: la ocupación observada no existe antes de operar |
| Verificación triangular del objetivo | Ocupación publicada, reconstruida desde comentarios y fracción bloqueada del calendario | Correlación de rangos entre las tres medidas | Confirma por qué vía se construye el objetivo. La coincidencia con la reconstrucción desde comentarios demuestra que la ocupación se deriva de las reseñas |
| Auditoría de columnas | Tipo, vacíos y valores distintos de las 90 columnas | Recorrido completo sobre entrenamiento | Catorce columnas llegan vacías y cuatro son constantes. Revisarlas todas evita construir sobre una columna que no tiene datos |

---

## Paso 3 — Limpieza

| Sección | Qué se calcula | Cómo | Por qué así |
|---|---|---|---|
| Formato de la tarifa | Conversión de texto a número | Se retiran el signo y el separador de miles | La tarifa interviene en el criterio de valores extremos aunque quede fuera de las variables explicativas |
| Duplicados | Filas idénticas e identificadores repetidos | Conteo directo | Comprobación de integridad de la extracción |
| Separación del régimen mensual **(nuevo)** | Tamaño, tarifa e ingreso de cada régimen | Partición por la bandera `is_monthly_priced` | El 16,2 % tiene tarifa 43 % menor e ingreso mediano cero. Está mal medido en los dos ejes a la vez y entra como segmento descriptivo, sin modelo de ingreso |
| Valores extremos | Tope de Tukey sobre el logaritmo de la tarifa por huésped | Tercer cuartil más tres veces el rango intercuartílico, aprendido en entrenamiento | El corte sobre la tarifa total elimina viviendas grandes legítimas. La tarifa por huésped compara alojamientos de tamaño distinto |
| Faltantes | Error de cuatro estrategias de relleno | Se esconde el 20 % de los valores conocidos y se mide el error absoluto medio de cada método | La convención de imputar con la media no tiene respaldo. La comparación convierte la decisión en medición |
| Exposición **(nuevo)** | Días publicados dentro de la ventana y objetivo como tasa | División del ingreso entre los días expuestos | Corrige la falta de comparabilidad sin borrar filas, que sería seleccionar sobre la variable dependiente |
| Fugas | Lista de columnas prohibidas, con demostración | Comprobación numérica de la identidad y del origen de cada columna | Treinta columnas quedan fuera. La disponibilidad observada, los comentarios y las calificaciones reconstruyen el objetivo por vías indirectas |

---

## Paso 4 — Columnas

| Sección | Qué se calcula | Cómo | Por qué así |
|---|---|---|---|
| Selección por mecanismo | Tabla de variables por tema, con el mecanismo y el signo esperado | Declaración escrita antes de medir cualquier asociación | Seleccionar por la asociación observada sobre los mismos datos con que se entrena infla el desempeño aparente y descarta efectos no monótonos |
| Señal medida | Correlación de rangos e información mutua de cada variable | Spearman e información mutua por vecinos más cercanos | La correlación lineal no ve lo no monótono. La antigüedad del anfitrión tiene correlación 0,012 e información mutua 0,458 |
| Ablación | Error con bloques de variables añadidos de uno en uno | Validación cruzada agrupada, misma semilla y mismo modelo | Convierte la conservación de cada bloque en una decisión medida. El criterio que propone un bloque no puede ser el que lo aprueba |
| Descartes empíricos **(nuevo)** | Bloques propuestos que aumentan el error | Se miden y se retiran | El texto por TF-IDF y descomposición en valores singulares empeora los tres objetivos; la mediana de la tarifa de los vecinos no aporta |
| Preprocesamiento | Columnas derivadas, relleno, escalado y codificación | Un único `Pipeline` que la validación cruzada reajusta en cada partición | Evita que una mediana o un vocabulario aprendido con todos los datos viaje a la partición de validación |

---

## Paso 5 — Modelos

| Sección | Qué se calcula | Cómo | Por qué así |
|---|---|---|---|
| Diseño en tres etapas **(nuevo)** | Riesgo, nivel y decisión | Clasificador, regresión sobre la tasa y composición de ambos | Un tercio de los anuncios tiene ingreso cero por no detección. Una regresión única promedia dos poblaciones distintas y asigna valores intermedios a las dos |
| Etapa A, riesgo | Probabilidad de que el anuncio registre actividad detectable | Clasificación con validación cruzada agrupada, medida con AUC | La etiqueta honesta es actividad detectada, no venta. El propietario necesita esa probabilidad para decidir |
| Etapa B, nivel | Ingreso por día expuesto entre los anuncios con exposición observable | Regresión sobre el logaritmo de la tasa | Es la estimación más estable de todas las ensayadas y separa el nivel del riesgo |
| Etapa C, decisión | Esperanza del ingreso anual | Producto de las dos etapas, expresado sobre 365 días | Es la única cifra comparable con un arriendo tradicional |
| Familias comparadas | Error de nueve familias de modelos | Misma partición, mismo preprocesamiento y misma validación cruzada agrupada | La comparación solo es válida si la vara es idéntica para todos |
| Enfoque directo y enfoque combinado | Ingreso estimado en un solo modelo, y como producto de tarifa por ocupación | Los dos con la misma validación | La descomposición aísla el factor bien medido, la tarifa, del factor censurado, la ocupación |
| Ajuste de parámetros | Mejor combinación del finalista | Búsqueda aleatoria sobre la misma validación cruzada | Recorre mejor el espacio que una rejilla cuando pocos parámetros concentran el efecto |

---

## Paso 6 — Evaluación

| Sección | Qué se calcula | Cómo | Por qué así |
|---|---|---|---|
| Medida final | Error del modelo elegido sobre la partición de prueba | Una sola evaluación, con el modelo ya ajustado | La partición de prueba pierde su papel en cuanto se usa para elegir |
| Métricas | Error absoluto medio, raíz del error cuadrático, R² en pesos y en escala logarítmica, y AUC en la etapa de riesgo | Cálculo directo | En pesos el error cuadrático queda dominado por la cola derecha. Las dos escalas juntas dicen cosas distintas y ambas hacen falta |
| Importancia | Deterioro del error al desordenar cada columna | Importancia por permutación sobre la partición de prueba | Mide el aporte al modelo ajustado, no la asociación bruta, y permite contrastar con el signo declarado en el paso 4 |
| Error por segmento | Error por tipo de alojamiento y por nivel de actividad | Agregación de los errores individuales | El error medio oculta que el objetivo tiene poblaciones con comportamiento distinto |
| Calibración | Valor observado y predicho por decil de predicción | Agrupación en diez partes iguales | Para una decisión de inversión importa que el orden y el nivel sean correctos, más que el R² |
| Limitaciones | Lo que el diseño no puede afirmar | Declaración | Un resultado sin sus límites no es interpretable |

---

## Decisiones descartadas, con su motivo

| Alternativa | Por qué se descartó |
|---|---|
| Borrar los anuncios con ingreso cero | Seleccionar sobre la variable dependiente. Produce una estimación optimista que ningún propietario puede usar, porque no sabe de antemano si quedará en el grupo seleccionado |
| Partición aleatoria por filas | Reparte anuncios del mismo anfitrión a ambos lados. Infla el R² en 0,098 sin mejorar la predicción |
| Modelar la estacionalidad de la tarifa | Hay una sola cotización por anuncio y el calendario de esta extracción no trae columna de precio. El efecto no es observable |
| Restringir las cotizaciones a un solo mes | Mejora el R² en 0,003 a cambio del 19 % de la muestra, e introduce su propia selección |
| Leer la estacionalidad del calendario | La fracción bloqueada decae con la distancia a la fecha. Con una sola extracción no se separa la temporada de la anticipación de la reserva |
| Usar el mes de la cotización como variable | La consulta pidió la primera fecha disponible, de modo que la fecha mide cuánto lleva cerrado el calendario, que es inventario dormido |
| Incluir el texto del anuncio por TF-IDF | Aumenta el error en los tres objetivos |
| Añadir la tarifa mediana de los vecinos cercanos | No mejora el error. Las coordenadas y las distancias ya la contienen |
| Reportar el ingreso anual con una sola regresión | Promedia dos poblaciones y deja el R² en pesos por debajo de una décima |

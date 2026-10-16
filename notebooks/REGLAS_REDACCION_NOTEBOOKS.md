# Reglas de redacción de los notebooks del proyecto

Análogas a las reglas de redacción de las fichas técnicas, trasladadas de un documento
descriptivo a un cuaderno de análisis. Aplican a las celdas de markdown, a los comentarios
del código y a los textos que el código imprime.

---

## 1. Principio general

**El notebook describe el análisis y sus decisiones técnicas. No describe el curso, ni el
equipo, ni la conversación que lo produjo.**

Un lector que nunca haya oído del programa debe poder seguirlo. Todo lo que solo tiene
sentido dentro del curso sobra: la rúbrica, los ejemplos vistos en clase, los nombres de
quienes escribieron cada parte y el reparto de tareas.

Prueba rápida para cada frase: ¿es un hecho del análisis, o es una explicación de por qué
el cuaderno quedó así? Si es lo segundo, se va.

La portada sí nombra el programa, la universidad y a los integrantes. La regla aplica dentro
del cuerpo del cuaderno.

---

## 2. Contaminación de chat

El defecto más difícil de ver, porque el texto parece técnico. Son frases que responden a una
consigna dada en conversación y no a la necesidad del lector. Tres variantes:

**2.1 Narrar el proceso de trabajo en primera persona.**
- Mal: «Primero cargo la tabla y miro qué trae.»
- Mal: «Reviso cuatro cosas: duplicados, formatos, faltantes y fugas.»
- Mal: «Me queda claro que el barrio solo no basta.»
- Bien: «La tabla trae una fila por alojamiento y 90 columnas.»
- Bien: «El barrio explica una fracción del precio; el tamaño y el tipo de alojamiento
  aportan el resto.»

**2.2 Anunciar o recordar lo que hace otra parte del cuaderno.**
- Mal: «Más adelante compruebo con los modelos que quitarlos no cambia la conclusión.»
- Mal: «Como cambió la tabla, repito la división entrenamiento/prueba.»
- Bien: el hecho en su lugar, sin anuncio.

**2.3 Justificar la organización del cuaderno o contar la lista.**
- Mal: «Hago una clase propia para que estas transformaciones queden dentro del Pipeline.»
- Mal: «Son los seis pasos que seguimos en los tres ejemplos.»
- Bien: «Las medianas se aprenden en `fit` y se aplican en `transform`, de modo que la
  partición de prueba no interviene en el relleno.»

---

## 3. Referencias cruzadas

Ningún cuaderno nombra al otro, ni por nombre de archivo ni como «el otro notebook».

Sí se nombra **el enfoque**, porque es terminología del análisis. «La estimación directa del
ingreso anual» es válido; «lo que dio el cuaderno rentabilidad_directo.ipynb» no lo es. Los
nombres de archivo viven en los comentarios del código, que son instrucciones de ejecución,
no prosa.

---

## 4. Cifras

**Ninguna cifra se escribe a mano en el markdown.** Toda cifra que el lector ve sale impresa
por el código de la celda anterior o de la misma celda.

Es la regla con más consecuencias prácticas. Un número pegado en el texto queda obsoleto en
cuanto cambia una semilla, un corte o una variable, y el cuaderno empieza a contradecir su
propia salida. Las conclusiones numéricas se redactan como `print(f"...")` sobre variables.

Corolario: ninguna gráfica recibe sus datos como literales. Si una barra vale 0,462, ese
valor llega desde la tabla de resultados, no escrito en una lista.

---

## 5. Atribución

**5.1 Las fuentes de datos conservan su nombre, su fecha de compilación y su licencia.**
Inside Airbnb, Bogotá, compilado el 14 de septiembre de 2026, Creative Commons Attribution
4.0.

**5.2 Los métodos conservan su autor y su año** cuando el criterio proviene de una
publicación. Tukey (1977) para el criterio de valores extremos, Breiman (2001) para el bosque
aleatorio, Chen y Guestrin (2016) para el potenciado por gradiente, Hutto y Gilbert (2014)
para los umbrales de sentimiento.

**5.3 La calidad del dato no es un hallazgo.** Fuera las frases cuyo sujeto es el dato y no el
fenómeno: «dato validado contra tres fuentes», «cifra que ahora respalda el cálculo».

---

## 6. Coherencia entre código y texto

**El código manda.** Si el texto afirma algo, la celda anterior lo demuestra. Al revisar una
sección se lee primero la salida y después el texto.

Ninguna afirmación sobre una variable sobrevive si la salida la contradice. Un texto que dice
que la limpieza bajó el error, sobre una tabla que muestra el mismo error antes y después, es
un error de redacción, no un matiz.

---

## 7. Estilo

1. **Sin contaminación de chat.** Ver el punto 2.
2. **Ejemplos orgánicos.** No se introducen con dos puntos, ni con guion, ni con paréntesis.
   - Mal: «las variables de actividad: calificación, disponibilidad y comentarios»
   - Bien: «Las variables de actividad reúnen la calificación, la disponibilidad a 90 días y
     el número de comentarios del último año.»
3. **Sin la fórmula «no solo X sino también Y».**
4. **Sin la fórmula «no es X, es Y».**
5. **Sin adjetivos de relleno**: fundamental, clave, crucial, vital, robusto, potente.
6. **Sin muletillas de arranque.** La frase empieza por su sujeto.
7. **Voz activa y concisión.** El análisis como sujeto, no el analista.
8. **Sin autorreferencia.** El cuaderno no dice «este notebook» ni «en esta celda».
9. **Sin negrita a media frase** y sin comillas de énfasis. Las comillas se reservan para la
   cita literal; el formato de código, para nombres de columnas, funciones y archivos.
10. **Sin énfasis en mayúsculas.**
11. **Análisis profundizado, no resultados enunciados.** Se dice el mecanismo, no solo el
    efecto. «El bosque supera a los lineales» es un resultado enunciado; «el precio depende de
    combinaciones de tamaño y ubicación que un modelo aditivo no representa» es el mecanismo.

---

## 8. Comentarios del código

El estilo heredado se conserva: **un comentario por línea, alineado a la derecha, en español,
que explica qué hace la línea** para un lector que está aprendiendo la biblioteca.

- El comentario explica la línea, no el propósito del proyecto.
- Nombres de variables y funciones en español, sin tildes.
- Las constantes de configuración van en mayúsculas al inicio del cuaderno.

---

## 9. Gráficas y esquemas

- Cada paso del análisis que produce una decisión lleva su gráfica o su esquema.
- Los esquemas de flujo se dibujan con código, no se insertan como imagen, para que el
  cuaderno corra de principio a fin sin archivos externos.
- Paleta única del proyecto, definida una sola vez. Tinta para el texto, un solo tono para
  magnitudes, hasta tres colores de serie con etiqueta directa.
- Sin dos ejes verticales en una misma gráfica. Dos magnitudes de escala distinta van en dos
  gráficas.
- Los datos de toda gráfica provienen de una variable calculada. Ver el punto 4.

---

## 10. Estructura

Los dos cuadernos siguen el mismo orden, y es el de la ruta del curso.

```
Portada y pregunta de investigación
Paso 1 — ¿Machine learning?        definición del problema y regla a mano
Paso 2 — Datos                     fuente, licencia, unidad de observación, objetivo
Paso 3 — Limpieza                  formatos, duplicados, extremos, faltantes, fugas
Paso 4 — Columnas                  selección por mecanismo, señal medida, transformaciones
Paso 5 — Modelos                   línea base, regla a mano y familias comparadas
Paso 6 — Evaluación                métricas, elección argumentada, importancia, respuesta
Conclusiones
```

---

## 11. Operación

Orden de ejecución y reglas de reproducibilidad.

```
notebooks/airbnb_comun.py          variables, transformadores, métricas y gráficos
notebooks/rentabilidad_directo.ipynb
notebooks/rentabilidad_combinado.ipynb   lee las salidas del enfoque directo si existen
```

- **Semilla fija** en la partición, en la validación cruzada y en todo modelo que la acepte.
- **El cuaderno corre de arriba abajo.** Ninguna celda depende de otra ejecutada antes fuera
  de orden; los contadores de ejecución quedan consecutivos.
- **Salidas versionadas** en `notebooks/salidas/`. Nunca se sobrescribe un archivo abierto.
- **`LOKY_MAX_CPU_COUNT`** se fija antes de importar scikit-learn, porque en Windows la
  detección de núcleos físicos lanza un aviso con traza que ensucia la salida.
- Las tablas largas se guardan además como `.csv` para poder citarlas sin volver a correr.

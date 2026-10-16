"""
Variables, transformadores, metricas y graficos del proyecto Airbnb Bogota.

Fuente de datos: Inside Airbnb, Bogota, Distrito Capital, Colombia, compilado el 14 de
septiembre de 2026. Licencia Creative Commons Attribution 4.0 (CC BY 4.0).

El modulo concentra lo que comparten los cuadernos de rentabilidad anual, de modo que el
preprocesamiento sea identico y las comparaciones entre enfoques sean validas.
"""

import os                                              # os permite fijar variables de entorno antes de importar otras librerias

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")       # en Windows la deteccion de nucleos fisicos lanza un aviso con traza; fijarlo lo evita

import json                                            # json lee la lista de comodidades, que viene como texto
import re                                              # re maneja expresiones regulares para limpiar el texto
import warnings                                        # warnings controla los avisos de Python
from pathlib import Path                               # Path maneja rutas de carpetas y archivos

import numpy as np                                     # numpy: operaciones matematicas y arreglos numericos
import pandas as pd                                    # pandas: manejo de tablas (DataFrames)
import matplotlib.pyplot as plt                        # matplotlib: graficos
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch   # piezas para dibujar los esquemas de flujo

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone   # clases base de scikit-learn
from sklearn.compose import ColumnTransformer, make_column_selector               # tratamiento distinto por grupo de columnas
from sklearn.decomposition import TruncatedSVD                                     # reduce la matriz de texto a pocas componentes
from sklearn.feature_extraction.text import TfidfVectorizer                        # convierte texto en pesos por palabra
from sklearn.impute import SimpleImputer                                           # rellena vacios
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score      # metricas de regresion
from sklearn.pipeline import Pipeline                                              # encadena pasos de preprocesamiento y modelo
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder     # codificacion y escalado

warnings.filterwarnings("ignore")                      # oculta los avisos para que la salida quede legible

# ---------------------------------------------------------------------------
# 1. Configuracion
# ---------------------------------------------------------------------------
SEMILLA = 42                                           # semilla unica para particion, validacion cruzada y modelos
DATOS = Path(__file__).resolve().parent.parent / "datos"   # carpeta de datos, relativa a la ubicacion del modulo
SALIDAS = Path(__file__).resolve().parent / "salidas"      # carpeta donde se guardan tablas y figuras
SALIDAS.mkdir(exist_ok=True)                           # la crea si no existe

# Anclas geograficas de Bogota, en grados decimales, para calcular distancias.
ZONA_T = (4.6675, -74.0535)                            # Zona T y Parque de la 93: nucleo de ocio y negocios
AEROPUERTO = (4.7016, -74.1469)                        # aeropuerto internacional El Dorado
CENTRO_HISTORICO = (4.5981, -74.0758)                  # La Candelaria y Centro Internacional

# ---------------------------------------------------------------------------
# 2. Paleta y estilo
# ---------------------------------------------------------------------------
# La paleta del proyecto. TINTA se reserva para texto y ejes; ACENTO es el color de magnitud;
# MORADO y GRIS completan las series. No se usa un cuarto color de serie: por encima de tres
# categorias el grafico se separa en varios paneles.
TINTA = "#1B1A2F"                                      # azul casi negro, para texto y ejes
ACENTO = "#E8963C"                                     # naranja, color principal de magnitud
MORADO = "#4C3F91"                                     # indigo, segunda serie
GRIS = "#8A879E"                                       # gris violaceo, series de referencia
SERIES = [ACENTO, MORADO, GRIS]                        # orden fijo de asignacion de color por serie
RAMPA = ["#F3E3CE", "#EFC894", "#E8963C", "#B96F23", "#7A4615"]   # rampa de un solo tono para magnitudes


def estilo():
    """Fija el estilo de todas las figuras del cuaderno."""
    plt.rcParams.update({                              # rcParams guarda los valores por defecto de matplotlib
        "figure.dpi": 110,                             # resolucion en pantalla
        "font.size": 10,                               # tamano base de texto
        "axes.titlesize": 12,                          # tamano del titulo de cada panel
        "axes.titleweight": "bold",                    # titulo en negrita
        "axes.titlelocation": "left",                  # titulo alineado a la izquierda
        "axes.labelcolor": TINTA,                      # color de las etiquetas de los ejes
        "axes.edgecolor": GRIS,                        # color del marco
        "axes.linewidth": 0.8,                         # grosor del marco
        "axes.spines.top": False,                      # sin linea superior
        "axes.spines.right": False,                    # sin linea derecha
        "axes.grid": True,                             # cuadricula visible
        "axes.axisbelow": True,                        # la cuadricula va detras de las barras
        "grid.color": "#E6E4EC",                       # cuadricula tenue
        "grid.linewidth": 0.7,                         # grosor de la cuadricula
        "text.color": TINTA,                           # color del texto
        "xtick.color": GRIS,                           # color de las marcas del eje x
        "ytick.color": GRIS,                           # color de las marcas del eje y
        "legend.frameon": False,                       # leyenda sin caja
        "figure.titlesize": 13,                        # tamano del titulo de la figura
    })


def miles(x, pos=None):
    """Formatea un numero con separador de miles, para los ejes de pesos."""
    return f"{x:,.0f}"                                 # coma como separador de miles y sin decimales


def caja(ax, x, y, ancho, alto, titulo, texto="", color=ACENTO, relleno=0.14):
    """Dibuja una caja redondeada con titulo y texto, para los esquemas de flujo."""
    ax.add_patch(FancyBboxPatch(                       # FancyBboxPatch dibuja un rectangulo de esquinas redondeadas
        (x, y), ancho, alto,                           # esquina inferior izquierda, ancho y alto
        boxstyle="round,pad=0.02,rounding_size=0.05",  # estilo de esquina
        linewidth=1.4, edgecolor=color,                # borde del color indicado
        facecolor=(*plt.matplotlib.colors.to_rgb(color), relleno),   # mismo color con transparencia como relleno
    ))
    ax.text(x + ancho / 2, y + alto * 0.68, titulo,    # titulo en la parte alta de la caja
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=TINTA)
    if texto:                                          # si se paso texto secundario
        ax.text(x + ancho / 2, y + alto * 0.28, texto, # lo escribe debajo del titulo
                ha="center", va="center", fontsize=8.2, color=TINTA, linespacing=1.35)


def flecha(ax, desde, hasta, texto="", color=GRIS):
    """Dibuja una flecha entre dos puntos, con etiqueta opcional."""
    ax.add_patch(FancyArrowPatch(                      # FancyArrowPatch dibuja la flecha
        desde, hasta,                                  # punto de origen y punto de destino
        arrowstyle="-|>", mutation_scale=13,           # punta de flecha y su tamano
        linewidth=1.2, color=color,                    # grosor y color
        shrinkA=2, shrinkB=2,                          # deja un margen en los extremos
    ))
    if texto:                                          # si la flecha lleva etiqueta
        mx, my = (desde[0] + hasta[0]) / 2, (desde[1] + hasta[1]) / 2   # punto medio de la flecha
        ax.text(mx, my + 0.018, texto, ha="center", va="bottom", fontsize=8, color=TINTA)


def lienzo(ancho=11, alto=4.2):
    """Crea una figura sin ejes, para dibujar esquemas."""
    fig, ax = plt.subplots(figsize=(ancho, alto))      # una sola figura con un solo panel
    ax.set_xlim(0, 1)                                  # coordenadas normalizadas en horizontal
    ax.set_ylim(0, 1)                                  # coordenadas normalizadas en vertical
    ax.axis("off")                                     # oculta los ejes
    return fig, ax                                     # devuelve la figura y el panel


# ---------------------------------------------------------------------------
# 3. Carga y limpieza de formato
# ---------------------------------------------------------------------------
def a_numero(serie):
    """Convierte el precio de texto a numero, quitando el signo de pesos y las comas."""
    return serie.str.replace("[$,]", "", regex=True).astype(float)   # elimina $ y , con una sola expresion regular


def _campo_cotizacion(texto, campo):
    """Extrae un campo del objeto JSON con que la fuente guarda la cotizacion de la tarifa."""
    try:                                               # intenta interpretar el texto como JSON
        return json.loads(texto)["quote"].get(campo)   # devuelve el campo pedido del objeto quote
    except (TypeError, ValueError, KeyError):          # si el texto falta o no tiene esa estructura
        return None                                    # devuelve vacio


def cargar_listings():
    """Carga la tabla de alojamientos, convierte la tarifa y abre la cotizacion que la produjo.

    La fuente no publica una tarifa de catalogo: publica el resultado de una consulta de
    disponibilidad, con su fecha, su duracion y su regimen de precio. Esas condiciones
    determinan que significa la cifra y se extraen aqui, junto a la tarifa.
    """
    tabla = pd.read_csv(                               # lee el archivo comprimido
        DATOS / "listings.csv.gz",                     # ruta del archivo
        dtype={"price": str},                          # fuerza price a texto, porque viene como "$274,830.78"
        low_memory=False,                              # lee el archivo completo para no inferir mal los tipos
    )
    tabla["precio_noche"] = a_numero(tabla["price"])   # tarifa por noche en pesos, como numero

    crudo = tabla["price_quote_raw"]                                           # texto JSON de la cotizacion
    tabla["es_mensual"] = crudo.apply(                                          # regimen de precio de la cotizacion
        lambda t: bool(_campo_cotizacion(t, "is_monthly_priced"))).astype(bool)   # True si la tarifa viene prorrateada de un mes
    tabla["descuento_cotizacion"] = pd.to_numeric(                              # descuento ya aplicado a la tarifa
        crudo.apply(lambda t: _campo_cotizacion(t, "discount_amount")), errors="coerce").fillna(0)
    entrada = pd.to_datetime(                                                   # fecha de entrada que devolvio la consulta
        crudo.apply(lambda t: _campo_cotizacion(t, "requested_checkin_date")), errors="coerce")
    salida = pd.to_datetime(                                                    # fecha de salida que devolvio la consulta
        crudo.apply(lambda t: _campo_cotizacion(t, "requested_checkout_date")), errors="coerce")
    tabla["noches_cotizadas"] = (salida - entrada).dt.days                      # duracion de la estancia cotizada
    # La consulta pidio la primera fecha con disponibilidad, de modo que la distancia hasta esa
    # fecha mide cuanto tiempo lleva bloqueado el calendario. Queda para el diagnostico y fuera
    # de cualquier modelo de ingreso, donde seria una medida del resultado.
    tabla["mes_cotizacion"] = entrada.dt.to_period("M").astype(str)             # mes de la fecha cotizada
    tabla["dias_hasta_cotizacion"] = (entrada - entrada.min()).dt.days          # dias desde la primera fecha disponible del archivo
    return tabla                                       # devuelve la tabla lista


def preparar_poblacion(tabla, corte=None):
    """Agrega las variables que definen la poblacion de analisis y el objetivo por exposicion.

    La exposicion es el numero de dias que el anuncio llevaba publicado dentro de la ventana de
    365 dias con que la fuente calcula el ingreso. Solo es observable para los anuncios que
    recibieron al menos un comentario, porque el archivo no trae la fecha de publicacion.
    """
    t = tabla.copy()                                                            # trabaja sobre una copia
    if corte is None:                                                           # si no se indico la fecha de corte
        corte = pd.Timestamp("2026-09-15")                                      # el dia anterior al inicio del calendario
    primera = pd.to_datetime(t["first_review"], errors="coerce")                # fecha del primer comentario
    ultima = pd.to_datetime(t["last_review"], errors="coerce")                  # fecha del ultimo comentario
    t["dias_expuesto"] = (corte - primera).dt.days.clip(lower=1, upper=365)     # dias dentro de la ventana, con tope de un ano
    t["exposicion_completa"] = t["dias_expuesto"] >= 365                        # True si estuvo publicado toda la ventana
    t["exposicion_observable"] = primera.notna()                                # True si hay con que medir la exposicion
    t["actividad_detectada"] = (ultima > corte - pd.Timedelta(days=365)).fillna(False)   # True si hubo comentario en la ventana
    t["ingreso_anual"] = t["estimated_revenue_l365d"]                           # ingreso bruto estimado de la ventana
    t["ingreso_por_dia"] = t["ingreso_anual"] / t["dias_expuesto"]              # ingreso por dia de exposicion
    t["noches_ocupadas"] = t["estimated_occupancy_l365d"].astype(float)         # noches ocupadas estimadas
    return t                                                                    # devuelve la tabla preparada


def ocupacion_calendario(fecha_corte=None):
    """Resume el calendario en la fraccion de los 365 dias siguientes marcados como no disponibles."""
    calendario = pd.read_csv(                          # carga el calendario, que tiene una fila por alojamiento y dia
        DATOS / "calendar.csv.gz",                     # ruta del archivo
        usecols=["listing_id", "date", "available"],   # solo tres columnas, para no agotar la memoria con 7,1 millones de filas
    )
    calendario["date"] = pd.to_datetime(calendario["date"])          # convierte la fecha de texto a fecha real
    if fecha_corte is None:                                          # si no se indico la fecha de corte
        fecha_corte = calendario["date"].min() - pd.Timedelta(days=1)   # el dia anterior al primer dia del calendario
    limite = fecha_corte + pd.Timedelta(days=365)                    # un ano despues de la fecha de corte
    ventana = calendario[calendario["date"] <= limite]                # se queda con los 365 dias siguientes
    bloqueado = ventana["available"] == "f"                           # marca los dias no disponibles, que son los ocupados o bloqueados
    resumen = bloqueado.groupby(ventana["listing_id"]).mean()         # por alojamiento, la fraccion de dias no disponibles
    resumen.name = "fraccion_bloqueada_365d"                          # nombre de la columna resultante
    return resumen, fecha_corte                                       # devuelve el resumen y la fecha de corte usada


def ocupacion_reviews(fecha_corte, tasa_resena=0.5, noches_por_resena=3.0):
    """Estima las noches ocupadas del ultimo ano a partir del numero de comentarios.

    El metodo multiplica los comentarios recibidos en los ultimos 365 dias por la inversa de
    la tasa de resena y por la estancia media, que es el procedimiento con que Inside Airbnb
    construye su propia estimacion de ocupacion.
    """
    resenas = pd.read_csv(DATOS / "reviews.csv.gz", usecols=["listing_id", "date"])   # solo el alojamiento y la fecha
    resenas["date"] = pd.to_datetime(resenas["date"])                                 # fecha de texto a fecha real
    ultimo_ano = resenas[resenas["date"] > fecha_corte - pd.Timedelta(days=365)]       # comentarios de los ultimos 365 dias
    conteo = ultimo_ano.groupby("listing_id").size()                                   # cuantos comentarios por alojamiento
    noches = conteo / tasa_resena * noches_por_resena                                  # comentarios a noches ocupadas
    noches.name = "noches_estimadas_resenas"                                           # nombre de la columna resultante
    return noches.clip(upper=365)                                                      # ningun alojamiento ocupa mas de 365 noches


# ---------------------------------------------------------------------------
# 4. Columnas descartadas por fuga
# ---------------------------------------------------------------------------
# Una fuga es una columna que solo se conoce despues del resultado, o que se calcula con el
# resultado. El objetivo ingreso anual es el producto del precio por las noches ocupadas, y la
# ocupacion estimada se deriva del numero de comentarios, de modo que la lista de columnas
# prohibidas es mas amplia que en un modelo de precio.
FUGAS_RENTABILIDAD = {
    "price": "es el precio por noche, uno de los dos factores del ingreso anual",
    "precio_noche": "version numerica del precio por noche",
    "price_quote_total_price": "cotizacion que reproduce el precio publicado",
    "price_quote_price_per_night": "coincide con el precio publicado en la practica totalidad de los casos",
    "price_quote_raw": "respuesta cruda de la cotizacion, contiene el precio",
    "price_quote_checkin_date": "la consulta pidio la primera fecha con disponibilidad, de modo "
                                "que la fecha cotizada mide cuanto lleva bloqueado el calendario",
    "price_quote_checkout_date": "cierra la estancia cotizada, que la regla de estancia minima "
                                 "del propio anuncio determina",
    "estimated_occupancy_l365d": "es el segundo factor del ingreso anual",
    "estimated_revenue_l365d": "es el objetivo",
    "availability_30": "disponibilidad observada, resultado de la operacion del alojamiento",
    "availability_60": "disponibilidad observada",
    "availability_90": "disponibilidad observada",
    "availability_365": "disponibilidad observada",
    "availability_eoy": "disponibilidad observada hasta fin de ano",
    "has_availability": "disponibilidad observada",
    "number_of_reviews": "los comentarios son el insumo con que se estima la ocupacion",
    "number_of_reviews_ltm": "comentarios del ultimo ano, insumo de la ocupacion estimada",
    "number_of_reviews_l30d": "comentarios del ultimo mes, insumo de la ocupacion estimada",
    "number_of_reviews_ly": "comentarios del ano anterior, insumo de la ocupacion estimada",
    "reviews_per_month": "comentarios por mes, insumo de la ocupacion estimada",
    "first_review": "fecha del primer comentario, solo existe si el alojamiento ya opero",
    "last_review": "fecha del ultimo comentario, solo existe si el alojamiento ya opero",
    "review_scores_rating": "calificacion, solo existe si el alojamiento ya recibio huespedes",
    "review_scores_accuracy": "calificacion por dimension",
    "review_scores_cleanliness": "calificacion por dimension",
    "review_scores_checkin": "calificacion por dimension",
    "review_scores_communication": "calificacion por dimension",
    "review_scores_location": "calificacion por dimension",
    "review_scores_value": "calificacion por dimension",
}

# Columnas descartadas sin ser fuga, con el motivo.
DESCARTES = {
    "vacias": [                                        # columnas sin un solo dato en la tabla
        "neighborhood_overview", "host_since", "host_response_time", "host_response_rate",
        "host_acceptance_rate", "host_thumbnail_url", "host_neighbourhood",
        "host_total_listings_count", "host_verifications", "neighbourhood",
        "neighbourhood_group_cleansed", "bathrooms", "calendar_updated", "instant_bookable",
    ],
    "constantes": [                                    # columnas con un solo valor distinto
        "scrape_id", "last_scraped", "source", "calendar_last_scraped",
    ],
    "identificadores": [                               # claves y enlaces, sin contenido predictivo
        "id", "listing_url", "picture_url", "host_id", "host_url", "host_profile_id",
        "host_profile_url", "host_name", "host_picture_url",
    ],
    "redundantes": [                                   # repiten casi exactamente otra columna ya incluida
        "minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights",
        "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
        "host_listings_count", "calculated_host_listings_count",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
        "hosts_time_as_user_years", "hosts_time_as_user_months",
    ],
    "sin_tratamiento": [                               # texto libre de baja cobertura o sin uso definido
        "host_location", "host_about", "host_has_profile_pic",
    ],
}

# ---------------------------------------------------------------------------
# 5. Tabla de variables, agrupadas por tematica
# ---------------------------------------------------------------------------
# Cada fila declara el tema, la variable, de donde sale y el mecanismo por el que se espera
# que afecte al ingreso anual. El signo esperado se declara antes de ver el resultado, para
# poder contrastarlo despues con la importancia del modelo.
VARIABLES = [
    # --- Estructura y capacidad -------------------------------------------------------
    ("Estructura", "accommodates", "original",
     "Capacidad maxima de huespedes. Acota el tamano del grupo que puede reservar y, con el, "
     "la tarifa que el alojamiento puede sostener.", "+"),
    ("Estructura", "bedrooms", "original",
     "Numero de habitaciones. Separa el apartamento familiar del estudio y determina cuantos "
     "grupos independientes caben.", "+"),
    ("Estructura", "beds", "original",
     "Numero de camas. A igual numero de habitaciones, mide si el espacio se densifico con "
     "camas adicionales.", "+"),
    ("Estructura", "n_banos", "creada de bathrooms_text",
     "Numero de banos extraido del texto, con medio bano valorado en 0,5. La dotacion "
     "sanitaria limita el numero de huespedes simultaneos que el espacio atiende sin conflicto.", "+"),
    ("Estructura", "bano_compartido", "creada de bathrooms_text",
     "Marca de bano compartido. Compartir bano elimina la privacidad y traslada el alojamiento "
     "al segmento de habitacion economica.", "-"),
    ("Estructura", "huesp_x_habitacion", "creada",
     "Huespedes por habitacion. Mide hacinamiento: a igual capacidad, un valor alto indica "
     "cuartos compartidos y tarifa por huesped mas baja.", "-"),
    ("Estructura", "camas_x_habitacion", "creada",
     "Camas por habitacion. Distingue la habitacion doble del dormitorio colectivo.", "-"),
    ("Estructura", "banos_x_huesped", "creada",
     "Banos por huesped. Dotacion sanitaria relativa al tamano del grupo.", "+"),

    # --- Equipamiento -----------------------------------------------------------------
    ("Equipamiento", "n_amenities", "creada de amenities",
     "Longitud de la lista de comodidades. Resume la inversion del anfitrion en dotacion y "
     "funciona como senal visible de calidad en el buscador.", "+"),
    ("Equipamiento", "am_* (dummies)", "creadas de amenities",
     "Una columna por comodidad frecuente. Cada comodidad cambia el costo del anfitrion y la "
     "disposicion a pagar del huesped, y el conteo no las distingue: cafetera y escritorio "
     "senalan estancia de trabajo, el acceso autonomo reduce la friccion operativa y habilita "
     "mas rotacion, el ascensor condiciona el acceso en edificios altos.", "+"),

    # --- Ubicacion --------------------------------------------------------------------
    ("Ubicacion", "latitude", "original",
     "Latitud. Bogota presenta un gradiente de renta y de oferta hotelera en direccion "
     "sur-norte que ninguna otra columna recoge con esa resolucion.", "+"),
    ("Ubicacion", "longitude", "original",
     "Longitud. Separa el eje oriental, contra los cerros, del occidente de la ciudad.", "+"),
    ("Ubicacion", "dist_zona_t", "creada",
     "Distancia en kilometros al nucleo de ocio y negocios de la Zona T y el Parque de la 93. "
     "La demanda de estancia corta se concentra alli y decae con la distancia.", "-"),
    ("Ubicacion", "dist_aeropuerto", "creada",
     "Distancia al aeropuerto El Dorado. Determina el atractivo para estancias de una noche "
     "ligadas a conexiones aereas.", "-"),
    ("Ubicacion", "dist_centro_hist", "creada",
     "Distancia al centro historico. Captura una demanda turistica distinta de la de negocios.", "-"),
    ("Ubicacion", "neighbourhood_cleansed", "original",
     "Localidad, en 20 categorias. Recoge seguridad percibida, comercio y densidad de oferta, "
     "que las distancias no representan.", "mixto"),

    # --- Reglas de reserva ------------------------------------------------------------
    ("Reglas de reserva", "log_min_noches", "creada de minimum_nights",
     "Logaritmo de la estancia minima. Selecciona el segmento de demanda: un minimo de una "
     "noche compite con la hoteleria, y uno de varias semanas compite con el arriendo.", "-"),
    ("Reglas de reserva", "estadia_min_larga", "creada de minimum_nights",
     "Marca de estancia minima de 28 noches o mas, que es el umbral comercial del arriendo "
     "mensual. Ese modelo de negocio cobra tarifa con descuento y recibe menos comentarios, "
     "de modo que reduce tanto el ingreso estimado como la ocupacion medida.", "-"),
    ("Reglas de reserva", "log_max_noches", "creada de maximum_nights",
     "Logaritmo de la estancia maxima. Un tope bajo restringe la estancia larga.", "mixto"),

    # --- Anfitrion y profesionalizacion -----------------------------------------------
    ("Anfitrion", "antiguedad_host", "creada de hosts_time_as_host_years/months",
     "Anos operando como anfitrion. La experiencia mejora la gestion de tarifa y calendario, "
     "mientras que el inventario mas antiguo tiende a estar menos renovado; la relacion no es "
     "monotona y por eso la correlacion lineal no la detecta.", "mixto"),
    ("Anfitrion", "log_anuncios_host", "creada de host_listings_count",
     "Logaritmo del numero de anuncios del anfitrion. Separa al arrendador ocasional del "
     "operador con cartera.", "+"),
    ("Anfitrion", "anuncios_completos", "original (calculated_host_listings_count_entire_homes)",
     "Numero de alojamientos completos del mismo anfitrion. Identifica al operador "
     "profesional, que fija tarifas con herramientas de precio dinamico y mantiene el "
     "calendario abierto todo el ano.", "+"),
    ("Anfitrion", "es_superhost", "creada de host_is_superhost",
     "Distintivo que exige calificacion alta, respuesta rapida y pocas cancelaciones. Actua "
     "como senal de calidad verificada y mejora la posicion en el buscador, de modo que su "
     "efecto se concentra en la ocupacion mas que en la tarifa.", "+"),
    ("Anfitrion", "tiene_licencia", "creada de license",
     "Registro nacional de turismo declarado. La formalidad permite facturar, atender demanda "
     "corporativa y operar sin riesgo de retiro del anuncio.", "+"),

    # --- Presentacion del anuncio -----------------------------------------------------
    ("Presentacion", "largo_nombre", "creada de name",
     "Numero de caracteres del titulo. Mide el esfuerzo del anfitrion en la ficha publica.", "+"),
    ("Presentacion", "largo_descripcion", "creada de description",
     "Numero de caracteres de la descripcion, sin etiquetas HTML. Una descripcion extensa "
     "acompana a la gestion profesional del anuncio.", "+"),
    ("Presentacion", "t_* (palabras del titulo)", "creadas de name",
     "Marcas de lujo, vista, nuevo, moderno y centro en el titulo. Anuncian atributos que "
     "ninguna columna estructurada registra.", "+"),
    ("Estructura", "metros_cuadrados", "creada de description",
     "Superficie declarada por el anfitrion en la descripcion, recuperada del texto. El area "
     "determina cuantos huespedes caben con holgura y es el atributo con que se tasa un "
     "inmueble en cualquier otro mercado; la fuente no la publica como columna.", "+"),
    ("Estructura", "declara_area", "creada de description",
     "Marca de que el anfitrion declara la superficie. Separa al anuncio redactado con detalle "
     "del anuncio minimo, y evita que el relleno del area se confunda con un area pequena.", "+"),

    # --- Tipologia --------------------------------------------------------------------
    ("Tipologia", "room_type", "original",
     "Alojamiento entero, habitacion privada, habitacion compartida o habitacion de hotel. "
     "Define el producto que se vende y es la particion mas gruesa del mercado.", "mixto"),
    ("Tipologia", "property_type", "original",
     "Tipo de inmueble en 65 categorias, agrupadas por debajo del 1 por ciento. Distingue "
     "loft, apartamento, casa y habitacion de hotel dentro de un mismo room_type.", "mixto"),
]


# Quien controla cada variable. La pregunta la hace un propietario que todavia no publico, de
# modo que la clasificacion decide que entra en su modelo: la dotacion que ya posee y las
# decisiones a las que puede comprometerse, pero no el desempeno que aun no tiene.
CONTROL = {
    "metros_cuadrados": "Dotacion", "declara_area": "Decision",
    "tiene_licencia": "Decision", "log_anuncios_host": "Decision",
    "anuncios_completos": "Decision", "antiguedad_host": "Desempeno",
    "es_superhost": "Desempeno", "room_type": "Decision", "property_type": "Dotacion",
    "accommodates": "Decision",
}
CONTROL_POR_TEMA = {                                   # para las variables que no estan en el diccionario
    "Estructura": "Dotacion", "Ubicacion": "Dotacion", "Tipologia": "Dotacion",
    "Equipamiento": "Decision", "Reglas de reserva": "Decision", "Presentacion": "Decision",
    "Anfitrion": "Decision",
}


def tabla_variables():
    """Devuelve la tabla de variables seleccionadas, con su tematica y quien las controla."""
    tabla = pd.DataFrame(VARIABLES, columns=["Tema", "Variable", "Origen",
                                             "Relacion teorica con el ingreso anual",
                                             "Signo esperado"])
    tabla["Quien la controla"] = [                     # dotacion, decision o desempeno
        CONTROL.get(v, CONTROL_POR_TEMA.get(t, "Dotacion"))
        for v, t in zip(tabla["Variable"], tabla["Tema"])
    ]
    return tabla


def tabla_descartes():
    """Devuelve la tabla de columnas descartadas, con el motivo de cada grupo."""
    filas = []                                         # lista donde se acumulan las filas
    for columna, motivo in FUGAS_RENTABILIDAD.items():  # recorre las columnas con fuga
        filas.append({"Columna": columna, "Grupo": "Fuga", "Motivo": motivo})
    motivos = {                                        # motivo unico por grupo de descarte
        "vacias": "sin un solo dato en la tabla",
        "constantes": "un solo valor distinto en toda la tabla",
        "identificadores": "clave o enlace, sin contenido predictivo",
        "redundantes": "repite casi exactamente otra columna ya incluida",
        "sin_tratamiento": "texto libre de baja cobertura, sin tratamiento definido",
    }
    for grupo, columnas in DESCARTES.items():          # recorre los demas grupos
        for columna in columnas:                       # y cada columna del grupo
            filas.append({"Columna": columna, "Grupo": grupo.capitalize(), "Motivo": motivos[grupo]})
    return pd.DataFrame(filas)                         # devuelve la tabla armada


# ---------------------------------------------------------------------------
# 6. Transformadores
# ---------------------------------------------------------------------------
def distancia_km(lat1, lon1, lat2, lon2):
    """Distancia en kilometros entre dos puntos, por la formula del semiverseno."""
    radio = 6371.0                                     # radio medio de la Tierra en kilometros
    rad = np.pi / 180                                  # factor para pasar de grados a radianes
    seno = (np.sin((lat2 - lat1) * rad / 2) ** 2       # termino de la diferencia de latitud
            + np.cos(lat1 * rad) * np.cos(lat2 * rad)  # ajuste por la latitud de cada punto
            * np.sin((lon2 - lon1) * rad / 2) ** 2)    # termino de la diferencia de longitud
    return 2 * radio * np.arcsin(np.sqrt(seno))        # distancia sobre la superficie


PALABRAS_TITULO = {                                    # palabras buscadas en el titulo y su expresion regular
    "t_lujo": r"lujo|luxur|premium|exclusiv",          # anuncia categoria alta
    "t_vista": r"vista|view|panoram",                  # anuncia vista
    "t_nuevo": r"nuevo|nueva|new|estren",              # anuncia inmueble reciente
    "t_moderno": r"modern|contempor|minimal",          # anuncia estilo
    "t_centro": r"centro|center|downtown|chapinero|zona\s*t|usaqu",   # anuncia ubicacion reconocible
}


class CrearColumnas(BaseEstimator, TransformerMixin):
    """Crea las columnas derivadas que no existen en la tabla original.

    No aprende nada de los datos, de modo que puede ir en cualquier posicion del Pipeline sin
    riesgo de trasladar informacion de la particion de prueba.
    """

    def fit(self, X, y=None):                          # fit: no hay nada que aprender
        return self                                    # devuelve el mismo objeto, como exige scikit-learn

    def transform(self, X):                            # transform: agrega las columnas derivadas
        X = X.copy()                                   # trabaja sobre una copia para no alterar la tabla original

        # Banos, extraidos del texto.
        texto_bano = X["bathrooms_text"].str.lower()                     # pasa el texto a minusculas
        banos = texto_bano.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)   # primer numero del texto, con decimales
        banos = banos.mask(texto_bano.str.contains("half", na=False) & banos.isna(), 0.5)   # "half-bath" sin numero vale 0,5
        banos = banos.mask(texto_bano.str.contains("no bathroom", na=False), 0.0)           # "no bathroom" vale 0
        X["n_banos"] = banos                                             # numero de banos
        X["bano_compartido"] = texto_bano.str.contains("shared", na=False).astype(int)      # 1 si el bano es compartido

        # Densidad de ocupacion del espacio.
        habitaciones = X["bedrooms"].replace(0, np.nan)                  # evita dividir por cero en estudios
        X["huesp_x_habitacion"] = X["accommodates"] / habitaciones       # huespedes por habitacion
        X["camas_x_habitacion"] = X["beds"] / habitaciones               # camas por habitacion
        X["banos_x_huesped"] = banos / X["accommodates"].clip(lower=1)   # banos por huesped

        # Comodidades.
        listas = X["amenities"].apply(self._leer_lista)                  # convierte el texto en lista de comodidades
        X["n_amenities"] = listas.apply(len)                             # cuantas comodidades tiene el alojamiento
        X["_amenities_lista"] = listas                                   # guarda la lista para el transformador de dummies

        # Reglas de reserva.
        X["log_min_noches"] = np.log1p(X["minimum_nights"])              # logaritmo de la estancia minima
        X["log_max_noches"] = np.log1p(X["maximum_nights"])              # logaritmo de la estancia maxima
        X["estadia_min_larga"] = (X["minimum_nights"] >= 28).astype(int)  # 1 si exige 28 noches o mas

        # Anfitrion.
        X["antiguedad_host"] = (X["hosts_time_as_host_years"]            # anos como anfitrion
                                + X["hosts_time_as_host_months"] / 12)   # mas los meses expresados en anos
        X["log_anuncios_host"] = np.log1p(X["host_listings_count"])      # logaritmo del numero de anuncios
        X["anuncios_completos"] = X["calculated_host_listings_count_entire_homes"]   # alojamientos completos del anfitrion
        X["es_superhost"] = (X["host_is_superhost"] == "t").astype(int)  # 1 si tiene el distintivo
        X["tiene_licencia"] = X["license"].notna().astype(int)           # 1 si declaro registro nacional de turismo

        # Geografia.
        X["dist_zona_t"] = distancia_km(X["latitude"], X["longitude"], *ZONA_T)              # km a la Zona T
        X["dist_aeropuerto"] = distancia_km(X["latitude"], X["longitude"], *AEROPUERTO)      # km al aeropuerto
        X["dist_centro_hist"] = distancia_km(X["latitude"], X["longitude"], *CENTRO_HISTORICO)   # km al centro historico

        # Texto del anuncio.
        titulo = X["name"].fillna("")                                    # titulo, con los vacios como cadena vacia
        descripcion = (X["description"].fillna("")                       # descripcion, con los vacios como cadena vacia
                       .str.replace(r"<[^>]+>", " ", regex=True))        # quita las etiquetas HTML que trae el 43,8 % de los casos
        X["largo_nombre"] = titulo.str.len()                             # caracteres del titulo
        X["largo_descripcion"] = descripcion.str.len()                   # caracteres de la descripcion

        # Superficie declarada en la descripcion. La fuente no trae una columna de area, y una
        # parte de los anfitriones la menciona en prosa. El area determina la capacidad y la
        # tarifa que el inmueble sostiene, de modo que recuperarla del texto anade una variable
        # estructural que no existe en ninguna columna.
        area = descripcion.str.extract(                                  # busca un numero seguido de la palabra metros
            r"(\d{1,4})\s*(?:m2|m²|mts|metros\s+cuadrados|metros)", flags=re.IGNORECASE)[0]
        area = pd.to_numeric(area, errors="coerce")                      # lo convierte a numero
        X["metros_cuadrados"] = area.where((area >= 10) & (area <= 600))  # descarta cifras imposibles para una vivienda
        X["declara_area"] = X["metros_cuadrados"].notna().astype(int)     # 1 si el anfitrion declara el area
        titulo_min = titulo.str.lower()                                  # titulo en minusculas para buscar palabras
        for columna, patron in PALABRAS_TITULO.items():                  # recorre las palabras buscadas
            X[columna] = titulo_min.str.contains(patron, regex=True).astype(int)   # 1 si el titulo la menciona
        X["texto"] = (titulo + " " + descripcion).str.strip()            # titulo y descripcion en una sola columna

        return X                                                         # devuelve la tabla con las columnas nuevas

    @staticmethod
    def _leer_lista(texto):
        """Convierte el texto de comodidades en lista; devuelve lista vacia si viene danado."""
        try:                                           # intenta interpretarlo como lista JSON
            valor = json.loads(texto)                  # convierte el texto en objeto de Python
            return valor if isinstance(valor, list) else []   # solo acepta listas
        except (TypeError, ValueError):                # si el texto esta vacio o mal formado
            return []                                  # devuelve lista vacia


class DummiesAmenities(BaseEstimator, TransformerMixin):
    """Crea una columna por comodidad frecuente.

    Entran las comodidades presentes en al menos uno de cada diez anuncios. Las muy raras no
    cambian el error y ensanchan la matriz, y las casi universales no distinguen a un
    alojamiento de otro. La lista se aprende en `fit` con la particion de entrenamiento, de
    modo que la de prueba no influye en que columnas existen.
    """

    def __init__(self, frec_min=0.10, frec_max=0.97):  # al crearla se fijan los dos umbrales
        self.frec_min = frec_min                       # frecuencia minima para que la comodidad entre
        self.frec_max = frec_max                       # frecuencia maxima para que la comodidad entre

    def fit(self, X, y=None):                          # fit: aprende cuales comodidades entran
        conteo = {}                                    # diccionario de comodidad a numero de alojamientos
        for lista in X["_amenities_lista"]:            # recorre la lista de cada alojamiento
            for comodidad in set(lista):               # set evita contar dos veces la misma comodidad
                conteo[comodidad] = conteo.get(comodidad, 0) + 1   # suma uno a esa comodidad
        total = len(X)                                 # numero de alojamientos de entrenamiento
        self.comodidades_ = sorted(                    # lista ordenada de las comodidades que entran
            c for c, n in conteo.items()               # recorre el conteo
            if self.frec_min <= n / total <= self.frec_max   # y filtra por frecuencia
        )
        self.nombres_ = ["am_" + self._limpiar(c) for c in self.comodidades_]   # nombre de columna de cada una
        return self                                    # devuelve el mismo objeto

    def transform(self, X):                            # transform: construye las columnas aprendidas
        X = X.copy()                                   # trabaja sobre una copia
        conjuntos = X["_amenities_lista"].apply(set)    # convierte cada lista en conjunto, para buscar mas rapido
        nuevas = {                                     # arma todas las columnas de una vez
            nombre: conjuntos.apply(lambda s, c=comodidad: int(c in s))   # 1 si el alojamiento tiene la comodidad
            for comodidad, nombre in zip(self.comodidades_, self.nombres_)
        }
        X = pd.concat([X, pd.DataFrame(nuevas, index=X.index)], axis=1)   # pega las columnas nuevas de un solo golpe
        return X.drop(columns=["_amenities_lista"])    # la lista ya no se necesita

    @staticmethod
    def _limpiar(texto):
        """Convierte el nombre de una comodidad en un nombre de columna valido."""
        limpio = re.sub(r"[^a-z0-9]+", "_", texto.lower())   # deja letras y numeros, el resto pasa a guion bajo
        return limpio.strip("_")[:40]                        # recorta y quita guiones de los extremos


class RellenarPorGrupo(BaseEstimator, TransformerMixin):
    """Rellena vacios con la mediana del grupo al que pertenece cada fila.

    Las medianas se aprenden en `fit` y se aplican en `transform`, de modo que la particion de
    prueba no interviene en el relleno.
    """

    def __init__(self, columnas, grupo):               # al crearla se indica que rellenar y por que agrupar
        self.columnas = columnas                       # lista de columnas a rellenar
        self.grupo = grupo                             # columna que define los grupos

    def fit(self, X, y=None):                          # fit: aprende las medianas
        self.medianas_ = {c: X.groupby(self.grupo)[c].median() for c in self.columnas}   # mediana por grupo
        self.globales_ = {c: X[c].median() for c in self.columnas}                        # mediana general de respaldo
        return self                                    # devuelve el mismo objeto

    def transform(self, X):                            # transform: rellena con lo aprendido
        X = X.copy()                                   # trabaja sobre una copia
        for c in self.columnas:                        # recorre cada columna a rellenar
            por_grupo = X[self.grupo].map(self.medianas_[c])        # mediana del grupo de cada fila
            X[c] = X[c].fillna(por_grupo).fillna(self.globales_[c])  # rellena con la del grupo y, si falta, con la general
        return X                                       # devuelve la tabla rellenada


# ---------------------------------------------------------------------------
# 7. Bloques de variables y preprocesamiento
# ---------------------------------------------------------------------------
# Los bloques permiten encender y apagar grupos de variables para medir cuanto aporta cada uno.
BLOQUES = {
    "estructura": ["accommodates", "bedrooms", "beds", "n_banos", "bano_compartido",
                   "huesp_x_habitacion", "camas_x_habitacion", "banos_x_huesped"],
    "equipamiento": ["n_amenities"],                   # el conteo de comodidades, sin abrirlas una por una
    "ubicacion": ["latitude", "longitude", "dist_zona_t", "dist_aeropuerto", "dist_centro_hist"],
    "reglas": ["log_min_noches", "log_max_noches", "estadia_min_larga"],
    # Decisiones de operacion. El propietario las elige antes de publicar y puede comprometerse
    # a ellas, de modo que son las palancas sobre las que el modelo informa una decision.
    "decision_anfitrion": ["tiene_licencia", "log_anuncios_host", "anuncios_completos"],
    # Consecuencias de haber operado. No existen para quien todavia no publico, y el distintivo
    # de superanfitrion exige diez estancias completadas en la misma ventana que mide el
    # objetivo, de modo que para la etapa de actividad la variable implica la etiqueta.
    "desempeno_pasado": ["es_superhost", "antiguedad_host"],
    "presentacion": ["largo_nombre", "largo_descripcion"] + list(PALABRAS_TITULO),
    "area": ["metros_cuadrados", "declara_area"],      # superficie recuperada de la descripcion
    # Reglas que el anfitrion fija dia a dia en el calendario. Son decisiones suyas, de modo
    # que entran sin fuga, a diferencia de la disponibilidad, que registra el resultado.
    "reglas_calendario": ["valores_estancia_minima", "rango_estancia_minima",
                          "varia_estancia_minima", "estancia_maxima_mediana"],
    # Conjunto de referencia: la seleccion mas estrecha, con las variables que una revision por
    # correlacion lineal con el precio deja en pie. Sirve de punto de comparacion de la ablacion.
    "referencia": ["latitude", "longitude", "accommodates", "bedrooms", "beds", "n_banos",
                   "bano_compartido", "n_amenities", "minimum_nights", "host_listings_count",
                   "es_superhost"],
}
CATEGORICAS = ["room_type", "property_type", "neighbourhood_cleansed"]   # columnas de texto con pocas categorias
# Bloques que no son listas de columnas numericas: las comodidades abiertas una por una, las
# categoricas y la representacion del texto del anuncio.
BLOQUES_ESPECIALES = ["comodidades", "tipologia", "texto"]
TODOS_LOS_BLOQUES = [b for b in BLOQUES if b != "referencia"] + BLOQUES_ESPECIALES
# Conjunto que la ablacion deja en pie. El bloque de texto queda fuera porque aumenta el error
# en los tres objetivos, y el conjunto se fija aqui para que los dos enfoques usen el mismo.
# Las variables se clasifican por quien las controla, porque la pregunta la hace un
# propietario que todavia no publico y necesita saber a que puede comprometerse.
DOTACION = ["estructura", "ubicacion", "area", "tipologia"]            # lo que ya posee y no elige
DECISION = ["reglas", "reglas_calendario", "equipamiento", "comodidades",   # a lo que se compromete
            "presentacion", "decision_anfitrion"]
DESEMPENO = ["desempeno_pasado"]                                       # resultado de haber operado

# Escenario de un propietario nuevo: dotacion y decisiones, sin desempeno pasado, que para el
# vale siempre lo mismo y no informa ninguna de las predicciones que va a pedir.
BLOQUES_ELEGIDOS = DOTACION + DECISION
# Escenario de un operador que amplia cartera: conoce su propio historial y puede usarlo.
BLOQUES_OPERADOR = DOTACION + DECISION + DESEMPENO
# Bloques propuestos y descartados por la medicion, con el motivo. La ablacion los evalua y la
# tabla deja constancia de que la decision tiene respaldo y no es una omision.
DESCARTES_EMPIRICOS = {
    "texto": "la representacion del titulo y la descripcion por TF-IDF y descomposicion en "
             "valores singulares aumenta el error en los tres objetivos",
    "precio_vecinos": "la mediana de la tarifa de los 25 anuncios mas cercanos no mejora el "
                      "error, porque las coordenadas y las distancias ya la contienen",
    "fecha_de_cotizacion": "el mes de la fecha cotizada mide cuanto lleva bloqueado el "
                           "calendario, de modo que describe el resultado y no el inmueble",
}


def armar_preprocesador(bloques=None, componentes_texto=15, codificacion="onehot"):
    """Arma el preprocesamiento con los bloques de variables indicados.

    bloques: lista de nombres de bloque; por defecto entran todos.
    componentes_texto: numero de componentes con que se resume el texto del anuncio.
    codificacion: "onehot" para columnas 0/1, "target" para codificacion por media del
    objetivo con ajuste cruzado interno.
    """
    bloques = list(TODOS_LOS_BLOQUES) if bloques is None else list(bloques)   # si no se indican, entran todos

    numericas = []                                     # lista de columnas numericas que entran
    for nombre in BLOQUES:                             # recorre los bloques numericos
        if nombre in bloques:                          # si el bloque esta encendido
            numericas += BLOQUES[nombre]               # agrega sus columnas

    partes = []                                        # lista de transformaciones del ColumnTransformer
    if numericas:                                      # si hay columnas numericas
        partes.append(("num", Pipeline([               # para las numericas, dos pasos
            ("imputar", SimpleImputer(strategy="median")),   # rellena lo que quede vacio con la mediana
            ("escalar", StandardScaler()),                   # y las deja en media 0 y desviacion 1
        ]), numericas))
    if "comodidades" in bloques:                       # las comodidades abiertas una por una entran por patron de nombre
        partes.append(("am", "passthrough", make_column_selector(pattern=r"^am_")))   # ya son 0/1, pasan sin cambios
    if "tipologia" in bloques:                         # categoricas
        if codificacion == "target":                   # codificacion por media del objetivo
            partes.append(("cat", TargetEncoder(random_state=SEMILLA), CATEGORICAS))   # ajuste cruzado interno
        else:                                          # codificacion en columnas 0/1
            partes.append(("cat", OneHotEncoder(       # una columna por categoria
                handle_unknown="infrequent_if_exist",  # una categoria no vista va al grupo de infrecuentes
                min_frequency=0.01,                    # las categorias por debajo del 1 % se agrupan
                sparse_output=False,                   # devuelve una tabla densa
            ), CATEGORICAS))
    if "texto" in bloques:                             # texto del anuncio
        partes.append(("txt", Pipeline([               # dos pasos sobre una sola columna de texto
            ("tfidf", TfidfVectorizer(                 # convierte el texto en pesos por palabra
                max_features=4000,                     # conserva las 4000 palabras mas frecuentes
                min_df=20,                             # ignora las que aparecen en menos de 20 anuncios
                ngram_range=(1, 2),                    # palabras solas y pares de palabras
                strip_accents="unicode",               # normaliza las tildes
            )),
            ("svd", TruncatedSVD(n_components=componentes_texto, random_state=SEMILLA)),   # resume en pocas componentes
        ]), "texto"))                                  # el nombre va sin lista, para pasar una sola columna

    columnas = ColumnTransformer(partes, remainder="drop")   # descarta todo lo que no se declaro

    return Pipeline([                                  # el preprocesamiento completo, en orden
        ("crear", CrearColumnas()),                                                   # 1) crea las columnas derivadas
        ("dummies", DummiesAmenities()),                                              # 2) abre las comodidades en columnas 0/1
        ("rellenar", RellenarPorGrupo(["beds", "bedrooms", "n_banos"], "accommodates")),   # 3) rellena por capacidad
        ("columnas", columnas),                                                       # 4) imputa, escala y codifica
    ])


# ---------------------------------------------------------------------------
# 8. Metricas y modelos de referencia
# ---------------------------------------------------------------------------
def metricas(y_real, y_pred):
    """Calcula las cuatro metricas de regresion sobre un vector de predicciones."""
    pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)   # recorta las predicciones negativas, que el logaritmo no admite
    real = np.asarray(y_real, dtype=float)                     # valores reales como arreglo
    return pd.Series({
        "MAE": mean_absolute_error(real, pred),                                        # error absoluto medio
        "MedAE": float(np.median(np.abs(real - pred))),                                # mediana del error absoluto, que la cola no arrastra
        "RMSE": float(np.sqrt(mean_squared_error(real, pred))),                        # raiz del error cuadratico medio
        "R2": r2_score(real, pred),                                                    # proporcion de varianza explicada
    })


def reglas_calendario():
    """Resume por alojamiento las reglas de reserva que el anfitrion fija dia a dia.

    El calendario trae la estancia minima y maxima de cada dia, de modo que registra si el
    anfitrion endurece o relaja la regla a lo largo del ano. Esas son decisiones suyas, no
    resultados de la operacion, y por eso pueden entrar como variables explicativas.
    """
    calendario = pd.read_csv(                          # carga solo las columnas de reglas
        DATOS / "calendar.csv.gz",
        usecols=["listing_id", "minimum_nights", "maximum_nights"],
    )
    por_anuncio = calendario.groupby("listing_id")     # agrupa por alojamiento
    reglas = pd.DataFrame({
        "valores_estancia_minima": por_anuncio["minimum_nights"].nunique(),   # cuantas reglas distintas usa en el ano
        "rango_estancia_minima": (por_anuncio["minimum_nights"].max()         # diferencia entre la regla mas dura y la mas blanda
                                  - por_anuncio["minimum_nights"].min()),
        "estancia_maxima_mediana": por_anuncio["maximum_nights"].median(),     # tope de estancia habitual
    })
    reglas["varia_estancia_minima"] = (reglas["valores_estancia_minima"] > 1).astype(int)   # 1 si la cambia alguna vez
    return reglas                                      # devuelve una fila por alojamiento


class ReglaPorGrupo(BaseEstimator, RegressorMixin):
    """Regla a mano: predice la mediana del objetivo en el grupo al que pertenece cada fila."""

    def __init__(self, columna="neighbourhood_cleansed"):   # al crearla se indica la columna que agrupa
        self.columna = columna                              # guarda el nombre de la columna

    def fit(self, X, y):                                    # aprende la mediana de cada grupo
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)   # asegura que el objetivo tenga el mismo indice que X
        self.medianas_ = y.groupby(X[self.columna]).median()       # mediana del objetivo por grupo
        self.global_ = float(y.median())                           # mediana general, para grupos no vistos
        return self                                                # devuelve el mismo objeto

    def predict(self, X):                                   # predice la mediana del grupo de cada fila
        return X[self.columna].map(self.medianas_).fillna(self.global_).to_numpy()   # mediana del grupo o la general


class ModeloDosPartes(BaseEstimator, RegressorMixin):
    """Modelo en dos partes para un objetivo con exceso de ceros.

    La primera parte estima la probabilidad de que el alojamiento registre actividad; la
    segunda, el valor esperado del objetivo entre los que la registran. La prediccion es el
    producto de ambas, que es la esperanza del objetivo cuando una fraccion alta de las
    observaciones vale cero y una regresion unica tiende a predecir valores intermedios para
    todas ellas.
    """

    def __init__(self, clasificador, regresor):        # recibe los dos estimadores ya armados
        self.clasificador = clasificador               # estima la probabilidad de actividad
        self.regresor = regresor                       # estima el valor entre los activos

    def fit(self, X, y):                               # entrena las dos partes
        y = np.asarray(y, dtype=float)                 # objetivo como arreglo
        activo = y > 0                                 # marca las filas con actividad
        self.clasificador_ = clone(self.clasificador).fit(X, activo)        # primera parte, sobre todas las filas
        self.regresor_ = clone(self.regresor).fit(X[activo], y[activo])     # segunda parte, solo sobre las activas
        return self                                                         # devuelve el mismo objeto

    def predict(self, X):                              # predice el producto de las dos partes
        probabilidad = self.clasificador_.predict_proba(X)[:, 1]   # probabilidad de que haya actividad
        valor = np.clip(self.regresor_.predict(X), 0, None)        # valor esperado entre los activos, sin negativos
        return probabilidad * valor                                 # esperanza del objetivo


def guardar(tabla, nombre):
    """Guarda una tabla de resultados en la carpeta de salidas y devuelve la ruta."""
    ruta = SALIDAS / nombre                            # ruta completa del archivo
    tabla.to_csv(ruta, encoding="utf-8")               # lo escribe en UTF-8 para conservar las tildes
    return ruta                                        # devuelve la ruta para poder imprimirla

"""
Sentimiento de comentarios Airbnb Bogota (espanol + ingles)
===========================================================

Metodo (estandar tecnico):
  - Ingles : VADER (Hutto & Gilbert, 2014) tal cual, sin cambios.
  - Espanol: el MISMO motor de VADER, adaptado al espanol:
      * lexico en espanol anclado a la escala de VADER (-4 a +4), construido sobre
        las palabras mas frecuentes de los propios comentarios,
      * negaciones (no, nunca, sin, ni...), intensificadores (muy, super...) y
        el conector contrastivo "pero" en espanol.
  - Puntaje: 'compound' de VADER, normalizado entre -1 y +1.
  - Clase  : umbrales estandar de VADER (Hutto & Gilbert, 2014):
        compound >=  0.05  -> positivo
        compound <= -0.05  -> negativo
        en otro caso       -> neutro
  - Otros idiomas (portugues, frances, aleman...): sin clasificar.

Uso:
    pip install pandas vaderSentiment lingua-language-detector
    python sentimiento_es.py reviews.csv.gz reviews_sentimiento.csv.gz
"""
import re
import sys
import unicodedata

import pandas as pd
from vaderSentiment import vaderSentiment as vs

UMBRAL = 0.05  # estandar VADER

# --------------------------------------------------------------------------
# 1. Lexico en espanol (sin tildes, minusculas). Valencia en escala VADER.
#    Cada palabra se ancla al equivalente en ingles del lexico de VADER.
# --------------------------------------------------------------------------
LEXICO_ES = """
# --- muy positivas (equivalentes: excellent, perfect, amazing, wonderful...) ---
excelente 2.7
excelentes 2.7
exelente 2.7
execelente 2.7
excelentemente 2.5
perfecto 2.7
perfecta 2.7
perfectos 2.7
perfectas 2.7
perfectamente 2.5
impecable 2.5
impecables 2.5
impecablemente 2.4
maravilloso 2.8
maravillosa 2.8
maravillosos 2.8
maravillosas 2.8
maravilla 2.7
espectacular 3.0
espectaculares 3.0
increible 2.8
increibles 2.8
increiblemente 2.6
genial 2.8
geniales 2.8
fantastico 2.7
fantastica 2.7
fenomenal 2.8
estupendo 2.6
estupenda 2.6
fabuloso 2.7
magnifico 2.8
magnifica 2.8
extraordinario 2.7
extraordinaria 2.7
excepcional 2.8
inmejorable 2.8
inigualable 2.7
insuperable 2.7
impresionante 2.3
inolvidable 2.3
divino 2.5
divina 2.5
magico 2.2
joya 2.2
bellisimo 2.9
hermoso 2.7
hermosa 2.7
hermosos 2.7
hermosas 2.7
precioso 2.7
preciosa 2.7
bello 2.4
bella 2.4
bonito 2.0
bonita 2.0
bonitos 2.0
bonitas 2.0
lindo 2.2
linda 2.2
lindos 2.2
lindas 2.2
encantador 2.6
encantadora 2.6
encantado 2.5
encantada 2.5
encantados 2.5
encanto 2.3
agrado 1.8
agrada 1.8
placido 1.8
super 1.5
encanta 2.5
encantaria 2.2
amamos 3.0
ame 2.8
amor 2.5
chevere 1.8
bacano 2.0
# --- positivas generales (good, nice, clean, comfortable, friendly...) ---
bien 1.6
gran 1.8
optimo 2.0
optima 2.0
optimos 2.0
optimas 2.0
envidiable 2.0
buen 1.9
buena 1.9
bueno 1.9
buenos 1.9
buenas 1.9
buenisimo 2.6
buenisima 2.6
mejor 1.9
mejores 1.9
comodo 1.8
comoda 1.8
comodos 1.8
comodas 1.8
comodamente 1.7
comodidad 1.6
comodidades 1.5
confortable 1.9
confortables 1.9
confort 1.5
acogedor 2.0
acogedora 2.0
acogedores 2.0
acojedor 2.0
calido 1.8
calida 1.8
calidez 2.0
limpio 1.8
limpia 1.8
limpios 1.8
limpias 1.8
limpieza 1.6
aseado 1.7
aseada 1.7
aseo 1.2
ordenado 1.5
ordenada 1.5
organizado 1.5
organizada 1.5
pulcro 1.8
higienico 1.4
# --- trato del anfitrion ---
amable 2.2
amables 2.2
amablemente 2.0
amabilidad 2.3
atento 2.0
atenta 2.0
atentos 2.0
atentas 2.0
atencion 1.2
atenciones 1.2
cordial 2.0
cordiales 2.0
cordialidad 2.0
gentil 2.2
gentiles 2.2
servicial 2.0
serviciales 2.0
hospitalidad 2.0
hospitalaria 2.0
calidad 1.4
educado 1.6
educada 1.6
educados 1.6
respetuoso 1.6
respetuosa 1.6
respetuosos 1.6
diligente 1.6
diligentes 1.6
proactivo 1.6
proactiva 1.6
colaborador 1.5
colaboradora 1.5
colaboradores 1.5
colaboracion 1.4
amigable 2.0
amigables 2.0
carino 2.0
empatia 1.8
comprensivo 1.6
comprensiva 1.6
profesional 1.5
responsable 1.4
puntual 1.3
eficiente 1.7
eficaz 1.6
bienvenida 1.8
# --- recomendacion y fidelidad ---
recomendado 1.9
recomendada 1.9
recomendados 1.9
recomendable 1.9
recomiendo 1.9
recomendamos 1.9
recomendaria 1.9
recomendariamos 1.9
recomendadisimo 2.6
recomendo 1.9
recomendando 1.9
recomendar 1.5
recomienda 1.6
recomendare 1.9
volveria 2.0
volveriamos 2.0
volveremos 2.0
volvere 2.0
volvemos 1.8
volver 1.2
regresaria 2.0
regresariamos 2.0
regresaremos 2.0
regresare 2.0
repetiria 2.0
repetire 2.0
repetir 1.4
# --- gratitud y emocion positiva ---
gracias 1.7
agradecido 1.9
agradecida 1.9
agradecidos 1.9
agradezco 1.9
agradecemos 1.9
agradecer 1.6
agradecimiento 1.8
felicitaciones 2.2
felicidades 2.2
bendiciones 2.2
bendiga 2.2
agradable 1.9
agradables 1.9
placentero 2.0
placentera 2.0
placer 1.8
grato 1.9
grata 1.9
ameno 1.6
amena 1.6
tranquilo 1.7
tranquila 1.7
tranquilos 1.7
tranquilas 1.7
tranquilidad 1.7
tranquilamente 1.5
silencioso 1.5
silenciosa 1.5
paz 1.8
relajante 1.8
relajarse 1.4
calmado 1.4
disfrutar 2.0
disfrutamos 2.2
disfrute 2.0
feliz 2.7
felices 2.7
contento 2.3
contenta 2.3
contentos 2.3
satisfecho 2.0
satisfecha 2.0
satisfechos 2.0
satisfactoria 1.8
gusto 1.2
gusta 1.6
gustado 1.7
# --- seguridad, comodidad funcional, ubicacion ---
seguro 1.4
segura 1.4
seguros 1.4
seguras 1.4
seguridad 1.4
confiable 1.8
confianza 1.6
ideal 2.0
ideales 2.0
practico 1.4
practica 1.4
funcional 1.3
conveniente 1.5
conveniencia 1.3
util 1.3
utiles 1.3
completo 1.3
completa 1.3
completos 1.3
equipado 1.4
equipada 1.4
dotado 1.2
dotada 1.2
moderno 1.5
moderna 1.5
modernos 1.5
modernas 1.5
amplio 1.4
amplia 1.4
amplios 1.4
espacioso 1.6
espaciosa 1.6
luminoso 1.6
iluminado 1.3
privilegiada 1.8
estrategica 1.3
estrategico 1.3
lujo 1.4
lujoso 1.6
elegante 1.6
# --- rapidez, ayuda, solucion ---
rapido 1.2
rapida 1.2
rapidamente 1.3
rapidez 1.4
inmediata 1.2
inmediato 1.2
inmediatamente 1.2
oportuna 1.2
oportunamente 1.2
prontitud 1.4
pronta 1.2
ayudo 1.4
ayudaron 1.4
ayuda 1.0
ayudar 0.8
apoyo 1.4
soluciono 1.0
solucionaron 1.0
resolvio 1.0
resolvieron 1.0
resuelve 1.0
soluciona 1.0
responde 0.8
respondio 1.0
respondia 1.0
respondieron 1.0
contesta 0.8
flexible 1.3
flexibles 1.3
flexibilidad 1.3
dispuesto 1.4
dispuesta 1.4
dispuestos 1.4
disposicion 1.4
facil 1.2
faciles 1.2
facilmente 1.2
facilidad 1.2
facilidades 1.2
facilita 1.2
accesible 1.0
accesibles 1.0
accesibilidad 1.0
claras 0.8
clara 0.8
claros 0.8
claridad 1.0
# --- cumplimiento de expectativas y precio ---
cumple 1.0
cumplio 1.2
cumplen 1.0
cumplia 1.0
coincide 0.8
coincidia 0.8
coinciden 0.8
supero 1.8
supera 1.8
destaco 1.3
destaca 1.3
destacar 1.3
economico 1.2
economicos 1.2
barato 1.2
razonable 1.0
plus 1.0
ventaja 1.2
beneficio 1.2
exito 1.8
# --- muy negativas (terrible, horrible, worst, dirty, disgusting...) ---
pesimo -3.0
pesima -3.0
horrible -2.8
terrible -2.6
espantoso -3.0
asqueroso -3.0
asco -2.8
lamentable -2.5
peor -2.8
estafa -3.0
engano -2.6
enganoso -2.5
mentira -2.5
insoportable -2.6
cucarachas -2.8
plaga -2.8
chinches -3.0
ratas -2.8
moho -2.2
# --- negativas generales (bad, dirty, noisy, broken, problem...) ---
malo -2.3
mala -2.3
malos -2.3
malas -2.3
mal -2.2
deficiente -2.0
desagradable -2.3
desagradables -2.3
feo -2.0
fea -2.0
sucio -2.4
sucia -2.4
sucios -2.4
sucias -2.4
suciedad -2.4
polvo -1.2
manchas -1.5
manchadas -1.6
manchado -1.6
basura -1.8
pelos -1.5
humedad -1.8
humedo -1.2
humedas -1.2
olor -0.8
olores -1.0
huele -1.0
olia -1.5
olian -1.5
ruido -1.4
ruidos -1.4
ruidoso -1.6
ruidosa -1.6
ruidosos -1.6
bulla -1.4
molesto -1.6
molesta -1.6
molestia -1.6
incomodo -1.8
incomoda -1.8
incomodas -1.8
incomodidad -1.6
inseguro -2.0
insegura -2.0
inseguridad -2.0
peligroso -2.0
peligrosa -2.0
riesgo -1.4
problema -1.4
problemas -1.5
inconveniente -1.2
inconvenientes -1.2
percance -1.4
complicaciones -1.2
complicado -1.2
complicada -1.2
dificil -1.0
dificultad -1.2
dificultades -1.2
dificulta -1.0
imposible -1.6
falla -1.5
inestable -1.4
intermitente -1.2
goteras -1.8
mosquitos -1.5
zancudos -1.5
fallas -1.6
fallo -1.4
danado -1.8
danada -1.8
dano -1.8
roto -1.8
rota -1.8
queja -1.8
quejas -1.8
quejarme -1.8
decepcion -2.0
decepcionante -2.2
decepcionado -2.2
decepcionada -2.2
lamentablemente -1.8
lastima -1.7
lastimosamente -1.8
desafortunadamente -1.8
falso -2.0
falsa -2.0
caro -1.4
costoso -1.2
cobro -1.0
cobraron -1.2
frio -0.8
fria -0.8
viejo -1.0
vieja -1.0
viejos -1.0
viejas -1.0
oscuro -1.2
oscura -1.2
pequeno -0.6
pequena -0.6
pequenos -0.6
pequenas -0.6
reducido -1.0
estrecho -1.0
incompleto -1.2
falta -1.2
faltan -1.2
falto -1.2
faltaba -1.2
faltaron -1.2
demora -1.2
demoro -1.2
retraso -1.2
lento -1.2
lenta -1.2
regular -1.2
mediocre -1.8
pesado -0.8
mejorar -0.8
mejoraria -0.8
mejoraran -0.8
mejore -0.6
"""

NEGADORES_ES = [
    "no", "nunca", "jamas", "ni", "tampoco", "nada", "nadie",
    "ninguno", "ninguna", "ningun", "ningunos", "ningunas", "sin",
]

INTENSIFICADORES_ES = {
    # aumentan
    "muy": vs.B_INCR, "bastante": vs.B_INCR,
    "demasiado": vs.B_INCR, "sumamente": vs.B_INCR, "extremadamente": vs.B_INCR,
    "absolutamente": vs.B_INCR, "totalmente": vs.B_INCR, "completamente": vs.B_INCR,
    "realmente": vs.B_INCR, "verdaderamente": vs.B_INCR, "excesivamente": vs.B_INCR,
    "especialmente": vs.B_INCR, "definitivamente": vs.B_INCR, "tan": vs.B_INCR,
    "mucho": vs.B_INCR, "muchisimo": vs.B_INCR, "mucha": vs.B_INCR,
    "muchos": vs.B_INCR, "muchas": vs.B_INCR, "altamente": vs.B_INCR,
    # disminuyen
    "poco": vs.B_DECR, "poca": vs.B_DECR, "pocos": vs.B_DECR, "pocas": vs.B_DECR,
    "algo": vs.B_DECR, "relativamente": vs.B_DECR, "ligeramente": vs.B_DECR,
    "apenas": vs.B_DECR, "casi": vs.B_DECR, "medianamente": vs.B_DECR,
}

# emojis frecuentes -> palabras con valencia (el lexico de emojis de VADER esta en ingles)
EMOJIS_ES = {
    "emoposfuerte": (3.0, "😍🥰❤💯🤩💖💕😘"),
    "emopos": (2.2, "😊😀😁😃😄🙂😌👍👌🙌👏⭐🌟✨🥳😎🤗🙏💙💚💛🧡💜😉🎉💫🌸"),
    "emonegfuerte": (-3.0, "🤮🤢😡😠👎"),
    "emoneg": (-2.0, "😞😢😭😒🙁☹😕😔😤😫😩"),
}


EMOTICONOS = {":)": "emopos", ":-)": "emopos", "=)": "emopos", ":D": "emopos", "(:": "emopos",
              "<3": "emoposfuerte", ":(": "emoneg", ":-(": "emoneg"}


def _plegar(texto):
    """Quita tildes y la n con tilde (excelente = excélente), deja mayusculas y emojis."""
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def _cargar_lexico():
    lex = {}
    for linea in LEXICO_ES.splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        palabra, valor = linea.rsplit(" ", 1)
        lex[palabra] = float(valor)
    for token, (valor, _) in EMOJIS_ES.items():
        lex[token] = valor
    return lex


# signos que cortan el alcance de una negacion o de un intensificador
CORTES = {",", ".", ";", ":", "!", "?", "pero", "|"}


def _escalar(palabra, valence, hay_mayusculas):
    """Efecto de un intensificador (muy, poco...) sobre la valencia de la palabra que sigue."""
    base = INTENSIFICADORES_ES.get(palabra.lower(), 0.0)
    if base == 0.0:
        return 0.0
    if valence < 0:
        base = -base
    if palabra.isupper() and hay_mayusculas:
        base += vs.C_INCR if base > 0 else -vs.C_INCR
    return base


class VaderEspanol(vs.SentimentIntensityAnalyzer):
    """Motor de VADER con lexico, negaciones, intensificadores y 'pero' en espanol."""

    def __init__(self):
        super().__init__()                       # carga lo de ingles (se reemplaza abajo)
        self.lexicon = _cargar_lexico()
        self.emojis = {e: token for token, (_, chars) in EMOJIS_ES.items() for e in chars}

    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        palabras = sentitext.words_and_emoticons
        bajas = [str(w).lower() for w in palabras]
        if bajas[i] in self.lexicon:
            valence = self.lexicon[bajas[i]]
            if item.isupper() and sentitext.is_cap_diff:            # PALABRA EN MAYUSCULAS
                valence += vs.C_INCR if valence > 0 else -vs.C_INCR
            # intensificadores: hasta 3 palabras antes (pesan menos con la distancia)
            for k in range(3):
                if i > k and bajas[i - (k + 1)] not in self.lexicon:
                    s = _escalar(palabras[i - (k + 1)], valence, sentitext.is_cap_diff)
                    s *= (1.0, 0.95, 0.9)[k]
                    valence += s
            # negacion: UNA sola vez, hasta 3 palabras antes, sin cruzar comas/puntos/'pero'
            for d in range(1, 4):
                if i - d < 0 or bajas[i - d] in CORTES:
                    break
                if bajas[i - d] in NEGADORES_ES:
                    valence *= vs.N_SCALAR
                    break
        sentiments.append(valence)
        return sentiments

    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        palabras = [str(w).lower() for w in words_and_emoticons]
        if "pero" in palabras:
            bi = palabras.index("pero")
            for si, s in enumerate(sentiments):
                if si < bi:
                    sentiments[si] = s * 0.5
                elif si > bi:
                    sentiments[si] = s * 1.5
        return sentiments

    def puntaje(self, texto):
        t = _plegar(texto)
        for emoticon, token in EMOTICONOS.items():                  # :) :( <3 ...
            t = t.replace(emoticon, f" {token} ")
        t = re.sub(r"(.)\1{2,}", r"\1", t)                         # encantooo -> encanto
        t = re.sub(r"(?i)\bsin (ninguna )?duda\b|\bsin dudarlo\b", " definitivamente ", t)
        t = re.sub(r"(?i)\bsin embargo\b|\bno obstante\b", " pero ", t)
        # VADER borra las comas y puntos al tokenizar; los dejamos como token '|' para que
        # una negacion no cruce de una frase a otra. '!' y '?' se conservan (enfasis).
        t = re.sub(r"[,.;:](?=\s|$)", " | ", t)
        t = re.sub(r"([!?]+)", r" \1 ", t)
        return self.polarity_scores(t)["compound"]


def clasificar(compound):
    """Umbrales estandar de VADER."""
    if compound >= UMBRAL:
        return "positivo"
    if compound <= -UMBRAL:
        return "negativo"
    return "neutro"


# --------------------------------------------------------------------------
# 2. Pipeline sobre la tabla de comentarios
# --------------------------------------------------------------------------
def _detector():
    from lingua import Language, LanguageDetectorBuilder
    langs = [Language.SPANISH, Language.ENGLISH, Language.PORTUGUESE, Language.FRENCH,
             Language.GERMAN, Language.ITALIAN, Language.DUTCH, Language.RUSSIAN,
             Language.CHINESE, Language.JAPANESE, Language.KOREAN]
    return LanguageDetectorBuilder.from_languages(*langs).with_low_accuracy_mode().build()


_DET = None


def _init():
    global _DET
    _DET = _detector()


def _idioma(texto):
    lang = _DET.detect_language_of(texto)
    return lang.iso_code_639_1.name.lower() if lang else "xx"


ES_STOP = set("el la los las de del que y en un una con es muy para por se lo al mi todo bien fue estuvo nos sus como pero".split())
EN_STOP = set("the and was is very are were with for this that had have from our you great place nice good stay we it of to in".split())
PT_MARCAS = re.compile(r"[ãõç]|\b(não|muito|você|obrigad[oa]|ótim[oa]|foi|tudo|estadia|fomos|também|muita)\b", re.I)


def corregir_idioma(texto, idioma):
    """lingua falla con textos cortos ('Excelente' sale como portugues): se corrige con palabras de apoyo."""
    toks = re.findall(r"[a-z]+", _plegar(texto).lower())
    if not toks:
        return idioma
    es_r = sum(t in ES_STOP for t in toks) / len(toks)
    en_r = sum(t in EN_STOP for t in toks) / len(toks)
    if idioma == "pt" and not PT_MARCAS.search(texto):
        return "es"                                  # sin marcas de portugues: en Bogota es espanol
    if idioma == "es" and en_r >= 0.25 and en_r > es_r + 0.10:
        return "en"
    if idioma == "en" and es_r >= 0.25 and es_r > en_r + 0.10:
        return "es"
    if idioma == "xx":
        return "es" if es_r > en_r else ("en" if en_r > es_r else "xx")
    return idioma


def agregar_sentimiento(reviews):
    """Recibe la tabla reviews (con 'comments') y agrega: texto, idioma, puntaje_sentimiento, sentimiento."""
    from multiprocessing import Pool

    r = reviews.copy()
    r["texto"] = (r["comments"].fillna("").astype(str)
                  .str.replace(r"<[^>]+>", " ", regex=True)       # quita <br/> y similares
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    with Pool(initializer=_init) as p:
        r["idioma"] = p.map(_idioma, r["texto"].tolist(), chunksize=2000)
    r["idioma"] = [corregir_idioma(t, i) for t, i in zip(r["texto"], r["idioma"])]
    r.loc[r["texto"] == "", "idioma"] = "vacio"

    en = vs.SentimentIntensityAnalyzer()
    # el ingles se puntua ANTES de cambiar las constantes del modulo para espanol
    r["puntaje_sentimiento"] = float("nan")
    m_en = r["idioma"] == "en"
    r.loc[m_en, "puntaje_sentimiento"] = [en.polarity_scores(t)["compound"] for t in r.loc[m_en, "texto"]]

    es = VaderEspanol()
    m_es = r["idioma"] == "es"
    r.loc[m_es, "puntaje_sentimiento"] = [es.puntaje(t) for t in r.loc[m_es, "texto"]]

    r["sentimiento"] = r["puntaje_sentimiento"].map(lambda x: clasificar(x) if x == x else "sin clasificar")
    return r.drop(columns=["texto"])


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "reviews.csv.gz"
    salida = sys.argv[2] if len(sys.argv) > 2 else "reviews_sentimiento.csv.gz"
    df = pd.read_csv(entrada)
    out = agregar_sentimiento(df)
    out.to_csv(salida, index=False)
    print(out["sentimiento"].value_counts(normalize=True).round(3))
    print("guardado en", salida)

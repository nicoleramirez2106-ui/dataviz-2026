"""Plan de entrenamiento: diagnóstico de una pareja frente a las parejas que ganan en la AVP.

Todo el diagnóstico sale de los datos, sin IA:
1. Se toman las estadísticas por set de las parejas de la AVP con estadísticas completas
   (las mismas filas que usan los perfiles de juego).
2. Cada estadística de la pareja se ubica en la distribución de las parejas GANADORAS
   (percentil: a qué porcentaje de ganadoras supera).
3. La prioridad de cada aspecto combina qué tan lejos está de la mediana de las ganadoras
   y cuánto separa esa estadística a ganadores de perdedores (|correlación rango-biserial|).
4. La pareja se asigna al perfil de juego más cercano (K-means de modelos/perfiles.json)
   y se mira cuánto ganaron las parejas que juegan parecido.

La IA (paso 3) solo redacta el plan a partir de este diagnóstico; no calcula nada.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import datos as D

MODELOS = Path(__file__).resolve().parent / "modelos"
STATS = ["kills", "attacks", "errors", "aces", "serve_errors", "blocks", "digs"]

# Cómo se lee cada estadística: 1 = más es mejor, -1 = menos es mejor, 0 = solo contexto
SENTIDO = {"kills_set": 1, "hitpct": 1, "aces_set": 1, "blocks_set": 1, "digs_set": 1,
           "errors_set": -1, "serve_errors_set": -1, "attacks_set": 0}
ETQ = {"kills_set": "Kills por set", "attacks_set": "Ataques por set", "errors_set": "Errores de ataque por set",
       "aces_set": "Aces por set", "serve_errors_set": "Errores de saque por set",
       "blocks_set": "Bloqueos por set", "digs_set": "Defensas por set", "hitpct": "% de ataque"}
N_PARECIDAS = 500          # parejas más parecidas con las que se compara
MIN_SETS = 1


# --------------------------------------------------------------------------
# Referencia: parejas de la AVP con estadísticas completas, por set
# --------------------------------------------------------------------------
def _referencia():
    a = D.E[(D.E.circuit.astype(str) == "AVP") & (D.E.tiene_stats_completas == 1)]
    a = a.merge(D.P[["match_id", "n_sets", "es_forfeit"]], on="match_id", how="left")
    a = a[(a.es_forfeit != 1) & (a.n_sets >= MIN_SETS)].copy()
    for s in STATS:
        a[f"{s}_set"] = a[s] / a.n_sets
    return a.replace([np.inf, -np.inf], np.nan).dropna(subset=list(SENTIDO)).reset_index(drop=True)


REF = _referencia()
GAN = REF[REF.resultado == "Ganador"]
PERD = REF[REF.resultado == "Perdedor"]


def _rbc(v):
    """Correlación rango-biserial ganadores vs perdedores (la misma medida del libro)."""
    u = mannwhitneyu(GAN[v], PERD[v], alternative="two-sided").statistic
    return 2 * u / (len(GAN) * len(PERD)) - 1


PODER = {v: abs(_rbc(v)) for v, s in SENTIDO.items() if s != 0}

try:
    PERFILES = json.loads((MODELOS / "perfiles.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    PERFILES = None

# Valores de ejemplo para el formulario: totales enteros de un partido típico a 2 sets
SETS_EJEMPLO = 2


def _ejemplo(filas):
    return {"sets": SETS_EJEMPLO, **{s: int(round(float(filas[f"{s}_set"].median()) * SETS_EJEMPLO)) for s in STATS}}


EJEMPLOS = {"tipico": _ejemplo(REF), "pierde": _ejemplo(PERD)}


# --------------------------------------------------------------------------
# Diagnóstico
# --------------------------------------------------------------------------
def _entero(v):
    """True si v es un número entero (acepta 3 y 3.0, rechaza 1.5)."""
    try:
        return float(v) == int(float(v))
    except (TypeError, ValueError):
        return False


def validar(valores):
    """Devuelve un mensaje de error o None si los datos tienen sentido.
    valores: totales (conteos enteros) de la pareja y número de sets jugados."""
    campos = ["sets"] + STATS
    if any(valores.get(c) is None for c in campos):
        return "Complete los sets jugados y las siete estadísticas."
    if not all(_entero(valores[c]) for c in campos):
        return "Use solo números enteros: son conteos (no se puede hacer 1,5 errores)."
    if any(valores[c] < 0 for c in campos):
        return "Las estadísticas no pueden ser negativas."
    if valores["sets"] < 1:
        return "Los sets jugados deben ser al menos 1."
    if valores["attacks"] <= 0:
        return "Los ataques deben ser mayores que 0."
    if valores["kills"] + valores["errors"] > valores["attacks"]:
        return "Kills + errores de ataque no pueden superar los ataques: revise los valores."
    return None


def _percentil(serie, x, sentido):
    """% de parejas ganadoras a las que la pareja supera (en el sentido en que la estadística es mejor)."""
    if sentido >= 0:
        return 100 * ((serie < x).mean() + 0.5 * (serie == x).mean())
    return 100 * ((serie > x).mean() + 0.5 * (serie == x).mean())


def _perfil(x):
    """Perfil de juego más cercano (centroides del K-means, en desviaciones estándar)."""
    if PERFILES is None:
        return None
    vs = PERFILES["variables"]
    z = np.array([(x[v] - PERFILES["media_global"][v]) / PERFILES["de_global"][v] for v in vs])
    dist = [np.linalg.norm(z - np.array([p["z"][v] for v in vs])) for p in PERFILES["perfiles"]]
    p = PERFILES["perfiles"][int(np.argmin(dist))]
    return {"nombre": p["nombre"], "pct_victorias": p["pct_victorias"], "n": p["n"]}


def _parecidas(x):
    """% de victorias de las N parejas de la AVP que juegan más parecido (distancia en desviaciones estándar)."""
    vs = list(SENTIDO)
    media, de = REF[vs].mean(), REF[vs].std()
    z_ref = (REF[vs] - media) / de
    z = (pd.Series(x)[vs] - media) / de
    d = np.sqrt(((z_ref - z) ** 2).sum(axis=1))
    cerca = REF.loc[d.nsmallest(N_PARECIDAS).index]
    return {"n": len(cerca), "pct_victorias": 100 * (cerca.resultado == "Ganador").mean()}


def diagnosticar(valores):
    """valores: totales de kills, attacks, errors, aces, serve_errors, blocks y digs, y los sets jugados.
    Se pasan a promedio por set para compararlos con la AVP (el % de ataque se calcula solo)."""
    sets = int(valores["sets"])
    x = {f"{s}_set": int(valores[s]) / sets for s in STATS}
    x["hitpct"] = (x["kills_set"] - x["errors_set"]) / x["attacks_set"]

    filas = []
    for v, sentido in SENTIDO.items():
        pct = _percentil(GAN[v], x[v], sentido) if sentido else None
        deficit = max(0.0, 50 - pct) / 50 if pct is not None else 0.0
        filas.append({"var": v, "etiqueta": ETQ[v], "sentido": sentido, "valor": x[v],
                      "mediana_ganadores": float(GAN[v].median()), "mediana_perdedores": float(PERD[v].median()),
                      "percentil": pct, "poder": PODER.get(v, 0.0), "prioridad": deficit * PODER.get(v, 0.0)})
    t = pd.DataFrame(filas)
    comparables = t[t.sentido != 0]
    brechas = (comparables[comparables.percentil < 50].sort_values("prioridad", ascending=False).head(3))
    fortalezas = comparables[comparables.percentil >= 60].sort_values("percentil", ascending=False).head(2)
    return {"valores": x, "tabla": t, "brechas": brechas.to_dict("records"),
            "fortalezas": fortalezas.to_dict("records"), "perfil": _perfil(x), "parecidas": _parecidas(x),
            "n_ganadores": len(GAN), "n_parejas": len(REF),
            "anios": (int(REF.year.min()), int(REF.year.max()))}


# --------------------------------------------------------------------------
# Paso 2: plan semanal con plantilla (sin IA)
# Cada estadística débil tiene ejercicios concretos de voleibol de playa. El plan reparte las
# prioridades del diagnóstico en las sesiones de la semana y fija metas medibles.
# --------------------------------------------------------------------------
EJERCICIOS = {
    "hitpct": [
        ("Ataque a zonas", "Marque tres objetivos con conos (línea, diagonal corta y fondo). Tras recepción y "
         "colocación, 3 series de 10 ataques; meta: 6 de 10 en el objetivo."),
        ("Decidir el golpe", "El compañero muestra tarde 'línea' o 'diagonal' como si bloqueara; el atacante elige "
         "golpe fuerte o shot al lado libre. 3 series de 8."),
    ],
    "kills_set": [
        ("Transición defensa-ataque", "El entrenador ataca desde un cajón, la pareja defiende, coloca y contraataca. "
         "4 series de 8; cuenta solo el punto ganado."),
        ("Ataque tras recepción difícil", "Saques fuertes a los pies y a la costura entre los dos; terminar la jugada "
         "con ataque. 3 series de 10."),
    ],
    "errors_set": [
        ("Serie sin errores", "Ataques al 80 % de potencia buscando la cancha libre: series de 10 en las que un error "
         "reinicia la cuenta. 4 series completas."),
        ("Leer el bloqueo", "Atacante contra bloqueador real: ante bloqueo cerrado, golpe seguro (shot o pokey) en "
         "lugar de potencia. 3 series de 8."),
    ],
    "aces_set": [
        ("Saque a la costura", "Saque dirigido al espacio entre los dos receptores y al fondo de la línea. 5 series de "
         "10 con conos; meta: 4 de 10 en zona."),
        ("Variar el saque", "Alternar flotado y en salto en la misma serie, eligiendo según el viento. 3 series de 10."),
    ],
    "serve_errors_set": [
        ("Saque bajo presión", "Series de 10 saques que deben entrar todos; un error reinicia la serie. Terminar 3 "
         "series completas."),
        ("Rutina de saque", "Fijar una rutina corta (respiración, botes, punto de mira) y repetirla en cada saque. "
         "40 saques con la misma rutina."),
    ],
    "blocks_set": [
        ("Desplazamiento y salto de bloqueo", "Pasos laterales en arena y salto con manos firmes sobre la red. "
         "4 series de 8 contra atacante."),
        ("Señales de bloqueo", "Practicar las señales (línea, diagonal, retirarse) y que el defensor cubra el lado "
         "libre. 3 series de 10 jugadas."),
    ],
    "digs_set": [
        ("Defensa de lectura", "El entrenador alterna ataque fuerte y shot desde un cajón; el defensor lee el brazo y "
         "se mueve antes del golpe. 4 series de 10."),
        ("Velocidad en la arena", "Sprints cortos laterales y hacia delante con caída al balón. 6 de 15 segundos con "
         "descanso de 45."),
    ],
}
JUEGO = {
    "hitpct": "Juego 2 contra 2: el punto ganado con ataque en zona marcada vale doble.",
    "kills_set": "Juego 2 contra 2 que empieza siempre con un ataque del entrenador: solo se cuenta el contraataque.",
    "errors_set": "Juego 2 contra 2: cada error de ataque le da dos puntos al rival.",
    "aces_set": "Juego a 15 puntos en el que cada saque en zona marcada suma un punto extra.",
    "serve_errors_set": "Juego a 15 puntos: un error de saque le da dos puntos al rival.",
    "blocks_set": "Juego 2 contra 2 en el que el bloqueo que termina en punto vale doble.",
    "digs_set": "Juego 2 contra 2 sin saque: el entrenador lanza ataques y la pareja gana el punto solo si defiende y contraataca.",
}
MANTENER = "Bloque corto de mantenimiento: unas pocas repeticiones de lo que ya hacen bien, sin cansarse."


def _meta(b):
    """Meta medible para una brecha: llegar a la mediana de las parejas ganadoras."""
    def f(x):
        if b["var"] == "hitpct":                       # el % de ataque se muestra como porcentaje
            return f"{100 * x:.1f} %".replace(".", ",")
        return f"{x:.1f}".replace(".", ",")
    verbo = "Subir" if b["sentido"] > 0 else "Bajar"
    return {"var": b["var"], "etiqueta": b["etiqueta"],
            "texto": f"{verbo} {b['etiqueta'].lower()} de {f(b['valor'])} a {f(b['mediana_ganadores'])}",
            "nota": "la mediana de las parejas que ganan en la AVP"}


def plantilla(diag, dias=3):
    """Plan de la semana a partir del diagnóstico. Devuelve un diccionario con metas y sesiones.

    Reparte las prioridades así: la prioridad 1 aparece en todas las sesiones (bloque principal o
    secundario), la 2 y la 3 se alternan, y cada sesión cierra con un juego que premia la prioridad del día.
    """
    brechas = diag["brechas"]
    fortalezas = diag["fortalezas"]
    if not brechas:          # sin brechas: mantenimiento con las dos estadísticas más importantes
        orden = sorted(PODER, key=lambda v: -PODER[v])[:2]
        brechas = [{"var": v, "etiqueta": ETQ[v]} for v in orden]
        metas = [{"var": None, "etiqueta": "Mantener", "texto": "Mantener el nivel en todas las estadísticas",
                  "nota": "su pareja ya está en la mitad alta de las ganadoras"}]
    else:
        metas = [_meta(b) for b in brechas]
    claves = [b["var"] for b in brechas]
    usados = {v: 0 for v in claves}      # rota los ejercicios de cada estadística para no repetirlos seguidos

    def siguiente(v):
        e = EJERCICIOS[v][usados[v] % len(EJERCICIOS[v])]
        usados[v] += 1
        return e
    sesiones = []
    for d in range(dias):
        principal = claves[d % len(claves)]
        secundaria = claves[(d + 1) % len(claves)] if len(claves) > 1 else claves[0]
        e1 = siguiente(principal)
        e2 = siguiente(secundaria)
        bloques = [
            (15, "Calentamiento", "Movilidad, desplazamientos en arena y toques en pareja."),
            (30, e1[0], e1[1]),
            (20, e2[0], e2[1]),
            (15, "Juego con regla", JUEGO[principal]),
            (10, "Vuelta a la calma", "Estiramientos y revisión rápida: ¿cuántas veces se logró la meta del día?"),
        ]
        if fortalezas and d == dias - 1:
            bloques.insert(3, (10, f"Mantener: {fortalezas[0]['etiqueta'].lower()}", MANTENER))
        sesiones.append({"titulo": f"Sesión {d + 1}", "foco": ETQ[principal], "bloques": bloques,
                         "minutos": sum(b[0] for b in bloques)})
    return {"metas": metas, "sesiones": sesiones, "dias": dias,
            "fortalezas": [f["etiqueta"] for f in fortalezas],
            "estilo": (diag["perfil"] or {}).get("nombre")}


def plan_texto(plan):
    """El plan en texto plano, para descargarlo."""
    lineas = ["PLAN DE ENTRENAMIENTO SEMANAL", "", "Metas para las próximas semanas:"]
    lineas += [f"- {m['texto']} ({m['nota']})." for m in plan["metas"]]
    if plan["fortalezas"]:
        lineas.append("Fortalezas a mantener: " + ", ".join(f.lower() for f in plan["fortalezas"]) + ".")
    for s in plan["sesiones"]:
        lineas += ["", f"{s['titulo'].upper()} · foco: {s['foco'].lower()} · {s['minutos']} min"]
        lineas += [f"  {m} min · {t}: {x}" for m, t, x in s["bloques"]]
    lineas += ["", "Plan generado a partir de datos de la AVP (2004–2019). Es una guía: ajústela a su pareja."]
    return "\n".join(lineas)

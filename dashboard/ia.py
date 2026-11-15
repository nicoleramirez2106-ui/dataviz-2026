"""Paso 3 del plan de entrenamiento: la IA (Gemini) redacta el plan.

La IA NO calcula nada: recibe el diagnóstico y la plantilla que ya armó plan.py con los datos,
y solo los redacta como lo haría un preparador. Si no hay clave, no hay internet o Gemini falla,
el dashboard sigue mostrando la plantilla.

La clave se lee de la variable de entorno GEMINI_API_KEY (nunca va escrita en el código ni en GitHub).
Opcional: GEMINI_MODEL para elegir otro modelo.
"""
import json
import os
import urllib.error
import urllib.request

import plan as PL

BASE = "https://generativelanguage.googleapis.com/v1beta"
URL = BASE + "/models/{modelo}:generateContent"
# Se prueban en orden: el primero que responda es el que se usa
MODELOS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-3.5-flash-lite", "gemini-3.8-flash",
                       "gemini-flash-latest", "gemini-2.5-flash"] if m]
TIEMPO_MAX = 45   # segundos

INSTRUCCIONES = """Eres un preparador de voleibol de playa que le escribe a un entrenador.
Vas a recibir el diagnóstico de una pareja (comparada con las parejas que ganan en la AVP) y un plan base
de una semana que ya fue calculado con datos. Tu trabajo es redactarlo de forma clara, práctica y motivadora.

Reglas:
- Usa SOLO las cifras, metas, ejercicios y tiempos del plan base. No inventes estadísticas, porcentajes ni datos.
- Puedes explicar cómo ejecutar cada ejercicio, qué corregir y cómo saber si se está mejorando.
- Trata al lector de "usted". Español neutro. Sin emojis.
- Máximo 450 palabras.
- Formato Markdown con estas secciones, en este orden:
  ## Resumen  (2 o 3 frases: estilo de juego y por qué esas prioridades)
  ## Metas  (una viñeta por meta, con sus cifras)
  ## Sesiones  (### Sesión N · foco; luego una viñeta por bloque con sus minutos y una indicación práctica)
  ## Cómo medir el progreso  (2 o 3 viñetas)
  ## Tenga en cuenta  (una frase: los datos muestran asociaciones con ganar, no garantías)
"""


def hay_clave():
    return bool(os.environ.get("GEMINI_API_KEY"))


def contexto(diag, plan):
    """Lo único que se le envía a la IA: estadísticas de juego y el plan base (nada personal)."""
    def cifra(v, x):
        t = f"{100 * x:.1f} %" if v == "hitpct" else f"{x:.1f}"
        return t.replace(".", ",")
    datos = {
        "estilo_de_juego_mas_parecido": plan["estilo"],
        "victorias_de_parejas_que_juegan_parecido": f"{diag['parecidas']['pct_victorias']:.0f} %",
        "prioridades": [{"estadistica": b["etiqueta"], "valor_de_la_pareja": cifra(b["var"], b["valor"]),
                         "mediana_de_las_ganadoras": cifra(b["var"], b["mediana_ganadores"]),
                         "supera_al_pct_de_ganadoras": f"{b['percentil']:.0f} %"} for b in diag["brechas"]],
        "fortalezas": plan["fortalezas"],
    }
    return ("DIAGNÓSTICO (JSON):\n" + json.dumps(datos, ensure_ascii=False, indent=1) +
            "\n\nPLAN BASE:\n" + PL.plan_texto(plan))


def disponibles(clave):
    """Modelos 'flash' que esta clave puede usar para generar texto (se le pregunta a Google)."""
    req = urllib.request.Request(BASE + "/models?pageSize=200", headers={"x-goog-api-key": clave})
    with urllib.request.urlopen(req, timeout=20) as r:
        datos = json.loads(r.read().decode("utf-8"))
    nombres = [m["name"].split("/")[-1] for m in datos.get("models", [])
               if "generateContent" in m.get("supportedGenerationMethods", [])]
    texto = [n for n in nombres if "flash" in n and not any(x in n for x in ("tts", "image", "live", "audio"))]
    # primero los estables (sin 'preview' ni 'exp'), los 'lite' antes por ser más livianos
    return sorted(texto, key=lambda n: ("preview" in n or "exp" in n, "lite" not in n, n))


def _detalle(err):
    """Descripción corta del error para mostrarla al usuario (nunca incluye la clave)."""
    if isinstance(err, urllib.error.HTTPError):
        try:
            msg = json.loads(err.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            msg = ""
        return f"HTTP {err.code}" + (f": {msg[:160]}" if msg else "")
    if isinstance(err, urllib.error.URLError):
        return f"sin conexión ({err.reason})"
    return type(err).__name__


def _llamar(modelo, texto, clave, instrucciones=None, contenidos=None, temperatura=0.4):
    """Una llamada a Gemini. 'contenidos' permite mandar una conversación (turnos user/model)."""
    cuerpo = {"contents": contenidos or [{"role": "user", "parts": [{"text": texto}]}],
              "generationConfig": {"temperature": temperatura, "maxOutputTokens": 2048}}
    if instrucciones:
        cuerpo["systemInstruction"] = {"parts": [{"text": instrucciones}]}
    req = urllib.request.Request(URL.format(modelo=modelo), data=json.dumps(cuerpo).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": clave})
    with urllib.request.urlopen(req, timeout=TIEMPO_MAX) as r:
        resp = json.loads(r.read().decode("utf-8"))
    partes = resp["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in partes).strip()


_MODELO_OK = None    # el último modelo que respondió: se prueba primero la próxima vez


def generar(instrucciones, contenidos, temperatura=0.4):
    """Prueba los modelos disponibles hasta que uno responda. Devuelve (texto, modelo) o lanza RuntimeError
    con un mensaje corto en español (sin la clave)."""
    global _MODELO_OK
    clave = os.environ.get("GEMINI_API_KEY")
    if not clave:
        raise RuntimeError("No encontré la clave de Gemini (variable GEMINI_API_KEY).")
    ultimo = None
    candidatos = ([_MODELO_OK] if _MODELO_OK else []) + [m for m in MODELOS if m != _MODELO_OK]
    try:                                   # se agregan los modelos que Google dice que esta clave puede usar
        candidatos += [m for m in disponibles(clave) if m not in candidatos]
    except Exception as err:               # si ni siquiera se puede listar, se intenta igual con la lista fija
        ultimo = err
    for modelo in candidatos:
        try:
            salida = _llamar(modelo, None, clave, instrucciones, contenidos, temperatura)
            if salida:
                _MODELO_OK = modelo
                return salida, modelo
        except urllib.error.HTTPError as err:
            ultimo = err
            if err.code == 402:                       # el proyecto de la clave está en prepago sin saldo
                raise RuntimeError("El proyecto de Google de la clave está en prepago y sin saldo; use una clave "
                                   "del nivel gratuito (sin facturación).") from err
            if err.code in (401, 403) or (err.code == 400 and "API key" in _detalle(err)):
                raise RuntimeError("Gemini rechazó la clave (revise que GEMINI_API_KEY esté bien copiada).") from err
            continue                                  # 404 (modelo no disponible), 429 (límite), 5xx: probar otro
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as err:
            ultimo = err
            continue
    if isinstance(ultimo, urllib.error.HTTPError) and ultimo.code == 429:
        raise RuntimeError("Se alcanzó el límite gratuito de Gemini por ahora; intente en un rato.")
    raise RuntimeError("No se pudo obtener respuesta de Gemini. "
                       f"Detalle técnico: {_detalle(ultimo) if ultimo else 'sin respuesta'}.")


def redactar(diag, plan):
    """Plan redactado por la IA. Devuelve (texto_markdown, modelo_usado) o lanza RuntimeError."""
    contenidos = [{"role": "user", "parts": [{"text": contexto(diag, plan)}]}]
    try:
        return generar(INSTRUCCIONES, contenidos)
    except RuntimeError as err:
        raise RuntimeError(f"{err} Mientras tanto, use el plan de arriba: es el mismo contenido, sin la "
                           "redacción de la IA.") from err


# --------------------------------------------------------------------------
# Chatbot: responde preguntas sobre el proyecto usando SOLO la ficha de datos que arma app.py
# --------------------------------------------------------------------------
INSTRUCCIONES_CHAT = """Eres el asistente del "Panel del entrenador", un dashboard sobre qué hace ganar a una pareja de
voleibol de playa (datos FIVB y AVP, 2000–2019). Hablas con entrenadores.

Reglas:
- Responde en español neutro, de "usted", claro y breve (máximo 130 palabras), con viñetas si ayudan.
- Usa SOLO la información de la FICHA DEL PROYECTO. No inventes cifras, nombres ni estudios.
- Si la respuesta no está en la ficha, dilo con honestidad y sugiere qué página del dashboard mirar.
- Puedes explicar conceptos (AUC, K-means, correlación, percentil) con palabras sencillas y ejemplos de voleibol.
- Recuerda cuando haga falta que los datos muestran asociaciones con ganar, no causas.
- No des consejos médicos ni de lesiones; para eso, un profesional de la salud.
- Si preguntan algo que no tiene que ver con voleibol o con el dashboard, responde amablemente que solo
  puedes ayudar con el proyecto.

FICHA DEL PROYECTO:
"""


def conversar(ficha, historial, pregunta):
    """historial: lista de {"rol": "user"|"model", "texto": ...}. Devuelve (respuesta, modelo)."""
    contenidos = [{"role": h["rol"], "parts": [{"text": h["texto"]}]} for h in historial[-10:]]
    contenidos.append({"role": "user", "parts": [{"text": pregunta}]})
    return generar(INSTRUCCIONES_CHAT + ficha, contenidos, temperatura=0.3)

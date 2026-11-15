"""Dashboard interactivo: Voleibol de playa, ¿qué hace ganar a una pareja?

Correr con:  python app.py   y abrir http://127.0.0.1:8050 en el navegador.

Cinco páginas: Inicio, Dashboard, Análisis, Predicción y Equipo. Los modelos de la
página Predicción y los perfiles de juego se entrenan aparte, con
entrenar_modelos.py; aquí solo se leen sus resultados guardados.
"""
import sys

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

import datos as D
import ia as IA
import plan as PL
import prediccion as M
from estilo import (ARENA, CORAL, ELEGIDO, ESCALA_CORR, ESCALA_SEC, ETAPAS, FUENTE, GENERO, GRIS, MAR, NEUTRO,
                    PAL, PAL_ETAPA, PAL_MODELO, PAL_PERFIL, PAL_PODER, PIZARRA, TURQUESA, fmt, fmt_p, fmt_pct,
                    vacia)

LIBRO = "https://nicoleramirez2106-ui.github.io/dataviz-2026/"
REPO = "https://github.com/nicoleramirez2106-ui/dataviz-2026"

app = Dash(__name__, title="Voleibol de playa · Dashboard",
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP,
                                 "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800"
                                 "&display=swap"],
           suppress_callback_exceptions=True,
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


# ==========================================================================
# Piezas reutilizables
# ==========================================================================
def tarjeta(titulo, cuerpo, subtitulo=None, fuente=True, clase=""):
    return dbc.Card([
        dbc.CardHeader([html.H6(titulo, className="tarjeta-titulo"),
                        html.Div(subtitulo, className="tarjeta-subtitulo") if subtitulo is not None else None]),
        dbc.CardBody(cuerpo),
        html.Div(FUENTE, className="tarjeta-fuente") if fuente else None,
    ], className=f"tarjeta mb-4 {clase}")


# Ícono por defecto de cada tono (el color dice QUÉ es; el ícono lo refuerza)
ICONO_TONO = {"tono-mar": "trofeo", "tono-coral": "rayo", "tono-turquesa": "balon", "tono-arena": "balon"}


def ico(nombre, clase=""):
    return html.Span(className=f"ico ico-{nombre} {clase}")


def kpi(valor, etiqueta, nota=None, id_=None, tono="", icono=None):
    """Indicador: ícono en un círculo del color del tono, etiqueta y número grande.
    Sin tono, el indicador es neutro (gris): solo se colorea cuando el color significa algo."""
    extra = {"id": id_} if id_ else {}
    tono = tono or "tono-gris"
    icono = icono or ICONO_TONO.get(tono, "datos")
    return html.Div([html.Div([html.Div(ico(icono), className="kpi-icono"),
                               html.Div(etiqueta, className="kpi-etiqueta")], className="kpi-cabeza"),
                     html.Div(valor, className="kpi-valor", **extra),
                     html.Div(nota, className="kpi-nota") if nota else None], className=f"kpi {tono}")


def grafico(id_, alto=380):
    # Mientras carga, un balón rebota en lugar del indicador genérico
    return dcc.Loading(dcc.Graph(id=id_, config=CONFIG, style={"height": f"{alto}px"}),
                       custom_spinner=html.Div(html.Img(src="/assets/balon.svg", className="balon-carga", alt=""),
                                                className="carga"))


def histograma(fig, s, ancho, color, nombre=None, opacidad=1.0):
    """Histograma calculado en Python: el navegador solo recibe las barras, no los 300.000 datos."""
    s = s.dropna()
    inicio = np.floor(s.min() / ancho) * ancho
    bordes = np.arange(inicio, s.max() + ancho * 1.001, ancho)
    n, bordes = np.histogram(s, bins=bordes)
    centros = (bordes[:-1] + bordes[1:]) / 2
    fig.add_bar(x=centros, y=n, width=ancho, name=nombre, marker=dict(color=color, line=dict(color="white", width=0.5)),
                opacity=opacidad, customdata=np.stack([bordes[:-1], bordes[1:]], axis=1), showlegend=nombre is not None,
                hovertemplate=(nombre + "<br>" if nombre else "") +
                "%{customdata[0]:.4~r} a %{customdata[1]:.4~r}<br>%{y:,} registros<extra></extra>")


def caja(fig, s, x, color, nombre, mostrar=True):
    """Diagrama de caja con los cuartiles ya calculados (más liviano que mandar todos los datos)."""
    s = s.dropna()
    q1, med, q3 = s.quantile([.25, .5, .75])
    iqr = q3 - q1
    bajo, alto = s[s >= q1 - 1.5 * iqr].min(), s[s <= q3 + 1.5 * iqr].max()
    fig.add_box(x=[x], q1=[q1], median=[med], q3=[q3], lowerfence=[bajo], upperfence=[alto],
                name=nombre, marker_color=color, offsetgroup=nombre, legendgroup=nombre, showlegend=mostrar,
                hoverinfo="y")


def entrenador(texto):
    return html.Div([html.I(className="bi bi-clipboard2-pulse me-2"),
                     html.Strong("Para el entrenador. "), texto], className="entrenador mb-4")


def encabezado(titulo, texto, subtitulo=None):
    """Título de la página; el subtítulo (opcional) va debajo, en el mismo color del título."""
    return html.Div([html.H2(titulo, className="pagina-titulo"),
                     html.Div(subtitulo, className="pagina-subtitulo") if subtitulo else None,
                     html.P(texto, className="pagina-intro")], className="mb-4")


def chips(id_, valor, ops):
    """Botones de filtro rápidos (un clic, sin desplegables)."""
    return dbc.RadioItems(id=id_, value=valor, inline=True, className="chips",
                          options=[{"label": l, "value": v} for v, l in ops],
                          inputClassName="btn-check", labelClassName="chip",
                          labelCheckedClassName="chip-activo")


def sin_modelos():
    return dbc.Alert([html.I(className="bi bi-exclamation-triangle me-2"),
                      "Todavía no hay resultados guardados de los modelos. Desde la carpeta del repositorio, corra ",
                      html.Code("python dashboard/entrenar_modelos.py"), " y vuelva a abrir el dashboard."],
                     color="warning")


# ==========================================================================
# Estructura: riel de navegación (línea de banda), marcador y filtros rápidos
# ==========================================================================
# Íconos sólidos propios (assets/iconos)
PAGINAS = [("/", "Inicio", "inicio"),
           ("/dashboard", "Dashboard", "dashboard"),
           ("/analisis", "Análisis", "analisis"),
           ("/prediccion", "Predicción", "prediccion"),
           ("/entrenamiento", "Entrenar", "silbato"),
           ("/equipo", "Equipo", "equipo")]

riel = html.Nav([
    dcc.Link([html.Img(src="/assets/balon.svg", className="balon-logo", alt=""),
              html.Div([html.Div("Panel del entrenador", className="marca-nombre"),
                        html.Div("Voleibol de playa", className="marca-sub")])],
             href="/", className="riel-marca", title="Inicio"),
    dbc.Nav([dbc.NavLink([ico(i), html.Span(txt)], href=ruta, active="exact", className="riel-item", id=f"nav-{n}")
             for n, (ruta, txt, i) in enumerate(PAGINAS)], vertical=True, className="riel-menu"),
    html.Div([ico("libro"), html.H6("Lea el libro"),
              html.P("El análisis completo, con el código y la metodología de cada gráfico."),
              html.A(["Abrir el libro", html.I(className="bi bi-arrow-right")], href=LIBRO, target="_blank")],
             className="riel-promo"),
    html.A([html.I(className="bi bi-github me-2"), "Repositorio del proyecto"], href=REPO, target="_blank",
           className="riel-repo"),
    html.Div([html.Div(id="coach-avatar", className="avatar"),
              html.Div([html.Div(id="coach-nombre", className="usuario-nombre"),
                        html.Div("Entrenador", className="usuario-rol")]),
              html.Button(ico("editar"), id="coach-editar", n_clicks=0, className="usuario-editar",
                          title="Cambiar el nombre")], className="riel-usuario"),
], className="riel", **{"aria-label": "Navegación principal"})

marcador = html.Header([
    html.Div([html.H1(id="saludo", className="saludo"),
              html.Div(f"Datos de {fmt(D.CIFRAS['n_partidos'], 0)} partidos de los circuitos FIVB y AVP, "
                       f"{D.ANIOS[0]}–{D.ANIOS[1]}.", className="saludo-sub")]),
    html.Div(id="marcador-tablero", className="tablero", **{"aria-live": "polite"}),
], className="marcador")

# Personalización: el nombre del coach se guarda en el navegador (no sale del computador)
ventana_coach = dbc.Modal(dbc.ModalBody([
    html.Img(src="/assets/balon.svg", className="balon-logo", alt=""),
    html.H5("¡Bienvenido al panel del entrenador!"),
    html.P("¿Cómo quiere que le llamemos? El nombre solo se guarda en este navegador."),
    dbc.Input(id="coach-input", placeholder="Nombre o apodo", maxLength=30, debounce=False),
    html.Div([html.Button("Ahora no", id="coach-omitir", n_clicks=0, className="boton-secundario"),
              html.Button("Guardar", id="coach-guardar", n_clicks=0, className="boton-principal")],
             className="modal-botones"),
]), id="coach-modal", is_open=False, centered=True, className="modal-coach")

filtros = html.Section([
    html.Div([html.Span("Circuito", className="filtro-etq"),
              chips("f-circuito", "Todos", [("Todos", "Todos"), ("FIVB", "FIVB"), ("AVP", "AVP")])],
             className="filtro-grupo"),
    html.Div([html.Span("Género", className="filtro-etq"),
              chips("f-genero", "Todos", [("Todos", "Todos"), ("M", "Hombres"), ("W", "Mujeres")])],
             className="filtro-grupo"),
    html.Div([html.Span("Etapa", className="filtro-etq"),
              chips("f-etapa", "Todas", [("Todas", "Todas")] + [(e, e) for e in ETAPAS])],
             className="filtro-grupo"),
    html.Div([html.Span("Años", className="filtro-etq"),
              dcc.RangeSlider(id="f-anios", min=D.ANIOS[0], max=D.ANIOS[1], step=1, value=list(D.ANIOS),
                              marks={a: str(a) for a in range(2000, 2020, 5)} | {2019: "2019"},
                              tooltip={"placement": "bottom", "always_visible": False})],
             className="filtro-grupo filtro-anios"),
    html.Button([html.I(className="bi bi-arrow-counterclockwise me-1"), "Restablecer"], id="f-reset",
                className="filtro-reset", n_clicks=0),
], id="filtros", className="filtros-banda")

app.layout = html.Div([
    dcc.Location(id="url"), dcc.Store(id="coach", storage_type="local"), ventana_coach, riel,
    html.Div([marcador, filtros, html.Main(id="contenido", className="contenido")], className="principal"),
], className="app")

@app.callback(Output("coach-modal", "is_open"), Output("coach", "data"), Output("coach-input", "value"),
              Input("coach-editar", "n_clicks"), Input("coach-guardar", "n_clicks"), Input("coach-omitir", "n_clicks"),
              Input("url", "pathname"), State("coach", "data"), State("coach-input", "value"))
def ventana_nombre(_e, _g, _o, _ruta, datos, escrito):
    """Pide el nombre la primera vez; después solo se abre con el lápiz del menú."""
    datos = datos or {}
    if ctx.triggered_id == "coach-editar":
        return True, no_update, datos.get("nombre", "")
    if ctx.triggered_id == "coach-guardar":
        return False, {"nombre": (escrito or "").strip()[:30], "visto": True}, no_update
    if ctx.triggered_id == "coach-omitir":
        return False, {**datos, "visto": True}, no_update
    return not datos.get("visto"), no_update, no_update


@app.callback(Output("saludo", "children"), Output("coach-nombre", "children"), Output("coach-avatar", "children"),
              Input("coach", "data"))
def saludar(datos):
    nombre = ((datos or {}).get("nombre") or "").strip()
    iniciales = "".join(p[0] for p in nombre.split()[:2]).upper() or "C"
    return (f"¡Hola, Coach {nombre}! 👋" if nombre else "¡Hola, Coach! 👋"), (nombre or "Coach"), iniciales


FILTROS = [Input("f-circuito", "value"), Input("f-genero", "value"), Input("f-anios", "value"),
           Input("f-etapa", "value")]


@app.callback(Output("f-circuito", "value"), Output("f-genero", "value"), Output("f-anios", "value"),
              Output("f-etapa", "value"), Input("f-reset", "n_clicks"), prevent_initial_call=True)
def restablecer(_):
    return "Todos", "Todos", list(D.ANIOS), "Todas"


@app.callback(Output("marcador-tablero", "children"), *FILTROS)
def actualizar_marcador(circuito, genero, anios, etapa):
    """Marcador superior: FIVB contra AVP y cuántas veces gana el favorito con los filtros actuales."""
    d = D.filtrar(D.P, circuito, genero, anios, etapa)
    n = d.circuit.astype(str).value_counts()
    r = d[d.rank_diff.notna()]
    fav = fmt_pct(100 * (1 - r.es_upset.mean()), 1) if len(r) else "—"
    return [html.Div([html.Span("FIVB", className="equipo equipo-fivb"),
                      html.Span(fmt(n.get("FIVB", 0), 0), className="digito"),
                      html.Span(":", className="dos-puntos"),
                      html.Span(fmt(n.get("AVP", 0), 0), className="digito"),
                      html.Span("AVP", className="equipo equipo-avp")], className="tablero-score",
                     title="Partidos con los filtros actuales"),
            html.Div([html.Span("Gana el favorito", className="tablero-etq"),
                      html.Span(fav, className="digito digito-chico")], className="tablero-dato"),
            html.Div([html.Span("Periodo", className="tablero-etq"),
                      html.Span(f"{anios[0]}–{anios[1]}", className="digito digito-chico")],
                     className="tablero-dato")]


# ==========================================================================
# Cifras fijas que se usan en Inicio y Conclusiones (todos los datos)
# ==========================================================================
C = D.CIFRAS
_r = D.P[D.P.rank_diff.notna()]
_grupos = pd.cut(_r.rank_diff.abs(), [0, 2, 5, 10, 20, np.inf])
_fav = _r.groupby(_grupos, observed=True).es_upset.apply(lambda s: 100 * (1 - s.mean()))
C["fav_cerca"], C["fav_lejos"] = _fav.iloc[0], _fav.iloc[-1]
_poder = D.tabla_poder("Todos", "Todos", D.ANIOS).set_index("var")
C["est10"] = 100 * (D.E[D.E.ventaja_estatura_cm >= 10].resultado == "Ganador").mean()
_perd = 100 * D.E.groupby("es_clasif").resultado.apply(lambda s: (s == "Perdedor").mean())
_vif, _n_vif = D.vif()
_n_circ = D.P.circuit.astype(str).value_counts()


# ==========================================================================
# Página: Inicio (una cancha: pregunta de un lado de la red, marcador del otro)
# ==========================================================================
def pagina_inicio():
    hallazgos = [
        ("Ranking", fmt_pct(C["pct_fav"]), "de los partidos los gana la pareja mejor rankeada. "
         "Casi uno de cada tres es una sorpresa."),
        ("Distancia con el rival", f"{fmt_pct(C['fav_cerca'], 0)} → {fmt_pct(C['fav_lejos'], 0)}",
         "gana el favorito con 1–2 posiciones de diferencia frente a más de 20."),
        ("Juego (AVP)", fmt(_poder.rbc["hitpct"]), "es la correlación rango-biserial del % de ataque: "
         "la variable que más separa al ganador del perdedor."),
    ]
    jugadas = [("/dashboard", "dashboard", "Dashboard",
                "Partidos por año, circuito y etapa, sorpresas y cobertura de datos con filtros rápidos."),
               ("/analisis", "analisis", "Análisis",
                "Univariado, bivariado, multivariado, estilos de juego de la AVP, metodología y conclusiones."),
               ("/prediccion", "prediccion", "Predicción",
                "Modelos previos al partido, simulador ¿Quién gana? y la calculadora histórica.")]

    def contador(valor, dec=0, sufijo=""):
        """Número que sube contando al abrir la página (assets/contador.js)."""
        return html.Span(fmt(valor, dec) + sufijo, className="contador",
                         **{"data-valor": str(valor), "data-dec": str(dec), "data-sufijo": sufijo})

    marcador_portada = [("Gana el favorito", contador(round(C["pct_fav"], 1), 1, " %"), "dato-mar"),
                        ("Sorpresas", contador(round(100 - C["pct_fav"], 1), 1, " %"), "dato-coral"),
                        ("Partidos FIVB", contador(int(_n_circ.get("FIVB", 0))), "dato-turquesa"),
                        ("Partidos AVP", contador(int(_n_circ.get("AVP", 0))), "dato-arena")]
    cinta = [f"{fmt(C['n_partidos'], 0)} partidos", f"{fmt(C['n_jugadores'], 0)} jugadores",
             f"{fmt(C['n_paises'], 0)} países sede", f"{fmt(C['n_rank'], 0)} partidos con ranking de las dos parejas",
             f"{fmt_pct(C['pct_stats'], 0)} con estadísticas de juego", "FIVB y AVP", "2000–2019"]
    piezas = [html.Span([html.Span(t), html.Span(className="ico ico-balon cinta-ico")], className="cinta-item") for t in cinta]

    return html.Div([
        html.Section([
            html.Img(src="/assets/cancha.svg", className="portada-escena", alt=""),
            html.Div([
                html.Div("Visualización de Datos y Toma de Decisiones, Universidad del Norte, 2026",
                         className="portada-curso"),
                html.H1("¿Qué hace ganar a una pareja de voleibol de playa?", className="portada-titulo"),
                html.P(f"{fmt(C['n_partidos'], 0)} partidos de los circuitos FIVB y AVP entre 2000 y 2019, "
                       "analizados para que un entrenador sepa qué tan sólida es una ventaja antes del "
                       "partido, cómo armar parejas y qué priorizar en el entrenamiento.", className="portada-texto"),
                html.Div([dcc.Link("Simular un partido", href="/prediccion", className="boton-principal"),
                          html.A("Leer el libro", href=LIBRO, target="_blank", className="boton-secundario")],
                         className="hero-botones"),
                html.Div("Nicole Ramírez y Roger Daza", className="hero-autores"),
            ], className="portada-texto-caja"),
        ], className="portada"),
        html.Div([html.Div([html.Div(v, className=f"portada-dato-valor {tono}"),
                            html.Div(etq, className="portada-dato-etq")], className="portada-dato")
                  for etq, v, tono in marcador_portada], className="portada-marcador"),
        html.Div(f"Favorito y sorpresa se calculan sobre los {fmt(C['n_rank'], 0)} partidos con ranking de las "
                 "dos parejas.", className="nota mb-3"),
        html.Div(html.Div(piezas + piezas, className="cinta-pista"), className="cinta mb-4",
                 **{"aria-label": " · ".join(cinta)}),
        html.H5("Elija su jugada", className="seccion-titulo"),
        html.Div([dcc.Link([html.Div(ico(icono), className="jugada-icono"), html.H6(t), html.P(x),
                            html.Span(f"Ir a {t}", className="jugada-ir")],
                           href=ruta, className="jugada")
                  for ruta, icono, t, x in jugadas] +
                 [html.Button([html.Div(ico("video"), className="jugada-icono"), html.H6("Video del proyecto"),
                               html.P("Recorrido de 11 minutos por el panel, contado por sus autores."),
                               html.Span("Ver el video", className="jugada-ir")],
                              className="jugada jugada-video", id="video-abrir", n_clicks=0)],
                 className="jugadas mb-4"),
        html.H5("Tres hallazgos para empezar", className="seccion-titulo"),
        html.Div([html.Div([html.Div(f"Set {i}", className="set-num"),
                            html.Div(etq, className="hallazgo-etq"),
                            html.Div(val, className="hallazgo-valor"),
                            html.Div(txt, className="hallazgo-texto")], className="hallazgo")
                  for i, (etq, val, txt) in enumerate(hallazgos, 1)], className="sets mb-4"),
        dbc.Row([
            dbc.Col(tarjeta("Pregunta de investigación", [
                html.P(html.Strong("¿Qué factores, medibles antes del partido, se asocian con que una "
                                   "pareja gane, y cómo puede usarlos un entrenador para preparar "
                                   "partidos y armar parejas?"), className="pregunta"),
                html.Ol([html.Li([html.Strong("Preparar un partido: "),
                                  "¿qué tan probable es ganar según la diferencia de ranking con el rival?"]),
                         html.Li([html.Strong("Armar parejas: "),
                                  "¿la estatura, la edad o la diferencia entre compañeros se asocian con ganar?"]),
                         html.Li([html.Strong("Priorizar el entrenamiento: "),
                                  "¿qué acciones de juego separan al ganador del perdedor?"]),
                         html.Li([html.Strong("Anticipar el resultado: "),
                                  "con lo que se sabe antes del partido, ¿se puede predecir quién gana mejor "
                                  "que con la regla \"gana el mejor rankeado\"?"])])],
                fuente=False), lg=7),
            dbc.Col(tarjeta("Objetivos", html.Ol([
                html.Li("Depurar y documentar el conjunto de datos."),
                html.Li("Describir la distribución de las variables del partido y de los jugadores."),
                html.Li("Comparar a las parejas ganadoras con las perdedoras."),
                html.Li("Cuantificar cuándo el ranking anticipa el resultado y cuándo no."),
                html.Li("Traducir los hallazgos en recomendaciones para el entrenador.")]), fuente=False), lg=5),
        ]),
    ])


# ==========================================================================
# Página: Dashboard (antes "Datos y limpieza")
# ==========================================================================
def pagina_dashboard():
    R = D.RES
    problemas = pd.DataFrame([
        ("Fechas de nacimiento en el futuro", f"{fmt(R['p1_partidos_bd_corregida'], 0)} partidos", "Restar 100 años y marcar"),
        ("Ranking guardado como texto (\"17, Q2\")", "Todos los partidos", "Separar ranking, posición Q y bandera"),
        ("Estadísticas de juego vacías", f"{fmt_pct(100 - C['pct_stats'])} de los partidos", "Bandera de cobertura; no se imputan"),
        ("Mismo jugador en ambos equipos", f"{fmt(R['p7_filas_invalidas'], 0)} partidos", "Eliminar las filas"),
        ("Estadísticas imposibles (kills > ataques)", f"{fmt(R['p8_celdas_invalidas'], 0)} celdas", "Anular y recalcular el % de ataque"),
        ("Forfeits y retiros", f"{fmt(R['p5_forfeit'], 0)} y {fmt(R['p5_retirado'], 0)}", "Banderas; forfeits fuera de duración"),
        ("Duración como texto HH:MM:SS", f"{fmt(R['p2_duracion_na'], 0)} vacías", "Convertir a minutos"),
        ("Edad que no cuadra con las fechas", f"{fmt(R['p9_edad_inconsistente'], 0)} partidos", "Marcar; se conservan")],
        columns=["Problema", "Alcance", "Decisión"])
    return html.Div([
        encabezado("Dashboard", "Panorama de los partidos con los filtros de arriba: un clic en circuito, "
                   "género, etapa o años actualiza el marcador y todos los gráficos.", subtitulo="Datos y limpieza"),
        html.Div(id="d-kpis", className="kpis mb-4"),
        dbc.Row([dbc.Col(tarjeta("Partidos por año y circuito", grafico("d-anio", 340),
                                 "La caída de la AVP en 2011–2012 coincide con su crisis financiera · "
                                 "haga clic en un año para ver su detalle"), lg=8),
                 dbc.Col(tarjeta("Detalle del año", html.Div(id="d-detalle"), fuente=False), lg=4)]),
        dbc.Row([dbc.Col(tarjeta("Sorpresas por año", grafico("d-sorpresas", 320),
                                 "% de partidos que gana la pareja peor rankeada (años con 30 o más partidos)"), lg=6),
                 dbc.Col(tarjeta("Cobertura de estadísticas de juego por año", grafico("d-cobertura", 320),
                                 "% de partidos con estadísticas completas"), lg=6)]),
        dbc.Row([dbc.Col(tarjeta("Partidos por circuito y género", grafico("d-circuito", 320)), lg=6),
                 dbc.Col(tarjeta("Partidos por etapa del torneo", grafico("d-etapa", 320)), lg=6)]),
        tarjeta("Problemas de calidad encontrados y decisiones", [
            html.Div([kpi(fmt(R["p0_filas_crudas"], 0), "Partidos en el archivo original"),
                      kpi(fmt(R["p7_filas_invalidas"], 0), "Partidos eliminados", "imposibles: mismo jugador en ambos equipos"),
                      kpi(fmt(R["p7_filas_finales"], 0), "Partidos analizados")], className="kpis mb-3"),
            dbc.Table.from_dataframe(problemas, striped=True, hover=True, className="tabla")], fuente=False),
        entrenador("Las estadísticas de juego (kills, aces, bloqueos) solo existen de forma confiable para la AVP. "
                   "Si su pareja compite en la FIVB, use el ranking, la edad, la estatura y la etapa, que están "
                   "disponibles para casi todos los partidos."),
    ])


@app.callback(Output("d-kpis", "children"), Output("d-circuito", "figure"), Output("d-etapa", "figure"),
              Output("d-anio", "figure"), Output("d-cobertura", "figure"), Output("d-sorpresas", "figure"),
              *FILTROS)
def actualizar_datos(circuito, genero, anios, etapa):
    d = D.filtrar(D.P, circuito, genero, anios, etapa)
    if d.empty:
        return [kpi("0", "Partidos con los filtros actuales")], vacia(), vacia(), vacia(), vacia(), vacia()
    r = d[d.rank_diff.notna()]
    jug = D.filtrar(D.J, circuito, genero, anios, etapa).player_id.nunique()
    kpis = [kpi(fmt(len(d), 0), "Partidos con los filtros actuales", id_="d-n", icono="balon"),
            kpi(fmt(jug, 0), "Jugadores distintos", icono="equipo"),
            kpi(fmt_pct(100 * (1 - r.es_upset.mean())) if len(r) else "—", "Gana el favorito",
                f"{fmt(len(r), 0)} partidos con ranking de ambas", tono="tono-mar"),
            kpi(fmt_pct(100 * r.es_upset.mean()) if len(r) else "—", "Sorpresas", tono="tono-coral"),
            kpi(f"{fmt(d[d.es_forfeit == 0].duracion_min.median(), 0)} min", "Duración mediana", icono="reloj"),
            kpi(fmt_pct(100 * d.tiene_stats_completas.mean()), "Con estadísticas de juego", icono="analisis")]

    t = d.groupby(["circuit", "gender"], observed=True).size().reset_index(name="n")
    t["circuit"] = pd.Categorical(t.circuit.astype(str), ["FIVB", "AVP"])
    t = t.sort_values("circuit")
    f1 = go.Figure()
    for g in ["M", "W"]:
        s = t[t.gender == g]
        f1.add_bar(x=s.circuit.astype(str), y=s.n, name=GENERO[g], marker_color=PAL[g],
                   text=[fmt(v, 0) for v in s.n], textposition="outside", cliponaxis=False,
                   hovertemplate="%{x} · " + GENERO[g] + "<br>%{y:,} partidos<extra></extra>")
    f1.update_layout(barmode="group", bargap=0.35, yaxis=dict(title="Partidos", tickformat=",d"))

    e = d.etapa.value_counts().reindex(ETAPAS).fillna(0)
    f2 = go.Figure(go.Bar(x=e.index, y=e.values, marker_color=[PAL_ETAPA[x] for x in e.index],
                          text=[fmt(v, 0) for v in e.values], textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}<br>%{y:,} partidos<extra></extra>"))
    f2.update_layout(yaxis=dict(title="Partidos", tickformat=",d"), showlegend=False)

    a = d.groupby(["year", "circuit"], observed=True).size().reset_index(name="n")
    f3 = go.Figure()
    for c in ["FIVB", "AVP"]:
        s = a[a.circuit == c]
        if len(s):
            f3.add_bar(x=s.year, y=s.n, name=c, marker_color=PAL[c],
                       hovertemplate="%{x} · " + c + "<br>%{y:,} partidos<extra></extra>")
    f3.update_layout(barmode="stack", yaxis=dict(title="Partidos", tickformat=",d"), hovermode="x unified",
                     clickmode="event")

    cob = d.groupby(["year", "circuit"], observed=True).tiene_stats_completas.mean().reset_index()
    f4 = go.Figure()
    for c in ["FIVB", "AVP"]:
        s = cob[cob.circuit == c]
        if len(s):
            f4.add_scatter(x=s.year, y=100 * s.tiene_stats_completas, name=c, mode="lines+markers",
                           line=dict(color=PAL[c], width=2), marker=dict(size=7),
                           hovertemplate="%{x} · " + c + "<br>%{y:.1f} % con estadísticas<extra></extra>")
    f4.update_layout(yaxis=dict(range=[0, 100], ticksuffix=" %"), hovermode="x unified")

    # Sorpresas por año (capítulo 7 del libro): total en coral, circuitos en su color
    f5 = go.Figure()
    if len(r):
        for c, nombre, color, ancho, guion in [(None, "Todos", CORAL, 3, "solid"),
                                              ("FIVB", "FIVB", PAL["FIVB"], 1.5, "dot"),
                                              ("AVP", "AVP", PAL["AVP"], 1.5, "dot")]:
            s = r if c is None else r[r.circuit == c]
            u = s.groupby("year").es_upset.agg(["mean", "size"])
            u = u[u["size"] >= 30]
            if len(u):
                f5.add_scatter(x=u.index, y=100 * u["mean"], name=nombre, mode="lines+markers",
                               line=dict(color=color, width=ancho, dash=guion), marker=dict(size=6 if c else 8),
                               customdata=u["size"],
                               hovertemplate="%{x} · " + nombre + "<br>Sorpresas: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>")
    f5.update_layout(yaxis=dict(ticksuffix=" %", rangemode="tozero"), hovermode="x unified")
    return kpis, f1, f2, f3, f4, (f5 if f5.data else vacia())


@app.callback(Output("d-detalle", "children"), Input("d-anio", "clickData"), *FILTROS)
def detalle_anio(click, circuito, genero, anios, etapa):
    d = D.filtrar(D.P, circuito, genero, anios, etapa)
    if d.empty:
        return html.Div("No hay partidos con estos filtros.", className="nota")
    anio = int(click["points"][0]["x"]) if click else int(d.year.max())
    y = d[d.year == anio]
    if y.empty:
        return html.Div(f"No hay partidos en {anio} con estos filtros.", className="nota")
    r = y[y.rank_diff.notna()]
    n = y.circuit.astype(str).value_counts()
    paises = y.country.astype(str).value_counts().head(3)
    return html.Div([
        html.Div(str(anio), className="detalle-anio"),
        html.Div([kpi(fmt(len(y), 0), "Partidos", icono="balon"),
                  kpi(fmt(n.get("FIVB", 0), 0), "FIVB", tono="tono-turquesa"),
                  kpi(fmt(n.get("AVP", 0), 0), "AVP", tono="tono-arena"),
                  kpi(fmt_pct(100 * r.es_upset.mean()) if len(r) >= 30 else "—", "Sorpresas", tono="tono-coral")],
                 className="kpis kpis-2 mb-2"),
        html.Div([html.Strong("Países sede con más partidos: "),
                  ", ".join(f"{p} ({fmt(v, 0)})" for p, v in paises.items())], className="nota"),
        html.Div("Mostrando el último año del filtro." if not click else "Año elegido en el gráfico.",
                 className="nota"),
    ])


# ==========================================================================
# Página: Análisis (univariado, bivariado, multivariado, estilos de juego,
# metodología y conclusiones)
# ==========================================================================
def contenido_uni():
    return html.Div([
        dbc.Row([dbc.Col([html.Label("Variable", className="control-etq"),
                          dcc.Dropdown(id="u-var", value="edad", clearable=False,
                                       options=[{"label": v[2], "value": k} for k, v in D.VARS_UNI.items()])],
                         md=5)], className="mb-3"),
        html.Div(id="u-kpis", className="kpis mb-3"),
        tarjeta("Distribución", grafico("u-fig", 400), html.Span(id="u-sub")),
        entrenador("La duración típica de un partido es de unos 42 minutos, pero uno de cada tres se va a "
                   "tercer set y dura cerca de 54: prepare la exigencia física para el partido largo."),
    ])


def contenido_bi():
    return html.Div([
        dbc.Row([dbc.Col([html.Label("Variable", className="control-etq"),
                          dcc.Dropdown(id="b-var", value="rank_main", clearable=False,
                                       options=[{"label": v[0] + (" (AVP)" if v[2] else ""), "value": k}
                                                for k, v in D.VARS_BI.items()])], md=5)], className="mb-3"),
        html.Div(id="b-kpis", className="kpis mb-3"),
        dbc.Row([dbc.Col(tarjeta("Ganador vs perdedor", grafico("b-caja", 400), html.Span(id="b-sub")), lg=6),
                 dbc.Col(tarjeta("Poder discriminante de todas las variables", grafico("b-poder", 400),
                                 "|correlación rango-biserial|; líneas en 0,3 y 0,5"), lg=6)]),
        dbc.Row([dbc.Col(tarjeta("¿Cuándo gana el favorito?", grafico("b-fav", 360),
                                 "% de victorias del mejor rankeado según la diferencia de ranking"), lg=6),
                 dbc.Col(tarjeta("Sorpresas por etapa y circuito", grafico("b-upset", 360),
                                 "% de partidos que gana la pareja peor rankeada (etapas con 30 o más partidos)"), lg=6)]),
        tarjeta("Variables categóricas: chi-cuadrado y V de Cramér", html.Div(id="b-chi"), fuente=False),
        entrenador("Lo que más informa antes del partido es la distancia de ranking con el rival, no el ranking "
                   "propio. Durante el partido, en la AVP, separa al ganador la eficacia en ataque: convertir más "
                   "y fallar menos."),
    ])


def contenido_multi():
    return html.Div([
        dbc.Row([dbc.Col([html.Label("Equipos", className="control-etq"),
                          chips("m-res", "Todos", [(x, x) for x in ["Todos", "Ganador", "Perdedor"]])],
                         md=6)], className="mb-3"),
        dbc.Row([dbc.Col(tarjeta("Correlación de Spearman entre variables del equipo", grafico("m-spear", 520)), lg=7),
                 dbc.Col(tarjeta("Redundancia entre variables previas al partido (VIF)", [
                     dbc.Table.from_dataframe(_vif.assign(VIF=_vif.VIF.map(lambda v: fmt(v))),
                                              striped=True, hover=True, size="sm", className="tabla"),
                     html.Div(f"Calculado con los {fmt(_n_vif, 0)} partidos que tienen ranking, edad y estatura "
                              "de las dos parejas. VIF ≤ 5: no hay redundancia.", className="nota")], fuente=False), lg=5)]),
        entrenador("Las variables que se conocen antes del partido no se repiten entre sí: ranking, edad y estatura "
                   "pueden mirarse juntas. Los partidos se alargan cuando la pareja que pierde es buena."),
    ])


# Perfiles de juego: se reconstruye a qué perfil pertenece cada fila (centroide más cercano del K-means
# guardado). Da exactamente los mismos grupos que el entrenamiento y permite graficar las variables reales.
if M.PERF is not None:
    _VP = M.PERF["variables"]
    _REFP = PL.REF[_VP + ["resultado"]].copy()
    _zp = (_REFP[_VP] - pd.Series(M.PERF["media_global"])[_VP]) / pd.Series(M.PERF["de_global"])[_VP]
    _cen = np.array([[r["z"][v] for v in _VP] for r in M.PERF["perfiles"]])
    _REFP["perfil"] = ((_zp.values[:, None, :] - _cen[None]) ** 2).sum(axis=2).argmin(axis=1)
    _MUESTRA_P = _REFP.sample(3000, random_state=2026)
    _OPC_P = [{"label": M.PERF["etiquetas"][v].capitalize() + ("" if v == "hitpct" else " por set"), "value": v}
              for v in _VP]


def etq_perfil(v):
    e = M.PERF["etiquetas"][v]
    return e.capitalize() + ("" if v == "hitpct" else " por set")


def rasgos_nombre(z):
    """Los rasgos que dieron el nombre al perfil (misma regla de entrenar_modelos.py:
    los dos más alejados del promedio, si superan 0,25 DE)."""
    orden = sorted(z, key=lambda v: -abs(z[v]))[:2]
    return {v for v in orden if abs(z[v]) >= 0.25}


def contenido_perfiles():
    """Los 4 estilos de juego de la AVP (K-means sobre 8 acciones de juego por set)."""
    if M.PERF is None:
        return sin_modelos()
    PF = M.PERF
    perf = PF["perfiles"]
    etq = PF["etiquetas"]
    vs = PF["variables"]

    def rasgos(z):
        orden = sorted(z, key=lambda v: -abs(z[v]))[:3]
        return html.Ul([html.Li([html.I(className=f"bi bi-arrow-{'up' if z[v] > 0 else 'down'}-short"),
                                 f"{'Más' if z[v] > 0 else 'Menos'} {etq[v]} que el promedio"])
                        for v in orden], className="perfil-rasgos")
    tarjetas = [html.Div([
        html.Div(f"Estilo {i + 1}", className="perfil-num"),
        html.H6(r["nombre"], className="perfil-nombre"),
        html.Div(fmt_pct(r["pct_victorias"]), className="perfil-valor"),
        html.Div(f"de partidos ganados · {fmt_pct(r['pct_filas'], 0)} de las actuaciones", className="kpi-nota"),
        rasgos(r["z"]),
    ], className="perfil-tarjeta perfil-activo", style={"--perfil": PAL_PERFIL[i]}) for i, r in enumerate(perf)]

    acciones = html.Div([html.Span("Se agrupó con 8 acciones de juego por set:", className="acciones-etq")] +
                        [html.Span(etq[v].capitalize(), className="accion") for v in vs], className="acciones mb-4")

    # Huella de cada estilo: cuánto se aleja del promedio en cada una de las 8 acciones
    fig_z = make_subplots(rows=1, cols=len(perf), shared_yaxes=True, horizontal_spacing=0.025,
                          subplot_titles=[f"E{i + 1}" for i in range(len(perf))])
    for i, r in enumerate(perf):
        z = [r["z"][v] for v in vs]
        clave = rasgos_nombre(r["z"])
        fig_z.add_bar(x=z, y=[etq_perfil(v) for v in vs], orientation="h", showlegend=False, row=1, col=i + 1,
                      marker_color=[PAL_PERFIL[i] if v in clave else PIZARRA[1] for v in vs],
                      text=[("+" if x > 0 else "") + fmt(x, 1) for x in z], textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}: %{x:+.2f} desviaciones estándar frente al promedio<extra></extra>")
        fig_z.add_vline(x=0, line_color=GRIS, row=1, col=i + 1)
    fig_z.update_xaxes(range=[-1.7, 1.7], tickvals=[-1, 0, 1], ticktext=["Menos", "Prom.", "Más"])
    fig_z.update_yaxes(autorange="reversed")
    fig_z.update_layout(margin=dict(l=10, r=10, t=30, b=10))

    # Las 8 acciones en cada estilo (cajas con los cuartiles ya calculados)
    fig_d = make_subplots(rows=2, cols=4, subplot_titles=[etq_perfil(v) for v in vs],
                          horizontal_spacing=0.06, vertical_spacing=0.18)
    for j, v in enumerate(vs):
        for i, r in enumerate(perf):
            sv = _REFP.loc[_REFP.perfil == i, v]
            q1, med, q3 = sv.quantile([.25, .5, .75])
            iqr = q3 - q1
            fig_d.add_box(x=[f"E{i + 1}"], q1=[q1], median=[med], q3=[q3],
                          lowerfence=[sv[sv >= q1 - 1.5 * iqr].min()], upperfence=[sv[sv <= q3 + 1.5 * iqr].max()],
                          marker_color=PAL_PERFIL[i], fillcolor=PAL_PERFIL[i], line=dict(width=1.3), opacity=0.85,
                          showlegend=False, row=j // 4 + 1, col=j % 4 + 1, hoverinfo="y")
    fig_d.update_annotations(font=dict(size=12.5))
    fig_d.update_layout(margin=dict(l=10, r=10, t=30, b=10))

    fig_v = go.Figure(go.Bar(
        x=[r["pct_victorias"] for r in perf], y=[f"E{i + 1}. {r['nombre']}" for i, r in enumerate(perf)],
        orientation="h", marker_color=PAL_PERFIL[:len(perf)],
        text=[fmt_pct(r["pct_victorias"]) for r in perf], textposition="outside", cliponaxis=False,
        customdata=[r["n"] for r in perf],
        hovertemplate="%{y}<br>Ganó el %{x:.1f} % de sus partidos<br>n = %{customdata:,}<extra></extra>"))
    fig_v.add_vline(x=50, line_dash="dash", line_color=GRIS)
    fig_v.update_layout(xaxis=dict(range=[0, 100], ticksuffix=" %"), yaxis=dict(autorange="reversed"),
                        margin=dict(l=10, r=40, t=10, b=10))

    medias = pd.DataFrame([{"Estilo": f"E{i + 1}. {r['nombre']}",
                            **{etq_perfil(v): fmt(r["medias"][v], 3 if v == "hitpct" else 2) for v in vs}}
                           for i, r in enumerate(perf)])

    return html.Div([
        html.P([f"Cuatro formas frecuentes de jugar un partido en la AVP, encontradas con K-means en "
                f"{fmt(PF['filas'], 0)} actuaciones (una pareja en un partido, {PF['anios'][0]}–{PF['anios'][1]}). "
                "Están ordenadas de la que más gana a la que menos. ",
                html.Strong("Los filtros de arriba no aplican a esta pestaña.")], className="pagina-intro mb-3"),
        html.Div(tarjetas, className="perfiles mb-3"),
        acciones,
        tarjeta("La huella de cada estilo en las 8 acciones",
                dcc.Graph(figure=fig_z, config=CONFIG, style={"height": "370px"}),
                "Cuánto hace cada estilo de cada acción frente a la pareja promedio · en color, las dos acciones que "
                "le dan el nombre"),
        dbc.Row([
            dbc.Col(tarjeta("Dónde cae cada estilo", [
                dbc.Row([dbc.Col([html.Label("Eje horizontal", className="control-etq"),
                                  dcc.Dropdown(id="pf-x", options=_OPC_P, value="hitpct", clearable=False)], sm=6),
                         dbc.Col([html.Label("Eje vertical", className="control-etq"),
                                  dcc.Dropdown(id="pf-y", options=_OPC_P, value="kills_set", clearable=False)], sm=6)],
                        className="mb-2"),
                html.Div([html.Label("Resaltar", className="control-etq me-2 d-inline"),
                          chips("pf-resaltar", "todos", [("todos", "Todos")] +
                                [(str(i), f"E{i + 1}") for i in range(len(perf))])], className="mb-1"),
                grafico("pf-disp", 380),
                html.Div([html.Span([html.Span(className="leyenda-punto", style={"background": PAL_PERFIL[i]}),
                                     f"E{i + 1}"], className="leyenda-item") for i in range(len(perf))] +
                         [html.Span([html.Span("✕", className="leyenda-x"), "Centro del estilo"],
                                    className="leyenda-item")], className="leyenda-html"),
            ], "Cada punto es una actuación (muestra de 3.000) · elija dos de las 8 acciones · en las zonas donde "
               "los colores se tocan, una actuación podría parecerse a dos estilos"), lg=7),
            dbc.Col(tarjeta("% de partidos ganados por estilo",
                            dcc.Graph(figure=fig_v, config=CONFIG, style={"height": "440px"}),
                            "Línea en 50 % · asociación observada, no causa: las acciones ocurren en el mismo partido"),
                    lg=5),
        ]),
        tarjeta("Las 8 acciones en cada estilo", dcc.Graph(figure=fig_d, config=CONFIG, style={"height": "520px"}),
                "Caja: el 50 % central de las actuaciones de cada estilo; línea: mediana"),
        tarjeta("Promedio por set de cada estilo",
                dbc.Table.from_dataframe(medias, striped=True, hover=True, size="sm",
                                         className="tabla tabla-desplazable"), fuente=True),
        entrenador("Use los estilos como lenguaje común para describir cómo jugó su pareja (\"hoy jugamos tipo E3\"), "
                   "no como una receta: lo que más separa al estilo que gana del que pierde es la eficacia en ataque, "
                   "la misma conclusión del análisis bivariado. En Entrenar, el diagnóstico le dice a qué estilo se "
                   "parece su pareja."),
    ])


@app.callback(Output("pf-disp", "figure"), Input("pf-x", "value"), Input("pf-y", "value"),
              Input("pf-resaltar", "value"))
def dispersion_perfiles(vx, vy, resaltar):
    """Cada actuación coloreada por el estilo que le asignó K-means, con el centro de cada estilo."""
    perf = M.PERF["perfiles"]
    m = _MUESTRA_P
    fig = go.Figure()
    for i, r in enumerate(perf):
        sel = m[m.perfil == i]
        activo = resaltar in ("todos", str(i))
        fig.add_scatter(x=sel[vx], y=sel[vy], mode="markers", name=f"E{i + 1}",
                        marker=dict(color=PAL_PERFIL[i] if activo else PIZARRA[0], size=5,
                                    opacity=0.6 if activo else 0.25),
                        hovertemplate=f"E{i + 1}. {r['nombre']}<br>{etq_perfil(vx)}: %{{x:.2f}}"
                                      f"<br>{etq_perfil(vy)}: %{{y:.2f}}<extra></extra>")
    for i, r in enumerate(perf):
        if resaltar in ("todos", str(i)):
            fig.add_scatter(x=[r["medias"][vx]], y=[r["medias"][vy]], mode="markers+text", showlegend=False,
                            text=[f"E{i + 1}"], textfont=dict(size=13, color=ELEGIDO),
                            textposition=["top right", "bottom right", "top left", "bottom left"][i % 4],
                            marker=dict(symbol="x", size=18, color=PAL_PERFIL[i], line=dict(color="white", width=2)),
                            hovertemplate=f"Centro de E{i + 1}<extra></extra>")
    fig.update_layout(showlegend=False, xaxis_title=etq_perfil(vx), yaxis_title=etq_perfil(vy),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def contenido_metodologia():
    pruebas = r"""
**Mann-Whitney U.** Compara una variable numérica entre ganadores y perdedores sin suponer
normalidad. Su tamaño del efecto es la **correlación rango-biserial**:

$$RBC = \frac{2U}{n_1 n_2} - 1$$

Va de −1 a 1. Poder discriminante: **alto** si |RBC| ≥ 0,5; **moderado** entre 0,3 y 0,5; **bajo** si es menor a 0,3.

**Chi-cuadrado y V de Cramér.** Miden si dos variables categóricas están asociadas:

$$V = \sqrt{\frac{\chi^2}{n\,(\min(r, c) - 1)}}$$

Despreciable si V < 0,1; pequeña hasta 0,3; media hasta 0,5; grande desde 0,5.

**Wilcoxon de rangos con signo.** Cada partido aporta un ganador y un perdedor que no son
independientes; por eso se compara también a cada pareja con su propio rival.
"""
    otras = r"""
**Solapamiento IQR.** Cuánto se superponen los rangos centrales (P25–P75) de ganadores y
perdedores: 0 si no se tocan, 1 si son idénticos.

**Correlación de Spearman** ($\rho$). Asociación monótona entre dos variables, usando rangos.

**Factor de inflación de la varianza.** Verifica que las variables previas al partido no sean
redundantes:

$$VIF_j = \frac{1}{1 - R^2_j}$$

Aceptable si VIF ≤ 5.

**¿Por qué tamaño del efecto y no p-valor?** Con más de 150.000 filas casi todo es
"significativo". El p-valor dice si una diferencia existe; el tamaño del efecto dice si importa.
"""
    fases = pd.DataFrame([
        ("1. Datos crudos", "Archivo original, nunca se modifica", "76.756 partidos"),
        ("2. Limpieza", "12 pasos: fechas, rankings, marcador, etapas, llaves y estadísticas",
         "Cifras de control comparables con R"),
        ("3. Archivos limpios", "Tres tablas con distinta unidad de análisis",
         "Partido · equipo-partido · jugador-partido"),
        ("4. Análisis", "Exploratorio univariado, bivariado y multivariado", "Libro y dashboard"),
        ("5. Modelado", "Modelo previo al partido y perfiles de juego, entrenados fuera del dashboard",
         "entrenar_modelos.py → dashboard/modelos/"),
        ("6. Conclusiones", "Recomendaciones para el entrenador", "Calculadora y hallazgos")],
        columns=["Fase", "Qué se hace", "Resultado"])
    refs = ["Vagnar, A. (2020). AVP & FIVB beach volleyball match database. BigTime Stats.",
            "R for Data Science Online Learning Community (2020). TidyTuesday: Beach volleyball.",
            "Mann, H. B. y Whitney, D. R. (1947). The Annals of Mathematical Statistics, 18(1), 50–60.",
            "Kerby, D. S. (2014). The simple difference formula. Comprehensive Psychology, 3.",
            "Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2.ª ed.).",
            "Cramér, H. (1946). Mathematical methods of statistics. Princeton University Press.",
            "Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6).",
            "Spearman, C. (1904). The American Journal of Psychology, 15(1), 72–101."]
    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta("Enfoque", [
                html.P("Análisis exploratorio de datos (EDA) en tres niveles, complementado con un modelo "
                       "previo al partido (página Predicción) y perfiles de juego descriptivos:"),
                html.Ul([html.Li([html.Strong("Univariado: "), "cada variable por separado."]),
                         html.Li([html.Strong("Bivariado: "), "cada variable frente al resultado."]),
                         html.Li([html.Strong("Multivariado: "), "varias variables a la vez."])]),
                html.P([html.Strong("Unidades de análisis: "), "partido, equipo-partido (una fila por "
                        "pareja) y jugador-partido."]),
                html.P([html.Strong("Favorito y sorpresa: "), "el favorito es la pareja con mejor ranking; "
                        "una sorpresa (upset) es un partido que gana la peor rankeada."], className="mb-0")],
                fuente=False), lg=4),
            dbc.Col(tarjeta("Pruebas no paramétricas", dcc.Markdown(pruebas, mathjax=True), fuente=False), lg=4),
            dbc.Col(tarjeta("Solapamiento, correlación y redundancia", dcc.Markdown(otras, mathjax=True),
                            fuente=False), lg=4),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Fases del procesamiento",
                            dbc.Table.from_dataframe(fases, striped=True, hover=True, className="tabla"),
                            fuente=False), lg=7),
            dbc.Col(tarjeta("Limitaciones", html.Ul([
                html.Li("Las estadísticas de juego describen casi solo a la AVP; no se imputan."),
                html.Li("Cobertura desigual en el tiempo: 2000 y 2019 son parciales y la AVP casi no tiene partidos en 2011–2012."),
                html.Li("Muchas parejas de clasificatoria no tienen ranking numérico."),
                html.Li("Asociación no es causalidad."),
                html.Li("Los datos fueron recopilados por terceros; no son un registro oficial.")]),
                fuente=False), lg=5),
        ]),
        tarjeta("Referencias principales", [html.Ul([html.Li(r) for r in refs]),
                                            html.A("Ver la lista completa en el libro →", href=LIBRO + "10_referencias.html",
                                                   target="_blank")], fuente=False),
    ])


def contenido_conclusiones():
    rho = D.E[VARS_M].corr(method="spearman").values
    rho_max = np.abs(rho[np.tril_indices_from(rho, k=-1)]).max()
    cv = D.chi_v(D.E.resultado, D.E.es_clasif)["v"]
    pct_fivb = 100 * (D.P.circuit == "FIVB").mean()
    datos_ = [
        ("El circuito define los datos", f"La FIVB tiene el {fmt_pct(pct_fivb, 2)} de los partidos, pero casi solo la AVP registra estadísticas de juego."),
        ("El faltante no es aleatorio", f"Solo el {fmt_pct(C['pct_stats'])} de los partidos tiene estadísticas completas; describen a la AVP de 2003 a 2019."),
        ("Los atípicos son legítimos", f"Jugadores muy altos, veteranos o partidos largos son reales. Solo se anularon {fmt(D.RES['p8_celdas_invalidas'], 0)} celdas imposibles."),
        ("Cobertura desigual en el tiempo", "La AVP casi desaparece en 2011–2012 por su crisis; 2000 y 2019 son años parciales."),
    ]
    ganar = [
        ("Una de cada tres es sorpresa", f"El mejor rankeado gana el {fmt_pct(C['pct_fav'])} de los {fmt(C['n_rank'], 0)} partidos con ranking."),
        ("Importa la distancia con el rival", f"El favorito gana el {fmt_pct(C['fav_cerca'])} con 1–2 posiciones de diferencia y el {fmt_pct(C['fav_lejos'])} con más de 20."),
        ("Edad y estatura pesan poco", f"Poder discriminante bajo. 10 cm o más de ventaja de estatura llevan la tasa de victoria al {fmt_pct(C['est10'])}."),
        ("Venir de clasificatoria pesa poco", f"Esas parejas pierden el {fmt_pct(_perd[1])} frente al {fmt_pct(_perd[0])}, con asociación despreciable (V = {fmt(cv, 3)})."),
        ("En la AVP gana la eficacia en ataque", f"% de ataque con poder alto (RBC = {fmt(_poder.rbc['hitpct'])}); le siguen errores de ataque, bloqueos, kills y aces."),
        ("Las variables no se repiten", f"Correlaciones de Spearman ≤ {fmt(rho_max)} y VIF ≤ {fmt(_vif.VIF.max())}."),
    ]

    def lista(items, inicio):
        return html.Div([html.Div([html.Div(str(i), className="num"),
                                   html.Div([html.Strong(t), html.Div(x, className="texto")])], className="hallazgo-item")
                         for i, (t, x) in enumerate(items, inicio)])
    recs = [("trofeo", "Preparar un partido", "Use la diferencia de ranking con el rival (calculadora). Con 1–2 posiciones el partido está abierto; solo con más de 20 el favorito es claro."),
            ("reloj", "Exigencia física", f"Un partido típico dura {fmt(C['dur_med'], 0)} minutos, pero uno de cada tres se va a tercer set y se acerca a {fmt(C['dur_3'], 0)}."),
            ("equipo", "Armar parejas", "Estatura y edad inclinan poco la balanza; no deben pesar más que el nivel reflejado en el ranking."),
            ("silbato", "Priorizar el entrenamiento", "Referencia AVP: convertir los ataques y reducir errores separa más al ganador que atacar más veces.")]
    return html.Div([
        html.Div([html.Div("Respuesta a la pregunta", className="hero-eyebrow"),
                  html.P("Antes del partido, el factor medible que más se asocia con ganar es la ventaja de ranking sobre "
                         "el rival; la estatura, la edad y el contexto aportan muy poco en comparación. Durante el partido, "
                         "al menos en la AVP, lo que separa al ganador es la eficacia en ataque.", className="respuesta")],
                 className="respuesta-caja mb-4"),
        dbc.Row([dbc.Col(tarjeta("Sobre los datos", lista(datos_, 1), fuente=False), lg=5),
                 dbc.Col(tarjeta("Sobre qué hace ganar a una pareja", lista(ganar, 5), fuente=False), lg=7)]),
        html.H5("Recomendaciones para el entrenador", className="seccion-titulo"),
        dbc.Row([dbc.Col(html.Div([html.Div(ico(icono), className="jugada-icono"), html.H6(t), html.P(x)], className="rec"), md=6, lg=3)
                 for icono, t, x in recs], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(tarjeta("Qué no se puede concluir", html.Ul([
                html.Li("Que las estadísticas de juego describan a la FIVB."),
                html.Li("Que un factor cause la victoria: todos los resultados son asociaciones."),
                html.Li("Que las relaciones se mantengan iguales hoy: el análisis agrupa veinte años.")]), fuente=False), lg=6),
            dbc.Col(tarjeta("Qué datos harían falta", html.Ul([
                html.Li("Estadísticas de juego de la FIVB."),
                html.Li("Ranking numérico de las parejas de clasificatoria."),
                html.Li("Contexto: viento, lesiones, tiempo de la pareja junta, historial entre rivales.")]), fuente=False), lg=6),
        ]),
    ])


PESTANAS_ANALISIS = [("uni", "Univariado", contenido_uni), ("bi", "Bivariado", contenido_bi),
                     ("multi", "Multivariado", contenido_multi), ("perfiles", "Estilos de juego (AVP)", contenido_perfiles),
                     ("metodologia", "Metodología", contenido_metodologia),
                     ("conclusiones", "Conclusiones", contenido_conclusiones)]


def pagina_analisis(pestana="uni"):
    return html.Div([
        encabezado("Análisis exploratorio", "Explore las variables una por una, compare ganadores con perdedores, vea qué "
                   "variables se mueven juntas y qué estilos de juego hay en la AVP. Los filtros de arriba aplican "
                   "a las pestañas univariada, bivariada y multivariada."),
        dbc.Tabs([dbc.Tab(f(), label=etq, tab_id=k) for k, etq, f in PESTANAS_ANALISIS],
                 active_tab=pestana, className="pestanas mb-4", id="a-pestanas"),
    ])


ANCHO = {"edad": 1, "estatura": 2.54, "ranking": 1, "duracion": 2, "puntos": 2, "kills": 1,
         "aces": 1, "blocks": 1, "digs": 1, "hitpct": 0.02}


@app.callback(Output("u-kpis", "children"), Output("u-fig", "figure"), Output("u-sub", "children"),
              Input("u-var", "value"), *FILTROS)
def actualizar_uni(clave, circuito, genero, anios, etapa):
    d, col = D.serie_uni(clave, circuito, genero, anios, etapa)
    _, _, etq, unidad, solo_stats = D.VARS_UNI[clave]
    s = d[col].dropna()
    if len(s) < 5:
        return [], vacia(), ""
    est = D.describir(s)
    dec = 3 if clave == "hitpct" else 1
    kpis = [kpi(fmt(est["n"], 0), "Observaciones"), kpi(fmt(est["Media"], dec), "Media"),
            kpi(fmt(est["Mediana"], dec), "Mediana"), kpi(fmt(est["DE"], dec), "Desv. estándar"),
            kpi(f"{fmt(est['P25'], dec)} – {fmt(est['P75'], dec)}", "50 % central (P25–P75)"),
            kpi(fmt(est["Asimetría"]), "Asimetría", "> 1: cola larga a la derecha" if est["Asimetría"] > 1 else None)]

    fig = go.Figure()
    if clave == "sets":
        v = s.value_counts().sort_index()
        fig.add_bar(x=[f"{int(k)} set" + ("" if k == 1 else "s") for k in v.index], y=v.values, marker_color=NEUTRO,
                    text=[fmt_pct(100 * x / v.sum()) for x in v.values], textposition="outside", cliponaxis=False,
                    hovertemplate="%{x}<br>%{y:,} partidos<extra></extra>")
    elif clave == "estatura" and genero == "Todos":
        for g in ["M", "W"]:
            histograma(fig, d.loc[d.gender == g, col], ANCHO[clave], PAL[g], GENERO[g], 0.65)
        fig.update_layout(barmode="overlay")
    else:
        histograma(fig, s, ANCHO[clave], NEUTRO)
        fig.add_vline(x=est["Mediana"], line_dash="dash", line_color=GRIS,
                      annotation_text=f"Mediana: {fmt(est['Mediana'], dec)}", annotation_position="top right")
    fig.update_layout(xaxis_title=f"{etq} ({unidad})", yaxis=dict(title="Frecuencia", tickformat=",d"), bargap=0)
    sub = "Equipos con estadísticas completas (casi todos de la AVP)" if solo_stats else \
        ("Partidos jugados (sin forfeits)" if D.VARS_UNI[clave][0] == "P" else
         "Una fila por jugador en cada partido" if D.VARS_UNI[clave][0] == "J" else "Una fila por pareja en cada partido")
    return kpis, fig, sub


@app.callback(Output("b-kpis", "children"), Output("b-caja", "figure"), Output("b-sub", "children"),
              Output("b-poder", "figure"), Output("b-fav", "figure"), Output("b-upset", "figure"),
              Output("b-chi", "children"),
              Input("b-var", "value"), *FILTROS)
def actualizar_bi(var, circuito, genero, anios, etapa):
    etq, unidad, solo_stats = D.VARS_BI[var]
    e = D.filtrar(D.E, circuito, genero, anios, etapa)
    d = D.solo_avp(e) if solo_stats else e
    r = D.comparar(d, var)
    if r is None:
        msg = "No hay estadísticas de juego con estos filtros (solo existen para la AVP)" if solo_stats else None
        fig_caja = vacia(msg) if msg else vacia()
        kpis = []
    else:
        dec = 3 if var == "hitpct" else 0 if var == "rank_main" else 1
        kpis = [kpi(fmt(r["med_g"], dec), "Mediana ganadores", tono="tono-mar"),
                kpi(fmt(r["med_p"], dec), "Mediana perdedores", tono="tono-coral"),
                kpi(fmt(r["rbc"]), "Correlación rango-biserial"),
                kpi(html.Span(r["poder"], className=f"poder poder-{r['poder'].lower()}"), "Poder discriminante"),
                kpi(fmt(r["solap"]), "Solapamiento IQR"), kpi(fmt_p(r["p"]), "p-valor (Mann-Whitney)")]
        fig_caja = go.Figure()
        partir = var == "estatura_media_cm" and genero == "Todos"
        for res in ["Ganador", "Perdedor"]:
            dd = d[d.resultado == res]
            if partir:
                for g in ["M", "W"]:
                    caja(fig_caja, dd.loc[dd.gender == g, var], GENERO[g], PAL[res], res, g == "M")
            else:
                caja(fig_caja, dd[var], res, PAL[res], res, False)
        fig_caja.update_layout(boxmode="group", yaxis_title=f"{etq} ({unidad})")
    sub = ("Solo AVP con estadísticas completas" if solo_stats else "Una fila por pareja en cada partido") + \
          (" · la estatura se separa por género" if var == "estatura_media_cm" and genero == "Todos" else "")

    t = D.tabla_poder(circuito, genero, anios, etapa)
    if t.empty:
        fp = vacia()
    else:
        t = t.assign(a=t.rbc.abs()).sort_values("a")
        fp = go.Figure(go.Bar(x=t.a, y=t.Variable, orientation="h", marker_color=[PAL_PODER[x] for x in t.poder],
                              customdata=np.stack([t.rbc, t.poder, t.n], axis=1),
                              hovertemplate="%{y}<br>RBC: %{customdata[0]:.3f}<br>Poder: %{customdata[1]}"
                                            "<br>n: %{customdata[2]:,}<extra></extra>"))
        for x in [0.3, 0.5]:
            fp.add_vline(x=x, line_dash="dot", line_color=GRIS)
        fp.update_layout(xaxis=dict(range=[0, 1], title="|RBC|"), margin=dict(l=10, r=20, t=20, b=10))

    p = D.filtrar(D.P, circuito, genero, anios, etapa)
    r_ = p[p.rank_diff.notna()]
    if len(r_) < 30:
        ff, fu, chi = vacia(), vacia(), html.Div("No hay suficientes partidos con estos filtros.")
    else:
        etqs = ["1–2", "3–5", "6–10", "11–20", "21 o más"]
        g = pd.cut(r_.rank_diff.abs(), [0, 2, 5, 10, 20, np.inf], labels=etqs)
        fav = (r_.assign(g=g, f=r_.rank_diff > 0).groupby("g", observed=True)
               .agg(pct=("f", "mean"), n=("f", "size")).reset_index())
        fav["ic"] = 196 * np.sqrt(fav.pct * (1 - fav.pct) / fav.n)
        fav["pct"] = 100 * fav.pct
        ff = go.Figure(go.Scatter(x=fav.g, y=fav.pct, mode="lines+markers+text", text=[fmt_pct(v) for v in fav.pct],
                                  textposition="top center", line=dict(color=PAL["Ganador"], width=2),
                                  marker=dict(size=9), error_y=dict(type="data", array=fav.ic, color=PAL["Ganador"]),
                                  customdata=fav.n,
                                  hovertemplate="Diferencia: %{x}<br>Gana el favorito: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>"))
        ff.add_hline(y=50, line_dash="dash", line_color=GRIS, annotation_text="50 %: moneda al aire",
                     annotation_position="bottom right")
        ff.update_layout(yaxis=dict(range=[40, 100], ticksuffix=" %"), xaxis_title="Diferencia de ranking (posiciones)")

        ue = (r_.groupby(["etapa", "circuit"], observed=True)
              .agg(pct=("es_upset", "mean"), n=("es_upset", "size")).reset_index())
        ue = ue[ue.n >= 30]
        fu = go.Figure()
        for c in ["FIVB", "AVP"]:
            s = ue[ue.circuit == c].set_index("etapa").reindex(ETAPAS).reset_index()
            if s.n.notna().any():
                fu.add_bar(x=s.etapa, y=100 * s.pct, name=c, marker_color=PAL[c], customdata=s.n,
                           text=[fmt_pct(100 * v) if pd.notna(v) else "" for v in s.pct], textposition="outside",
                           cliponaxis=False, textfont=dict(size=11),
                           hovertemplate="%{x} · " + c + "<br>Sorpresas: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>")
        fu.update_layout(barmode="group", yaxis=dict(ticksuffix=" %"), bargap=0.3, uniformtext=dict(minsize=10, mode="show"))

        cruces = [("Sorpresa × circuito", r_.es_upset, r_.circuit), ("Sorpresa × género", r_.es_upset, r_.gender),
                  ("Sorpresa × etapa", r_.es_upset, r_.etapa),
                  ("Resultado × viene de clasificatoria", e.resultado, e.es_clasif)]
        filas = []
        for nombre, x, y in cruces:
            cv = D.chi_v(x, y)
            if cv:
                mag = "Despreciable" if cv["v"] < 0.1 else "Pequeña" if cv["v"] < 0.3 else "Media" if cv["v"] < 0.5 else "Grande"
                filas.append({"Cruce": nombre, "V de Cramér": fmt(cv["v"], 3), "p-valor": fmt_p(cv["p"]), "Magnitud": mag})
        chi = dbc.Table.from_dataframe(pd.DataFrame(filas), striped=True, hover=True, size="sm", className="tabla") \
            if filas else html.Div("Con estos filtros no hay cruces que comparar (cada variable queda con una sola categoría).")
    return kpis, fig_caja, sub, fp, ff, fu, chi


VARS_M = ["rank_main", "rank_rival", "edad_media", "estatura_media_cm", "dif_edad_pareja",
          "dif_estatura_pareja_cm", "duracion_min"]
ETQ_M = ["Ranking propio", "Ranking rival", "Edad media", "Estatura media", "Dif. edad pareja",
         "Dif. estatura pareja", "Duración"]


@app.callback(Output("m-spear", "figure"), Input("m-res", "value"), *FILTROS)
def actualizar_multi(res, circuito, genero, anios, etapa):
    d = D.filtrar(D.E, circuito, genero, anios, etapa)
    if res != "Todos":
        d = d[d.resultado == res]
    if len(d) < 30:
        return vacia()
    rho = d[VARS_M].corr(method="spearman").values.copy()
    rho[np.triu_indices_from(rho, k=1)] = np.nan
    fig = go.Figure(go.Heatmap(x=ETQ_M, y=ETQ_M, z=rho, colorscale=ESCALA_CORR, zmin=-1, zmax=1,
                               texttemplate="%{z:.2f}", colorbar=dict(title="rho"), xgap=2, ygap=2,
                               hovertemplate="%{y} – %{x}<br>rho = %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-30)
    return fig


# ==========================================================================
# Página: Predicción (modelos, simulador ¿Quién gana? y calculadora histórica)
# Todo lo que se muestra sale de dashboard/modelos/, generado por entrenar_modelos.py
# ==========================================================================
METRICAS = [("accuracy", "Accuracy"), ("precision", "Precisión"), ("recall", "Exhaustividad (recall)"),
            ("f1", "F1"), ("roc_auc", "ROC-AUC")]


def contenido_modelos():
    """El modelo que usa el dashboard, explicado para el entrenador; el detalle técnico va plegado."""
    if M.MET is None or M.CUR is None:
        return sin_modelos()
    MET, rec = M.MET, M.MET["recomendado"]
    pol, regla = MET["politica"], MET["regla"]
    res = MET["modelos"]
    r = res[rec]["prueba"]
    n_prueba = pol["n_prueba"]

    # ¿En qué se fija? (importancia por permutación)
    imp = pd.DataFrame(MET["importancias"][rec]).sort_values("media")
    imp["pct"] = 100 * imp.media.clip(lower=0) / imp.media.clip(lower=0).sum()
    fimp = go.Figure(go.Bar(x=imp.pct, y=imp.etiqueta, orientation="h", marker_color=NEUTRO,
                            text=[fmt_pct(x, 0) if x >= 1 else "< 1 %" for x in imp.pct], textposition="outside",
                            cliponaxis=False, customdata=imp.media,
                            hovertemplate="%{y}<br>%{x:.1f} % de lo que usa el modelo"
                                          "<br>(caída de AUC al desordenarla: %{customdata:.4f})<extra></extra>"))
    fimp.update_layout(xaxis=dict(title="Peso en las predicciones", ticksuffix=" %", range=[0, 105]),
                       margin=dict(l=10, r=30, t=10, b=10))

    # ¿Se puede confiar en el porcentaje? (calibración con los partidos de prueba)
    dist = M.CUR[rec]["distribucion"]
    bb = np.array(dist["bordes"])
    cen = (bb[:-1] + bb[1:]) / 2
    g, pdr = np.array(dist["gana"]), np.array(dist["pierde"])
    filas_c = []
    for lo in np.arange(0, 1, 0.1):
        m = (cen >= lo) & (cen < lo + 0.1)
        n = g[m].sum() + pdr[m].sum()
        if n >= 50:
            filas_c.append((lo + 0.05, 100 * g[m].sum() / n, int(n), f"{int(round(lo * 100))}–{int(round(lo * 100)) + 10} %"))
    cal = pd.DataFrame(filas_c, columns=["x", "real", "n", "rango"])
    fcal = go.Figure()
    fcal.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=GRIS, dash="dash"), hoverinfo="skip",
                     showlegend=False)
    fcal.add_bar(x=100 * cal.x, y=cal.real, width=8, marker_color=MAR, customdata=np.stack([cal.n, cal.rango], axis=1),
                 text=[fmt_pct(v, 0) for v in cal.real], textposition="outside", cliponaxis=False, showlegend=False,
                 hovertemplate="El modelo dijo %{customdata[1]}<br>Ganaron de verdad: %{y:.1f} %"
                               "<br>%{customdata[0]:,} partidos<extra></extra>")
    fcal.update_layout(xaxis=dict(title="Probabilidad de ganar que dio el modelo", range=[0, 100], ticksuffix=" %",
                                  dtick=10),
                       yaxis=dict(title="% que de verdad ganó", range=[0, 105], ticksuffix=" %"),
                       margin=dict(l=10, r=10, t=10, b=10))

    # --- Detalle técnico (plegado) ---
    filas = [{"Modelo": n + (" ★" if n == rec else ""),
              "AUC validación cruzada": f"{fmt(res[n]['auc_cv_media'], 3)} ± {fmt(res[n]['auc_cv_de'], 3)}",
              "AUC prueba": fmt(res[n]["prueba"]["roc_auc"], 3),
              "Accuracy prueba": fmt(res[n]["prueba"]["accuracy"], 3),
              "Precisión": fmt(res[n]["prueba"]["precision"], 3), "Recall": fmt(res[n]["prueba"]["recall"], 3),
              "F1": fmt(res[n]["prueba"]["f1"], 3)} for n in M.NOMBRES]
    filas.append({"Modelo": "Regla \"gana el mejor rankeado\"", "AUC validación cruzada": "—",
                  "AUC prueba": fmt(regla["roc_auc"], 3), "Accuracy prueba": fmt(regla["accuracy"], 3),
                  "Precisión": fmt(regla["precision"], 3), "Recall": fmt(regla["recall"], 3),
                  "F1": fmt(regla["f1"], 3)})
    tabla = pd.DataFrame(filas)

    cm = np.array(res[rec]["matriz_confusion"]["valores"])
    total = cm.sum()
    fcm = go.Figure(go.Heatmap(z=cm, x=["Predice: gana", "Predice: pierde"], y=["Real: gana", "Real: pierde"],
                               colorscale=ESCALA_SEC, showscale=False, xgap=3, ygap=3,
                               text=[[f"{fmt(v, 0)}<br>{fmt_pct(100 * v / total)}" for v in f] for f in cm],
                               texttemplate="%{text}", textfont=dict(size=15),
                               hovertemplate="%{y} · %{x}<br>%{z:,} partidos<extra></extra>"))
    fcm.update_yaxes(autorange="reversed", showgrid=False)
    fcm.update_xaxes(showgrid=False, side="top")

    fd = go.Figure()
    for clave, etq, color in [("gana", "Real: gana", PAL["Ganador"]), ("pierde", "Real: pierde", PAL["Perdedor"])]:
        fd.add_bar(x=cen, y=dist[clave], width=bb[1] - bb[0], name=etq, marker_color=color, opacity=0.65,
                   hovertemplate=etq + "<br>Probabilidad %{x:.3f}<br>%{y:,} partidos<extra></extra>")
    fd.add_vline(x=0.5, line_dash="dash", line_color=GRIS)
    fd.update_layout(barmode="overlay", bargap=0, xaxis=dict(range=[0, 1], title="Probabilidad predicha de ganar"),
                     yaxis=dict(title="Partidos de prueba", tickformat=",d"))

    kpis_tec = [kpi(fmt(r[k], 3), etq) for k, etq in METRICAS] + [
        kpi(f"{fmt(res[rec]['auc_cv_media'], 3)} ± {fmt(res[rec]['auc_cv_de'], 3)}", "AUC validación cruzada",
            "5 pliegues en entrenamiento")]
    limitaciones = [
        "Solo usa información previa al partido. No hay datos de forma reciente, lesiones, viento ni historial "
        "entre rivales.",
        f"Solo incluye partidos con ranking, edad y estatura de las dos parejas "
        f"({fmt(pol['descartes']['partidos_modelo'], 0)} de {fmt(pol['descartes']['partidos_iniciales'], 0)}).",
        "Son asociaciones observacionales: el modelo no dice que cambiar una variable cause la victoria.",
        f"La AUC de prueba ({fmt(r['roc_auc'], 3)}) es menor que la de validación cruzada "
        f"({fmt(res[rec]['auc_cv_media'], 3)}): en 2017–2019 hubo más sorpresas y el ranking anticipa peor.",
    ]
    tecnico = dbc.Accordion([
        dbc.AccordionItem([
            html.P(MET["razon"], className="mb-2"),
            dbc.Table.from_dataframe(tabla, striped=True, hover=True, size="sm", className="tabla tabla-desplazable"),
            html.Div(f"★ modelo usado · prueba 2017–2019 ({fmt(n_prueba, 0)} partidos). Se compararon tres modelos y "
                     "se eligió el de mejor AUC en validación cruzada; los otros dos no se usan en el dashboard.",
                     className="nota")], title="¿Por qué XGBoost? Comparación con los otros modelos"),
        dbc.AccordionItem([
            html.Div(kpis_tec, className="kpis mb-3"),
            dbc.Row([dbc.Col([html.Div("Matriz de confusión (umbral 0,5)", className="control-etq"),
                              dcc.Graph(figure=fcm, config=CONFIG, style={"height": "300px"})], lg=5),
                     dbc.Col([html.Div("Probabilidad predicha según el resultado real", className="control-etq"),
                              dcc.Graph(figure=fd, config=CONFIG, style={"height": "300px"})], lg=7)]),
            dbc.Row([dbc.Col([html.Div("Curvas de desempeño frente a la regla del ranking", className="control-etq"),
                              chips("p-curva", "roc", [("roc", "ROC"), ("pr", "Precisión-exhaustividad")]),
                              grafico("p-curvas", 340)], lg=6),
                     dbc.Col([html.Div("Umbral de decisión", className="control-etq"),
                              dcc.Slider(id="p-umbral", min=0.05, max=0.95, step=0.01, value=0.5,
                                         marks={v: fmt(v, 1) for v in [0.1, 0.3, 0.5, 0.7, 0.9]},
                                         tooltip={"placement": "top", "always_visible": False}),
                              html.Div(id="p-umbral-txt", className="nota mb-2"),
                              grafico("p-umbral-fig", 280)], lg=6)]),
        ], title="Métricas completas en la prueba 2017–2019"),
        dbc.AccordionItem(html.Ul([
            html.Li([html.Strong("Alcance: "), pol["alcance"]]),
            html.Li([html.Strong("Unidad: "), pol["unidad"]]),
            html.Li([html.Strong("División: "), pol["division"],
                     f" Entrenamiento: {fmt(pol['n_entrenamiento'], 0)} partidos; prueba: {fmt(n_prueba, 0)}."]),
            html.Li([html.Strong("Validación cruzada: "), pol["validacion_cruzada"]]),
            html.Li([html.Strong("Preprocesamiento: "), pol["preprocesamiento"]]),
            html.Li([html.Strong("Excluidas por fuga de información: "), ", ".join(pol["excluidas_por_fuga"]), "."]),
            html.Li([html.Strong("Hiperparámetros: "), str(res[rec]["hiperparametros"])]),
            html.Li([html.Strong("Semilla: "), f"{pol['semilla']}, la misma del libro."]),
        ], className="lista-politica"), title="Cómo se entrenó"),
        dbc.AccordionItem(html.Ul([html.Li(x) for x in limitaciones]), title="Limitaciones"),
    ], start_collapsed=True, always_open=True, className="tecnico")

    predictores = [pol["etiquetas_predictores"][p_] for p_ in pol["predictores"]]
    return html.Div([
        html.Div([
            html.Div("El modelo que usa el dashboard", className="recomendado-etq"),
            html.Div(rec, className="recomendado-nombre"),
            html.P("Combina cientos de árboles de decisión pequeños. Cada árbol hace preguntas sencillas sobre el "
                   "partido (por ejemplo, \"¿la ventaja de ranking es mayor que 5?\") y cada árbol nuevo corrige los "
                   "errores de los anteriores. Solo usa lo que se sabe antes de empezar:", className="recomendado-razon"),
            html.Div([html.Span(t, className="accion") for t in predictores], className="acciones mt-2"),
        ], className="recomendado mb-4"),
        html.Div([
            kpi(f"{fmt(100 * r['accuracy'], 0)} de 100", "Partidos en los que acierta el ganador", icono="trofeo",
                tono="tono-mar"),
            kpi(f"{fmt(100 * regla['accuracy'], 0)} de 100", "Aciertos de la regla \"gana el mejor rankeado\"",
                icono="analisis"),
            kpi(f"{fmt(100 * r['roc_auc'], 0)} de 100", "Veces que le da más probabilidad al ganador que al perdedor",
                f"AUC = {fmt(r['roc_auc'], 3)}", icono="prediccion"),
            kpi(fmt(n_prueba, 0), "Partidos de 2017–2019 con los que se probó", "el modelo no los vio al aprender",
                icono="balon"),
        ], className="kpis mb-4"),
        html.Div([html.Div("Entonces, ¿para qué sirve si acierta casi lo mismo que el ranking?", className="hero-eyebrow"),
                  html.P(["Porque además de decir quién es el favorito dice ", html.Strong("qué tan favorito es"),
                          ": no es lo mismo un 52 % que un 85 %. Con eso el entrenador sabe si una ventaja es sólida o "
                          "frágil. Es lo que usa la pestaña ", html.Strong("¿Quién gana?"), "."], className="respuesta")],
                 className="respuesta-caja mb-4"),
        dbc.Row([
            dbc.Col(tarjeta("¿En qué se fija el modelo?", dcc.Graph(figure=fimp, config=CONFIG, style={"height": "330px"}),
                            "Cuánto pesa cada dato en sus predicciones: casi todo es la ventaja de ranking"), lg=6),
            dbc.Col(tarjeta("¿Se puede confiar en el porcentaje?", dcc.Graph(figure=fcal, config=CONFIG,
                                                                             style={"height": "330px"}),
                            "Partidos de prueba agrupados por la probabilidad que dio el modelo · si las barras tocan la "
                            "línea punteada, el porcentaje es confiable"), lg=6)]),
        entrenador("Cuando el modelo dice 70 %, las parejas ganaron cerca del 70 % de esos partidos: el porcentaje se "
                   "puede leer tal cual. Úselo para medir qué tan sólida es una ventaja, no como una certeza: incluso "
                   "con 85 %, una de cada siete veces gana el rival."),
        html.H5("Detalle técnico", className="seccion-titulo mt-2"),
        tecnico,
    ])


@app.callback(Output("p-curvas", "figure"), Input("p-curva", "value"))
def actualizar_curvas(tipo):
    rec = M.MET["recomendado"]
    fig = go.Figure()
    for n, color, guion in [(rec, PAL_MODELO[rec], "solid"), ("Regla del ranking", PAL_MODELO["Regla del ranking"], "dash")]:
        cur = M.CUR[n]["con_ranking" if n == "Regla del ranking" else "todos"][tipo]
        x, y = (cur["fpr"], cur["tpr"]) if tipo == "roc" else (cur["recall"], cur["precision"])
        fig.add_scatter(x=x, y=y, mode="lines", name=n, line=dict(color=color, width=3, dash=guion),
                        hovertemplate=n + "<br>%{x:.2f} · %{y:.2f}<extra></extra>")
    if tipo == "roc":
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                        line=dict(color=PIZARRA[0], dash="dot"))
        fig.update_layout(xaxis=dict(range=[0, 1], title="Tasa de falsos positivos"),
                          yaxis=dict(range=[0, 1], title="Tasa de verdaderos positivos"))
    else:
        fig.update_layout(xaxis=dict(range=[0, 1], title="Exhaustividad (recall)"),
                          yaxis=dict(range=[0.4, 1], title="Precisión"))
    return fig


@app.callback(Output("p-umbral-fig", "figure"), Output("p-umbral-txt", "children"), Input("p-umbral", "value"))
def actualizar_umbral(u):
    t = M.CUR[M.MET["recomendado"]]["umbrales"]
    fig = go.Figure()
    # Métricas del mismo modelo: un solo tono (pizarra) y se distinguen por el tipo de línea
    for clave, etq, color, guion in [("accuracy", "Accuracy", PIZARRA[4], "solid"), ("precision", "Precisión", PIZARRA[3], "dash"),
                                     ("recall", "Exhaustividad", PIZARRA[2], "dot"), ("f1", "F1", PIZARRA[4], "dashdot")]:
        fig.add_scatter(x=t["umbral"], y=t[clave], name=etq, mode="lines", line=dict(color=color, width=2, dash=guion),
                        hovertemplate="Umbral %{x:.2f}<br>" + etq + ": %{y:.3f}<extra></extra>")
    fig.add_vline(x=u, line_color=ELEGIDO)
    fig.update_layout(xaxis=dict(range=[0, 1], title="Umbral"), yaxis=dict(range=[0, 1]), hovermode="x unified")
    i = int(np.argmin(np.abs(np.array(t["umbral"]) - u)))
    prec = "no definida (no predice ningún \"gana\")" if t["precision"][i] is None else fmt(t["precision"][i], 3)
    txt = (f"Con umbral {fmt(t['umbral'][i], 2)}: accuracy {fmt(t['accuracy'][i], 3)}, precisión "
           f"{prec}, exhaustividad {fmt(t['recall'][i], 3)}, F1 {fmt(t['f1'][i], 3)}; el modelo "
           f"predice \"gana\" en el {fmt_pct(t['pct_pred_gana'][i])} de los partidos de prueba.")
    return fig, txt


def contenido_simulador():
    if M.MODELO is None or M.MET is None:
        return sin_modelos()

    def pareja(lado, titulo, rank, edad, est):
        return html.Div([
            html.Div(titulo, className="lado-titulo"),
            html.Label("Ranking de la pareja", className="control-etq"),
            dbc.Input(id=f"s-rank-{lado}", type="number", min=1, max=300, step=1, value=rank),
            html.Label("Edad media de la pareja (años)", className="control-etq mt-2"),
            dbc.Input(id=f"s-edad-{lado}", type="number", min=16, max=50, step=0.5, value=edad),
            html.Label("Estatura media de la pareja (cm)", className="control-etq mt-2"),
            dbc.Input(id=f"s-est-{lado}", type="number", min=150, max=215, step=1, value=est),
            dbc.Switch(id=f"s-clasif-{lado}", label="Viene de clasificatoria", value=False, className="mt-2"),
        ], className=f"lado lado-{lado}")
    return html.Div([
        html.P(["Simulador del modelo recomendado (", html.Strong(M.MET["recomendado"]), "). Solo usa información "
                "previa al partido. La probabilidad se calcula desde los dos lados y se promedia, así que la de su pareja "
                "y la del rival suman 100 %."], className="nota mb-3"),
        html.Div([pareja("a", "Su pareja", 8, 28, 190), html.Div(className="red red-chica", **{"aria-hidden": "true"}),
                  pareja("b", "Rival", 12, 28, 188)], className="simulador-cancha mb-3"),
        dbc.Row([dbc.Col([html.Span("Circuito", className="filtro-etq"),
                          chips("s-circuito", "FIVB", [("FIVB", "FIVB"), ("AVP", "AVP")])], md=3),
                 dbc.Col([html.Span("Género", className="filtro-etq"),
                          chips("s-genero", "M", [("M", "Hombres"), ("W", "Mujeres")])], md=3),
                 dbc.Col([html.Span("Etapa", className="filtro-etq"),
                          chips("s-etapa", "Cuadro principal", [(e, e) for e in ETAPAS])], md=6)], className="mb-3"),
        html.Div(id="s-resultado", className="resultado mb-4"),
    ])


@app.callback(Output("s-resultado", "children"),
              Input("s-rank-a", "value"), Input("s-rank-b", "value"), Input("s-edad-a", "value"),
              Input("s-edad-b", "value"), Input("s-est-a", "value"), Input("s-est-b", "value"),
              Input("s-clasif-a", "value"), Input("s-clasif-b", "value"), Input("s-circuito", "value"),
              Input("s-genero", "value"), Input("s-etapa", "value"))
def simular(ra, rb, ea, eb, sa, sb, ca, cb, circuito, genero, etapa):
    if None in (ra, rb, ea, eb, sa, sb):
        return dbc.Alert("Ingrese el ranking, la edad y la estatura media de las dos parejas: el modelo se "
                         "entrenó con partidos que tienen esos tres datos.", color="warning")
    p = M.probabilidad(rank_propio=ra, rank_rival=rb, edad_propia=float(ea), edad_rival=float(eb),
                       est_propia=float(sa), est_rival=float(sb), clasif_propia=bool(ca), clasif_rival=bool(cb),
                       circuito=circuito, genero=genero, etapa=etapa)
    pct = 100 * p
    if pct >= 75:
        lectura = (f"Favorito claro según el modelo, aunque le asigna a la sorpresa un {fmt_pct(100 - pct, 0)} "
                   "de probabilidad.")
    elif pct >= 55:
        lectura = "Ligera ventaja: el modelo inclina la balanza, pero el partido está lejos de decidido."
    elif pct > 45:
        lectura = "Partido abierto: el modelo no separa a las dos parejas."
    else:
        lectura = "El modelo pone a su pareja en desventaja; estos partidos también se ganan."
    r = M.MET["modelos"][M.MET["recomendado"]]["prueba"]
    return html.Div([
        html.Div("Probabilidad estimada de ganar", className="resultado-etq"),
        html.Div([html.Div([html.Span("Su pareja", className="score-etq"),
                            html.Span(fmt_pct(pct, 0), className="digito score-a", id="s-pct")]),
                  html.Span("–", className="score-guion"),
                  html.Div([html.Span(fmt_pct(100 - pct, 0), className="digito score-b"),
                            html.Span("Rival", className="score-etq")])], className="score"),
        html.Div([html.Div(style={"width": f"{pct}%"}, className="barra-a"),
                  html.Div(style={"width": f"{100 - pct}%"}, className="barra-b")], className="barra-duelo mb-2"),
        html.Div(lectura, className="resultado-lectura"),
        html.Div(f"Modelo: {M.MET['recomendado']}, con ROC-AUC {fmt(r['roc_auc'], 3)} y accuracy "
                 f"{fmt_pct(100 * r['accuracy'])} en {fmt(r['n'], 0)} partidos de prueba (2017–2019). Es una "
                 "estimación a partir de asociaciones históricas, no una garantía.", className="resultado-detalle"),
    ])


def contenido_calculadora():
    def sel(id_, valor, ops):
        return dbc.Select(id=id_, value=valor, options=[{"label": l, "value": v} for v, l in ops])
    controles = tarjeta("Su partido", [
        dbc.Row([dbc.Col([html.Label("Ranking de su pareja", className="control-etq"),
                          dbc.Input(id="c-propio", type="number", min=1, max=64, step=1, value=8)]),
                 dbc.Col([html.Label("Ranking del rival", className="control-etq"),
                          dbc.Input(id="c-rival", type="number", min=1, max=64, step=1, value=12)])], className="mb-3"),
        html.Label("Circuito", className="control-etq"),
        sel("c-circuito", "Todos", [("Todos", "Todos"), ("FIVB", "FIVB"), ("AVP", "AVP")]),
        html.Label("Género", className="control-etq mt-3"),
        sel("c-genero", "Todos", [("Todos", "Todos"), ("M", "Hombres"), ("W", "Mujeres")]),
        html.Label("Etapa del torneo", className="control-etq mt-3"),
        sel("c-etapa", "Todas", [("Todas", "Todas")] + [(e, e) for e in ETAPAS]),
        html.Label("Diferencia de estatura media con el rival (cm)", className="control-etq mt-3"),
        dcc.Slider(id="c-estatura", min=-20, max=20, step=1, value=0,
                   marks={v: f"{v:+d}" if v else "0" for v in range(-20, 21, 10)},
                   tooltip={"placement": "top", "always_visible": False}),
        html.Div(id="c-estatura-txt", className="nota"),
    ], fuente=False)
    return html.Div([
        dbc.Alert([html.I(className="bi bi-clock-history me-2"), html.Strong("Calculadora del entrenador. "),
                   "Ingrese el ranking de su pareja y el del rival. La calculadora busca los partidos históricos con una "
                   "ventaja parecida y le muestra cuántos ganó la pareja en su posición. No es el modelo predictivo: "
                   "son frecuencias observadas, sin ajustar por otras variables."], color="light", className="aviso-solape"),
        dbc.Row([
            dbc.Col(controles, lg=4),
            dbc.Col([
                html.Div(id="c-resultado", className="resultado mb-4"),
                tarjeta("% de victorias según la ventaja de ranking sobre el rival", grafico("c-curva", 330),
                        "El punto marca su partido · ventana móvil de ±2 posiciones"),
                tarjeta("% de victorias según la ventaja de estatura", grafico("c-est", 280),
                        "Mismos filtros de circuito, género y etapa · el punto marca su pareja"),
            ], lg=8),
        ]),
        html.Div("Son frecuencias históricas de 2000 a 2019, no una predicción: describen qué pasó en partidos "
                 "parecidos, no garantizan el resultado de uno nuevo.", className="nota mb-4"),
    ])


def pagina_prediccion(pestana="modelos"):
    return html.Div([
        encabezado("Predicción del partido", "¿Se puede anticipar quién gana con lo que se sabe antes del partido? Vea qué "
                   "tan bien lo hace el modelo, simule un partido o consulte la calculadora histórica."),
        dbc.Tabs([dbc.Tab(contenido_modelos(), label="Modelos", tab_id="modelos"),
                  dbc.Tab(contenido_simulador(), label="¿Quién gana?", tab_id="simulador"),
                  dbc.Tab(contenido_calculadora(), label="Calculadora histórica", tab_id="calculadora")],
                 active_tab=pestana, className="pestanas mb-4", id="p-pestanas"),
    ])


@app.callback(Output("c-estatura-txt", "children"), Input("c-estatura", "value"))
def texto_estatura(v):
    if not v:
        return "Seleccionado: misma estatura media que la pareja rival."
    return f"Seleccionado: su pareja es {abs(v)} cm más {'alta' if v > 0 else 'baja'} que la rival."


@app.callback(Output("c-resultado", "children"), Output("c-curva", "figure"), Output("c-est", "figure"),
              Input("c-propio", "value"), Input("c-rival", "value"), Input("c-circuito", "value"),
              Input("c-genero", "value"), Input("c-etapa", "value"), Input("c-estatura", "value"))
def calcular(propio, rival, circuito, genero, etapa, estatura):
    if not propio or not rival:
        return dbc.Alert("Ingrese los dos rankings (de 1 a 64).", color="warning"), vacia(), vacia()
    propio, rival = int(propio), int(rival)
    v = rival - propio                              # positivo: su pareja está mejor rankeada
    d = D.filtrar(D.E, circuito, genero, D.ANIOS, etapa)
    dr = d[d.ventaja_rank.notna()]
    if len(dr) < 30:
        return dbc.Alert("Hay muy pocos partidos con esos filtros. Pruebe con 'Todos'.", color="warning"), vacia(), vacia()

    ancho = 0 if abs(v) <= 2 else 1 if abs(v) <= 6 else 2 if abs(v) <= 15 else 4
    while True:
        sub = dr[dr.ventaja_rank.between(v - ancho, v + ancho)]
        if len(sub) >= 50 or ancho >= 10:
            break
        ancho += 1
    if len(sub) < 10:
        return dbc.Alert("No hay partidos históricos con esa diferencia de ranking y esos filtros.",
                         color="warning"), vacia(), vacia()
    p = (sub.resultado == "Ganador").mean()
    ic = 1.96 * np.sqrt(p * (1 - p) / len(sub))
    pct = 100 * p
    if pct >= 75:
        lectura, clase = "Ventaja clara: su pareja gana la gran mayoría de estos partidos.", "fuerte"
    elif pct >= 55:
        lectura, clase = "Ventaja real pero frágil: uno de cada tres o cuatro partidos así termina en sorpresa.", "media"
    elif pct > 45:
        lectura, clase = "Partido abierto: históricamente es casi una moneda al aire.", "abierto"
    else:
        lectura, clase = "Su pareja parte en desventaja, pero estos partidos también se ganan.", "baja"
    rango = "exactamente" if ancho == 0 else f"entre {v - ancho:+d} y {v + ancho:+d}"
    resultado = html.Div([
        html.Div("Probabilidad histórica de ganar", className="resultado-etq"),
        html.Div(fmt_pct(pct), className=f"resultado-valor resultado-{clase}"),
        dbc.Progress(value=pct, className="resultado-barra mb-2", color=None,
                     style={"--bs-progress-bar-bg": PAL["Ganador"]}),
        html.Div(lectura, className="resultado-lectura"),
        html.Div(f"Basado en {fmt(len(sub), 0)} partidos en que una pareja tenía una ventaja de ranking {rango} "
                 f"posiciones (la suya es {v:+d}). Intervalo de confianza del 95 %: "
                 f"{fmt_pct(max(0, pct - 100 * ic))} a {fmt_pct(min(100, pct + 100 * ic))}.", className="resultado-detalle"),
    ])

    cur = D.curva_victoria(dr, "ventaja_rank", 2)
    cur = cur[(cur.index >= -40) & (cur.index <= 40) & (cur.n >= 100)]
    f1 = go.Figure()
    f1.add_scatter(x=cur.index, y=cur.pct, mode="lines", line=dict(color=PAL["Ganador"], width=2), name="Histórico",
                   customdata=cur.n, hovertemplate="Ventaja %{x:+d}<br>Victorias: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>")
    f1.add_hline(y=50, line_dash="dash", line_color=GRIS)
    f1.add_vline(x=0, line_color="#D1D5DB")
    f1.add_scatter(x=[v], y=[pct], mode="markers", name="Su partido", showlegend=False,
                   marker=dict(size=14, color=ELEGIDO, line=dict(color="white", width=3)),
                   hovertemplate="Su partido<br>Ventaja %{x:+d}<br>%{y:.1f} %<extra></extra>")
    f1.update_layout(showlegend=False, yaxis=dict(range=[0, 100], ticksuffix=" %", title="% de victorias"),
                     xaxis_title="Ventaja de ranking (ranking del rival − ranking propio)")

    de = d[d.ventaja_estatura_cm.notna()]
    cortes = [-np.inf, -10, -5, -2, 2, 5, 10, np.inf]
    etqs = ["≤ −10", "−10 a −5", "−5 a −2", "−2 a 2", "2 a 5", "5 a 10", "≥ 10"]
    g = pd.cut(de.ventaja_estatura_cm, cortes, labels=etqs, right=False)
    t = de.assign(g=g, w=de.resultado == "Ganador").groupby("g", observed=True).agg(pct=("w", "mean"), n=("w", "size"))
    t["pct"] *= 100
    suyo = etqs[int(np.searchsorted(cortes[1:-1], estatura, side="right"))]
    f2 = go.Figure(go.Bar(x=t.index.astype(str), y=t.pct, customdata=t.n,
                          marker_color=[ELEGIDO if k == suyo else PIZARRA[1] for k in t.index],
                          text=[fmt_pct(x) for x in t.pct], textposition="outside", cliponaxis=False,
                          hovertemplate="Ventaja %{x} cm<br>Victorias: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>"))
    f2.add_hline(y=50, line_dash="dash", line_color=GRIS, layer="below")
    f2.update_layout(yaxis=dict(range=[0, 80], ticksuffix=" %"), xaxis_title="Ventaja de estatura sobre el rival (cm)")
    return resultado, f1, f2


# ==========================================================================
# Página: Equipo (por ahora solo la pestaña, sin contenido)
# ==========================================================================
# ==========================================================================
# Página: Plan de entrenamiento (diagnóstico con los datos de la AVP)
# ==========================================================================
CAMPOS_PLAN = [("sets", "Sets jugados", "si suma varios partidos, el total de sets"),
               ("kills", "Kills", "ataques que terminan en punto"),
               ("attacks", "Ataques", "todos los ataques intentados"),
               ("errors", "Errores de ataque", "ataques que se van fuera, a la red o bloqueados"),
               ("aces", "Aces", "saques que terminan en punto directo"),
               ("serve_errors", "Errores de saque", "saques fuera o a la red"),
               ("blocks", "Bloqueos", "bloqueos que terminan en punto"),
               ("digs", "Defensas", "balones defendidos")]


def pagina_entrenamiento():
    campos = [dbc.Col([html.Label(etq, className="control-etq", htmlFor=f"e-{c}"),
                       dbc.Input(id=f"e-{c}", type="number", min=1 if c == "sets" else 0, step=1,
                                 inputMode="numeric", value=PL.EJEMPLOS["tipico"][c]),
                       html.Div(ayuda, className="campo-ayuda")], xs=6, md=4, lg=6, xl=4, className="mb-3")
              for c, etq, ayuda in CAMPOS_PLAN]
    formulario = tarjeta("Estadísticas de su pareja", [
        html.P("Escriba los totales de la pareja (sume los dos jugadores) en uno o varios partidos, y cuántos sets "
               "jugaron. Solo números enteros; el promedio por set y el % de ataque se calculan solos. "
               "Si no las tiene a mano, cargue un ejemplo.",
               className="nota mb-2"),
        html.Div([html.Span("Ejemplos", className="filtro-etq me-2"),
                  chips("e-ejemplo", "tipico", [("tipico", "Pareja típica AVP"), ("pierde", "Pareja que suele perder")])],
                 className="mb-3"),
        dbc.Row(campos),
        html.Button("Diagnosticar", id="e-diagnosticar", n_clicks=0, className="boton-principal mt-1"),
    ], "Una fila por pareja, no por jugador", fuente=False)
    return html.Div([
        encabezado("Plan de entrenamiento",
                   "Compare las estadísticas de su pareja con las de las parejas que ganan en la AVP, vea a qué estilo "
                   "de juego se parece, qué conviene priorizar y llévese un plan para la semana."),
        dbc.Row([dbc.Col(formulario, lg=5),
                 dbc.Col(dcc.Loading(html.Div(id="e-resultado"), custom_spinner=html.Div(
                     html.Img(src="/assets/balon.svg", className="balon-carga", alt=""), className="carga")), lg=7)]),
        dcc.Store(id="e-datos"),
        html.Section([
            html.Div([html.Div([html.H5("Plan para la semana", className="seccion-titulo mb-1"),
                                html.Div("Armado con los datos del diagnóstico: cada prioridad tiene ejercicios y una "
                                         "meta medible.", className="nota mt-0")]),
                      html.Div([html.Span("Días por semana", className="filtro-etq me-2"),
                                chips("e-dias", 3, [(3, "3"), (4, "4"), (5, "5")]),
                                html.Button([html.I(className="bi bi-download me-2"), "Descargar plan"],
                                            id="e-descargar", n_clicks=0, className="boton-secundario ms-2"),
                                dcc.Download(id="e-descarga")], className="plan-controles")],
                     className="plan-cabeza mb-3"),
            dcc.Loading(html.Div(id="e-plan"), custom_spinner=html.Div(
                html.Img(src="/assets/balon.svg", className="balon-carga", alt=""), className="carga")),
            html.Div([
                html.Div([html.Div([html.Span(className="ia-chispa"), "Redactar el plan con IA"], className="ia-titulo"),
                          html.Div("Gemini convierte el plan de arriba en indicaciones listas para la cancha. No cambia "
                                   "las metas ni los ejercicios, y solo recibe las estadísticas: nada personal.",
                                   className="nota mt-0")]),
                html.Button("Redactar con IA", id="e-redactar", n_clicks=0, className="boton-principal"),
            ], className="ia-cabeza mt-4"),
            dcc.Store(id="e-ia-texto"),
            dcc.Loading(html.Div(id="e-ia"), custom_spinner=html.Div(
                [html.Img(src="/assets/balon.svg", className="balon-carga", alt=""),
                 html.Span("Redactando… puede tardar unos segundos", className="nota ms-2")], className="carga")),
        ], id="e-plan-zona", className="oculto mb-4"),
        entrenador(f"La comparación usa {fmt(len(PL.REF), 0)} filas pareja-partido de la AVP "
                   f"({PL.REF.year.min()}–{PL.REF.year.max()}) con estadísticas completas; la FIVB no tiene estas "
                   "estadísticas. Las brechas muestran asociaciones con ganar, no causas: el plan es una guía, no "
                   "reemplaza el criterio del entrenador."),
    ])


@app.callback([Output(f"e-{c}", "value") for c, _, _ in CAMPOS_PLAN], Input("e-ejemplo", "value"),
              prevent_initial_call=True)
def cargar_ejemplo(ejemplo):
    return [PL.EJEMPLOS[ejemplo][c] for c, _, _ in CAMPOS_PLAN]


def figura_percentiles(diag):
    t = diag["tabla"][diag["tabla"].sentido != 0].copy()
    brechas = {b["var"] for b in diag["brechas"]}
    fuertes = {f["var"] for f in diag["fortalezas"]}
    t["color"] = [CORAL if v in brechas else MAR if v in fuertes else PIZARRA[0] for v in t["var"]]
    t = t.iloc[::-1]
    dec = [3 if v == "hitpct" else 1 for v in t["var"]]
    fig = go.Figure(go.Bar(
        x=t.percentil, y=t.etiqueta, orientation="h", marker_color=t.color,
        text=[f"{fmt(p, 0)}" for p in t.percentil], textposition="outside", cliponaxis=False,
        customdata=np.stack([[fmt(v, d) for v, d in zip(t.valor, dec)],
                             [fmt(m, d) for m, d in zip(t.mediana_ganadores, dec)]], axis=1),
        hovertemplate="%{y}<br>Su pareja: %{customdata[0]}<br>Mediana de las ganadoras: %{customdata[1]}"
                      "<br>Supera al %{x:.0f} % de las ganadoras<extra></extra>"))
    fig.add_vline(x=50, line_dash="dash", line_color=GRIS, annotation_text="Mediana de las ganadoras",
                  annotation_position="top")
    fig.update_layout(height=330, margin=dict(l=10, r=30, t=30, b=10), showlegend=False,
                      xaxis=dict(range=[0, 105], title="Percentil frente a las parejas ganadoras (0–100)"))
    return fig


@app.callback(Output("e-resultado", "children"), Input("e-diagnosticar", "n_clicks"),
              [State(f"e-{c}", "value") for c, _, _ in CAMPOS_PLAN])
def diagnosticar(_, *valores):
    datos = dict(zip([c for c, _, _ in CAMPOS_PLAN], valores))
    error = PL.validar(datos)
    if error:
        return dbc.Alert(error, color="warning")
    diag = PL.diagnosticar(datos)
    per, par = diag["perfil"], diag["parecidas"]

    def dato(v, var):
        return fmt_pct(100 * v, 1) if var == "hitpct" else fmt(v, 1)

    tarjetas_brecha = [html.Div([
        html.Div(f"Prioridad {i}", className="brecha-num"),
        html.Div(b["etiqueta"], className="brecha-titulo"),
        html.Div([html.Span(dato(b["valor"], b["var"]), className="brecha-valor"),
                  html.Span(f" frente a {dato(b['mediana_ganadores'], b['var'])} de la ganadora típica",
                            className="brecha-ref")]),
        html.Div(f"Supera solo al {fmt(b['percentil'], 0)} % de las parejas ganadoras.", className="brecha-nota"),
    ], className="brecha") for i, b in enumerate(diag["brechas"], 1)]
    if not tarjetas_brecha:
        tarjetas_brecha = [html.Div("No hay brechas claras: en todas las estadísticas su pareja está en la mitad "
                                    "alta de las ganadoras. Conviene mantener el nivel.", className="nota")]
    fuertes = ", ".join(f"{f['etiqueta'].lower()} (supera al {fmt(f['percentil'], 0)} %)" for f in diag["fortalezas"])

    return html.Div([
        dbc.Row([
            dbc.Col(kpi(per["nombre"] if per else "—", "Estilo de juego más parecido",
                        f"Las parejas con este estilo ganaron el {fmt_pct(per['pct_victorias'])} de sus partidos"
                        if per else None, tono="kpi-perfil tono-gris", icono="analisis"), md=7),
            dbc.Col(kpi(fmt_pct(par["pct_victorias"], 0), "Ganaron las parejas que juegan parecido",
                        f"Las {fmt(par['n'], 0)} parejas de la AVP con estadísticas más cercanas", tono="tono-mar"), md=5),
        ], className="g-3 mb-3"),
        tarjeta("Dónde está su pareja frente a las ganadoras",
                dcc.Graph(figure=figura_percentiles(diag), config=CONFIG),
                "Coral: brechas prioritarias · azul: fortalezas · en los errores, más a la derecha = menos errores"),
        html.H5("Qué priorizar", className="seccion-titulo mb-2"),
        html.Div(tarjetas_brecha, className="brechas mb-2"),
        html.Div(f"Fortalezas: {fuertes}." if fuertes else "", className="nota mb-3"),
        html.Div("La prioridad combina qué tan lejos está su pareja de la ganadora típica y cuánto separa esa "
                 "estadística a ganadores de perdedores en la AVP (correlación rango-biserial).", className="nota"),
    ])


@app.callback(Output("e-datos", "data"), Output("e-plan-zona", "className"), Input("e-diagnosticar", "n_clicks"),
              [State(f"e-{c}", "value") for c, _, _ in CAMPOS_PLAN])
def guardar_datos_plan(_, *valores):
    """Guarda los datos ya validados para armar el plan (y, en el paso siguiente, pasárselo a la IA)."""
    datos = dict(zip([c for c, _, _ in CAMPOS_PLAN], valores))
    if PL.validar(datos):
        return None, "oculto"
    return datos, "mb-4"


@app.callback(Output("e-plan", "children"), Input("e-datos", "data"), Input("e-dias", "value"))
def mostrar_plan(datos, dias):
    if not datos:
        return None
    plan = PL.plantilla(PL.diagnosticar(datos), int(dias))
    metas = html.Div([html.Div([html.Div(f"Meta {i}" if m["var"] else "Meta", className="meta-num"),
                                html.Div(m["texto"], className="meta-texto"),
                                html.Div(m["nota"][0].upper() + m["nota"][1:], className="meta-nota")], className="meta")
                      for i, m in enumerate(plan["metas"], 1)], className="metas mb-3")
    sesiones = html.Div([html.Div([
        html.Div([html.Span(ss["titulo"], className="sesion-titulo"),
                  html.Span(f"{ss['minutos']} min", className="sesion-min")], className="sesion-cabeza"),
        html.Div(["Foco: ", html.Strong(ss["foco"].lower())], className="sesion-foco"),
        html.Ol([html.Li([html.Span(f"{m}'", className="bloque-min"),
                          html.Div([html.Div(t, className="bloque-titulo"), html.Div(x, className="bloque-texto")])],
                         className="bloque") for m, t, x in ss["bloques"]], className="bloques"),
    ], className="sesion") for ss in plan["sesiones"]], className="sesiones")
    extra = []
    if plan["fortalezas"]:
        extra.append(html.Div("Fortalezas que conviene mantener: " +
                              ", ".join(f.lower() for f in plan["fortalezas"]) + ".", className="nota mt-3"))
    return [metas, sesiones] + extra


@app.callback(Output("e-ia", "children"), Output("e-ia-texto", "data"), Input("e-redactar", "n_clicks"),
              Input("e-datos", "data"), Input("e-dias", "value"), prevent_initial_call=True)
def redactar_con_ia(_, datos, dias):
    """Solo llama a la IA con el botón; si cambian los datos o los días, borra la redacción anterior."""
    if ctx.triggered_id != "e-redactar" or not datos:
        return None, None
    diag = PL.diagnosticar(datos)
    plan = PL.plantilla(diag, int(dias))
    try:
        texto, modelo = IA.redactar(diag, plan)
    except RuntimeError as err:
        return dbc.Alert([html.I(className="bi bi-info-circle me-2"), str(err)], color="light",
                         className="aviso-solape mt-3"), None
    return html.Div([
        html.Div([html.Span("Redactado por IA", className="ia-sello"),
                  html.Span(f"{modelo} · revise el texto antes de usarlo", className="nota m-0")], className="mb-2"),
        dcc.Markdown(texto, className="ia-texto"),
    ], className="ia-caja mt-3"), texto


@app.callback(Output("e-descarga", "data"), Input("e-descargar", "n_clicks"), State("e-datos", "data"),
              State("e-dias", "value"), State("e-ia-texto", "data"), prevent_initial_call=True)
def descargar_plan(_, datos, dias, texto_ia):
    """Descarga la redacción de la IA si existe; si no, la plantilla."""
    if not datos:
        return no_update
    if texto_ia:
        return dcc.send_string(texto_ia, "plan_de_entrenamiento_IA.md")
    plan = PL.plantilla(PL.diagnosticar(datos), int(dias))
    return dcc.send_string(PL.plan_texto(plan), "plan_de_entrenamiento.txt")


EQUIPO = [
    {"nombre": "Nicole Ramírez Álvarez", "iniciales": "NR",
     "linea": "Ciencia de Datos · 6.º semestre · Universidad del Norte",
     "rol": "Versión en Python",
     "texto": "Desarrolló el análisis en Python: el libro en Jupyter Book, con la limpieza de los datos, el análisis "
              "exploratorio, los estilos de juego y los modelos de predicción, y este dashboard en Dash, incluidos "
              "el plan de entrenamiento y el asistente con IA.",
     "github": "nicoleramirez2106-ui"},
    {"nombre": "Roger Alberto Daza Polo", "iniciales": "RD",
     "linea": "Ciencia de Datos · 6.º semestre · Universidad del Norte",
     "rol": "Versión en R",
     "texto": "Desarrolló el análisis en R: el libro en bookdown, con la limpieza, el análisis exploratorio y los "
              "modelos, y la aplicación interactiva en Shiny, que replica el dashboard en ese lenguaje.",
     "github": "RogerDaZa"},
]


def pagina_equipo():
    tarjetas = [html.Div([
        html.Div(p["iniciales"], className="miembro-avatar"),
        html.Div(p["rol"], className="miembro-rol"),
        html.H5(p["nombre"], className="miembro-nombre"),
        html.Div(p["linea"], className="miembro-linea"),
        html.P(p["texto"], className="miembro-texto"),
        html.A([html.I(className="bi bi-github me-2"), f"github.com/{p['github']}"],
               href=f"https://github.com/{p['github']}", target="_blank", className="boton-secundario miembro-enlace"),
    ], className="miembro") for p in EQUIPO]
    return html.Div([
        encabezado("Equipo", "Proyecto del curso Visualización de Datos y Toma de Decisiones, Universidad del Norte. "
                   "El mismo análisis se hizo en dos lenguajes: Python y R."),
        html.Div(tarjetas, className="equipo-grid mb-4"),
        html.Div([html.A([ico("libro"), "Libro del proyecto (Python)"], href=LIBRO, target="_blank",
                         className="boton-principal"),
                  html.A([html.I(className="bi bi-github"), "Repositorio"], href=REPO, target="_blank",
                         className="boton-secundario")], className="hero-botones"),
    ])


# ==========================================================================
# Chatbot: asistente que responde con la ficha del proyecto (Gemini, ver ia.py)
# ==========================================================================
_FICHA = None


def ficha_proyecto():
    """Resumen en texto de las cifras del dashboard: es lo único que el asistente puede usar para responder."""
    global _FICHA
    if _FICHA is not None:
        return _FICHA
    t = D.tabla_poder("Todos", "Todos", D.ANIOS)
    poder = "\n".join(f"- {r.Variable}: mediana ganadores {fmt(r.med_g, 3 if r['var'] == 'hitpct' else 1)}, "
                      f"perdedores {fmt(r.med_p, 3 if r['var'] == 'hitpct' else 1)}, correlación rango-biserial "
                      f"{fmt(r.rbc, 2)}, poder {r.poder.lower()}" for _, r in t.iterrows())
    lineas = [
        "DATOS: " + f"{fmt(C['n_partidos'], 0)} partidos de los circuitos FIVB ({fmt(_n_circ.get('FIVB', 0), 0)}) y AVP "
        f"({fmt(_n_circ.get('AVP', 0), 0)}), {D.ANIOS[0]}–{D.ANIOS[1]}, {fmt(C['n_jugadores'], 0)} jugadores, "
        f"{C['n_paises']} países sede. Fuente: vb_matches.csv (BigTimeStats vía TidyTuesday). Solo el "
        f"{fmt_pct(C['pct_stats'], 0)} de los partidos tiene estadísticas de juego, casi todos de la AVP.",
        f"RANKING: en los {fmt(C['n_rank'], 0)} partidos con ranking de las dos parejas, gana la mejor rankeada el "
        f"{fmt_pct(C['pct_fav'])}; el {fmt_pct(100 - C['pct_fav'])} son sorpresas. Con 1–2 posiciones de diferencia "
        f"gana el favorito el {fmt_pct(C['fav_cerca'], 0)}; con más de 20, el {fmt_pct(C['fav_lejos'], 0)}.",
        f"DURACIÓN: un partido típico dura {fmt(C['dur_med'], 0)} minutos; los de tres sets, {fmt(C['dur_3'], 0)}.",
        f"ESTATURA Y CLASIFICATORIA: con 10 cm o más de ventaja de estatura sobre el rival se gana el "
        f"{fmt_pct(C['est10'])}. Las parejas que vienen de clasificatoria pierden el {fmt_pct(_perd[1])} frente al "
        f"{fmt_pct(_perd[0])} del resto (asociación despreciable).",
        "QUÉ SEPARA A GANADORES DE PERDEDORES (por pareja en cada partido; estadísticas de juego solo AVP):\n" + poder,
    ]
    if M.PERF:
        est = "\n".join(f"- E{i + 1}. {r['nombre']}: gana el {fmt_pct(r['pct_victorias'])} de sus partidos, "
                        f"{fmt_pct(r['pct_filas'], 0)} de las actuaciones" for i, r in enumerate(M.PERF["perfiles"]))
        lineas.append("ESTILOS DE JUEGO (K-means, k = 4, sobre 8 acciones por set: kills, aces, bloqueos, defensas, "
                      "ataques, errores de ataque, errores de saque y % de ataque; AVP; una fila = una pareja en un "
                      "partido; los estilos se solapan):\n" + est)
    if M.MET:
        rec, r, regla = M.MET["recomendado"], M.MET["modelos"][M.MET["recomendado"]]["prueba"], M.MET["regla"]
        imp = ", ".join(f"{x['etiqueta']}" for x in M.MET["importancias"][rec][:3])
        lineas.append(f"MODELO DE PREDICCIÓN: {rec}, con datos previos al partido (ventaja de ranking, edad, estatura, "
                      f"clasificatoria, circuito, género, etapa). Entrenado con 2000–2016 y probado con 2017–2019: "
                      f"acierta el {fmt_pct(100 * r['accuracy'], 0)} de los partidos (la regla 'gana el mejor rankeado' "
                      f"el {fmt_pct(100 * regla['accuracy'], 0)}), AUC {fmt(r['roc_auc'], 3)}. Lo que más pesa: {imp}. "
                      "Sus porcentajes están bien calibrados: cuando dice 70 %, ganan cerca del 70 %. Su valor es "
                      "decir qué tan favorita es una pareja, no solo quién.")
    lineas.append("PÁGINAS DEL DASHBOARD: Inicio (resumen); Dashboard (partidos por año, circuito, etapa y sorpresas, "
                  "con filtros); Análisis (univariado, bivariado, multivariado, estilos de juego, metodología, "
                  "conclusiones); Predicción (el modelo, simulador '¿Quién gana?' y calculadora histórica por "
                  "diferencia de ranking); Entrenar (diagnóstico de una pareja frente a las ganadoras de la AVP, "
                  "prioridades, plan semanal con ejercicios y redacción con IA); Equipo.")
    _FICHA = "\n\n".join(lineas)
    return _FICHA


SUGERENCIAS = ["¿Qué es lo que más separa a un ganador de un perdedor?",
               "¿Qué tan confiable es el modelo de predicción?",
               "Explíqueme los 4 estilos de juego",
               "¿Cómo uso la página Entrenar?"]

chat = html.Div([
    html.Button([ico("chat"), html.Span("Pregúntele al asistente", className="chat-boton-txt")], id="chat-boton",
                n_clicks=0, className="chat-boton", title="Asistente del coach"),
    html.Div([
        html.Div([html.Div([html.Img(src="/assets/balon.svg", className="chat-avatar", alt=""),
                            html.Div([html.Div("Asistente del coach", className="chat-titulo"),
                                      html.Div("Responde con los datos del proyecto · IA (Gemini)",
                                               className="chat-sub")])], className="chat-cabeza-izq"),
                  html.Button("×", id="chat-cerrar", n_clicks=0, className="chat-cerrar", title="Cerrar")],
                 className="chat-cabeza"),
        html.Div(id="chat-mensajes", className="chat-mensajes"),
        html.Div(id="chat-estado", className="chat-estado"),
        html.Div([dcc.Input(id="chat-input", placeholder="Escriba su pregunta…", maxLength=500, debounce=False,
                            className="chat-input", n_submit=0, autoComplete="off"),
                  html.Button(ico("enviar"), id="chat-enviar", n_clicks=0, className="chat-enviar", title="Enviar")],
                 className="chat-pie"),
    ], id="chat-panel", className="chat-panel oculto"),
    dcc.Store(id="chat-historial", storage_type="session", data=[]),
], className="chat")
app.layout.children.append(chat)

# ---------- Video del proyecto ----------
ventana_video = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Video del proyecto"), close_button=True),
    dbc.ModalBody(html.Video(src="/assets/video/video_proyecto.mp4", poster="/assets/video/portada.jpg",
                             controls=True, preload="none", id="video-reproductor", className="video-reproductor")),
], id="video-ventana", size="xl", centered=True, is_open=False, className="ventana-video")
app.layout.children.append(ventana_video)


@app.callback(Output("video-ventana", "is_open"), Input("video-abrir", "n_clicks"), prevent_initial_call=True)
def abrir_video(n):
    return bool(n)

          # el asistente flota sobre todas las páginas

# Al llegar un mensaje nuevo, la conversación baja sola hasta el final
app.clientside_callback(
    """function(hijos) {
        setTimeout(function () { var c = document.getElementById("chat-mensajes");
                                 if (c) { c.scrollTop = c.scrollHeight; } }, 60);
        return window.dash_clientside.no_update;
    }""", Output("chat-mensajes", "title"), Input("chat-mensajes", "children"))


@app.callback(Output("chat-panel", "className"), Input("chat-boton", "n_clicks"), Input("chat-cerrar", "n_clicks"),
              State("chat-panel", "className"), prevent_initial_call=True)
def abrir_chat(_a, _c, clase):
    if ctx.triggered_id == "chat-cerrar":
        return "chat-panel oculto"
    return "chat-panel" if "oculto" in (clase or "") else "chat-panel oculto"


@app.callback(Output("chat-historial", "data"), Output("chat-input", "value"), Output("chat-estado", "children"),
              Input("chat-enviar", "n_clicks"), Input("chat-input", "n_submit"),
              Input({"type": "chat-sug", "index": ALL}, "n_clicks"),
              State("chat-input", "value"), State("chat-historial", "data"), prevent_initial_call=True)
def preguntar(_e, _s, sug, texto, historial):
    historial = historial or []
    if isinstance(ctx.triggered_id, dict):            # clic en una pregunta sugerida
        if not any(sug):                              # (las sugerencias desaparecen al redibujar: no es un clic)
            return no_update, no_update, no_update
        texto = SUGERENCIAS[ctx.triggered_id["index"]]
    texto = (texto or "").strip()
    if not texto:
        return no_update, no_update, no_update
    try:
        resp, _ = IA.conversar(ficha_proyecto(), historial, texto)
    except RuntimeError as err:
        resp = f"_No pude responder ahora: {err}_"
    return historial + [{"rol": "user", "texto": texto}, {"rol": "model", "texto": resp}], "", ""


@app.callback(Output("chat-mensajes", "children"), Input("chat-historial", "data"))
def mostrar_chat(historial):
    if not historial:
        return [html.Div("Hola, Coach. Puedo explicarle los resultados del proyecto: qué pesa más para ganar, cómo "
                         "leer el modelo o los estilos de juego. Pruebe con:", className="chat-msg chat-bot"),
                html.Div([html.Button(t, id={"type": "chat-sug", "index": i}, n_clicks=0, className="chat-sug")
                          for i, t in enumerate(SUGERENCIAS)], className="chat-sugerencias")]
    return [html.Div(dcc.Markdown(h["texto"]) if h["rol"] == "model" else h["texto"],
                     className=f"chat-msg {'chat-bot' if h['rol'] == 'model' else 'chat-yo'}") for h in historial] + \
           [html.Div(id="chat-fin")]


# ==========================================================================
# Navegación entre páginas (las rutas antiguas llevan a su nuevo lugar)
# ==========================================================================
RUTAS = {"/": pagina_inicio, "/dashboard": pagina_dashboard, "/analisis": pagina_analisis,
         "/prediccion": pagina_prediccion, "/entrenamiento": pagina_entrenamiento, "/equipo": pagina_equipo,
         "/datos": pagina_dashboard,
         "/exploratorio": pagina_analisis,
         "/metodologia": lambda: pagina_analisis("metodologia"),
         "/conclusiones": lambda: pagina_analisis("conclusiones"),
         "/calculadora": lambda: pagina_prediccion("calculadora")}
CON_FILTROS = {"/dashboard", "/datos", "/analisis", "/exploratorio"}


@app.callback(Output("contenido", "children"), Output("filtros", "className"), Input("url", "pathname"))
def mostrar_pagina(ruta):
    clase = "filtros-banda" + ("" if ruta in CON_FILTROS else " oculto")
    return RUTAS.get(ruta, pagina_inicio)(), clase


if __name__ == "__main__":
    # Puerto opcional: python app.py 8051 (por si el 8050 está ocupado)
    puerto = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    print(f"\nDashboard listo. Ábralo en el navegador: http://127.0.0.1:{puerto}\n(Para detenerlo: Ctrl + C)\n")
    app.run(debug=False, port=puerto)

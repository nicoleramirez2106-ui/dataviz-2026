"""Colores, formato de números y estilo de los gráficos (los mismos del libro)."""
import plotly.graph_objects as go
import plotly.io as pio

# Paleta con significado: cada familia de color representa UNA sola dimensión, en todo el dashboard.
# Si dos cosas comparten color, es porque significan lo mismo.
#   Resultado  -> MAR (gana / a favor) y CORAL (pierde / sorpresa / en contra)
#   Circuito   -> TURQUESA (FIVB) y ARENA (AVP)
#   Género     -> morados (solo se usan para hombres y mujeres)
#   Cantidad u orden -> PIZARRA, de claro a oscuro = menos a más (etapa, poder, complejidad, conteos)
#   Dato sin categoría -> NEUTRO (un solo gris azulado); lo elegido por el usuario -> TINTA (casi negro)
MAR = "#0B3B5C"        # gana, ganador, a favor (y color de la marca en botones y menú)
CORAL = "#E4572E"      # pierde, perdedor, sorpresa, brecha
TURQUESA = "#1A8A9A"   # FIVB
ARENA = "#D9A441"      # AVP
PIZARRA = ["#DCE3EA", "#B4C1CF", "#8798AB", "#5B6F86", "#33475E"]   # menos -> más
NEUTRO = "#5B6F86"     # una sola serie sin categoría (histogramas, conteos)
ELEGIDO = "#111827"    # lo que el usuario eligió o el valor elegido (k, umbral...)
GRIS = "#64748B"       # líneas de referencia (mediana, 50 %)

PAL = {"FIVB": TURQUESA, "AVP": ARENA,
       "Ganador": MAR, "Perdedor": CORAL,
       "M": "#5E548E", "W": "#A98BD1"}
ETAPAS = ["Clasificatoria", "Fase de grupos", "Cuadro principal", "Semifinales", "Finales"]
# Etapas: más oscuro = más avanzado en el torneo
PAL_ETAPA = dict(zip(ETAPAS, PIZARRA))
# Poder discriminante: más oscuro = separa más a ganadores de perdedores
PAL_PODER = {"Alto": PIZARRA[4], "Moderado": PIZARRA[2], "Bajo": PIZARRA[0]}
# Modelos: más oscuro = más complejo; la regla de referencia en gris
PAL_MODELO = {"Regresión logística": "#AEBFD0", "Random Forest": "#6B84A0",
              "XGBoost": "#2C3E52", "Regla del ranking": "#9CA3AF"}
# Perfiles de juego, ordenados por % de victorias: de mar (más gana) a coral (más pierde)
PAL_PERFIL = [MAR, "#5B8DB0", "#EFA58C", CORAL]
# Correlación entre variables: verde (negativa) – blanco – pizarra (positiva); colores exclusivos
ESCALA_CORR = [[0, "#6E9A74"], [0.5, "#F7F7F7"], [1, "#33475E"]]
# Escala de una sola dirección (por ejemplo, matriz de confusión): más oscuro = más casos
ESCALA_SEC = [[0, "#F1F4F7"], [1, "#33475E"]]
GENERO = {"M": "Hombres", "W": "Mujeres"}

TINTA = "#1F2937"
TINTA_SUAVE = "#6B7280"
REJILLA = "#ECEEF1"
FUENTE = "Fuente: vb_matches.csv (BigTimeStats vía TidyTuesday). Elaboración propia."
LETRA = "'Plus Jakarta Sans', 'Segoe UI', Roboto, Arial, sans-serif"


def fmt(x, d=2):
    """Número con coma decimal y punto de miles: fmt(76749, 0) -> '76.749'."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x, d=1):
    return f"{fmt(x, d)} %"


def fmt_p(p):
    return "< 0,001" if p < 0.001 else fmt(p, 3)


# Plantilla de plotly para todos los gráficos
pio.templates["vb"] = go.layout.Template(layout=dict(
    font=dict(family=LETRA, size=13, color=TINTA),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", separators=",.",
    xaxis=dict(gridcolor=REJILLA, zeroline=False, linecolor="#D1D5DB",
               ticks="outside", tickcolor="#D1D5DB", automargin=True),
    yaxis=dict(gridcolor=REJILLA, zeroline=False, automargin=True),
    legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                title=None),
    margin=dict(l=10, r=20, t=40, b=10),
    hoverlabel=dict(bgcolor="white", bordercolor="#D1D5DB",
                    font=dict(family=LETRA, color=TINTA)),
    colorway=[NEUTRO, PIZARRA[4], PIZARRA[2], GRIS]))
pio.templates.default = "vb"


def vacia(mensaje="No hay datos para estos filtros"):
    """Figura en blanco con un mensaje, para cuando un filtro deja 0 filas."""
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(size=14, color=TINTA_SUAVE))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=320)
    return fig

# Dashboard Shiny: Voleibol de playa, ¿qué hace ganar a una pareja?
# Réplica en R del dashboard de Dash. Lee datos_app.rds (creado con preparar_datos.R).

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(plotly)
  library(dplyr)
})

# ---------------------------------------------------------------- datos
D <- readRDS("datos_app.rds")
P <- D$p
P$circuito <- as.character(P$circuito)
P$genero   <- as.character(P$genero)
P$etapa    <- as.character(P$etapa)
P$upset <- ifelse(!is.na(P$w_rank) & !is.na(P$l_rank) & P$w_rank != P$l_rank,
                  P$w_rank > P$l_rank, NA)
P$dist <- abs(P$w_rank - P$l_rank)
ANIOS <- range(P$year, na.rm = TRUE)
ET <- unique(P$etapa)
ET <- c(intersect(c("Cuadro eliminatorio", "Clasificatoria", "Fase de grupos", "Sin dato"), ET),
        setdiff(ET, c("Cuadro eliminatorio", "Clasificatoria", "Fase de grupos", "Sin dato")))
RES <- D$res
COMP <- RES$comparacion

LIBRO <- "https://nicoleramirez2106-ui.github.io/dataviz-2026/EDA_Beach_Volley_R/index.html"
REPO  <- "https://github.com/nicoleramirez2106-ui/dataviz-2026"

# Paleta fija: un color = una sola cosa
MAR <- "#0B3B5C"; CORAL <- "#E4572E"; TURQ <- "#1A8A9A"; ARENA <- "#D9A441"
COL_CIRC <- c(AVP = ARENA, FIVB = TURQ)
COL_GEN  <- c(Hombres = MAR, Mujeres = CORAL)

# ---------------------------------------------------------------- ayudas
nz <- function(x, d) if (is.null(x) || length(x) == 0) d else x
fmt <- function(x, d = 0) {
  ifelse(is.na(x), "—", formatC(x, format = "f", digits = d, big.mark = ".", decimal.mark = ","))
}
pct <- function(x, d = 1) ifelse(is.na(x), "—", paste0(fmt(x, d), " %"))

kpi <- function(etq, valor, nota = NULL, tono = "tono-mar") {
  div(class = paste("kpi", tono),
      div(class = "kpi-etiqueta", etq),
      div(class = "kpi-valor", valor),
      if (!is.null(nota)) div(class = "kpi-nota", nota))
}

tarjeta <- function(titulo, ..., sub = NULL) {
  div(class = "card tarjeta mb-4",
      div(class = "card-header",
          h6(class = "tarjeta-titulo", titulo),
          if (!is.null(sub)) div(class = "tarjeta-subtitulo", sub)),
      div(class = "card-body", ...))
}

encabezado <- function(titulo, texto, saque = NULL) {
  div(class = "mb-4",
      if (!is.null(saque)) div(class = "pagina-saque", saque),
      h2(class = "pagina-titulo", titulo),
      p(class = "pagina-intro", texto))
}

entrenador <- function(texto) {
  div(class = "entrenador mb-4", strong("Para el entrenador. "), texto)
}

chips <- function(id, opciones, activo = opciones[1]) {
  div(class = "chips", `data-chip` = id,
      lapply(opciones, function(o) {
        tags$span(class = if (o == activo) "chip chip-activo" else "chip", `data-val` = o, o)
      }))
}

lay <- function(fig, ...) {
  fig %>%
    layout(font = list(family = "Barlow, Segoe UI, sans-serif", color = "#1F2937"),
           paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)",
           legend = list(orientation = "h", y = -0.22), ...) %>%
    config(displayModeBar = FALSE)
}

grafico <- function(id, alto = 340) plotlyOutput(id, height = paste0(alto, "px"))

vacio <- function(txt = "No hay partidos con estos filtros.") {
  validate(need(FALSE, txt))
}

# Cifras fijas (todos los datos)
Pr <- P[!is.na(P$upset), ]
N_RANK <- nrow(Pr)
PCT_FAV <- 100 * mean(!Pr$upset)
FAV_CERCA <- 100 * mean(!Pr$upset[Pr$dist <= 2])
FAV_LEJOS <- 100 * mean(!Pr$upset[Pr$dist > 20])
N_CIRC <- table(factor(P$circuito, levels = c("FIVB", "AVP")))
PCT_STATS <- 100 * mean(P$tiene_stats, na.rm = TRUE)

ETQ_AVP <- c(kills = "los ataques ganadores", errores_ataque = "los errores de ataque", aces = "los aces",
             errores_saque = "los errores de saque", bloqueos = "los bloqueos", defensas = "las defensas")
AV <- RES$avp_tab
TOP <- AV[which.max(AV$or_1de), ]
TOP_NOM <- if (TOP$variable %in% names(ETQ_AVP)) ETQ_AVP[[TOP$variable]] else TOP$variable

fila_modelo <- function(nombre) COMP[COMP$modelo == nombre, ]
AUC_REGLA <- fila_modelo("Regla del ranking")$auc_test
MEJOR <- RES$nombres[[RES$mejor]]
AUC_MEJOR <- fila_modelo(MEJOR)$auc_test

# ---------------------------------------------------------------- UI: riel, marcador, filtros
PAGINAS <- list(c("inicio", "Inicio", "balon"), c("dashboard", "Dashboard", "tablero"),
                c("analisis", "Análisis", "pizarra"), c("prediccion", "Predicción", "saque"),
                c("equipo", "Equipo", "equipo"))

riel <- tags$nav(class = "riel", `aria-label` = "Navegación principal",
  tags$a(class = "riel-marca", href = "#", `data-ir` = "inicio", title = "Inicio",
         img(src = "balon.svg", class = "balon-logo", alt = ""),
         span(class = "riel-sigla", "VP")),
  div(class = "riel-menu nav flex-column",
      lapply(PAGINAS, function(x) {
        tags$a(class = paste("riel-item nav-link", if (x[1] == "inicio") "active" else ""),
               href = "#", `data-ir` = x[1],
               span(class = paste0("ico ico-", x[3])), span(x[2]))
      })),
  div(class = "riel-enlaces",
      tags$a(href = LIBRO, target = "_blank", title = "Leer el libro", "Libro"),
      tags$a(href = REPO, target = "_blank", title = "Repositorio", "Repo"))
)

marcador <- div(class = "marcador",
  div(class = "marcador-marca",
      div(class = "marcador-titulo", "Voleibol de playa"),
      div(class = "marcador-sub", "¿Qué hace ganar a una pareja?")),
  uiOutput("tablero", class = "tablero")
)

filtros <- div(id = "filtros", class = "filtros-banda oculto",
  div(class = "filtro-grupo", span(class = "filtro-etq", "Circuito"),
      chips("f_circuito", c("Todos", "AVP", "FIVB"))),
  div(class = "filtro-grupo", span(class = "filtro-etq", "Género"),
      chips("f_genero", c("Todos", "Hombres", "Mujeres"))),
  div(class = "filtro-grupo", span(class = "filtro-etq", "Etapa"),
      chips("f_etapa", c("Todas", ET))),
  div(class = "filtro-grupo filtro-anios", span(class = "filtro-etq", "Años"),
      sliderInput("f_anios", NULL, min = ANIOS[1], max = ANIOS[2], value = ANIOS, step = 1,
                  sep = "", ticks = FALSE, width = "100%")),
  actionButton("f_reset", "Restablecer", class = "filtro-reset")
)

# ---------------------------------------------------------------- página: Inicio
pag_inicio <- function() {
  contador <- function(valor, dec = 0, sufijo = "") {
    span(class = "contador", `data-valor` = as.character(valor), `data-dec` = as.character(dec),
         `data-sufijo` = sufijo, paste0(fmt(valor, dec), sufijo))
  }
  hallazgos <- list(
    list("Ranking", pct(PCT_FAV, 1),
         paste0("de los partidos los gana la pareja mejor rankeada; el ", fmt(100 - PCT_FAV, 0), " % restante son sorpresas.")),
    list("Distancia con el rival", paste0(fmt(FAV_CERCA, 0), " % → ", fmt(FAV_LEJOS, 0), " %"),
         "gana el favorito con 1–2 posiciones de diferencia frente a más de 20."),
    list("Juego (AVP)", paste0(fmt(TOP$or_1de, 2), "×"),
         paste0("es cuánto aumenta la razón de ganar por cada desviación estándar en ", TOP_NOM,
                ": la acción que más separa al ganador del perdedor."))
  )
  jugadas <- list(
    list("dashboard", "tablero", "Dashboard",
         "Partidos por año, circuito y etapa, sorpresas y cobertura de datos con filtros rápidos."),
    list("analisis", "pizarra", "Análisis",
         "Univariado, bivariado, multivariado, estilos de juego de la AVP, metodología y conclusiones."),
    list("prediccion", "saque", "Predicción",
         "Modelos previos al partido, simulador ¿Quién gana? y la calculadora histórica.")
  )
  cinta <- c(paste(fmt(nrow(P)), "partidos"),
             paste(fmt(N_RANK), "partidos con ranking de las dos parejas"),
             paste(fmt(PCT_STATS, 0), "% con estadísticas de juego"), "FIVB y AVP",
             paste0(ANIOS[1], "–", ANIOS[2]))
  piezas <- lapply(cinta, function(t) {
    span(class = "cinta-item", span(t), span(class = "ico ico-balon cinta-ico"))
  })

  div(
    tags$section(class = "portada",
      img(src = "cancha.svg", class = "portada-escena", alt = ""),
      div(class = "portada-texto-caja",
          div(class = "portada-curso", "Visualización de Datos y Toma de Decisiones, Universidad del Norte, 2026"),
          h1(class = "portada-titulo", "¿Qué hace ganar a una pareja de voleibol de playa?"),
          p(class = "portada-texto",
            paste0(fmt(nrow(P)), " partidos de los circuitos FIVB y AVP entre ", ANIOS[1], " y ", ANIOS[2],
                   ", analizados para que un entrenador sepa qué tan sólida es una ventaja antes del ",
                   "partido, cómo armar parejas y qué priorizar en el entrenamiento.")),
          div(class = "hero-botones",
              tags$a(class = "boton-principal", href = "#", `data-ir` = "prediccion", "Simular un partido"),
              tags$a(class = "boton-secundario", href = LIBRO, target = "_blank", "Leer el libro")),
          div(class = "hero-autores", "Nicolle Ramírez y Roger Daza"))
    ),
    div(class = "portada-marcador",
        div(class = "portada-dato", div(class = "portada-dato-valor dato-mar", contador(round(PCT_FAV, 1), 1, " %")),
            div(class = "portada-dato-etq", "Gana el favorito")),
        div(class = "portada-dato", div(class = "portada-dato-valor dato-coral", contador(round(100 - PCT_FAV, 1), 1, " %")),
            div(class = "portada-dato-etq", "Sorpresas")),
        div(class = "portada-dato", div(class = "portada-dato-valor dato-turquesa", contador(as.integer(N_CIRC[["FIVB"]]))),
            div(class = "portada-dato-etq", "Partidos FIVB")),
        div(class = "portada-dato", div(class = "portada-dato-valor dato-arena", contador(as.integer(N_CIRC[["AVP"]]))),
            div(class = "portada-dato-etq", "Partidos AVP"))),
    div(class = "nota mb-3",
        paste0("Favorito y sorpresa se calculan sobre los ", fmt(N_RANK),
               " partidos con ranking de las dos parejas.")),
    div(class = "cinta mb-4", div(class = "cinta-pista", piezas, piezas)),
    h5(class = "seccion-titulo", "Elija su jugada"),
    div(class = "jugadas mb-4",
        lapply(jugadas, function(j) {
          tags$a(class = "jugada", href = "#", `data-ir` = j[[1]],
                 span(class = paste0("ico ico-", j[[2]], " jugada-icono")),
                 h6(j[[3]]), p(j[[4]]), span(class = "jugada-ir", paste("Ir a", j[[3]])))
        }),
        div(class = "jugada jugada-video",
            span(class = "ico ico-video jugada-icono"), h6("Video del proyecto"),
            div(class = "proximamente", "Próximamente"))),
    h5(class = "seccion-titulo", "Tres hallazgos para empezar"),
    div(class = "sets mb-4",
        lapply(seq_along(hallazgos), function(i) {
          h <- hallazgos[[i]]
          div(class = "hallazgo", div(class = "set-num", paste("Set", i)),
              div(class = "hallazgo-etq", h[[1]]), div(class = "hallazgo-valor", h[[2]]),
              div(class = "hallazgo-texto", h[[3]]))
        })),
    fluidRow(
      column(7, tarjeta("Pregunta de investigación",
        p(class = "pregunta", strong("¿Qué factores, medibles antes del partido, se asocian con que una pareja gane, y cómo puede usarlos un entrenador para preparar partidos y armar parejas?")),
        tags$ol(
          tags$li(strong("Preparar un partido: "), "¿qué tan probable es ganar según la diferencia de ranking con el rival?"),
          tags$li(strong("Armar parejas: "), "¿la estatura, la edad o la diferencia entre compañeros se asocian con ganar?"),
          tags$li(strong("Priorizar el entrenamiento: "), "¿qué acciones de juego separan al ganador del perdedor?"),
          tags$li(strong("Anticipar el resultado: "), "con lo que se sabe antes del partido, ¿se puede predecir quién gana mejor que con la regla \"gana el mejor rankeado\"?")))),
      column(5, tarjeta("Objetivos",
        tags$ol(
          tags$li("Depurar y documentar el conjunto de datos."),
          tags$li("Describir la distribución de las variables del partido y de los jugadores."),
          tags$li("Comparar a las parejas ganadoras con las perdedoras."),
          tags$li("Cuantificar cuándo el ranking anticipa el resultado y cuándo no."),
          tags$li("Traducir los hallazgos en recomendaciones para el entrenador."))))
    )
  )
}

# ---------------------------------------------------------------- página: Dashboard
pag_dashboard <- function() {
  div(
    encabezado("Dashboard",
               "Panorama de los partidos con los filtros de arriba: un clic en circuito, género, etapa o años actualiza el marcador y todos los gráficos.",
               "Datos y limpieza"),
    uiOutput("d_kpis", class = "kpis mb-4"),
    fluidRow(
      column(8, tarjeta("Partidos por año y circuito", grafico("d_anio", 340),
                        sub = "La caída de la AVP en 2011–2012 coincide con su crisis financiera · haga clic en un año para ver su detalle")),
      column(4, tarjeta("Detalle del año", uiOutput("d_detalle")))),
    fluidRow(
      column(6, tarjeta("Sorpresas por año", grafico("d_sorp", 320),
                        sub = "% de partidos que gana la pareja peor rankeada (años con 30 o más partidos)")),
      column(6, tarjeta("Cobertura de estadísticas de juego por año", grafico("d_cob", 320),
                        sub = "% de partidos con estadísticas de ataque, saque, bloqueo y defensa"))),
    fluidRow(
      column(6, tarjeta("Partidos por circuito y género", grafico("d_circ", 320))),
      column(6, tarjeta("Partidos por etapa del torneo", grafico("d_etapa", 320)))),
    tarjeta("Problemas de calidad del archivo",
            div(class = "kpis mb-3",
                kpi("Partidos analizados", fmt(RES$n_total), "después de limpiar")),
            tableOutput("t_cal"))
  )
}

# ---------------------------------------------------------------- página: Análisis
VARS_UNI <- list("Edad (años)" = c("w_age", "l_age"), "Estatura (cm)" = c("w_hgt", "l_hgt"),
                 "Ranking" = c("w_rank", "l_rank"), "Duración (min)" = "duration_min",
                 "Sets por partido" = "n_sets")
VARS_BI <- VARS_UNI[1:3]

tab_univ <- function() {
  div(
    div(class = "mb-3", style = "max-width:380px;",
        tags$label(class = "control-etq", "Variable"),
        selectInput("u_var", NULL, choices = names(VARS_UNI), width = "100%")),
    uiOutput("u_kpis", class = "kpis mb-3"),
    tarjeta("Distribución", grafico("u_fig", 380),
            sub = "Para edad, estatura y ranking se juntan las dos parejas de cada partido")
  )
}

tab_biv <- function() {
  div(
    div(class = "mb-3", style = "max-width:380px;",
        tags$label(class = "control-etq", "Variable"),
        selectInput("b_var", NULL, choices = names(VARS_BI), width = "100%")),
    uiOutput("b_kpis", class = "kpis mb-3"),
    tarjeta("Ganador contra perdedor", grafico("b_caja", 400),
            sub = "En el ranking, un número menor es una pareja mejor clasificada")
  )
}

tab_multi <- function() {
  div(
    tarjeta("Correlación entre variables del partido", grafico("m_cor", 460),
            sub = "Correlación de Pearson con los datos disponibles de cada par de variables")
  )
}

tab_perfiles <- function() {
  div(
    div(class = "nota mb-3",
        strong("Los filtros de arriba no aplican a esta pestaña. "),
        paste0("K-means con 4 grupos sobre estadísticas por set de la AVP (", fmt(D$est$n),
               " filas, una por pareja y partido). Silueta media: ", fmt(D$est$sil, 2), ".")),
    div(class = "alert aviso-solape mb-4",
        "Los grupos se solapan bastante (silueta baja): son estilos de juego dominantes, no categorías cerradas."),
    uiOutput("p_tarjetas", class = "perfiles mb-4"),
    fluidRow(
      column(6, tarjeta("Perfil de cada estilo", grafico("p_radar", 420),
                        sub = "Promedio de cada grupo, estandarizado")),
      column(6, tarjeta("Tabla de grupos", tableOutput("p_tabla"),
                        sub = "Promedios por set de cada pareja")))
  )
}

tab_metodo <- function() {
  tarjeta("Metodología",
    tags$ol(
      tags$li(strong("Datos. "), paste0(fmt(RES$n_total), " partidos FIVB y AVP entre ", ANIOS[1], " y ", ANIOS[2], ".")),
      tags$li(strong("Limpieza. "), "Se corrigen fechas de nacimiento, se separa el ranking del texto, se eliminan filas inválidas y se marcan forfeits y retiros."),
      tags$li(strong("Exploración. "), "Análisis univariado, bivariado y multivariado; comparación de ganadores y perdedores con pruebas no paramétricas."),
      tags$li(strong("Estilos de juego. "), "K-means con 4 grupos sobre estadísticas por set de la AVP."),
      tags$li(strong("Modelos. "), paste0("Regla del ranking, regresión logística, Random Forest y XGBoost. División temporal: entrenamiento hasta 2016 (", fmt(RES$n_train), " partidos) y prueba 2017–2019 (", fmt(RES$n_test), " partidos); validación cruzada estratificada de 5 grupos optimizando el AUC.")),
      tags$li(strong("Orientación de la pareja A. "), "La pareja A es la ganadora cuando el identificador del partido es par; así la variable respuesta queda balanceada.")),
    div(class = "nota", "Detalle completo en el libro."))
}

tab_concl <- function() {
  hall <- list(
    c("El ranking anticipa, pero no decide.", paste0("Gana el mejor rankeado en ", pct(PCT_FAV, 1), " de los partidos; con 1–2 posiciones de diferencia baja a ", pct(FAV_CERCA, 0), " y con más de 20 sube a ", pct(FAV_LEJOS, 0), ".")),
    c("Modelos frente a la regla del ranking.", paste0("El mejor modelo (", MEJOR, ") logra un AUC de ", fmt(AUC_MEJOR, 3), " en prueba, frente a ", fmt(AUC_REGLA, 3), " de la regla del ranking.")),
    c("En la AVP importa la eficacia.", paste0("Entre las acciones de juego, ", TOP_NOM, " es lo que más separa al ganador del perdedor.")),
    c("La cobertura de datos es desigual.", paste0("Solo ", pct(PCT_STATS, 0), " de los partidos tienen estadísticas de juego."))
  )
  recs <- list(
    c("Preparar el partido", "Usar la diferencia de ranking como punto de partida, sabiendo que una ventaja pequeña es frágil."),
    c("Armar parejas", "Revisar el efecto de la edad, la estatura y venir de clasificatoria en la pestaña Predicción antes de decidir."),
    c("Entrenar", paste0("Entre las acciones de juego de la AVP, ", TOP_NOM, " es lo que más se asocia con ganar.")),
    c("Decidir con cautela", "Los modelos son estimaciones a partir de asociaciones históricas, no garantías.")
  )
  div(
    div(class = "respuesta-caja mb-4",
        div(class = "pagina-saque", "Respuesta a la pregunta"),
        p(class = "respuesta",
          "El ranking es una señal previa al partido, pero deja espacio a la sorpresa; ",
          paste0("al menos en la AVP, entre las acciones de juego, ", TOP_NOM, " es lo que más separa al ganador."))),
    tarjeta("Hallazgos",
            lapply(seq_along(hall), function(i) {
              div(class = "hallazgo-item", div(class = "num", i),
                  div(strong(hall[[i]][1]), div(class = "texto", hall[[i]][2])))
            })),
    h5(class = "seccion-titulo", "Recomendaciones para el entrenador"),
    fluidRow(lapply(recs, function(r) {
      column(3, div(class = "rec", h6(r[1]), p(r[2])))
    }))
  )
}

pag_analisis <- function() {
  div(
    encabezado("Análisis", "Explore las variables del partido, compare ganadores y perdedores, conozca los estilos de juego de la AVP y lea las conclusiones.", "Exploración"),
    tabsetPanel(id = "a_tab",
      tabPanel("Univariado", tab_univ()),
      tabPanel("Bivariado", tab_biv()),
      tabPanel("Multivariado", tab_multi()),
      tabPanel("Estilos de juego", tab_perfiles()),
      tabPanel("Metodología", tab_metodo()),
      tabPanel("Conclusiones", tab_concl()))
  )
}

# ---------------------------------------------------------------- página: Predicción
tab_modelos <- function() {
  div(
    div(class = "recomendado mb-4",
        div(class = "recomendado-etq", "Modelo recomendado"),
        div(class = "recomendado-nombre", MEJOR),
        p(class = "recomendado-razon",
          paste0("Elegido por el mayor AUC en validación cruzada. Entrenado con partidos de ", ANIOS[1],
                 " a 2016 y evaluado con los de 2017 a 2019 (", fmt(RES$n_test), " partidos)."))),
    div(class = "filtro-grupo mb-3", span(class = "filtro-etq", "Modelo"),
        chips("p_modelo", unname(unlist(RES$nombres)), MEJOR)),
    uiOutput("p_kpis", class = "kpis mb-4"),
    fluidRow(
      column(6, tarjeta("Curva ROC", grafico("p_roc", 360),
                        sub = "Todos los modelos con los partidos de prueba")),
      column(6, tarjeta(paste("Matriz de confusión:", MEJOR), grafico("p_cm", 360),
                        sub = "Partidos de prueba, umbral 0,5"))),
    fluidRow(
      column(6, tarjeta("Importancia por permutación", grafico("p_imp", 340),
                        sub = "Cuánto cae el AUC al desordenar cada variable")),
      column(6, tarjeta("Comparación de modelos", grafico("p_comp", 340)))),
    tarjeta("Efecto de cada variable (regresión logística)", grafico("p_or", 320),
            sub = "Razón de odds con IC 95 %; variables numéricas por cada desviación estándar. Mayor que 1 favorece a la pareja A."),
    tarjeta("Umbral de decisión",
            sliderInput("p_umbral", "Umbral de probabilidad", min = 0.05, max = 0.95, value = 0.5,
                        step = 0.01, width = "100%"),
            uiOutput("p_umbral_kpis", class = "kpis mb-2")),
    tarjeta("Tabla de resultados", tableOutput("p_tabla_comp"))
  )
}

lado <- function(id, titulo, rank, edad, est) {
  div(class = paste0("lado lado-", id),
      div(class = "lado-titulo", titulo),
      numericInput(paste0("s_rank_", id), tags$span(class = "control-etq", "Ranking de la pareja"),
                   rank, min = 1, max = 300, step = 1),
      numericInput(paste0("s_edad_", id), tags$span(class = "control-etq", "Edad media de la pareja (años)"),
                   edad, min = 16, max = 50, step = 0.5),
      numericInput(paste0("s_est_", id), tags$span(class = "control-etq", "Estatura media de la pareja (cm)"),
                   est, min = 150, max = 215, step = 1),
      checkboxInput(paste0("s_clasif_", id), "Viene de clasificatoria", FALSE))
}

tab_simulador <- function() {
  div(
    div(class = "nota mb-3",
        "Escriba los datos de las dos parejas. La probabilidad se calcula con la regresión logística y se promedia desde los dos lados, así que la de su pareja y la del rival suman 100 %."),
    div(class = "simulador-cancha mb-3",
        lado("a", "Su pareja", 8, 28, 190),
        div(class = "red red-chica", `aria-hidden` = "true"),
        lado("b", "Rival", 12, 28, 188)),
    fluidRow(
      column(4, span(class = "filtro-etq", "Circuito"), chips("s_circ", c("AVP", "FIVB"))),
      column(4, span(class = "filtro-etq", "Género"), chips("s_gen", c("Hombres", "Mujeres"))),
      column(4, span(class = "filtro-etq", "Etapa"),
             chips("s_etapa", c("Cuadro eliminatorio", "Clasificatoria", "Fase de grupos")))),
    div(class = "mt-3"),
    uiOutput("s_res", class = "resultado mb-4")
  )
}

tab_calc <- function() {
  div(
    div(class = "alert aviso-solape mb-4",
        strong("Calculadora del entrenador. "),
        "Cuenta, en los partidos históricos con una ventaja de ranking parecida, cuántas veces ganó la pareja mejor rankeada. Son frecuencias observadas, sin ajustar por otras variables."),
    fluidRow(
      column(5,
        fluidRow(
          column(6, numericInput("c_propio", tags$span(class = "control-etq", "Ranking de su pareja"), 8, min = 1, max = 64, step = 1)),
          column(6, numericInput("c_rival", tags$span(class = "control-etq", "Ranking del rival"), 12, min = 1, max = 64, step = 1))),
        tags$label(class = "control-etq", "Circuito"), chips("c_circ", c("Todos", "AVP", "FIVB")),
        tags$label(class = "control-etq mt-3", "Género"), chips("c_gen", c("Todos", "Hombres", "Mujeres")),
        tags$label(class = "control-etq mt-3", "Etapa del torneo"), chips("c_etapa", c("Todas", ET))),
      column(7,
        uiOutput("c_res", class = "resultado mb-4"),
        tarjeta("Victoria del favorito según la distancia de ranking", grafico("c_curva", 300)))),
    div(class = "nota mb-4",
        paste0("Son frecuencias históricas de ", ANIOS[1], " a ", ANIOS[2], ", no una predicción: describen qué pasó en partidos parecidos, no garantizan el resultado de uno nuevo."))
  )
}

pag_pred <- function() {
  div(
    encabezado("Predicción", "Qué tan bien se puede anticipar el resultado con lo que se sabe antes del partido.", "Modelos"),
    tabsetPanel(id = "p_tab",
      tabPanel("Modelos", tab_modelos()),
      tabPanel("¿Quién gana?", tab_simulador()),
      tabPanel("Calculadora histórica", tab_calc()))
  )
}

# ---------------------------------------------------------------- UI final
ui <- bootstrapPage(
  theme = bs_theme(version = 5),
  title = "Voleibol de playa · Dashboard",
  tags$head(
    tags$link(rel = "preconnect", href = "https://fonts.googleapis.com"),
    tags$link(rel = "stylesheet",
              href = "https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Barlow+Condensed:ital,wght@0,600;0,700;0,800;1,700;1,800&display=swap"),
    tags$link(rel = "stylesheet", href = "estilo.css"),
    tags$link(rel = "stylesheet", href = "extra.css"),
    tags$link(rel = "stylesheet", href = "shiny.css"),
    tags$link(rel = "icon", href = "balon.svg")
  ),
  riel,
  div(class = "principal",
      marcador,
      filtros,
      tags$main(class = "contenido",
        tabsetPanel(id = "pag", type = "hidden",
          tabPanelBody("inicio", pag_inicio()),
          tabPanelBody("dashboard", pag_dashboard()),
          tabPanelBody("analisis", pag_analisis()),
          tabPanelBody("prediccion", pag_pred()),
          tabPanelBody("equipo", div(class = "pagina-vacia"))))),
  tags$script(src = "app.js")
)

# ---------------------------------------------------------------- servidor
server <- function(input, output, session) {

  observeEvent(input$ir, {
    updateTabsetPanel(session, "pag", selected = input$ir)
  })

  dat <- reactive({
    d <- P
    circ <- nz(input$f_circuito, "Todos")
    gen <- nz(input$f_genero, "Todos")
    eta <- nz(input$f_etapa, "Todas")
    an <- nz(input$f_anios, ANIOS)
    if (circ != "Todos") d <- d[d$circuito == circ, ]
    if (gen != "Todos") d <- d[d$genero == gen, ]
    if (eta != "Todas") d <- d[d$etapa == eta, ]
    d[d$year >= an[1] & d$year <= an[2], ]
  })

  observeEvent(input$f_reset, {
    session$sendCustomMessage("chips_set", list(id = "f_circuito", value = "Todos"))
    session$sendCustomMessage("chips_set", list(id = "f_genero", value = "Todos"))
    session$sendCustomMessage("chips_set", list(id = "f_etapa", value = "Todas"))
    updateSliderInput(session, "f_anios", value = ANIOS)
  })

  # ---- marcador
  output$tablero <- renderUI({
    d <- dat()
    n <- table(factor(d$circuito, levels = c("FIVB", "AVP")))
    r <- d[!is.na(d$upset), ]
    fav <- if (nrow(r) > 0) pct(100 * mean(!r$upset), 1) else "—"
    an <- nz(input$f_anios, ANIOS)
    tagList(
      div(class = "tablero-score", title = "Partidos con los filtros actuales",
          span(class = "equipo equipo-fivb", "FIVB"),
          span(class = "digito", fmt(n[["FIVB"]])),
          span(class = "dos-puntos", ":"),
          span(class = "digito", fmt(n[["AVP"]])),
          span(class = "equipo equipo-avp", "AVP")),
      div(class = "tablero-dato", span(class = "tablero-etq", "Gana el favorito"),
          span(class = "digito digito-chico", fav)),
      div(class = "tablero-dato", span(class = "tablero-etq", "Periodo"),
          span(class = "digito digito-chico", paste0(an[1], "–", an[2]))))
  })

  # ---- Dashboard
  output$d_kpis <- renderUI({
    d <- dat()
    if (nrow(d) == 0) return(div(class = "nota", "No hay partidos con estos filtros."))
    r <- d[!is.na(d$upset), ]
    dur <- d$duration_min[!d$walkover & !is.na(d$duration_min) & d$duration_min > 0]
    tagList(
      kpi("Partidos", fmt(nrow(d)), "con los filtros actuales"),
      kpi("Gana el favorito", if (nrow(r) > 0) pct(100 * mean(!r$upset), 1) else "—",
          "pareja mejor rankeada", "tono-turquesa"),
      kpi("Sorpresas", if (nrow(r) > 0) pct(100 * mean(r$upset), 1) else "—",
          "gana la peor rankeada", "tono-coral"),
      kpi("Duración mediana", if (length(dur) > 0) paste(fmt(median(dur), 0), "min") else "—"),
      kpi("Con estadísticas", pct(100 * mean(d$tiene_stats, na.rm = TRUE), 0), "de ataque, saque y defensa",
          "tono-arena"))
  })

  output$d_anio <- renderPlotly({
    d <- dat(); if (nrow(d) == 0) vacio()
    a <- d %>% count(year, circuito)
    plot_ly(a, x = ~year, y = ~n, color = ~circuito, colors = COL_CIRC, type = "bar", source = "anio") %>%
      lay(barmode = "stack", xaxis = list(title = "", dtick = 2), yaxis = list(title = "Partidos")) %>%
      event_register("plotly_click")
  })

  output$d_detalle <- renderUI({
    d <- dat()
    if (nrow(d) == 0) return(div(class = "nota", "No hay partidos con estos filtros."))
    ev <- event_data("plotly_click", source = "anio")
    click <- !is.null(ev) && !is.null(ev$x)
    anio <- if (click) round(as.numeric(ev$x[1])) else max(d$year)
    y <- d[d$year == anio, ]
    if (nrow(y) == 0) return(div(class = "nota", paste("No hay partidos en", anio, "con estos filtros.")))
    r <- y[!is.na(y$upset), ]
    tagList(
      div(class = "detalle-anio", anio),
      div(class = "kpis kpis-2 mb-2",
          kpi("Partidos", fmt(nrow(y))),
          kpi("Sorpresas", if (nrow(r) >= 10) pct(100 * mean(r$upset), 1) else "—", "gana la peor rankeada", "tono-coral"),
          kpi("FIVB", fmt(sum(y$circuito == "FIVB")), NULL, "tono-turquesa"),
          kpi("AVP", fmt(sum(y$circuito == "AVP")), NULL, "tono-arena")),
      div(class = "nota", if (!click) "Mostrando el último año del filtro." else "Año elegido en el gráfico."))
  })

  output$d_sorp <- renderPlotly({
    d <- dat(); r <- d[!is.na(d$upset), ]
    if (nrow(r) == 0) vacio()
    a <- r %>% group_by(year, circuito) %>%
      summarise(n = n(), s = 100 * mean(upset), .groups = "drop") %>% filter(n >= 30)
    validate(need(nrow(a) > 0, "Sin años con 30 o más partidos con estos filtros."))
    plot_ly(a, x = ~year, y = ~s, color = ~circuito, colors = COL_CIRC, type = "scatter", mode = "lines+markers") %>%
      lay(xaxis = list(title = "", dtick = 2), yaxis = list(title = "% de sorpresas", rangemode = "tozero"))
  })

  output$d_cob <- renderPlotly({
    d <- dat(); if (nrow(d) == 0) vacio()
    a <- d %>% group_by(year) %>% summarise(s = 100 * mean(tiene_stats, na.rm = TRUE), .groups = "drop")
    plot_ly(a, x = ~year, y = ~s, type = "bar", marker = list(color = MAR)) %>%
      lay(xaxis = list(title = "", dtick = 2), yaxis = list(title = "% con estadísticas", range = c(0, 100)))
  })

  output$d_circ <- renderPlotly({
    d <- dat(); if (nrow(d) == 0) vacio()
    a <- d %>% count(circuito, genero)
    plot_ly(a, x = ~circuito, y = ~n, color = ~genero, colors = COL_GEN, type = "bar") %>%
      lay(barmode = "group", xaxis = list(title = ""), yaxis = list(title = "Partidos"))
  })

  output$d_etapa <- renderPlotly({
    d <- dat(); if (nrow(d) == 0) vacio()
    a <- d %>% count(etapa)
    plot_ly(a, y = ~etapa, x = ~n, type = "bar", orientation = "h", marker = list(color = TURQ)) %>%
      lay(yaxis = list(title = "", autorange = "reversed"), xaxis = list(title = "Partidos"))
  })

  output$t_cal <- renderTable({
    cal <- D$calidad
    data.frame(Problema = cal$Problema, Partidos = fmt(cal$Partidos), Porcentaje = pct(cal$Porcentaje, 1))
  }, striped = TRUE, hover = TRUE, spacing = "s")

  # ---- Análisis: univariado
  output$u_kpis <- renderUI({
    d <- dat(); v <- nz(input$u_var, names(VARS_UNI)[1])
    cols <- VARS_UNI[[v]]
    x <- unlist(d[cols], use.names = FALSE)
    if (v == "Duración (min)") x <- x[!d$walkover & x > 0]
    x <- x[!is.na(x)]
    if (length(x) == 0) return(div(class = "nota", "No hay datos con estos filtros."))
    tagList(kpi("Observaciones", fmt(length(x))), kpi("Media", fmt(mean(x), 1)),
            kpi("Mediana", fmt(median(x), 1)), kpi("Desviación estándar", fmt(sd(x), 1)))
  })

  output$u_fig <- renderPlotly({
    d <- dat(); v <- nz(input$u_var, names(VARS_UNI)[1])
    cols <- VARS_UNI[[v]]
    x <- unlist(d[cols], use.names = FALSE)
    if (v == "Duración (min)") x <- x[!d$walkover & x > 0]
    x <- x[!is.na(x)]
    validate(need(length(x) > 0, "No hay datos con estos filtros."))
    plot_ly(x = x, type = "histogram", nbinsx = 40, marker = list(color = MAR)) %>%
      lay(xaxis = list(title = v), yaxis = list(title = "Frecuencia"), bargap = 0.05)
  })

  # ---- Análisis: bivariado
  par_bi <- reactive({
    d <- dat(); v <- nz(input$b_var, names(VARS_BI)[1])
    cols <- VARS_BI[[v]]
    a <- d[[cols[1]]]; b <- d[[cols[2]]]
    ok <- !is.na(a) & !is.na(b)
    list(a = a[ok], b = b[ok], v = v)
  })

  output$b_kpis <- renderUI({
    z <- par_bi()
    if (length(z$a) < 30) return(div(class = "nota", "No hay suficientes partidos con estos filtros."))
    i <- if (length(z$a) > 20000) sample(length(z$a), 20000) else seq_along(z$a)
    pv <- tryCatch(suppressWarnings(wilcox.test(z$a[i], z$b[i], paired = TRUE, exact = FALSE)$p.value),
                   error = function(e) NA_real_)
    tagList(
      kpi("Partidos comparados", fmt(length(z$a))),
      kpi("Ganador (media)", fmt(mean(z$a), 1), NULL, "tono-mar"),
      kpi("Perdedor (media)", fmt(mean(z$b), 1), NULL, "tono-coral"),
      kpi("Prueba de Wilcoxon pareada", if (is.na(pv)) "—" else if (pv < 0.001) "p < 0,001" else paste("p =", fmt(pv, 3))))
  })

  output$b_caja <- renderPlotly({
    z <- par_bi()
    validate(need(length(z$a) >= 30, "No hay suficientes partidos con estos filtros."))
    df <- data.frame(valor = c(z$a, z$b), rol = rep(c("Ganador", "Perdedor"), each = length(z$a)))
    plot_ly(df, y = ~valor, color = ~rol, colors = c(Ganador = MAR, Perdedor = CORAL), type = "box") %>%
      lay(yaxis = list(title = z$v), xaxis = list(title = ""))
  })

  # ---- Análisis: multivariado
  output$m_cor <- renderPlotly({
    d <- dat()
    validate(need(nrow(d) > 30, "No hay suficientes partidos con estos filtros."))
    m <- d[, c("w_rank", "l_rank", "w_age", "l_age", "w_hgt", "l_hgt", "duration_min", "n_sets")]
    nombres <- c("Ranking ganador", "Ranking perdedor", "Edad ganador", "Edad perdedor",
                 "Estatura ganador", "Estatura perdedor", "Duración", "Sets")
    cm <- suppressWarnings(cor(m, use = "pairwise.complete.obs"))
    plot_ly(x = nombres, y = nombres, z = cm, type = "heatmap", zmin = -1, zmax = 1,
            colorscale = list(c(0, CORAL), c(0.5, "#FFFFFF"), c(1, MAR))) %>%
      lay(yaxis = list(autorange = "reversed"), xaxis = list(tickangle = -30),
          margin = list(l = 120, b = 110, t = 10, r = 10))
  })

  # ---- Análisis: estilos de juego
  EST <- D$est$tab
  COL_PERFIL <- c(MAR, CORAL, TURQ, ARENA)
  FEAT <- c("kills", "errors", "aces", "serve_errors", "blocks", "digs", "attacks", "pct_ataque")
  ETQ_FEAT <- c("Ataques ganadores", "Errores de ataque", "Aces", "Errores de saque", "Bloqueos",
                "Defensas", "Volumen de ataque", "% de ataque")

  output$p_tarjetas <- renderUI({
    tagList(lapply(seq_len(nrow(EST)), function(i) {
      nom <- sub("^Grupo [0-9]+: ", "", EST$perfil[i])
      div(class = "perfil-tarjeta", style = paste0("--perfil:", COL_PERFIL[(i - 1) %% 4 + 1], ";"),
          div(class = "perfil-num", paste("Perfil", i)),
          h6(class = "perfil-nombre", paste0(toupper(substr(nom, 1, 1)), substr(nom, 2, nchar(nom)))),
          div(class = "perfil-valor", pct(EST$victorias[i], 1)),
          div(class = "kpi-nota", "de victorias"),
          div(class = "kpi-nota", paste(fmt(EST$partidos[i]), "filas")))
    }))
  })

  output$p_radar <- renderPlotly({
    Z <- scale(as.matrix(EST[, FEAT]))
    Z[is.nan(Z)] <- 0
    fig <- plot_ly(type = "scatterpolar", mode = "lines")
    for (i in seq_len(nrow(EST))) {
      fig <- add_trace(fig, type = "scatterpolar", mode = "lines",
                       r = c(as.numeric(Z[i, ]), Z[i, 1]), theta = c(ETQ_FEAT, ETQ_FEAT[1]),
                       name = sub("^Grupo ", "Perfil ", EST$perfil[i]), fill = "toself", opacity = 0.55,
                       line = list(color = COL_PERFIL[(i - 1) %% 4 + 1]))
    }
    lay(fig, polar = list(radialaxis = list(visible = TRUE, showticklabels = FALSE)),
        margin = list(l = 40, r = 40, t = 20, b = 60))
  })

  output$p_tabla <- renderTable({
    t <- EST
    data.frame(Perfil = sub("^Grupo ", "", t$perfil), Filas = fmt(t$partidos), `Victorias (%)` = fmt(t$victorias, 1),
               Ataques = fmt(t$kills, 1), Errores = fmt(t$errors, 1), Aces = fmt(t$aces, 2),
               Bloqueos = fmt(t$blocks, 2), Defensas = fmt(t$digs, 1), `% ataque` = fmt(100 * t$pct_ataque, 1),
               check.names = FALSE)
  }, striped = TRUE, hover = TRUE, spacing = "s")

  # ---- Predicción: modelos
  fila_sel <- reactive(fila_modelo(nz(input$p_modelo, MEJOR)))

  output$p_kpis <- renderUI({
    f <- fila_sel()
    tagList(kpi("AUC en validación cruzada", fmt(f$auc_cv, 3), if (is.na(f$auc_cv)) "la regla no se entrena" else NULL),
            kpi("AUC en prueba", fmt(f$auc_test, 3), NULL, "tono-turquesa"),
            kpi("Exactitud", pct(100 * f$exactitud, 1), NULL, "tono-arena"),
            kpi("F1", fmt(f$f1, 3), NULL, "tono-coral"))
  })

  output$p_roc <- renderPlotly({
    r <- RES$roc
    sel <- nz(input$p_modelo, MEJOR)
    fig <- plot_ly()
    cols <- c(MAR, CORAL, TURQ, ARENA)
    for (i in seq_along(unique(r$modelo))) {
      m <- unique(r$modelo)[i]
      s <- r[r$modelo == m, ]
      fig <- add_trace(fig, x = s$fpr, y = s$tpr, type = "scatter", mode = "lines", name = m,
                       line = list(color = cols[(i - 1) %% 4 + 1], width = if (m == sel) 4 else 1.5),
                       opacity = if (m == sel) 1 else 0.6)
    }
    fig <- add_trace(fig, x = c(0, 1), y = c(0, 1), type = "scatter", mode = "lines", name = "Azar",
                     line = list(color = "#9CA3AF", dash = "dash"))
    lay(fig, xaxis = list(title = "Tasa de falsos positivos"), yaxis = list(title = "Tasa de verdaderos positivos"))
  })

  output$p_cm <- renderPlotly({
    cf <- RES$conf_mejor
    m <- xtabs(Freq ~ prediccion + real, cf)
    z <- unclass(m)
    ann <- list()
    k <- 0
    for (i in seq_len(nrow(z))) for (j in seq_len(ncol(z))) {
      k <- k + 1
      ann[[k]] <- list(x = j - 1, y = i - 1, text = fmt(z[i, j]), showarrow = FALSE,
                       font = list(size = 18, color = if (z[i, j] > max(z) / 2) "white" else "#1F2937"))
    }
    plot_ly(x = c("Pierde A", "Gana A"), y = c("Predice: pierde A", "Predice: gana A"), z = z, type = "heatmap",
            colorscale = list(c(0, "#F6EFE2"), c(1, MAR)), showscale = FALSE) %>%
      lay(annotations = ann, yaxis = list(autorange = "reversed"), xaxis = list(title = "Resultado real"),
          margin = list(l = 120, b = 50, t = 10, r = 10))
  })

  output$p_imp <- renderPlotly({
    imp <- RES$imp
    nom <- c(dif_rank = "Diferencia de ranking", dif_edad = "Diferencia de edad",
             dif_estatura_cm = "Diferencia de estatura", dif_clasif = "Clasificatoria",
             circuito = "Circuito", genero = "Género", etapa = "Etapa")
    etq <- ifelse(names(imp) %in% names(nom), nom[names(imp)], names(imp))
    df <- data.frame(v = etq, i = as.numeric(imp))
    df <- df[order(df$i), ]
    df$v <- factor(df$v, levels = df$v)
    plot_ly(df, x = ~i, y = ~v, type = "bar", orientation = "h", marker = list(color = TURQ)) %>%
      lay(xaxis = list(title = "Caída del AUC"), yaxis = list(title = ""),
          margin = list(l = 150, b = 40, t = 10, r = 10))
  })

  output$p_comp <- renderPlotly({
    plot_ly(COMP, x = ~modelo, y = ~auc_test, type = "bar", name = "AUC en prueba", marker = list(color = MAR)) %>%
      add_trace(y = ~exactitud, name = "Exactitud", marker = list(color = ARENA)) %>%
      lay(barmode = "group", xaxis = list(title = ""), yaxis = list(title = "", range = c(0.4, 0.8)))
  })

  output$p_or <- renderPlotly({
    o <- RES$or_tab
    k <- rep(1, nrow(o))
    i <- match(o$termino, names(RES$sds))
    k[!is.na(i)] <- RES$sds[i[!is.na(i)]]
    b <- o$coef * k; e <- o$ee * k
    df <- data.frame(t = o$termino, or = exp(b), lo = exp(b - 1.96 * e), hi = exp(b + 1.96 * e))
    df$t <- factor(df$t, levels = rev(df$t))
    plot_ly(df, x = ~or, y = ~t, type = "scatter", mode = "markers", marker = list(color = MAR, size = 9),
            error_x = list(type = "data", symmetric = FALSE, array = df$hi - df$or, arrayminus = df$or - df$lo,
                           color = MAR)) %>%
      lay(xaxis = list(title = "Razón de odds", type = "log"), yaxis = list(title = ""),
          margin = list(l = 190, b = 40, t = 10, r = 10), shapes = list(list(type = "line", x0 = 1, x1 = 1,
          y0 = 0, y1 = 1, yref = "paper", line = list(color = "#9CA3AF", dash = "dash"))))
  })

  output$p_umbral_kpis <- renderUI({
    pr <- RES$prob_mejor
    u <- nz(input$p_umbral, 0.5)
    pred <- pr$p >= u
    y <- pr$a_gana == 1
    tp <- sum(pred & y); fp <- sum(pred & !y); fn <- sum(!pred & y); tn <- sum(!pred & !y)
    tagList(kpi("Exactitud", pct(100 * (tp + tn) / length(y), 1)),
            kpi("Precisión", pct(if (tp + fp > 0) 100 * tp / (tp + fp) else NA, 1), NULL, "tono-turquesa"),
            kpi("Exhaustividad", pct(if (tp + fn > 0) 100 * tp / (tp + fn) else NA, 1), NULL, "tono-arena"))
  })

  output$p_tabla_comp <- renderTable({
    data.frame(Modelo = COMP$modelo, `AUC validación cruzada` = fmt(COMP$auc_cv, 3),
               `AUC prueba` = fmt(COMP$auc_test, 3), Exactitud = fmt(COMP$exactitud, 3), F1 = fmt(COMP$f1, 3),
               check.names = FALSE)
  }, striped = TRUE, hover = TRUE, spacing = "s")

  # ---- Predicción: simulador
  CO <- D$coef
  cf <- function(n) if (n %in% names(CO)) unname(CO[n]) else 0
  logit_a <- function(ra, rb, ea, eb, ha, hb, ca, cb, circ, gen, eta) {
    cf("(Intercept)") + cf("dif_rank") * (rb - ra) + cf("dif_edad") * (ea - eb) +
      cf("dif_estatura_cm") * (ha - hb) + cf("dif_clasif") * (ca - cb) +
      (if (circ == "FIVB") cf("circuitoFIVB") else 0) +
      (if (gen == "Mujeres") cf("generoMujeres") else 0) + cf(paste0("etapa", eta))
  }

  output$s_res <- renderUI({
    ra <- input$s_rank_a; rb <- input$s_rank_b
    ea <- input$s_edad_a; eb <- input$s_edad_b
    ha <- input$s_est_a; hb <- input$s_est_b
    req(ra, rb, ea, eb, ha, hb)
    ca <- as.integer(isTRUE(input$s_clasif_a)); cb <- as.integer(isTRUE(input$s_clasif_b))
    circ <- nz(input$s_circ, "AVP"); gen <- nz(input$s_gen, "Hombres"); eta <- nz(input$s_etapa, "Cuadro eliminatorio")
    p1 <- plogis(logit_a(ra, rb, ea, eb, ha, hb, ca, cb, circ, gen, eta))
    p2 <- 1 - plogis(logit_a(rb, ra, eb, ea, hb, ha, cb, ca, circ, gen, eta))
    pa <- 100 * mean(c(p1, p2))
    lectura <- if (pa >= 65) "Ventaja clara para su pareja."
               else if (pa >= 55) "Ligera ventaja para su pareja."
               else if (pa > 45) "Partido muy parejo."
               else if (pa > 35) "Ligera ventaja para el rival."
               else "Ventaja clara para el rival."
    tagList(
      div(class = "resultado-etq", "Probabilidad estimada de ganar"),
      div(class = "score",
          div(span(class = "score-etq", "Su pareja"), span(class = "digito score-a", pct(pa, 0))),
          span(class = "score-guion", "–"),
          div(span(class = "digito score-b", pct(100 - pa, 0)), span(class = "score-etq", "Rival"))),
      div(class = "barra-duelo mb-2",
          div(class = "barra-a", style = paste0("width:", round(pa, 1), "%")),
          div(class = "barra-b", style = paste0("width:", round(100 - pa, 1), "%"))),
      div(class = "resultado-lectura", lectura),
      div(class = "resultado-detalle",
          paste0("Modelo: regresión logística, con AUC de ", fmt(fila_modelo("Regresión logística")$auc_test, 3),
                 " en los partidos de prueba. Es una estimación a partir de asociaciones históricas, no una garantía.")))
  })

  # ---- Predicción: calculadora histórica
  BINS <- c(0, 2, 5, 10, 20, Inf)
  BIN_ETQ <- c("1–2", "3–5", "6–10", "11–20", "Más de 20")

  base_calc <- reactive({
    d <- Pr
    circ <- nz(input$c_circ, "Todos"); gen <- nz(input$c_gen, "Todos"); eta <- nz(input$c_etapa, "Todas")
    if (circ != "Todos") d <- d[d$circuito == circ, ]
    if (gen != "Todos") d <- d[d$genero == gen, ]
    if (eta != "Todas") d <- d[d$etapa == eta, ]
    d
  })

  output$c_res <- renderUI({
    rp <- input$c_propio; rr <- input$c_rival
    req(rp, rr)
    d <- base_calc()
    gap <- abs(rp - rr)
    if (gap == 0) {
      return(tagList(div(class = "resultado-etq", "Probabilidad histórica de ganar"),
                     div(class = "resultado-valor", "50 %"),
                     div(class = "resultado-lectura", "Mismo ranking: no hay favorito.")))
    }
    lo <- max(1, round(gap * 0.8)); hi <- round(gap * 1.2) + 1
    sub <- d[d$dist >= lo & d$dist <= hi, ]
    if (nrow(sub) < 30) {
      return(tagList(div(class = "resultado-etq", "Probabilidad histórica de ganar"),
                     div(class = "resultado-lectura", "Con estos filtros hay muy pocos partidos parecidos para calcularla.")))
    }
    fav <- mean(!sub$upset)
    prob <- if (rp < rr) fav else 1 - fav
    ic <- 1.96 * sqrt(fav * (1 - fav) / nrow(sub))
    lectura <- if (rp < rr) "Su pareja es la favorita: así le fue históricamente a la mejor rankeada."
               else "El rival es el favorito: así le fue históricamente a la peor rankeada."
    tagList(
      div(class = "resultado-etq", "Probabilidad histórica de ganar"),
      div(class = "resultado-valor", pct(100 * prob, 1)),
      div(class = "barra-duelo mb-2",
          div(class = "barra-a", style = paste0("width:", round(100 * prob, 1), "%")),
          div(class = "barra-b", style = paste0("width:", round(100 - 100 * prob, 1), "%"))),
      div(class = "resultado-lectura", lectura),
      div(class = "resultado-detalle",
          paste0("Basado en ", fmt(nrow(sub)), " partidos con una diferencia de ranking entre ", lo, " y ", hi,
                 " posiciones. Intervalo de confianza de 95 %: ", pct(100 * max(0, prob - ic), 1), " a ",
                 pct(100 * min(1, prob + ic), 1), ".")))
  })

  output$c_curva <- renderPlotly({
    d <- base_calc()
    validate(need(nrow(d) > 30, "No hay suficientes partidos con estos filtros."))
    d$bin <- cut(d$dist, BINS, labels = BIN_ETQ)
    a <- d %>% group_by(bin) %>% summarise(n = n(), s = 100 * mean(!upset), .groups = "drop") %>% filter(n >= 10)
    gap <- abs(nz(input$c_propio, 8) - nz(input$c_rival, 12))
    mi <- as.character(cut(gap, BINS, labels = BIN_ETQ))
    cols <- ifelse(as.character(a$bin) == mi, CORAL, MAR)
    plot_ly(a, x = ~bin, y = ~s, type = "bar", marker = list(color = cols),
            text = ~paste0(fmt(s, 0), " %"), textposition = "outside") %>%
      lay(xaxis = list(title = "Diferencia de ranking", categoryorder = "array", categoryarray = BIN_ETQ),
          yaxis = list(title = "% que gana el favorito", range = c(0, 100)), showlegend = FALSE)
  })
}

shinyApp(ui, server)

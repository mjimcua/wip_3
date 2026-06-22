# Auditoría del HITO 2 — qué hace cada paso y qué KPI optimiza (2026-06-13)
*Según el código actual. El HITO 2 = elegir el método y aplicarlo. Nota: el
código empaqueta H2+H3+H4 conceptuales en la clase StratifiedForecastHito2.*

## Las métricas de calidad de una forecast series (el marco que faltaba)
Cada serie se evalúa por estos KPI; cada paso del HITO 2 mueve uno u otro:

| KPI | Qué mide | Bueno si |
|---|---|---|
| **soporte (n)** | unidades de pipeline observadas | alto |
| **error binomial** | error irreducible por muestra: √(p(1−p)/n) | bajo |
| **evidence weight (z)** | cuánta voz tiene el dato propio vs el grupo | alto = serie habla sola |
| **WAPE $** | error del forecast en dólares sobre test | bajo |
| **sesgo $** | sobre/infra-estimación sistemática | ≈0 |
| **aggregation_cost** | $ que cuesta agregar a grano grueso | bajo (recuperado) |
| **cobertura de banda** | si la banda contiene el real (futuro/walk-fwd) | ~95% |

## Los 12 pasos, en orden de ejecución

### 1. step_h2_fit_baseline_mandatory
- **Hace:** tasa plana Σren/Σpipe por celda mandatory (el método tradicional).
- **Salida:** tasa de referencia por celda (el "ANTES", rival a batir = M1).
- **KPI que toca:** ninguno aún — es la línea base contra la que se miden los demás.

### 2. step_h2_fit_shrunk
- **Hace:** tasa por serie fina con evidence weight z=n/(n+k) hacia su padre.
- **Salida:** tasa shrunk por FS (= M3).
- **KPI optimizado:** baja la VARIANZA / error binomial efectivo (las series
  pobres dejan de hacer ruido). Sube z donde hay evidencia.

### 3. step_h2_fit_uplift_covariates
- **Hace:** factor de uplift por combinación de covariables (desde df_fine);
  fallback global si <200 renovaciones.
- **Salida:** uplift_ajustado por (FS, mes).
- **KPI optimizado:** baja el SESGO $ (la revalorización deja de ser plana).

### 4. step_h2_fit_ts
- **Hace:** EWM (media móvil con decaimiento) para FS con ≥18 meses reales,
  mezclada con z; las cortas se quedan en shrunk.
- **Salida:** tasa con componente temporal (= M4).
- **KPI optimizado:** baja WAPE $ en series con historia (capta el nivel
  reciente). LIMITACIÓN: sigue tendencia con retardo, no la extrapola.

### 5. step_h2_quality_photo
- **Hace:** foto antes/después: nº series, soporte mediano, %$ en soporte<30/
  <100, error de muestreo presupuestado (SE·z).
- **Salida:** tabla comparativa de calidad estructural.
- **KPI optimizado:** ninguno — MIDE soporte y error binomial para evidenciar
  qué compró el evidence weight. Es el "antes/después" en calidad.

### 6. step_h2_backtest_test_months
- **Hace:** backtest ESCALONADO M1→M4 en $ reales sobre el test reservado.
- **Salida:** WAPE y sesgo $ por método; elige el ganador.
- **KPI optimizado:** ES el juez de WAPE $ y sesgo $. Decide qué método se usa.
  LIMITACIÓN: elige y evalúa en el mismo test (sesgo de selección; falta walk-fwd).

### 7. step_h2_forecast_projection
- **Hace:** aplica el método a la pipeline de projection (unidades predichas).
- **Salida:** columnas h2_pred_units_mandatory / _shrunk en el df.
- **KPI:** ninguno nuevo — aplica lo ya elegido.

### 8. step_h2_forecast_bands
- **Hace:** banda = ±z·max(WAPE·pred, suelo binomial), asimétrica por tendencia.
- **Salida:** lo/hi por celda y total 2026.
- **KPI optimizado:** cobertura de banda (declara la incertidumbre del WAPE +
  error binomial). Honesta, no frecuentista exacta.

### 9-11. forecast_next_year / extend_AB / assemble_2027  (HITO 4 conceptual)
- **Hacen:** horizonte 2027 = C1 multi-año firme + C2 re-renovación 1Y +
  C3 adquisición 1Y, con sus aportaciones.
- **Salida:** tabla 2027 desglosada por componente.
- **KPI:** banda declarada (más ancha; estimación sobre estimación).

### 12. step_h2_export_final_table  (HITO 3 conceptual: trazabilidad)
- **Hace:** back-annotation a la tabla fina: raw + predicciones + z + tasa de
  celda + uplift → tabla_final_forecast.csv.
- **Salida:** la tabla explicada línea a línea.
- **KPI:** ninguno — trazabilidad.

## El árbol de decisión que reduce los problemas (resumen, ver COLAPSO_MULTINIVEL)
El soporte se resuelve ANTES (en el HITO 1) bajando por el árbol: N1 sustrato →
N2 ANOVA → N3 levels/mandatory → evidence weight → heurística. El HITO 2 ya
recibe el soporte resuelto y solo ELIGE técnica según soporte+histórico+forma.

## Hallazgos de la auditoría (honestos)
1. **Falta el puente:** el HITO 2 NO re-evalúa el soporte tras el colapso del
   HITO 1 ni etiqueta estado_final (resuelta/evidence-weight/heurística). Es el
   Paso A del cierre pendiente.
2. **Falta el resumen económico antes/después** del colapso (Paso B).
3. **La tendencia/estacionalidad** (Paso C-D) no está integrada en la elección
   de técnica: fit_ts usa EWM, no Holt-Winters; no hay extrapolación de
   pendiente ni límites duros 0-100% (Paso E).
4. **fit_shrunk encoge al mandatory crudo,** no al padre del árbol de colapso.
5. **Backtest con sesgo de selección** (mismo test para elegir y evaluar).
6. El KPI **cobertura de banda** no se mide aún (requiere walk-forward).

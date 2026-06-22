# Cierre del HITO 2 — esquema (2026-06-13)
*Para discutir antes de implementar. El HITO 2 = elegir el método; aquí se
cierra: re-evaluar soporte tras el colapso, etiquetar, estudiar tendencia/
estacionalidad, elegir técnica, y resumir el antes/después económico.*

## Recordatorio del reparto
- Lo conocido: la PIPELINE (unidades que expiran en el futuro).
- Lo que estimamos: TASA de renovación y UPLIFT.
- El HITO 2 elige CÓMO; el HITO 3 aplica y produce el forecast.

## Secuencia del cierre (en orden)

### Paso A — RE-EVALUAR soporte tras los 3 niveles de colapso
Una vez las series quedan como quedan (tras N1 sustrato, N2 ANOVA, N3 levels/
mandatory), volver a medir el soporte de cada grupo final (fs_id_group).
Etiquetar el estado final de cada serie:
  - RESUELTA: soporte >= umbral → se le puede estimar tasa propia.
  - EVIDENCE WEIGHT: sigue pobre → tasa mezclada con el padre (z).
  - HEURÍSTICA: soporte ínfimo o sin train → la resuelve regla/horizonte.
Salida: columna `estado_final` + `nivel_resuelto` (en qué rama paró).

### Paso B — RESUMEN antes/después (impacto económico)
Retomar las métricas del HITO 1 y mostrar la mejora:
  - soporte mediano ANTES (fino) vs DESPUÉS (tras colapso).
  - % del $ en series con soporte<100 (binomial) antes vs después.
  - el aggregation_cost / coste de agregar: ¿cuánto se ha recuperado?
  - error binomial presupuestado antes vs después.
Es el "qué hemos mejorado" en dinero, con las mismas varas del HITO 1.

### Paso C — ESTUDIAR tendencia y estacionalidad (reinsertar el banco)
Aquí entran los 3 steps que sacamos del HITO 1 (BANCO_STEPS_TENDENCIA_HITO2):
rate_trend, top_trend, recent_slope. Ahora SÍ tienen sentido: con el soporte
resuelto, se evalúa la forma de cada serie para ELEGIR técnica.

### Paso D — ELEGIR la técnica por serie (según soporte + histórico + forma)
  - serie con BUEN dinero Y BUEN histórico (>=18-24m) → SERIE TEMPORAL que
    capture estacionalidad Y tendencia a la vez (el deseo explícito de Miguel).
  - serie con histórico medio / tendencia clara → modelo de nivel+pendiente.
  - serie corta o pobre → tasa shrunk (evidence weight), sin pretender forma.
  - heurística → hereda padre / regla.

### Paso E — LÍMITES DUROS de la tasa (sanidad)
  - tasa de renovación NUNCA > 100% (en unidades).
  - tasa > ~90% es SOSPECHOSA → marca de revisión (no error, pero se señala).
  - estos límites se aplican a la SALIDA de cualquier técnica, incluida la
    extrapolación de tendencia (que sin tope se dispararía).

## Decisiones técnicas a fijar antes de construir
1. **Técnica de serie temporal concreta:** ¿Holt-Winters (nivel+tendencia+
   estacionalidad, clásica y explicable) o algo más simple (EWM + índice
   estacional)? Recomiendo Holt-Winters para las FUs con dinero+histórico,
   por ser estándar y capturar las dos cosas que Miguel quiere.
2. **Umbrales de elegibilidad:** ¿cuánto dinero y cuánto histórico para que una
   serie use serie temporal? (liga con umbrales auto por métrica).
3. **Cómo aplicar el tope 100% / sospecha 90%:** ¿clip duro + flag, o
   transformación que satura (logit) al extrapolar? El logit evita el corte
   feo pero es menos transparente. Recomiendo clip + flag para esta etapa.
4. **El backtest escalonado (M1-M4) ya existe:** ¿se reordena para reflejar el
   nuevo árbol de colapso como peldaños? (hoy M3 es "dims finas shrunk").

## Lo que ya está construido y se reaprovecha
- Colapso N1 y N2 (esta sesión); N3 pendiente (parent_fs_ids existe).
- fit_shrunk (evidence weight), fit_ts (EWM), backtest escalonado, foto de
  calidad, bandas, assemble_2027 — del HITO 2 ya implementado en sesiones
  previas. Hay que RECONECTARLOS al árbol de colapso nuevo.

---
## ESTADO DE IMPLEMENTACIÓN (2026-06-13) — versión para probar
- **Paso A (re-evaluar soporte + etiquetar):** IMPLEMENTADO
  (step_h2_reassess_support). Columnas step_h2_fs_group, step_h2_estado_final.
- **Paso B (resumen económico antes/después):** IMPLEMENTADO
  (step_h2_improvement_summary).
- **Paso C (tendencia/estacionalidad):** NO IMPLEMENTADO. Requiere reinsertar
  el banco (BANCO_STEPS_TENDENCIA_HITO2) — decisión pendiente.
- **Paso D (elegir técnica, Holt-Winters):** NO IMPLEMENTADO. Hoy fit_ts usa
  EWM. Falta decidir Holt-Winters vs EWM y umbrales de elegibilidad.
- **Paso E (límites duros 0-100% / sospecha 90%):** NO IMPLEMENTADO. Falta
  decidir clip+flag vs logit.
- **fit_shrunk al padre del árbol** (no mandatory crudo): NO HECHO.
- **Backtest walk-forward** (quitar sesgo de selección): NO HECHO.

Lo implementado corre de principio a fin (run_all) y es PROBABLE. Lo no
implementado está aislado: no rompe la ejecución, solo no añade esas mejoras.

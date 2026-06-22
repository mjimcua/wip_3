"""Ejecución completa del framework (estado a 2026-06-12).

Corre HITO 1 (preparar y valorar) + HITO 2 (elegir método, predecir, bandas,
horizonte) y muestra las cuatro cifras que el negocio pide para una iteración:

  1. Forecast de RENOVACIÓN 2026 (projection) CON banda de confianza.
  2. Aportaciones SIMULADAS de 2027, desglosadas:
       C1 multi-año 2Y/3Y ya firmado (firme),
       C2 re-renovación 1Y (renovaciones 2026 que vuelven a vencer en 2027),
       C3 adquisición 1Y (altas 2026 que vencerán en 2027),
       TOTAL 2027 = C1 + C2 + C3.

Para datos REALES: cambiar la ruta del CSV y ajustar SOLO el bloque CONFIG
con los nombres/valores reales de tus columnas (ver comentarios)."""
import pandas as pd
from stratified_forecast import ForecastConfiguration, Step1Config, run_all

# ----------------------------- DATOS -----------------------------
CSV = "portfolio_v2.csv"                 # ← ruta de tu extract real
df = pd.read_csv(CSV)

# ----------------------------- CONFIG ----------------------------
# Ajusta a tus columnas reales. Lo que NO cambie de nombre, déjalo igual.
config = ForecastConfiguration(
    cfg=Step1Config(
        business_mandatory_dims=["regional_level_1", "product_level_1"],  # grano padre del shrinkage
        pending_date="2026-07-01",       # primer mes a predecir (corta train/test/projection)
        test_months=3,                   # meses reales reservados para el backtest escalonado
        covariate_cols=["desc_bucket", "price_cap", "msrp_increased"],     # → uplift (tabla fina)
    ),
    raw_data_path=None, verbosity="execution")

# --------------------------- EJECUCIÓN ---------------------------
sf1, sf2 = run_all(config, df_raw=df)

# ----------------------- LECTURA DE RESULTADOS -------------------
# (todo queda también en el log por pantalla y en tabla_final_forecast.csv)
print("\n" + "=" * 60)
print("RESUMEN DE LA ITERACIÓN")
print("=" * 60)

# 1) Renovación 2026 con banda
b = sf2.step_metadata.get("step_h2_forecast_bands", {})
if b:
    print(f"1) RENOVACIÓN 2026 (projection): ${b['total']:,.0f}")
    print(f"   banda: ${b['lo']:,.0f} … ${b['hi']:,.0f}  "
          f"(método {b['metodo']}, WAPE {b['wape']:.1%})")

# 2) Aportaciones 2027 desglosadas
a = sf2.step_metadata.get("step_h2_assemble_2027", {})
if a:
    print(f"\n2) AÑO COMPLETO {a['year']}: ${a['total']:,.0f}")
    print(f"   C1 multi-año firme  : ${a['c1']:,.0f}")
    print(f"   C2 re-renovación 1Y : ${a['c2']:,.0f}  (renovaciones 2026 → pipeline 2027)")
    print(f"   C3 adquisición 1Y   : ${a['c3']:,.0f}  (altas 2026 que vencerán)")

# 3) Series a vigilar / explicar a negocio
sf1.describe_portfolio()

"""Ejecución PASO A PASO del framework (no agregada).

Filosofía: cada step se llama explícitamente sobre el MISMO objeto `sf`. Así,
en un notebook, puedes re-ejecutar UN solo step (o desde el que cambiaste)
sin relanzar todo desde cero — el estado vive en `sf` y se va acumulando.

Hay UN solo objeto, `sf`. (Antes había sf1/sf2: eran el mismo objeto en dos
momentos; el HITO 2 hereda del HITO 1. Ya no se usa esa distinción.)

Si trabajas en notebook: ejecuta el bloque SETUP una vez, y luego cada step
en su propia celda. Para recalcular, vuelve a correr solo esa celda y las
siguientes — no las anteriores.
"""
import pandas as pd
from stratified_forecast import ForecastConfiguration, Step1Config, StratifiedForecastHito2

# ============================ SETUP (una vez) ============================
# keep_default_na=False: que "NA" (Norteamérica), "N/A", etc. NO se lean como
# NaN. Si tu extract no tiene esos valores, puedes quitarlo.
df = pd.read_csv("portfolio_v3.csv", keep_default_na=False, na_values=[])  # ← tu extract real

config = ForecastConfiguration(
    cfg=Step1Config(
        business_mandatory_dims=["tr_regional_level_1", "tr_product_level_1"],
        # roles (train/test/projection) vienen de SQL, no se derivan por fecha:
        use_raw_roles=True,
        # mes EN CURSO (incompleto): va en train pero fuera del backtest. La
        # columna la trae SQL (0/1). Es instrumental, NO una dimensión.
        current_month_col="is_current_month",
        covariate_cols=["discount_interval", "price_cap", "msrp_increased", "prev_OperationGroup"],
        # CLAVE: declara las señales de comportamiento para activar el COLAPSO N1.
        structural_timevarying_dims={"softcancel": "negative", "dormant": "negative"},
        timevarying_positive_values=("si", "yes", "1", 1, True, "True"),
    ),
    raw_data_path=None, verbosity="execution")

sf = StratifiedForecastHito2(config)   # UN objeto para todo (H1 + H2)
sf.df_raw = df

# ===================== HITO 1 — paso a paso (preparar y valorar) =========
sf.step_0_validate_input()
sf.step_1_normalize_period()
sf.step_1_collapse_covariates()
sf.step_1_derive_roles_from_period()
sf.step_2_report_only_projection_top()  # cobertura temprana: $ solo en projection (id al vuelo)
sf.step_1_add_universe()
sf.step_1_add_coverage_pattern()
sf.step_1_add_forecast_route()
sf.step_1_report_money_by_route()
sf.step_1_drop_no_impact()
sf.step_1_fill_gaps()
sf.step_1_add_rates()
sf.step_1_add_auv()
sf.step_1_add_synthetic_flag()
sf.step_1_assert_coherence()
sf.step_1_report_density()
sf.step_2_build_identity()
sf.step_2_build_support()
sf.step_2_collapse_signal_support()        # COLAPSO Nivel 1 (sustrato)
sf.step_2_report_density_money()
sf.step_2_report_gap_density_money()
sf.step_2_report_support_profile()
sf.step_2_report_no_training_top()
sf.step_2_report_density_mandatory_vs_full()
sf.step_2_report_history_length()
sf.step_2_report_aggregation_cost()
sf.step_3_anova_rate()
sf.step_3_collapse_anova()                 # COLAPSO Nivel 2 (ANOVA)
sf.step_3_report_dim_fragmentation()
sf.step_3_classify_small_series()
sf.step_3_report_level_coverage()
sf.step_3_report_covariate_value()
sf.step_6_add_story_columns()
sf.step_6_report_story_figures()
sf.describe_portfolio()                     # utilidad (no step)

# ===================== HITO 2 — paso a paso (elegir método y predecir) ===
sf.step_3_report_recent_slope()            # TENDENCIA: series con más pendiente reciente
sf.step_3_report_top_trend()               #            (para evaluación gráfica en projection)
sf.step_h2_fit_baseline_mandatory()        # M1: tasa plana del grano grueso
sf.step_h2_fit_shrunk()                    # M3: encogida hacia el grupo del ÁRBOL de colapso
sf.step_h2_fit_uplift_covariates()         # uplift por combinación de covariables
sf.step_h2_fit_ts()                        # M4: serie temporal (EWM) en series con historia
sf.step_h2_reassess_support()              # Paso A: re-evaluar soporte tras colapso
sf.step_h2_improvement_summary()           # Paso B: antes/después económico
sf.step_h2_quality_photo()
sf.step_h2_backtest_test_months()          # elige método por WAPE en el test
sf.step_h2_forecast_projection()           # predicción projection con el método elegido
sf.step_h2_forecast_bands()                # banda POR SERIE + 2 totales (coherente / cuadratura)
sf.step_h2_total_2026()                    # NÚMERO FINAL 2026 = real + proyectado
sf.step_h2_forecast_next_year()            # 2027: regeneración de pipeline
sf.step_h2_extend_AB()                     # 2027: extiende horizonte (1Y que re-renuevan en 2027)
sf.step_h2_assemble_2027()                 # 2027: C1 firmado + C2 re-renovación + C3 adquisición
sf.step_h2_export_final_table()            # tabla final para Power BI

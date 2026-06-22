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
df = pd.read_csv("portfolio_v2.csv")               # ← tu extract real
config = ForecastConfiguration(
    cfg=Step1Config(
        business_mandatory_dims=["regional_level_1", "product_level_1"],
        pending_date="2026-07-01",
        test_months=3,
        covariate_cols=["desc_bucket", "price_cap", "msrp_increased"],
        timevarying_positive_values=("si", "yes", "1", 1, True),  # ajusta a tu dato
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

# ===================== HITO 2 — paso a paso (elegir método) ==============
# (descomenta cuando vayas a ejecutar el HITO 2)
# sf.step_h2_fit_baseline_mandatory()
# sf.step_h2_fit_shrunk()
# sf.step_h2_fit_uplift_covariates()
# sf.step_h2_fit_ts()
# sf.step_h2_reassess_support()           # Paso A: re-evaluar soporte tras colapso
# sf.step_h2_improvement_summary()        # Paso B: antes/después económico
# sf.step_h2_quality_photo()
# sf.step_h2_backtest_test_months()
# sf.step_h2_forecast_projection()
# sf.step_h2_forecast_bands()
# sf.step_h2_forecast_next_year()
# sf.step_h2_extend_AB()
# sf.step_h2_assemble_2027()
# sf.step_h2_export_final_table()

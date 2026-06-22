"""STRATIFIED FORECAST FRAMEWORK — fichero único (consolidado 2026-06-12).

Mapa del fichero:
  1. CONFIGURACIÓN  (Step1Config, ForecastConfiguration)
  2. GLOSARIO + STEP_META  (metadatos de output: pregunta, defs, criterio, §)
  3. HITO 1  (preparar y valorar: roles, covariables→vista FU, diagnósticos,
     escalera) + describe_portfolio() (utilidad no-step) + helpers _require/
     _done/_table (contrato de output de TODOS los steps)
  4. HITO 2  (método y valor: credibilidad, uplift por covariables, TS,
     foto, backtest escalonado, projection, 2027 A+B, tabla final)
  5. RUNNERS  (run_hito_1, run_hito_2, run_all)

Guía extendida: FLUJO_DE_TRABAJO_FRAMEWORK.md (generada por introspección).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import os
import numpy as np
import pandas as pd

# ============================ 1. CONFIGURACIÓN ============================


@dataclass
class Step1Config:
    period_col: str = "period"
    dataset_role_col: str = "dataset_role"
    # Roles que cuentan como HISTORIA OBSERVADA (soporte de la tasa: densidad,
    # huecos, coherencia). Por defecto train ∪ test: la frontera train/test es
    # un artefacto de evaluación (backtest), no dice qué dato existe — un mes
    # "test" es una observación real de la tasa, y a veces es lo único que hay.
    # Más adelante la frontera train/test se decidirá por presupuesto de
    # historia (6 meses → todo es usable; 3 años → eliges split); medir el
    # soporte sobre este pool es robusto a dónde caiga ese split.
    history_roles: tuple = ("train", "test")
    # ---- roles por FECHA PENDING ----
    # pending_date: "YYYY-MM-DD" o "YYYY-MM" = día 1 del primer mes PENDIENTE.
    #   projection = desde pending_date en adelante (incl.)
    #   test       = los test_months meses inmediatamente anteriores
    #   train      = todo lo anterior
    # use_raw_roles=True ignora esta lógica y respeta la columna del raw.
    pending_date: Optional[str] = None   # None = mes natural de hoy
    test_months: int = 2
    # Recorte del borde izquierdo: ignora TODO lo anterior a esta fecha
    # ("YYYY-MM"). Útil si el inicio de la ventana del extract tiene mala
    # cobertura (arranques frágiles que fabrican huecos). None = sin recorte.
    # Detección/eliminación de PILOTOS: prefijo débil antes del arranque real.
    use_raw_roles: bool = False          # False = calcula; True = usa columna del raw
    # compatibilidad (aliases; no usar en config nuevas):
    derive_roles_from_period: Optional[bool] = None
    role_as_of_period: Optional[str] = None
    role_test_months: Optional[int] = None
    pipeline_units_col: str = "total_tr_units"
    pipeline_usd_col: str = "total_tr_usd"
    renewed_units_col: str = "total_renewed_units"
    renewed_usd_col: str = "total_renewed_usd"

    reacq_units_col: str = "total_reacquired_units"
    reacq_usd_col: str = "total_reacquired_usd"
    auv_pipeline_col: str = "TR_AUV"
    auv_renewed_col: str = "REN_AUV"
    auv_reacq_col: str = "ReAC_AUV"

    auv_tolerance_pct: float = 0.01

    # step_4 freeze: volume tier breakpoints (above min_volume).
    # Default 500/2000 mirrors the original framework. Adjust if your
    # business has very different scale.
    volume_tier_small_max: int = 500
    volume_tier_medium_max: int = 2000

    # step_4 stratified freeze: number of deciles. Default 10. Setting to
    # 5 makes the policy coarser but more robust on small portfolios.
    n_deciles: int = 10

    # step_2_get_recommended_threshold: cascade parameters
    threshold_candidates: list = field(default_factory=lambda: [
        30, 50, 100, 150, 200, 300, 500, 750, 1000, 2000,
    ])
    threshold_strict_coverage_pct: float = 90.0
    threshold_strict_se_max: float = 0.05
    threshold_relaxed_coverage_pct: float = 85.0

    flag_time_series_col: str = "flag_time_series"
    # Valores que la columna flag_time_series_col puede tener en el raw.
    # Cada fila debe ser exactamente uno de estos dos valores. Si tu raw
    # trae otros valores (strings, booleanos), normalízalos a estos en tu
    # subclase override de `load_raw_forecast_units` antes de devolver
    # el df.
    flag_time_series_normal_value: int = 0
    flag_time_series_ts_value: int = 1
    ts_revenue_col: str = "total_renewed_usd"
    ts_region_col: str = "regional_level_1"

    # ---- dimensiones por COMPLEMENTO (no por lista blanca) ----
    # Una columna es DIMENSIÓN si NO es una columna de rol/medida ni está en
    # ignore_cols. El conjunto de dimensiones NO se declara aquí: se resuelve
    # contra las columnas del raw en step_0 (ver resolve_dimensions) y se
    # congela. Así una dimensión nueva en la vista entra sola y nunca se olvida
    # una en el id (que fusionaría filas y corrompería las tasas).
    #
    # ignore_cols: columnas del raw que NO son ni rol/medida ni dimensión
    # (claves subrogadas, timestamps de carga, notas...). Por defecto vacío:
    # el contrato es que el raw trae solo columnas de rol + dimensiones.
    ignore_cols: list = field(default_factory=list)
    # COVARIABLES (uplift): columnas que NO segmentan las FU. El raw viene
    # multiplicado por sus combinaciones; el framework colapsa a una vista
    # FU interna y conserva la tabla fina (df_fine) con fu_id/comb_id.
    covariate_cols: list = field(default_factory=list)

    # ---- tiers = ANOTACIONES opcionales (no definen el conjunto) ----
    # Solo clasifican dims con rol especial aguas abajo. Cada dim listada debe
    # existir en el complemento (step_0 lo verifica). Vacías = sin anotación.
    #   - business_mandatory_dims: backstop del colapso (shrinkage, futuro).
    #   - structural_stable_dims:  estructural sin rol especial.
    #   - structural_timevarying_dims: {dim: direction} — la DIRECCIÓN no se
    #     deriva del dato, hace falta para la dinámica/régimen. Es la única
    #     anotación imprescindible hoy.
    #   - exploratory_dims: tentativas.
    business_mandatory_dims: list = field(default_factory=list)
    structural_stable_dims: list = field(default_factory=list)
    structural_timevarying_dims: dict = field(
        default_factory=lambda: {"softcancel": "negative", "dormant": "negative"}
    )
    exploratory_dims: list = field(default_factory=list)

    timevarying_positive_values: tuple = (
        True, "True", "true", "yes", "Yes", "YES",
        "Y", "y", "1", 1, "T", "t",
    )

    def __post_init__(self):
        if isinstance(self.structural_timevarying_dims, list):
            if len(self.structural_timevarying_dims) == 0:
                self.structural_timevarying_dims = {}
            else:
                raise ValueError(
                    "structural_timevarying_dims must be a dict {dim: direction}. "
                    "Each direction must be 'negative' or 'positive'."
                )
        if not isinstance(self.structural_timevarying_dims, dict):
            raise ValueError(
                f"structural_timevarying_dims must be dict, "
                f"got {type(self.structural_timevarying_dims).__name__}"
            )
        for dim, direction in self.structural_timevarying_dims.items():
            if direction not in {"negative", "positive"}:
                raise ValueError(
                    f"Invalid direction '{direction}' for dim '{dim}'. "
                    "Must be 'negative' or 'positive'."
                )
        # covariables: no pueden ser dimensión a la vez → se retiran de los
        # tiers automáticamente (error solo si chocan con una mandatory)
        covs = set(self.covariate_cols or [])
        if covs:
            choque = covs & set(self.business_mandatory_dims or [])
            if choque:
                raise ValueError(f"covariate_cols en conflicto con "
                                 f"business_mandatory_dims: {sorted(choque)}")
            if self.structural_timevarying_dims:
                self.structural_timevarying_dims = {
                    k: v for k, v in self.structural_timevarying_dims.items()
                    if k not in covs}
        # compatibilidad: traduce params viejos a los nuevos
        if self.derive_roles_from_period is not None:
            self.use_raw_roles = not self.derive_roles_from_period
        if self.role_as_of_period is not None and self.pending_date is None:
            self.pending_date = self.role_as_of_period
        if self.role_test_months is not None:
            self.test_months = self.role_test_months

    def non_dimension_columns(self) -> set:
        """Conjunto fijo de columnas de rol/medida a EXCLUIR del complemento.
        Ojo: ts_region_col NO entra (apunta a una dimensión real, p.ej.
        regional_level_1; el track TS solo la nombra). ts_revenue_col sí
        entra, pero suele ser alias de renewed_usd (duplicado inofensivo).
        Columnas a None (medidas ausentes) se ignoran."""
        cols = {self.period_col, self.dataset_role_col,
                self.pipeline_units_col, self.pipeline_usd_col,
                self.renewed_units_col, self.renewed_usd_col,
                self.flag_time_series_col, self.ts_revenue_col}
        for c in (self.reacq_units_col, self.reacq_usd_col,
                  self.auv_pipeline_col, self.auv_renewed_col, self.auv_reacq_col):
            if c is not None:
                cols.add(c)
        cols.discard(None)
        return cols

    def resolve_dimensions(self, all_columns) -> list:
        """Dimensiones = columnas del raw que NO son rol/medida ni ignore_cols.
        Conserva el orden del raw. Se llama en step_0 y se congela."""
        exclude = (self.non_dimension_columns() | set(self.ignore_cols or [])
                   | set(self.covariate_cols or []))
        return [c for c in all_columns if c not in exclude]

    def tier_order(self) -> list:
        """Orden de anotaciones de tier (para componer el fs_id de forma legible):
        mandatory, stable, timevarying, exploratory. Sin duplicados."""
        ordered = []
        for c in (self.business_mandatory_dims
                  + self.structural_stable_dims
                  + list(self.structural_timevarying_dims.keys())
                  + self.exploratory_dims):
            if c and c not in ordered:
                ordered.append(c)
        return ordered

    @property
    def annotated_dims(self) -> list:
        """Todas las dims con alguna anotación de tier (para la aserción
        ⊆ complemento en step_0)."""
        return self.tier_order()

    @property
    def timevarying_dim_names(self) -> list:
        return list(self.structural_timevarying_dims.keys())

    def tier_of(self, col: str) -> Optional[str]:
        if col in self.business_mandatory_dims:
            return "business_mandatory"
        if col in self.structural_stable_dims:
            return "structural_stable"
        if col in self.structural_timevarying_dims:
            return "structural_timevarying"
        if col in self.exploratory_dims:
            return "exploratory"
        return None


@dataclass
class ForecastConfiguration:
    cfg: Step1Config = field(default_factory=Step1Config)
    raw_data_path: Optional[str | Path] = "synthetic_portfolio.csv"
    table_prefix: str = "forecast"
    verbosity: str = "execution"
    language: str = "es"
    explanations_path: str | Path = "explanations.yaml"

    # ---- freeze decile policy ----
    # Stratify freeze threshold by decile of annual pipeline USD.
    # tolerances_usd: max USD error tolerated per decile (when projecting
    # the rate of the decile's FUs). Default 100_000 per decile (1M total
    # over 10 deciles). Set None to fall back to uniform mode.
    # r_method: how to estimate the rate r per decile when computing
    # N_min from SE_max via binomial inverse.
    #   "median" -> median of rate_renewal of FUs in decile (default)
    #   "conservative" -> 0.5 (worst case Var=p(1-p))
    #   float in (0, 1) -> use that exact value
    freeze_decile_tolerances_usd: Optional[dict] = field(default_factory=lambda: {
        i: 100_000.0 for i in range(1, 11)
    })
    freeze_decile_r_method: dict = field(default_factory=lambda: {
        i: "median" for i in range(1, 11)
    })

    def __post_init__(self):
        if self.verbosity not in ("execution", "debug", "explain"):
            raise ValueError(
                f"verbosity='{self.verbosity}' not supported. "
                f"Use 'execution', 'debug' or 'explain'."
            )
        if self.language not in ("es", "en"):
            raise ValueError(
                f"language='{self.language}' not supported. Use 'es' or 'en'."
            )
        if self.raw_data_path is not None:
            self.raw_data_path = Path(self.raw_data_path)
        self.explanations_path = Path(self.explanations_path)

    def table_name_for_step(self, step: str) -> str:
        if not step or not step.strip():
            raise ValueError("table_name_for_step: 'step' must not be empty")
        return f"{self.table_prefix}_{step.strip()}"

# ====================== 2-3. GLOSARIO, METADATOS, HITO 1 ==================



class DataNotReadyError(ValueError):
    """El raw no está preparado para predecir (incoherencias bloqueantes)."""



GLOSARIO = {
 "FU": "Forecast Unit: una celda (combinación de dims) en UN mes.",
 "FS": "Forecast Series: la serie temporal de una combinación de dims.",
 "soporte": "unidades de pipeline de la FU/FS (n de la binomial).",
 "cota binomial": "error mínimo del caso más favorable (mundo plano): sqrt(p(1-p)/n). Error≤cota=ruido; error>>cota=error real añadido.",
 "z (evidence weight)": "PESO POR EVIDENCIA (término oficial; la literatura lo llama credibilidad o shrinkage): cuánta voz tiene el dato PROPIO de la serie frente a su celda: tasa_publicada = z·propia + (1−z)·celda, con z=n/(n+k). n=soporte propio en train; k=soporte al que ambas fuentes pesan igual, estimado del dato como ruido-dentro/varianza-entre (Bühlmann): grupo homogéneo→k grande→shrink fuerte; series realmente distintas→k pequeño→el dato propio manda antes.",
 "uplift": "concepto SIEMPRE en inglés: factor de cambio de valor = AUV_renovado/AUV_pipeline (revalorizar es la acción, uplift el concepto).",
 "WAPE": "suma de |error| / suma de real (ponderado por tamaño).",
 "eta2": "η²: % de la varianza de la tasa que explica una dimensión.",
 "celda mandatory": "agregado a las dims obligatorias de negocio.",
 "gate": "primer criterio de la escalera que la celda NO supera.",
}

STEP_META = {
 "step_1_report_density": {"ref": "§1.11", "preg": "¿Qué calidad de MUESTRA tienen las FU/FS?", "defs": ["FU","FS","soporte"], "crit": "soporte<30=ruidoso; ≥100=utilizable; mediana del portfolio como termómetro."},
 "step_2_report_gap_density_money": {"ref": "§1.21", "preg": "¿Cuánto DINERO vive en series intermitentes?", "defs": ["FS"], "crit": "huecos=ceros legítimos (contrato vigente); preocupa solo si cargan $ material."},
 "step_3_classify_small_series": dict(ref="", preg="¿Cuántas series pequeñas hay y cuánto $ son: fusionables, encogibles o polvo?", defs=["FS", "soporte"], crit="3 destinos de GESTIÓN, no 3 métodos: fusionable (subir grano si hay dim sin señal), encogible (evidence weight ya resuelve), polvo (heredar y reportar agregado). El nº de series es incomodidad; el $ es lo que decide.", formula="fusionable: dust<=sop<small Y existe dim anulable; polvo: sop<dust Y padre<mediana; resto encogible"),
 "step_3_collapse_anova": dict(ref="", preg="¿Ganamos soporte colapsando la dimensión no-mandatory que no separa la tasa?", defs=["FS","eta2","soporte"], crit="NIVEL 2 del árbol: ejecuta lo que el ANOVA informaba. Colapsa con '*' la dim no-mandatory de menor η² para las series aún pobres. No toca geografía/producto.", formula="drop_dim = argmin η² entre no-mandatory con η²<eta_low; '*' en su posición si soporte<floor"),
 "step_3_anova_rate": {"ref": "§1.29", "preg": "¿Qué dimensiones SEPARAN la tasa (no colapsar)?", "defs": ["eta2"], "crit": "η²≥0.10 separa; 0.03–0.10 intermedia; <0.03 anulable si además fragmenta."},
 "step_3_report_dim_fragmentation": {"ref": "§1.31", "preg": "¿Qué dimensión FABRICA el grano fino?", "defs": ["soporte"], "crit": "factor alto SIN señal = quitar; CON señal = shrink, no quitar."},
 "step_6_add_story_columns": {"ref": "§1.35", "preg": "¿Qué calidad tiene cada celda y cuál es su MODO DE FALLO?", "defs": ["gate","cota binomial"], "crit": "escalera: soporte→temporal→estacionalidad→tendencia→mezcla→apto_promedio; el gate es el PRIMER fallo, los flags story_* guardan todos."},
 "step_h2_fit_shrunk": {"ref": "§2.2", "preg": "¿Cuánta voz tiene el dato propio de cada serie?", "defs": ["z (evidence weight)"], "crit": "k por momentos; vigilar z ponderado por $."},
 "step_h2_fit_uplift_covariates": {"ref": "§2.3", "preg": "¿Cómo mueve el descuento/destope la REVALORIZACIÓN?", "defs": ["uplift"], "crit": "factor por combinación con n≥200; si no, fallback global; no_info usa el base de la celda."},
 "step_h2_reassess_support": dict(ref="", preg="Tras el colapso, ¿cuánto $ tiene tasa propia, cuánto hereda y cuánto es heurística?", defs=["soporte","FS"], crit="PASO A del cierre: re-evalúa soporte del grupo colapsado y etiqueta estado_final.", formula="RESUELTA si soporte_grupo>=floor; HEURISTICA si <dust o sin train; resto EVIDENCE_WEIGHT"),
 "step_h2_improvement_summary": dict(ref="", preg="¿Qué soporte hemos ganado con el colapso (antes/después)?", defs=["soporte"], crit="PASO B: compara grano fino vs colapsado en las varas del HITO 1.", formula="soporte mediano y %$ bajo soporte<100, fino vs grupo"),
 "step_h2_quality_photo": {"ref": "§2.7", "preg": "¿Qué calidad tienen las series ANTES (mandatory) vs DESPUÉS (framework)?", "defs": ["soporte","z (evidence weight)"], "crit": "el fino fragmenta; la credibilidad devuelve el error presupuestado a escala."},
 "step_h2_forecast_bands": dict(ref="", preg="¿Cuál es el forecast 2026 y su banda de confianza?", defs=["WAPE", "z (evidence weight)"], crit="banda anclada en el error REAL del backtest (no inventada); suelo binomial irreducible por serie; asimétrica donde hay tendencia reciente. Declarada como cuantificación honesta, no IC frecuentista exacto.", formula="half = z·max(WAPE·pred, raíz(Σ SE_usd²)); lo/hi escalados por 1+trend_skew·|pendiente| en el lado correspondiente"),
 "step_h2_assemble_2027": dict(ref="", preg="¿Cuánto saldrá el año completo siguiente y cuánto aporta cada parte?", defs=[], crit="C1 multi-año ya firmado (firme) + C2 re-renovación 1Y + C3 adquisición 1Y (ambas simuladas, banda ancha); total = C1+C2+C3.", formula="C1 = pred_shrunk×AUV×uplift de projection 2Y/3Y del año objetivo; C2/C3 = pred_usd de extend_AB partido por share de A vs B"),
 "step_h2_backtest_test_months": {"ref": "§2.8", "preg": "¿Cuánto MEJORA cada bloque de variables, en $ reales?", "defs": ["WAPE"], "crit": "escalonado M1→M4 sobre test reservado; gana el WAPE $ mínimo (validación out-of-sample pendiente, ver doc)."},
}


for _k, _v in {
 "step_2_report_only_projection_top": dict(ref="", preg="¿Qué forecast series viven SOLO en projection (heurística pura) y cuáles son las 3 de más $?", defs=["FS"], crit="ni train ni test en todo el histórico → no hay nada que aprender; top 3 con clave completa para revisar si falta una variable a futuro. Solo informa.", formula="solo-projection = FS con pipeline>0 en projection y 0 en train+test"),
 "step_2_report_no_training_top": dict(ref="§1.22", preg="¿Qué dinero hay que predecir SIN ninguna evidencia de entrenamiento?", defs=["FS"], crit="en un negocio continuo es ANÓMALO: revisar en origen (migración entre celdas por dims time-varying, producto nuevo, clave mal informada). El framework las cubre con la tasa del grupo (z=0), pero la revisión manda.", formula="sin train = 0 meses reales con pipeline>0 en rol train; ranking por $ de projection"),
 "step_2_report_support_profile": dict(ref="", preg="¿Cómo es la materia prima por tramos de soporte y cuánto error arrastramos solo por tamaño?", defs=["FS", "soporte"], crit="describe sin predecir; el error binomial (±z·√(p(1−p)/n)) es el error irreducible por muestra → motiva el HITO 2.", formula="error_pp = 1.96·√(0.5·0.5/n)·100 (p=0,5 peor caso, 95%)"),
 "step_2_collapse_signal_support": dict(ref="", preg="¿Ganamos soporte uniendo las series de señal negativa (o positiva) del mismo grupo?", defs=["FS", "soporte"], crit="NIVEL 1 del árbol de colapso: une por signo (concatena señales activas, conserva traza) solo si soporte bajo; un movimiento a prueba, mide antes/después.", formula="candidata = señal activa Y soporte_FS < floor; grupo = concat(señales activas) + resto de dims iguales"),
 "step_2_report_density_money": dict(ref="§1.20", preg="¿Cuánto dinero vive en cada nivel de soporte y cómo de concentrado está?", defs=["FS", "soporte"], crit="tabla CANÓNICA por cortes 30/100/200/500 (la foto del HITO 2 compara antes/después con estos mismos cortes); esperado Gini alto (0,8-0,95). El Gini SOLO INFORMA: no condiciona ningún cálculo; decide dónde va la atención HUMANA (cabeza auditable serie a serie vs cola industrial) y es referencia comparativa entre granos y entre pasadas (si cambia mucho, la forma del dinero cambió: revisar)."),
 "step_2_report_history_length": dict(ref="§1.25", preg="¿Cuánta historia tiene cada serie y cuánto $ puede optar a la técnica de serie temporal?", defs=["FS"], crit="<12m: sin ciclo estacional observable; >=18m: elegible para EWM (HITO 2)."),
}.items():
    STEP_META.setdefault(_k, {}).update(_v)
for _k, _v in {
 "step_2_report_density_money": "Gini = 2*Suma(i*x_i)/(n*Suma(x)) - (n+1)/n con x = $ por serie ordenado ASC (0=uniforme, 1=todo en una serie); dolar mediano = soporte donde el $ acumulado (soporte asc) cruza el 50%",
 "step_2_report_gap_density_money": "huecos de la serie = nº de meses sintéticos en su historia; tabla = $ agrupado por nº EXACTO de meses de hueco",
 "step_2_report_history_length": "longitud = meses entre nacimiento y fin de historia de la serie",
 "step_h2_fit_shrunk": "z = n/(n+k); k = ruido-dentro/varianza-entre (momentos), recortado a [5, 5000]; tasa publicada = z*propia + (1-z)*celda",
 "step_3_anova_rate": "eta2 = varianza-entre-niveles / varianza-total de la tasa, ponderada por soporte",
 "step_h2_backtest_test_months": "WAPE = Suma|pred-real|/Suma(real) en $; sesgo = (Suma pred - Suma real)/Suma real; pred$ = tasa*pipeline*AUV_pipe*uplift",
 "step_h2_fit_uplift_covariates": "uplift_comb = (Suma $ren/Suma u_ren)/(Suma $pipe/Suma u_pipe) del train de la combinación; ajustado = share_info*factor + (1-share_info)*base de la celda",
 "step_h2_quality_photo": "error presupuestado = Suma_series SE*$, con SE = raiz(p(1-p)/n) (en DESPUÉS, SE*z)",
}.items():
    STEP_META.setdefault(_k, {})["formula"] = _v

class StratifiedForecastHito1:
    """Pipeline del HITO 1. Un único self.df; cada método valida precondiciones,
    transforma y loguea. Ver main_hito_1.py para la secuencia."""

    def __init__(self, config: Optional[ForecastConfiguration] = None):
        self.config = config or ForecastConfiguration()
        self._assert_config_generation()
        self.df: Optional[pd.DataFrame] = None
        # df_raw: dato precargado por el usuario para no releer en cada corrida.
        # step_0 lo COPIA a self.df (los steps mutan; el original queda intacto).
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_fine: Optional[pd.DataFrame] = None  # tabla fina con fu_id/comb_id
        self.steps_completed: set = set()
        self.step_metadata: dict = {}
        self._has_reacq = False
        self._dimension_cols: Optional[list] = None  # resuelto y congelado en step_0

    # ---- carga (override en subclase para SQL) ----
    def load_raw_forecast_units(self) -> pd.DataFrame:
        path = self.config.raw_data_path
        if path is None:
            raise ValueError("load_raw_forecast_units: raw_data_path es None y no se "
                             "ha sobreescrito el método; devuelve aquí tu DataFrame.")
        return pd.read_csv(path)

    # ---- logging ----
    def _log(self, msg: str) -> None:
        if self.config.verbosity in ("execution", "debug", "explain"):
            print(msg)

    @staticmethod
    def _fmt_dict(d, indent=6) -> str:
        """Pretty-print compacto de un dict para logs: un par por línea, alineado."""
        if not d:
            return "{}"
        kw = max(len(str(k)) for k in d)
        pad = " " * indent
        lines = [f"{pad}{str(k):<{kw}}  {v}" for k, v in d.items()]
        return "{\n" + "\n".join(lines) + "\n" + " " * (indent - 2) + "}"

    # ---- panel de métricas constante (antes/después por paso) ----
    _PANEL_STEPS = {  # pasos donde el panel es informativo (tocan grano/tasa/$)
        "step_1_collapse_covariates", "step_1_drop_no_impact", "step_1_fill_gaps",
        "step_2_build_identity", "step_3_classify_small_series",
        "step_h2_fit_baseline_mandatory", "step_h2_fit_shrunk",
        "step_h2_fit_ts", "step_h2_forecast_projection"}

    def _panel_metrics(self):
        """5 métricas de estado del df actual; None si aún no calculables."""
        cfg = self.cfg
        df = self.df
        m = {}
        try:
            m["filas"] = float(len(df))
            if "step_2_fs_id" in df.columns:
                m["n_series"] = float(df["step_2_fs_id"].nunique())
                sop = df.groupby("step_2_fs_id")[cfg.pipeline_units_col].median()
                m["soporte_med"] = float(sop.median())
                m["pct_series_sop_lt30"] = float(100*(sop < 30).mean())
            if "step_1_fs_with_synthetic_months" in df.columns or "step_1_synthetic" in df.columns:
                col = "step_1_synthetic" if "step_1_synthetic" in df.columns else None
                if col:
                    real = df[df[col].fillna(0).astype(int) == 0]
                    tot = max(len(df), 1)
                    m["pct_filas_sinteticas"] = float(100*(len(df)-len(real))/tot)
        except Exception:
            pass
        return m

    def _panel(self, method):
        if method not in self._PANEL_STEPS:
            return
        cur = self._panel_metrics()
        prev = getattr(self, "_panel_prev", {})
        if cur:
            self._log("  ── panel de métricas (Δ vs paso anterior) ──")
            etiquetas = {"filas": "filas df", "n_series": "nº series",
                         "soporte_med": "soporte mediano", 
                         "pct_series_sop_lt30": "% series soporte<30",
                         "pct_filas_sinteticas": "% filas sintéticas"}
            for k, lab in etiquetas.items():
                if k in cur:
                    d = ""
                    if k in prev and prev[k] != cur[k]:
                        d = f"  (Δ {cur[k]-prev[k]:+,.1f})"
                    self._log(f"    {lab:<22} {cur[k]:>12,.1f}{d}")
            self._panel_prev = cur

    def _done(self, method: str, result: str = "OK") -> None:
        """Cierre estándar de cada paso: VEREDICTO para la siguiente etapa."""
        self._panel(method)
        self._log("")
        self._log(f"  ✔ {method} | VEREDICTO → {result}")
        self._log("")

    # ---- infraestructura ----
    def _require(self, method: str, steps: list) -> None:
        """Contrato de invocación de cada paso: valida dependencias y anuncia el
        OBJETIVO (la 1ª línea del docstring del método). Así todo paso emite
        OBJETIVO → datos del proceso (_log) → VEREDICTO (_done)."""
        missing = [s for s in steps if s not in self.steps_completed]
        if missing:
            raise RuntimeError(f"{method}: faltan pasos previos {missing}")
        doc = (getattr(getattr(self, method, None), "__doc__", "") or "").strip()
        if doc:
            # primer PÁRRAFO (líneas hasta el primer blanco), unido en una frase
            partes = []
            for ln in doc.splitlines():
                s = ln.strip()
                if not s:
                    break
                partes.append(s)
            obj = " ".join(partes)
            # cortar en el primer fin de frase si cabe; si no, tope con elipsis
            punto = obj.find(". ")
            if 0 < punto <= 238:
                obj = obj[:punto + 1]
            elif len(obj) > 240:
                obj = obj[:237] + "..."
        else:
            obj = "(sin docstring)"
        meta = STEP_META.get(method, {})
        self._log("")
        self._log("=" * 76)
        self._log(f"■ {method}" + (f"   [{meta['ref']} de FLUJO_DE_TRABAJO_FRAMEWORK.md]" if meta.get("ref") else ""))
        self._log(f"  PROPÓSITO: {obj}")
        if meta.get("preg"):
            self._log(f"  PREGUNTA DE NEGOCIO: {meta['preg']}")
        for t in meta.get("defs", []):
            self._log(f"  DEF · {t}: {GLOSARIO.get(t, '')}")
        if meta.get("crit"):
            self._log(f"  CRITERIO: {meta['crit']}")
        if meta.get("formula"):
            self._log(f"  FÓRMULA: {meta['formula']}")
        self._log("")
        self._log("  --- datos ---")

    def _mark(self, step: str, metadata: Optional[dict] = None) -> None:
        self.steps_completed.add(step)
        if metadata is not None:
            self.step_metadata[step] = metadata

    @property
    def cfg(self):
        return self.config.cfg

    def _assert_config_generation(self) -> None:
        """Los ficheros van en PAREJA: si el forecast_configuration cargado es de
        una generación anterior (sin la API del complemento), fallamos AQUÍ, en
        la construcción, con el path real del módulo cargado — no más tarde con
        un AttributeError críptico ('Step1Config object has no attribute ...')."""
        required = ("resolve_dimensions", "non_dimension_columns",
                    "annotated_dims", "history_roles")
        missing = [m for m in required if not hasattr(self.cfg, m)]
        if missing:
            import inspect
            try:
                src = inspect.getfile(type(self.cfg))
            except Exception:
                src = "<desconocido>"
            raise RuntimeError(
                f"forecast_configuration DESPAREJADO: a {type(self.cfg).__name__} le falta(n) "
                f"{missing}. Python está cargando: {src} — esa copia es de una generación "
                f"anterior a este stratified_forecast_hito_1.py. Sustituye ESE fichero por "
                f"el de esta entrega y reinicia el kernel/proceso (un import ya hecho no se "
                f"refresca al cambiar el .py en disco).")

    @property
    def dimension_cols(self) -> list:
        """Dimensiones resueltas por complemento y congeladas en step_0.
        Se materializan una vez, antes de que existan columnas step_1_/step_2_,
        para que las derivadas no se lean nunca como dimensiones."""
        if self._dimension_cols is None:
            raise RuntimeError(
                "dimension_cols no resuelto: corre step_0_validate_input primero "
                "(las dimensiones se resuelven por complemento sobre el raw).")
        return self._dimension_cols

    # ========================================================
    # FASE A — step_0 (validar) y step_1 (preparar fila a fila)
    # ========================================================
    def step_0_validate_input(self, df: Optional[pd.DataFrame] = None) -> None:
        """Valida el raw (columnas, dims, NaN en dims, flag_time_series, rol) y
        normaliza el rol a minúsculas. No transforma nada más."""
        self._require("step_0_validate_input", [])
        if df is not None:
            self.df = df
        elif self.df_raw is not None:
            self._log("  step_0: usando df_raw precargado (copia defensiva, sin releer)")
            self.df = self.df_raw.copy()
        if self.df is None:
            self.df = self.load_raw_forecast_units()
        if not isinstance(self.df, pd.DataFrame) or len(self.df) == 0:
            raise ValueError("step_0_validate_input: input debe ser DataFrame no vacío")
        cfg = self.cfg
        flag, nv, tv = (cfg.flag_time_series_col, cfg.flag_time_series_normal_value,
                        cfg.flag_time_series_ts_value)
        if nv == tv:
            raise ValueError("step_0: flag_time_series normal y ts deben diferir")
        if flag not in self.df.columns:
            raise ValueError(f"step_0: falta columna '{flag}'")
        if self.df[flag].isna().any():
            raise ValueError(f"step_0: '{flag}' tiene nulos")
        valid = {nv, tv, float(nv) if isinstance(nv, int) else nv,
                 float(tv) if isinstance(tv, int) else tv}
        bad = set(self.df[flag].unique()) - valid
        if bad:
            raise ValueError(f"step_0: '{flag}' con valores inesperados: {bad}")
        required = [cfg.period_col, cfg.dataset_role_col, cfg.pipeline_units_col,
                    cfg.pipeline_usd_col, cfg.renewed_units_col, cfg.renewed_usd_col]
        miss = [c for c in required if c not in self.df.columns]
        if miss:
            raise ValueError(f"step_0: columnas requeridas ausentes: {miss}")
        # ---- resolver DIMENSIONES por complemento y congelarlas ----
        # Dimensión = toda columna del raw que no es rol/medida ni ignore_cols.
        dims = cfg.resolve_dimensions(self.df.columns)
        if not dims:
            raise ValueError(
                "step_0: el complemento no deja ninguna dimensión. Revisa las "
                "columnas de rol/medida del config o el raw (¿solo trae medidas?).")
        # Aserción: cada dim anotada en un tier debe existir en el complemento.
        # Caza placeholders (p.ej. 'dummy_field') y typos en las anotaciones.
        phantom = [d for d in cfg.annotated_dims if d not in dims]
        if phantom:
            raise ValueError(
                f"step_0: dims anotadas en un tier que NO están en el raw "
                f"(o son columnas de rol): {phantom}. Quítalas del tier o "
                f"corrige el nombre; el conjunto de dims sale del complemento.")
        self._dimension_cols = dims  # CONGELADO: nadie lo recalcula después
        # NULOS en dimensiones: el groupby los descartaría sin avisar.
        nan_dims = {c: int(self.df[c].isna().sum())
                    for c in dims if self.df[c].isna().any()}
        if nan_dims:
            raise ValueError(
                f"step_0: NULOS en dimensiones {nan_dims}. El groupby los "
                f"descartaría sin avisar. Rellena con un centinela o corrige el "
                f"origen. (pandas lee 'NA'/'N/A'/'NULL' como NaN; si es valor "
                f"real, carga con keep_default_na=False.)")
        self._has_reacq = (cfg.reacq_units_col is not None
                           and cfg.reacq_usd_col is not None)
        if self._has_reacq:
            for c in [cfg.reacq_units_col, cfg.reacq_usd_col]:
                if c not in self.df.columns:
                    raise ValueError(f"step_0: columna reacq '{c}' ausente")
        if self.df[cfg.dataset_role_col].isna().any():
            raise ValueError(f"step_0: '{cfg.dataset_role_col}' tiene nulos")
        # normalizar rol a minúsculas (in-place)
        self.df[cfg.dataset_role_col] = (
            self.df[cfg.dataset_role_col].astype(str).str.strip().str.lower())
        self._mark("step_0_validate_input", metadata={"dimension_cols": list(dims)})
        self._log(f"  step_0_validate_input: {len(dims)} DIMENSIONES por complemento: {dims}")
        covs = [c for c in (cfg.covariate_cols or []) if c in self.df.columns]
        cov_missing = [c for c in (cfg.covariate_cols or []) if c not in self.df.columns]
        if covs:
            self._log(f"    COVARIABLES (no segmentan; cualifican el uplift): {covs}")
        if cov_missing:
            self._log(f"    ⚠ covariables declaradas AUSENTES del raw: {cov_missing}")
        instrum = sorted(c for c in cfg.non_dimension_columns() if c in self.df.columns)
        self._log(f"    campos INSTRUMENTALES (rol/medida) presentes ({len(instrum)}): {instrum}")
        ign = sorted(getattr(cfg, "ignore_cols", None) or [])
        if ign:
            self._log(f"    columnas IGNORADAS por config ({len(ign)}): {ign}")
        # Diagnóstico: una MEDIDA resuelta como dimensión revienta el grano
        # (floats casi únicos por fila → cada fila es su propia serie → soporte 1,
        # proyección sin historia que casar, todo a heurística). Avisar fuerte.
        sospechosas, constantes = [], []
        for d in dims:
            s = self.df[d]
            nun = int(s.nunique(dropna=False))
            if pd.api.types.is_float_dtype(s) or nun > 1000:
                sospechosas.append(f"{d} ({s.dtype}, {nun:,} valores)")
            elif nun <= 1:
                constantes.append(d)
        if sospechosas:
            self._log(f"    ⚠ dims que PARECEN MEDIDAS sin declarar (float o cardinalidad "
                      f"enorme): {sospechosas}. Si son medidas, NO pases su *_col a None "
                      f"(los defaults ya traen los nombres de producción) o mételas en "
                      f"ignore_cols; como dims, cada fila se vuelve su propia serie.")
        if constantes:
            self._log(f"    nota: dims CONSTANTES (1 valor): {constantes} — no separan "
                      f"nada; candidatas a ignore_cols.")
        self._done("step_0_validate_input",
                   f"{len(self.df):,} filas, {len(dims)} dims, rol normalizado")

    def step_1_normalize_period(self) -> None:
        """Convierte la columna de periodo a Period[M] (in-place)."""
        self._require("step_1_normalize_period", ["step_0_validate_input"])
        pc = self.cfg.period_col
        if self.df[pc].dtype.name != "period[M]":
            self.df[pc] = pd.PeriodIndex(self.df[pc].astype(str), freq="M")
        self._mark("step_1_normalize_period")
        self._done("step_1_normalize_period", "periodo a Period[M]")

    def step_1_collapse_covariates(self) -> None:
        """Colapsa el raw multiplicado por covariables a la VISTA FU interna
        (grano dims_estables × mes): Σ unidades/USD, AUVs RECALCULADAS del
        agregado (jamás promediadas). La tabla FINA original se conserva en
        self.df_fine con fu_id (dims estables) y comb_id (covariables) para
        Fase 0, auditoría y back-annotation futura. Asserts: conservación de
        pipeline/valor. Sin covariables declaradas = no hace nada."""
        self._require("step_1_collapse_covariates", ["step_1_normalize_period"])
        cfg = self.cfg
        covs = [c for c in (cfg.covariate_cols or []) if c in self.df.columns]
        if not covs:
            self.df_fine = None
            self._mark("step_1_collapse_covariates")
            self._done("step_1_collapse_covariates", "sin covariables declaradas")
            return
        dc, pc, rc = self.dimension_cols, cfg.period_col, cfg.dataset_role_col
        # normalizar incompletitud SOLO en la copia de trabajo (df_raw intacto)
        for c in covs:
            self.df[c] = self.df[c].fillna("no_info").replace("", "no_info").astype(str)
        fine = self.df.copy()
        fine["fu_id"] = fine[dc].astype(str).agg("|".join, axis=1)
        fine["comb_id"] = fine[covs].astype(str).agg("|".join, axis=1)
        self.df_fine = fine
        n_fino = len(fine)
        sumcols = [c for c in (cfg.pipeline_units_col, cfg.pipeline_usd_col,
                               cfg.renewed_units_col, cfg.renewed_usd_col,
                               cfg.reacq_units_col, cfg.reacq_usd_col)
                   if c and c in fine.columns]
        conserva_pre = {c: float(fine[c].sum()) for c in sumcols}
        keys = dc + [pc]
        otras = [c for c in (rc, cfg.flag_time_series_col)
                 if c and c in fine.columns]
        agg = {c: "sum" for c in sumcols} | {c: "first" for c in otras}
        chk = fine.groupby(keys, observed=True)[rc].nunique()
        if (chk > 1).any():
            self._log(f"    ⚠ {int((chk > 1).sum())} grupos (FU,mes) con roles "
                      f"mezclados en el raw; se toma el primero.")
        vista = fine.groupby(keys, observed=True, as_index=False).agg(agg)
        # AUVs recalculadas del agregado (coherencia: Σusd/Σunits)
        pares = [(cfg.pipeline_usd_col, cfg.pipeline_units_col, "TR_AUV"),
                 (cfg.renewed_usd_col, cfg.renewed_units_col, "REN_AUV"),
                 (cfg.reacq_usd_col, cfg.reacq_units_col, "ReAC_AUV")]
        for usd, uni, auv in pares:
            if usd in vista.columns and uni in vista.columns and auv in fine.columns:
                vista[auv] = np.where(vista[uni] > 0, vista[usd] / vista[uni], 0.0)
        for c, v in conserva_pre.items():
            v2 = float(vista[c].sum())
            if abs(v - v2) > max(1e-6 * abs(v), 1e-6):
                raise AssertionError(f"collapse: pipeline/valor NO conservado en {c}: {v} → {v2}")
        self._log(f"  step_1_collapse_covariates: {n_fino:,} filas finas → "
                  f"{len(vista):,} filas FU-mes (covariables: {covs})")
        self._log(f"    combinaciones medias por (FU,mes): {n_fino / max(len(vista),1):.2f} | "
                  f"pipeline/valor conservado en {len(sumcols)} métricas ✓ | tabla fina en self.df_fine")
        self.df = vista
        self._mark("step_1_collapse_covariates",
                   metadata={"covs": covs, "n_fino": n_fino, "n_vista": len(vista)})
        self._done("step_1_collapse_covariates",
                   f"vista FU: {n_fino:,}→{len(vista):,} filas; pipeline/valor conservado")

    def step_3_report_covariate_value(self, min_n=200) -> None:
        """FASE 0 (solo informa): cobertura mensual de las covariables (% de
        unidades con dato vs no_info) y tabla de FACTORES EMPÍRICOS de uplift
        por COMBINACIÓN observada en la historia: uplift = AUV_renovado /
        AUV_pipeline del agregado de la combinación. Con soporte; las
        combinaciones con n<min_n se marcan (candidatas a fallback marginal)."""
        self._require("step_3_report_covariate_value", ["step_1_collapse_covariates"])
        cfg = self.cfg
        if self.df_fine is None:
            self._mark("step_3_report_covariate_value")
            self._done("step_3_report_covariate_value", "sin covariables")
            return
        covs = self.step_metadata["step_1_collapse_covariates"]["covs"]
        pc, rc = cfg.period_col, cfg.dataset_role_col
        pu, pus = cfg.pipeline_units_col, cfg.pipeline_usd_col
        ru, rus = cfg.renewed_units_col, cfg.renewed_usd_col
        hist = self.df_fine[self.df_fine[rc].isin(cfg.history_roles)]
        tot_m = hist.groupby(pc)[pu].sum()
        self._log(f"  step_3_report_covariate_value: cobertura mensual (% unidades con dato):")
        for c in covs:
            con = hist[hist[c] != "no_info"].groupby(pc)[pu].sum()
            pct = (100 * con.reindex(tot_m.index).fillna(0) / tot_m)
            self._log(f"    {c}: media {pct.mean():.1f}% | primer mes con dato: "
                      f"{(pct[pct > 0].index.min() if (pct > 0).any() else '—')} | "
                      f"último: {pct.iloc[-1]:.1f}%")
        g = hist.groupby("comb_id", observed=True)
        t = g.agg(n=(pu, "sum"), tr_usd=(pus, "sum"),
                  ren_u=(ru, "sum"), ren_usd=(rus, "sum"))
        t = t[t["n"] > 0]
        auv_p = t["tr_usd"] / t["n"]
        auv_r = np.where(t["ren_u"] > 0, t["ren_usd"] / t["ren_u"], np.nan)
        t["uplift"] = auv_r / auv_p
        t = t.sort_values("n", ascending=False)
        self._log(f"    FACTORES EMPÍRICOS por combinación ({'|'.join(covs)}) — "
                  f"historia, {len(t)} combinaciones:")
        self._log(f"      {'combinación':<34} {'n(units)':>9}  {'uplift':>7}  soporte")
        for cid, r in t.head(15).iterrows():
            up = f"×{r['uplift']:.2f}" if np.isfinite(r["uplift"]) else "  —  "
            flag = "ok" if r["n"] >= min_n else f"<{min_n}→fallback"
            self._log(f"      {str(cid)[:34]:<34} {int(r['n']):>9,}  {up:>7}  {flag}")
        self._mark("step_3_report_covariate_value")
        self._done("step_3_report_covariate_value",
                   f"{len(t)} combinaciones; factores empíricos calculados")


    def step_1_derive_roles_from_period(self) -> None:
        """Asigna roles train/test/projection a partir de pending_date.
          projection = desde pending_date en adelante (mes incluido)
          test       = los test_months meses inmediatamente anteriores
          train      = todo lo anterior
        Con use_raw_roles=True solo DIAGNOSTICA (respeta la columna del raw)."""
        self._require("step_1_derive_roles_from_period", ["step_1_normalize_period"])
        cfg, df = self.cfg, self.df
        rc, pc = cfg.dataset_role_col, cfg.period_col
        antes = df[rc].astype(str).value_counts(dropna=False).to_dict()
        self._log(f"  step_1_derive_roles_from_period: roles en el raw ANTES:\n{self._fmt_dict(antes)}")
        if not df[rc].isin(cfg.history_roles).any():
            self._log(f"    ⚠ NINGUNA fila casa con history_roles={cfg.history_roles}: "
                      f"sin historia, todo caería a heurística/shrink. ¿Valores o "
                      f"columna de roles distintos a los esperados?")
        if cfg.use_raw_roles:
            self._mark("step_1_derive_roles_from_period", metadata={"derived": False, "antes": antes})
            self._done("step_1_derive_roles_from_period",
                       f"use_raw_roles=True — se respeta la columna '{rc}' del raw")
            return
        pdate = (pd.Period(str(cfg.pending_date)[:7], freq="M")
                 if cfg.pending_date else pd.Period(pd.Timestamp.now(), freq="M"))
        t_months = cfg.test_months
        t0 = pdate - t_months                # primer mes de test
        t1 = pdate - 1                       # último mes de test
        per = df[pc]
        df[rc] = np.where(per >= pdate, "projection",
                          np.where(per >= t0, "test", "train"))
        despues = df[rc].value_counts().to_dict()
        self._log(f"    pending_date={pdate} | test_months={t_months}")
        self._log(f"    → train ≤ {t0 - 1} | test = {t0}..{t1} | projection ≥ {pdate}")
        self._log(f"    roles DESPUÉS:\n{self._fmt_dict(despues)}")
        for r in ("train", "test", "projection"):
            if despues.get(r, 0) == 0:
                self._log(f"    ⚠ rol '{r}' quedó VACÍO: revisa pending_date/ventana del extract.")
        self.df = df
        self._mark("step_1_derive_roles_from_period",
                   metadata={"derived": True, "pending_date": str(pdate),
                             "test": [str(t0), str(t1)], "antes": antes, "despues": despues})
        self._done("step_1_derive_roles_from_period",
                   f"roles derivados: pending={pdate}, test={t0}..{t1}")

    def step_1_add_universe(self) -> None:
        """Añade step_1_universe (normal | time_series): cómo se predice la fila."""
        self._require("step_1_add_universe", ["step_0_validate_input"])
        cfg = self.cfg
        is_normal = self.df[cfg.flag_time_series_col] == cfg.flag_time_series_normal_value
        self.df["step_1_universe"] = np.where(is_normal, "normal", "time_series")
        self._mark("step_1_add_universe")
        n_norm = int(is_normal.sum())
        self._done("step_1_add_universe",
                   f"columna step_1_universe (normal={n_norm:,}, ts={len(self.df)-n_norm:,})")

    def step_1_add_coverage_pattern(self) -> None:
        """Añade step_1_coverage_pattern (7 categorías por roles cubiertos)."""
        self._require("step_1_add_coverage_pattern", ["step_1_add_universe"])
        cfg = self.cfg
        real = (self.df[self.df.get("step_1_synthetic", 0) == 0]
                if "step_1_synthetic" in self.df.columns else self.df)
        presence = (real.groupby(self.dimension_cols)[cfg.dataset_role_col]
                    .apply(lambda x: frozenset(x.unique())))

        def _label(roles):
            parts = [r for r in ("train", "test", "projection") if r in roles]
            if len(parts) == 3: return "train_test_projection"
            if len(parts) == 1: return f"{parts[0]}_only"
            return "_".join(parts) if parts else "unknown"

        pmap = presence.apply(_label).to_dict()
        keys = (self.df[self.dimension_cols].apply(tuple, axis=1)
                if len(self.dimension_cols) > 1 else self.df[self.dimension_cols[0]])
        self.df["step_1_coverage_pattern"] = keys.map(pmap).fillna("unknown")
        self._mark("step_1_add_coverage_pattern")
        vc = self.df.drop_duplicates(self.dimension_cols)["step_1_coverage_pattern"].value_counts().to_dict()
        self._done("step_1_add_coverage_pattern", f"columna step_1_coverage_pattern:\n{self._fmt_dict(vc)}")

    def step_1_add_forecast_route(self) -> None:
        """Añade step_1_forecast_route (veredicto de cobertura):
        no_impact (sin projection) | heuristic (projection sin train/test) |
        trainable (projection con historia)."""
        self._require("step_1_add_forecast_route", ["step_1_add_coverage_pattern"])
        cov = self.df["step_1_coverage_pattern"]
        no_proj = cov.isin(("train_only", "test_only", "train_test"))
        only_proj = (cov == "projection_only")
        route = np.where(no_proj, "no_impact", np.where(only_proj, "heuristic", "trainable"))
        route = np.where(cov == "unknown", "trainable", route)
        self.df["step_1_forecast_route"] = route
        self._mark("step_1_add_forecast_route")
        cfg = self.cfg
        vc = self.df.drop_duplicates(self.dimension_cols)["step_1_forecast_route"].value_counts().to_dict()
        self._done("step_1_add_forecast_route", f"columna step_1_forecast_route:\n{self._fmt_dict(vc)}")

    def step_1_report_money_by_route(self) -> dict:
        """Pinta el dinero a predecir por ruta (pipeline_usd de projection del año
        en curso, agregado por forecast_route). La magnitud del problema."""
        self._require("step_1_report_money_by_route", ["step_1_add_forecast_route"])
        cfg = self.cfg
        cy = _derive_current_year(self.df, cfg)
        normal = self.df[self.df["step_1_universe"] == "normal"]
        proj = normal[normal[cfg.dataset_role_col] == "projection"]
        if cy is not None:
            proj = proj[proj[cfg.period_col].apply(lambda x: pd.notna(x) and x.year == cy)]
        money = proj.groupby("step_1_forecast_route")[cfg.pipeline_usd_col].sum().to_dict()
        n_fs = normal.drop_duplicates(self.dimension_cols)["step_1_forecast_route"].value_counts().to_dict()
        total = float(sum(money.values()))
        self._log(f"  step_1_report_money_by_route: dinero a predecir (projection {cy}): ${total:,.0f}")
        out = {}
        for r in ["trainable", "heuristic", "no_impact", "unknown"]:
            usd, n = float(money.get(r, 0.0)), int(n_fs.get(r, 0))
            if n == 0 and usd == 0:
                continue
            pct = 100 * usd / total if total else 0
            w = " (no_impact debería ser 0)" if (r == "no_impact" and usd > 0) else ""
            self._log(f"    {r}: ${usd:,.0f} ({pct:.1f}%) {n} FS{w}")
            out[r] = {"usd_proj": round(usd, 0), "usd_pct": round(pct, 1), "n_fs": n}
        self._mark("step_1_report_money_by_route", metadata={"money_by_route": out,
                                                              "total_usd_proj": round(total, 0)})
        return out

    def step_1_drop_no_impact(self) -> None:
        """Borra del df las FS no_impact (sin projection): ya no se usan."""
        self._require("step_1_drop_no_impact", ["step_1_report_money_by_route"])
        before = len(self.df)
        mask_drop = self.df["step_1_forecast_route"] == "no_impact"
        n_fs = self.df[mask_drop].drop_duplicates(self.dimension_cols).shape[0]
        self.df = self.df[~mask_drop].reset_index(drop=True)
        self._mark("step_1_drop_no_impact")
        self._done("step_1_drop_no_impact",
                   f"eliminadas {n_fs:,} FS no_impact ({before - len(self.df):,} filas)")


    def step_1_fill_gaps(self) -> None:
        """Rellena huecos con filas sintéticas SOLO en series trainable y SOLO
        dentro del tramo de historia (train/test), no hasta projection."""
        self._require("step_1_fill_gaps", ["step_1_add_forecast_route",
                                            "step_1_normalize_period"])
        cfg = self.cfg
        if "step_1_synthetic" not in self.df.columns:
            self.df["step_1_synthetic"] = 0
        dc, pc, rc = self.dimension_cols, cfg.period_col, cfg.dataset_role_col
        new_rows = []
        for combo, group in self.df.groupby(dc):
            if group["step_1_forecast_route"].iloc[0] != "trainable":
                continue
            g = group.sort_values(pc)
            hist = g[g[rc].isin(cfg.history_roles)]
            if len(hist) == 0:
                continue
            observed = set(hist[pc].values)
            full = pd.period_range(hist[pc].min(), hist[pc].max(), freq="M")
            missing = sorted(set(full) - observed)
            if not missing:
                continue
            dimvals = dict(zip(dc, combo)) if isinstance(combo, tuple) else {dc[0]: combo}
            real = list(g[pc].values)
            p2r = dict(zip(real, list(g[rc].values)))
            for period in missing:
                prior = [p for p in real if p < period]
                role = p2r[max(prior)] if prior else "train"
                row = {pc: period, rc: role, "step_1_synthetic": 1,
                       "step_1_universe": "normal",
                       "step_1_forecast_route": "trainable",
                       "step_1_coverage_pattern": g["step_1_coverage_pattern"].iloc[0],
                       cfg.flag_time_series_col: cfg.flag_time_series_normal_value}
                row.update(dimvals)
                for c in [cfg.pipeline_units_col, cfg.pipeline_usd_col,
                          cfg.renewed_units_col, cfg.renewed_usd_col]:
                    row[c] = 0
                if self._has_reacq:
                    row[cfg.reacq_units_col] = 0
                    row[cfg.reacq_usd_col] = 0
                new_rows.append(row)
        if not new_rows:
            self._mark("step_1_fill_gaps")
            self._done("step_1_fill_gaps", "sin huecos")
            return
        syn = pd.DataFrame(new_rows)
        syn[pc] = pd.PeriodIndex(syn[pc], freq="M")
        for c in self.df.columns:
            if c not in syn.columns:
                syn[c] = np.nan
        self.df = pd.concat([self.df, syn[self.df.columns]], ignore_index=True)
        self.df = self.df.sort_values([pc] + [c for c in dc]).reset_index(drop=True)
        n_series = len(set(tuple(r[d] for d in dc) for r in new_rows))
        # porcentaje que representan los huecos en las series que QUEDAN (trainable):
        rc = cfg.dataset_role_col
        hist_tr = self.df[(self.df["step_1_forecast_route"] == "trainable")
                          & (self.df["step_1_universe"] == "normal")
                          & self.df[rc].isin(cfg.history_roles)]
        total_meses = int(len(hist_tr))                      # meses de historia incl. huecos
        n_tr_series = int(hist_tr.groupby([c for c in dc]).ngroups)
        pct_meses = 100 * len(new_rows) / max(total_meses, 1)
        pct_series = 100 * n_series / max(n_tr_series, 1)
        self._log(f"    intermitencia: los huecos son el {pct_meses:.1f}% de los meses de "
                  f"historia de las series trainable; afectan al {pct_series:.1f}% de las series.")
        self._mark("step_1_fill_gaps")
        self._done("step_1_fill_gaps",
                   f"{len(new_rows):,} huecos ({pct_meses:.1f}% de los meses) en "
                   f"{n_series:,} series ({pct_series:.1f}% de las trainable)")

    def step_2_report_gap_examples(self, top_n=10, min_usd=10000, dims=None) -> None:
        """¿POR QUÉ tienen huecos las series grandes? Timeline mensual de las
        top_n series por $ a predecir con algún mes-hueco. El parámetro dims
        fija el GRANO de la serie y se DECLARA en el log: None = grano FU
        COMPLETO (todas las dimensiones, donde viven los huecos); 'mandatory'
        = business_mandatory_dims (agrega el resto: comparable con tu SQL de
        reporting); o lista explícita. A grano agregado, un mes es HUECO solo
        si TODAS sus hijas son sintéticas ese mes."""
        self._require("step_2_report_gap_examples",
                      ["step_1_fill_gaps", "step_2_build_support"])
        cfg, df = self.cfg, self.df
        pu, ru, pc, rc = (cfg.pipeline_units_col, cfg.renewed_units_col,
                          cfg.period_col, cfg.dataset_role_col)
        if dims is None:
            kd, grano = list(self.dimension_cols), "FU COMPLETO"
        elif dims == "mandatory":
            kd, grano = list(cfg.business_mandatory_dims), "MANDATORY"
        else:
            kd, grano = list(dims), "personalizado"
        desconocidas = [d for d in kd if d not in self.dimension_cols]
        if desconocidas:
            raise ValueError(f"gap_examples: dims desconocidas {desconocidas}")
        tr = ((df["step_1_forecast_route"] == "trainable")
              & (df["step_1_universe"] == "normal")
              & df[rc].isin(cfg.history_roles))
        hist = df[tr].copy()
        syn_col = "step_1_synthetic"
        if syn_col not in hist.columns or not (hist[syn_col] == 1).any():
            self._mark("step_2_report_gap_examples")
            self._done("step_2_report_gap_examples", "sin huecos que inspeccionar")
            return
        hist["_dk"] = hist[kd].astype(str).agg("|".join, axis=1)
        # $ a predecir por serie = SUMA del usd_proj de sus FS hijas
        fs_usd = (df.drop_duplicates("step_2_fs_id")
                  .set_index("step_2_fs_id")["step_2_usd_proj"].astype(float))
        uni = hist.drop_duplicates("step_2_fs_id")[["step_2_fs_id", "_dk"]].copy()
        uni["_usd"] = uni["step_2_fs_id"].map(fs_usd).fillna(0)
        k_usd = uni.groupby("_dk")["_usd"].sum().sort_values(ascending=False)
        # hueco al grano elegido: mes donde TODAS las filas de la serie son sintéticas
        es_real = 1 - hist[syn_col].fillna(0).astype(int)
        mes_real = es_real.groupby([hist["_dk"], hist[pc]]).max()
        con_hueco = set(mes_real[mes_real == 0].index.get_level_values(0))
        k_usd = k_usd[k_usd.index.isin(con_hueco)]
        sobre = k_usd[k_usd >= min_usd]
        if len(sobre):
            k_usd = sobre.head(top_n)
        else:
            self._log(f"    (ninguna serie con huecos alcanza ${min_usd:,.0f} a grano "
                      f"{grano} — muestro las top {top_n} por $ igualmente)")
            k_usd = k_usd.head(top_n)
        self._log(f"  step_2_report_gap_examples: GRANO de la serie = {grano} "
                  f"({len(kd)} dims): {kd}")
        self._log(f"    TOP {len(k_usd)} series CON HUECOS (por $ a predecir, "
                  f"mín ${min_usd:,.0f}):")
        for dk, u in k_usd.items():
            sub = hist[hist["_dk"] == dk]
            agg = (sub.assign(_real=1 - sub[syn_col].fillna(0).astype(int))
                   .groupby(pc).agg(pipeline=(pu, "sum"), renovadas=(ru, "sum"),
                                    reales=("_real", "max")).sort_index())
            vals = str(dk).split("|")
            self._log(f"\n    serie (clave completa del grano {grano}, replicable en SQL):")
            for i in range(0, len(kd), 4):
                chunk = " | ".join(f"{kd[j]}={vals[j]}"
                                   for j in range(i, min(i + 4, len(kd))))
                self._log(f"      {chunk}")
            n_g = int((agg["reales"] == 0).sum())
            self._log(f"    $ a predecir: ${u:,.0f} | meses: {len(agg)} | huecos: {n_g}")
            self._log(f"    {'mes':>10}  {'pipeline':>9}  {'renovadas':>9}  {'estado':>8}")
            for p, r in agg.iterrows():
                estado = "HUECO" if int(r["reales"]) == 0 else "real"
                self._log(f"    {str(p):>10}  {int(r['pipeline']):>9,}  "
                          f"{int(r['renovadas']):>9,}  {estado:>8}")
        # FRONTERA + distribución mensual: SIEMPRE al grano FU completo
        # (los sintéticos viven ahí, sea cual sea el grano de display)
        hist["_dk_full"] = hist[self.dimension_cols].astype(str).agg("|".join, axis=1)
        first_real = hist[hist[syn_col] == 0].groupby("_dk_full")[pc].min()
        gaps_df = hist[hist[syn_col] == 1][["_dk_full", pc]]
        if len(gaps_df):
            off = (pd.PeriodIndex(gaps_df[pc]).asi8
                   - pd.PeriodIndex(gaps_df["_dk_full"].map(first_real)).asi8)
            pct_early = 100 * float((off <= 3).mean())
            tmp = pd.DataFrame({"k": gaps_df["_dk_full"].values, "o": off})
            s1 = set(tmp.loc[tmp.o == 1, "k"]); s2 = set(tmp.loc[tmp.o == 2, "k"])
            fragiles = s1 & s2
            n_gapped = gaps_df["_dk_full"].nunique()
            self._log(f"\n    FRONTERA (grano FU completo): {pct_early:.1f}% de los huecos "
                      f"caen en los 3 primeros meses de vida de su serie.")
            self._log(f"    arranques frágiles (1ª observación seguida de ≥2 huecos): "
                      f"{len(fragiles):,} series ({100*len(fragiles)/max(n_gapped,1):.1f}% "
                      f"de las que tienen huecos)")
            self._log(f"    → si estos % son altos Y el pico está al inicio del dataset, "
                  f"los huecos del arranque se limpian en ORIGEN (decisión 2026-06-12).")
            gap_months = gaps_df[pc].value_counts().sort_index()
            self._log(f"\n    HUECOS por mes (cronológico, grano FU completo):")
            for p, c in gap_months.items():
                barra = "#" * min(60, int(60 * c / max(gap_months.max(), 1)))
                self._log(f"      {p}: {c:>6,}  {barra}")
            top3 = gap_months.sort_values(ascending=False).head(3)
            self._log(f"    pico de huecos: " +
                      " | ".join(f"{p} ({c:,})" for p, c in top3.items()))
        self._mark("step_2_report_gap_examples")
        self._done("step_2_report_gap_examples",
                   f"{len(k_usd)} series inspeccionadas a grano {grano}")


    def _table(self, df, title="", col_defs=None, max_rows=20):
        """Pinta una tabla como DataFrame (visualización consistente) con
        definiciones de columnas debajo."""
        if title:
            self._log(f"  {title}")
        for line in df.head(max_rows).to_string().splitlines():
            self._log(f"    {line}")
        if len(df) > max_rows:
            self._log(f"    ... ({len(df) - max_rows} filas más)")
        for c, d in (col_defs or {}).items():
            self._log(f"    col {c}: {d}")




    def step_2_report_no_training_top(self, top=10) -> dict:
        """RIESGO: dinero a predecir SIN NINGUNA evidencia de entrenamiento.
        Series con $ en projection y CERO meses reales en train. En un
        negocio continuo esto es anómalo: candidatas a revisión en origen
        (migración de población entre celdas por dims time-varying, producto
        nuevo, o clave mal informada). Lista el top por $ con la clave
        completa dim=valor para poder buscarlas en el origen."""
        self._require("step_2_report_no_training_top", ["step_2_build_support"])
        cfg, df = self.cfg, self.df
        rc = cfg.dataset_role_col
        real = df[df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0] \
            if "step_1_synthetic" in df.columns else df
        tr_m = (real[(real[rc] == "train") & (real[cfg.pipeline_units_col] > 0)]
                .groupby("step_2_fs_id").size())
        prj = df[(df[rc] == "projection") & df["step_2_fs_id"].notna()]
        usd = prj.groupby("step_2_fs_id")[cfg.pipeline_usd_col].sum()
        meses = prj.groupby("step_2_fs_id")[cfg.period_col].nunique()
        sin = usd[~usd.index.isin(tr_m.index)].sort_values(ascending=False)
        total_prj = max(usd.sum(), 1)
        if len(sin) == 0:
            self._log("    ninguna serie a predecir carece de entrenamiento.")
        else:
            t = pd.DataFrame({"usd_proj": sin,
                              "meses_proj": meses.reindex(sin.index)})
            self._table(t.head(top).round(0),
                        f"TOP {top} series a predecir SIN entrenamiento:",
                        col_defs={"usd_proj": "$ de pipeline en projection de la serie",
                                  "meses_proj": "meses de projection con pipeline"})
            self._log("    claves completas (para buscar en origen):")
            ej = (prj[prj["step_2_fs_id"].isin(sin.head(top).index)]
                  .drop_duplicates("step_2_fs_id").set_index("step_2_fs_id"))
            for fsid in sin.head(top).index:
                if fsid in ej.index:
                    fila = ej.loc[fsid]
                    self._log("      " + " | ".join(
                        f"{d}={fila[d]}" for d in self.dimension_cols))
        pct = 100*float(sin.sum())/total_prj
        self._mark("step_2_report_no_training_top",
                   metadata={"n_series": int(len(sin)),
                             "usd": float(sin.sum()), "pct": pct})
        self._done("step_2_report_no_training_top",
                   f"{len(sin):,} series sin train = ${sin.sum():,.0f} "
                   f"({pct:.1f}% del $ a predecir)")
        return {"n": int(len(sin)), "usd": float(sin.sum()), "pct": pct}

    # ---- utilidad descriptiva (NO es un step; no altera nada) ----
    def describe_portfolio(self) -> None:
        """Describe pipeline, FUs y FSs del df actual: totales mensuales,
        nº de series, soporte, terms/purchase_type/covariables y % no_info.
        Información de referencia (p.ej. para regenerar sintéticos)."""
        cfg, df = self.cfg, self.df
        pc, pu = cfg.period_col, cfg.pipeline_units_col
        print(f"describe_portfolio: {len(df):,} filas | dims={self.dimension_cols}")
        m = df.groupby(pc)[pu].sum()
        print(f"  pipeline mensual: min={m.min():,.0f} p50={m.median():,.0f} "
              f"max={m.max():,.0f} | meses={len(m)} ({m.index.min()}..{m.index.max()})")
        if "step_2_fs_id" in df.columns:
            print(f"  FUs={len(df):,} | FSs={df['step_2_fs_id'].nunique():,} | "
                  f"soporte mediano FS-mes={df[pu].median():,.0f}")
        for c in ("term", "term_level_1", "purchase_type"):
            if c in df.columns:
                print(f"  {c}: {df.groupby(c)[pu].sum().to_dict()}")
        covs = getattr(cfg, "covariate_cols", []) or []
        src_df = self.df_fine if self.df_fine is not None else df
        for c in covs:
            if c in src_df.columns:
                ni = (src_df[c].astype(str) == "no_info")
                print(f"  covariable {c}: niveles={src_df[c].nunique()} | "
                      f"no_info={100*src_df.loc[ni, pu].sum()/max(src_df[pu].sum(),1):.0f}% del pipeline")

    # alias de compatibilidad (nombre viejo)
    def step_1_report_gap_examples(self, **kw):
        return self.step_2_report_gap_examples(**kw)

    def step_1_report_density(self, thresholds=(30, 100, 200, 500)) -> dict:
        """VALORACIÓN 2 — densidad de las FU por sample size.

        El soporte de una FU (mes) es el volumen de pipeline: el denominador
        binomial de la tasa que vamos a predecir. Se mide en train/test de las
        series trainable, INCLUYENDO los huecos sintéticos (el mes vacío es real:
        penaliza la intermitencia). Resume por FS —agrupando por dimensiones, sin
        necesidad de fs_id— qué proporción de meses pasa cada umbral, el soporte
        mediano y el peor mes. No borra ni toca tasas: clasifica para mejorar la
        capacidad predictiva. El cruce con el dinero de projection (usd_proj) es
        de capa FS y se hace en step_2; aquí el peso de dinero es el histórico.

        Materializa columnas FS (broadcast a las filas de la serie):
        step_1_fs_density_median, _min, _prop_ge_{umbral}. NaN fuera de las
        series trainable (heurísticas y demás quedan fuera, como en el resto)."""
        self._require("step_1_report_density",
                      ["step_1_fill_gaps", "step_1_add_forecast_route"])
        cfg, df = self.cfg, self.df
        pu_col, pusd_col, rc = cfg.pipeline_units_col, cfg.pipeline_usd_col, cfg.dataset_role_col
        dims = self.dimension_cols
        thr = sorted(int(t) for t in thresholds)
        noisy = thr[0]  # umbral de "mes ruidoso" = el más bajo (típico 30)

        mask = ((df["step_1_universe"] == "normal")
                & (df["step_1_forecast_route"] == "trainable")
                & (df[rc].isin(cfg.history_roles)))
        sub = df.loc[mask].copy()
        if len(sub) == 0:
            self._mark("step_1_report_density", metadata={"empty": True})
            self._done("step_1_report_density", "sin FU trainable en train/test")
            return {}
        n = sub[pu_col].astype(float).fillna(0.0)
        usd = sub[pusd_col].astype(float).fillna(0.0)
        total_usd = float(usd.sum())

        # ---- distribución de las FU por nivel de soporte ----
        edges = [-np.inf] + thr + [np.inf]
        labels = ([f"<{thr[0]}"]
                  + [f"{thr[i]}-{thr[i+1]}" for i in range(len(thr) - 1)]
                  + [f">={thr[-1]}"])
        bucket = pd.cut(n, bins=edges, labels=labels, right=False)
        fu_by_level = {}
        self._log(f"  step_1_report_density: densidad de las FU por sample size "
                  f"(train/test, trainable, incl. huecos)")
        self._log(f"    FU={len(sub):,} | $hist=${total_usd:,.0f} | dims={len(dims)}")
        self._log(f"    nivel de soporte (n=unidades) | FU | %FU | %$hist")
        for lab in labels:
            m = (bucket == lab).values
            cnt = int(m.sum())
            pct_fu = 100 * cnt / len(sub)
            pct_usd = 100 * float(usd[m].sum()) / total_usd if total_usd else 0.0
            fu_by_level[lab] = {"n_fu": cnt, "pct_fu": round(pct_fu, 1),
                                "pct_usd_hist": round(pct_usd, 1)}
            self._log(f"      {lab:>10} | {cnt:>7,} | {pct_fu:5.1f}% | {pct_usd:5.1f}%")

        # ---- resumen por FS (agrupando por dimensiones, sin fs_id) ----
        sub["_n"] = n.values
        g = sub.groupby(dims, sort=False)["_n"]
        median = g.median()
        minimum = g.min()
        prop_ge = {t: g.apply(lambda s, t=t: float((s >= t).mean())) for t in thr}
        n_fs = int(median.shape[0])
        prop_noisy_series = 1.0 - prop_ge[noisy]
        all_noisy = int((prop_ge[noisy] == 0.0).sum())
        median_of_medians = float(median.median())
        self._log(f"    por FS (series trainable): N={n_fs:,}")
        self._log(f"      soporte mediano de la serie — mediana del portfolio: {median_of_medians:,.0f}")
        self._log(f"      series con TODOS los meses ruidosos (<{noisy}): {all_noisy:,} ({100*all_noisy/n_fs:.1f}%)")
        for t in thr:
            share = float((prop_ge[t] >= 0.8).mean())
            self._log(f"      series con >=80% de meses >={t}: {100*share:.1f}%")

        # ---- materializar columnas FS (broadcast a filas) ----
        keys = (df[dims].apply(tuple, axis=1) if len(dims) > 1 else df[dims[0]])
        def _mapcol(series):
            d = series.to_dict()
            if len(dims) > 1:
                d = {(k if isinstance(k, tuple) else (k,)): v for k, v in d.items()}
            return keys.map(d)
        df["step_1_fs_density_median"] = _mapcol(median)
        df["step_1_fs_density_min"] = _mapcol(minimum)
        for t in thr:
            df[f"step_1_fs_density_prop_ge_{t}"] = _mapcol(prop_ge[t])
        self.df = df

        meta = {"fu_by_level": fu_by_level, "n_fs": n_fs,
                "median_of_medians": round(median_of_medians, 1),
                "fs_all_noisy": all_noisy, "thresholds": thr,
                "noisy_threshold": noisy}
        self._mark("step_1_report_density", metadata=meta)
        self._done("step_1_report_density",
                   f"densidad medida ({n_fs:,} FS); columnas step_1_fs_density_median, _min, "
                   f"_prop_ge_{{{','.join(map(str, thr))}}}")
        return meta

    def step_1_add_rates(self) -> None:
        """Añade step_1_rate_renewal y step_1_rate_reacq (NaN en projection,
        0 en sintéticas)."""
        self._require("step_1_add_rates", ["step_1_add_universe"])
        cfg, df = self.cfg, self.df
        pu = df[cfg.pipeline_units_col].values.astype(float)
        ru = df[cfg.renewed_units_col].values.astype(float)
        is_proj = df[cfg.dataset_role_col].values == "projection"
        is_syn = (df.get("step_1_synthetic", 0).values == 1
                  if "step_1_synthetic" in df.columns else np.zeros(len(df), bool))
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(pu > 0, ru / pu, np.nan)
            if self._has_reacq:
                rate_re = np.where(pu > 0, df[cfg.reacq_units_col].values.astype(float) / pu, np.nan)
            else:
                rate_re = np.full(len(df), np.nan)
        rate[is_syn] = 0.0; rate_re[is_syn] = 0.0
        rate[is_proj] = np.nan; rate_re[is_proj] = np.nan
        df["step_1_rate_renewal"] = rate
        df["step_1_rate_reacq"] = rate_re
        self._mark("step_1_add_rates")
        self._done("step_1_add_rates", "columnas step_1_rate_renewal, step_1_rate_reacq")

    def step_1_add_auv(self) -> None:
        """Añade step_1_auv_pipeline, step_1_auv_renewed, step_1_auv_uplift_ratio
        (uplift=1 si no renovó nadie, 0 prohibido; NaN en projection)."""
        self._require("step_1_add_auv", ["step_1_add_universe"])
        cfg, df = self.cfg, self.df
        pu = df[cfg.pipeline_units_col].values.astype(float)
        ru = df[cfg.renewed_units_col].values.astype(float)
        pusd = df[cfg.pipeline_usd_col].values.astype(float)
        rusd = df[cfg.renewed_usd_col].values.astype(float)
        is_proj = df[cfg.dataset_role_col].values == "projection"
        is_syn = (df.get("step_1_synthetic", 0).values == 1
                  if "step_1_synthetic" in df.columns else np.zeros(len(df), bool))
        with np.errstate(divide="ignore", invalid="ignore"):
            ap = np.where(pu > 0, pusd / pu, np.nan)
            ar = np.where(ru > 0, rusd / ru, np.nan)
            uplift = np.where(ru > 0, np.where(ap > 0, ar / ap, np.nan), 1.0)
        ap[is_syn] = 0.0; ar[is_syn] = 0.0; uplift[is_syn] = 1.0
        ar[is_proj] = np.nan; uplift[is_proj] = np.nan
        df["step_1_auv_pipeline"] = ap
        df["step_1_auv_renewed"] = ar
        df["step_1_auv_uplift_ratio"] = uplift
        self._mark("step_1_add_auv")
        self._done("step_1_add_auv",
                   "columnas step_1_auv_pipeline, step_1_auv_renewed, step_1_auv_uplift_ratio")

    def step_1_add_synthetic_flag(self) -> None:
        """Añade step_1_fs_with_synthetic_months (1 si la serie tiene imputación)."""
        self._require("step_1_add_synthetic_flag", ["step_1_add_universe"])
        cfg, df = self.cfg, self.df
        df["step_1_fs_with_synthetic_months"] = 0
        if "step_1_synthetic" not in df.columns:
            self._mark("step_1_add_synthetic_flag")
            self._done("step_1_add_synthetic_flag", "columna step_1_fs_with_synthetic_months (0 imputadas)")
            return
        df["step_1_synthetic"] = df["step_1_synthetic"].fillna(0).astype(int)
        nmask = df["step_1_universe"] == "normal"
        nd = df.loc[nmask]
        if len(nd) > 0:
            aff = nd.groupby(self.dimension_cols)["step_1_synthetic"].transform("max").astype(int)
            df.loc[nmask, "step_1_fs_with_synthetic_months"] = aff.values
        n_aff = int(df.drop_duplicates(self.dimension_cols)["step_1_fs_with_synthetic_months"].sum())
        self._mark("step_1_add_synthetic_flag")
        self._done("step_1_add_synthetic_flag",
                   f"columna step_1_fs_with_synthetic_months ({n_aff:,} series con imputación)")

    def step_1_assert_coherence(self) -> None:
        """Valida la fórmula de 4 factores en filas reales train/test; si falla,
        lista las filas a corregir y lanza DataNotReadyError."""
        self._require("step_1_assert_coherence", ["step_1_add_auv"])
        cfg, df = self.cfg, self.df
        normal = df[(df["step_1_universe"] == "normal")
                    & (df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0)
                    & (df[cfg.dataset_role_col].isin(cfg.history_roles))]
        id_cols = [cfg.period_col, cfg.dataset_role_col] + [c for c in self.dimension_cols if c in normal.columns]
        MAX = 30

        def _fmt(mask, col, label):
            bad = normal[mask]
            lines = [f"  ── {len(bad)} filas reales train/test con {label} ──"]
            for _, r in bad[id_cols + [col]].head(MAX).iterrows():
                lines.append("     " + " | ".join(f"{c}={r[c]}" for c in id_cols) + f" → {col}={r[col]}")
            if len(bad) > MAX:
                lines.append(f"     ... y {len(bad)-MAX:,} más")
            return "\n".join(lines)

        pu = normal[cfg.pipeline_units_col].astype(float)
        pusd = normal[cfg.pipeline_usd_col].astype(float)
        blocks = []
        if ((pu <= 0) | pu.isna()).any():
            blocks.append(_fmt(((pu <= 0) | pu.isna()).values, cfg.pipeline_units_col, "volumen nulo o 0"))
        if ((pusd <= 0) | pusd.isna()).any():
            blocks.append(_fmt(((pusd <= 0) | pusd.isna()).values, cfg.pipeline_usd_col, "AUV_pipeline nulo o 0"))
        if "step_1_auv_uplift_ratio" in df.columns and (normal["step_1_auv_uplift_ratio"] == 0).any():
            blocks.append(_fmt((normal["step_1_auv_uplift_ratio"] == 0).values,
                               "step_1_auv_uplift_ratio", "uplift=0 (renovar gratis)"))
        if blocks:
            raise DataNotReadyError(
                "El raw no está preparado para predecir (fórmula de 4 factores). "
                "Filas a corregir:\n\n" + "\n\n".join(blocks))
        self._mark("step_1_assert_coherence")
        self._done("step_1_assert_coherence", "OK")

    # ========================================================
    # FASE B — step_2 (construir Forecast Series)
    # ========================================================
    def step_2_build_identity(self, extra_dims=None) -> None:
        """Añade step_2_fs_id y step_2_parent_fs_ids (identidad y caminos de
        agregación) sobre el universo normal."""
        self._require("step_2_build_identity", ["step_1_assert_coherence"])
        cfg = self.cfg
        fs_dims = _resolve_fs_dims(cfg, self.dimension_cols, extra_dims)
        miss = [d for d in fs_dims if d not in self.df.columns]
        if miss:
            raise ValueError(f"step_2_build_identity: dims ausentes: {miss}")
        df = self.df
        nmask = df["step_1_universe"] == "normal"
        work = df[nmask].copy()
        work["step_2_fs_id"] = work[fs_dims].astype(str).agg("|".join, axis=1)
        collapsible = [d for d in fs_dims if "_level_" in d]
        pos = {d: i for i, d in enumerate(fs_dims)}

        def _parents(fid):
            parts = fid.split("|")
            out = {}
            for d in collapsible:
                p = parts.copy(); p[pos[d]] = "*"; out[d] = "|".join(p)
            return json.dumps(out)

        pmap = {fid: _parents(fid) for fid in work["step_2_fs_id"].unique()}
        work["step_2_parent_fs_ids"] = work["step_2_fs_id"].map(pmap)
        for c in ["step_2_fs_id", "step_2_parent_fs_ids"]:
            df.loc[nmask, c] = work[c].values
        self.df = df
        n_fs = work["step_2_fs_id"].nunique()
        n_history_periods = int(df[df[cfg.dataset_role_col].isin(cfg.history_roles)][cfg.period_col].nunique())
        self._mark("step_2_build_identity", metadata={"fs_dims": fs_dims, "n_fs": int(n_fs),
                                                      "n_history_periods": n_history_periods})
        self._done("step_2_build_identity",
                   f"columnas step_2_fs_id, step_2_parent_fs_ids ({n_fs:,} FS, {len(fs_dims)} dims)")

    def step_2_build_support(self) -> None:
        """Añade soporte y economía de cada FS sobre la HISTORIA OBSERVADA
        (cfg.history_roles = train ∪ test): n_avg, usd_avg, usd_proj (dinero a
        predecir, este sí sobre projection), n_periods, history_months,
        first_period, k_share_usd. n_avg/usd_avg incluyen sintéticas."""
        self._require("step_2_build_support", ["step_2_build_identity"])
        cfg, df = self.cfg, self.df
        pu, pusd = cfg.pipeline_units_col, cfg.pipeline_usd_col
        pc, rc = cfg.period_col, cfg.dataset_role_col
        hist = df[df[rc].isin(cfg.history_roles)]
        grp = hist.groupby("step_2_fs_id")
        # agregaciones VECTORIZADAS (evitan apply-lambda que en pandas reciente
        # devuelve objetos 2D y rompe el .map posterior con shape (0, N))
        n_periods = grp[pc].nunique()
        n_sum = grp[pu].sum()
        usd_sum = grp[pusd].sum()
        denom = n_periods.replace(0, 1)
        n_avg = (n_sum / denom).rename("n_avg")
        usd_avg = (usd_sum / denom).rename("usd_avg")
        first = grp[pc].min()
        last = grp[pc].max()
        # history en nº de meses: (último − primero) en meses + 1.
        # Resta elemento a elemento de periodos → offset; .n da el entero.
        def _months_between(a, b):
            try:
                return (a.year - b.year) * 12 + (a.month - b.month) + 1
            except Exception:
                return 1
        history = pd.Series(
            [_months_between(last[i], first[i]) for i in first.index],
            index=first.index, name="history")
        cy = _derive_current_year(df, cfg)
        if cy is not None:
            pcurr = df[(df[rc] == "projection") &
                       (df[pc].apply(lambda x: pd.notna(x) and x.year == cy))]
            usd_proj = pcurr.groupby("step_2_fs_id")[pusd].sum()
        else:
            usd_proj = pd.Series(dtype=float)
        df["step_2_n_avg"] = df["step_2_fs_id"].map(n_avg)
        df["step_2_usd_avg"] = df["step_2_fs_id"].map(usd_avg)
        df["step_2_usd_proj"] = df["step_2_fs_id"].map(usd_proj).fillna(0.0)
        df["step_2_n_periods"] = df["step_2_fs_id"].map(n_periods)
        df["step_2_history_months"] = df["step_2_fs_id"].map(history)
        df["step_2_first_period"] = df["step_2_fs_id"].map(first).astype(str)
        total = usd_avg.sum()
        df["step_2_k_share_usd"] = df["step_2_fs_id"].map(
            (usd_avg / total) if total > 0 else usd_avg * 0)
        self.df = df
        self._mark("step_2_build_support")
        self._done("step_2_build_support",
                   "columnas step_2_n_avg, step_2_usd_avg, step_2_usd_proj, "
                   "step_2_n_periods, step_2_history_months, step_2_first_period, step_2_k_share_usd")

    def step_2_report_fu_audit(self, match: dict, top: int = 6) -> None:
        """AUDITORÍA de una combinación PARCIAL de dims (las de tu consulta SQL).
        Compara el agregado mensual del RAW intacto vs el procesado (¿se borró
        algo?), lista cuántas FUs FINAS conviven bajo ese paraguas con su
        presencia mes a mes, y señala qué dims difieren entre las dos mayores
        (la dim que mueve población entre celdas). Para investigar huecos."""
        self._require("step_2_report_fu_audit", ["step_2_build_support"])
        cfg = self.cfg
        pc, pu, ru = cfg.period_col, cfg.pipeline_units_col, cfg.renewed_units_col
        dc = self.dimension_cols
        bad = [k for k in match if k not in dc]
        if bad:
            raise ValueError(f"fu_audit: dims desconocidas {bad}; dims = {dc}")
        def _f(d):
            m = pd.Series(True, index=d.index)
            for k, v in match.items():
                m &= d[k].astype(str) == str(v)
            return d[m]
        self._log(f"  step_2_report_fu_audit: paraguas {match}")
        raw = self.df_raw
        agg_raw = None
        if raw is not None:
            r = _f(raw).copy()
            r["_p"] = pd.PeriodIndex(r[pc].astype(str), freq="M")
            agg_raw = r.groupby("_p")[[pu, ru]].sum().sort_index()
            self._log(f"    RAW intacto: {len(r):,} filas | agregado mensual "
                      f"(comparar 1:1 con tu SQL):")
            self._log(f"      {'mes':>8}  {'pipeline':>9}  {'renovadas':>9}")
            for p, row in agg_raw.iterrows():
                self._log(f"      {str(p):>8}  {int(row[pu]):>9,}  {int(row[ru]):>9,}")
        d = _f(self.df)
        d = d[d.get("step_1_synthetic", 0).fillna(0).astype(int) == 0]
        if agg_raw is not None:
            delta = float(agg_raw[pu].sum()) - float(d[pu].sum())
            self._log(f"    PROCESADO: {len(d):,} filas | Δ pipeline RAW−PROCESADO = "
                      f"{delta:,.0f} (≠0 ⇒ series no_impact borradas bajo el paraguas)")
        hist = d[d[cfg.dataset_role_col].isin(cfg.history_roles)].copy()
        if not len(hist):
            self._mark("step_2_report_fu_audit")
            self._done("step_2_report_fu_audit", "sin historia bajo el paraguas")
            return
        hist["_dk"] = hist[dc].astype(str).agg("|".join, axis=1)
        vol = hist.groupby("_dk")[pu].sum().sort_values(ascending=False)
        meses = sorted(hist[pc].unique())
        pres = hist.groupby("_dk")[pc].apply(set)
        self._log(f"    FUs FINAS con historia bajo el paraguas: {len(vol):,} | "
                  f"ventana {meses[0]}..{meses[-1]} ({len(meses)} meses)")
        self._log(f"    presencia mensual de las top {min(top, len(vol))} (#=dato, ·=nada):")
        for dk in vol.head(top).index:
            line = "".join("#" if m in pres[dk] else "·" for m in meses)
            self._log(f"      [{line}] {int(vol[dk]):>9,} u")
        if len(vol) >= 2:
            va = vol.index[0].split("|"); vb = vol.index[1].split("|")
            difs = [f"{dc[i]}: '{va[i]}' vs '{vb[i]}'"
                    for i in range(len(dc)) if va[i] != vb[i]]
            self._log(f"    dims que SEPARAN las dos mayores → {difs}")
            self._log(f"    (si sus presencias se alternan en el tiempo, esa dim está "
                      f"MIGRANDO la población entre celdas: huecos estructurales, no borrado)")
        self._mark("step_2_report_fu_audit")
        self._done("step_2_report_fu_audit",
                   f"{len(vol):,} FUs finas bajo el paraguas; ver Δ y presencia")

    def step_2_collapse_signal_support(self, support_floor=30) -> dict:
        """COLAPSO NIVEL 1 — ganar soporte uniendo señales del MISMO SIGNO.
        Cuando una forecast series con señal time-varying activa (dormant,
        softcancel...) tiene soporte bajo, se une con sus pares del mismo
        signo que COMPARTEN el resto de dimensiones. La unión se marca en una
        columna nueva `step_2_grupo_colapso` con la CONCATENACIÓN de las
        señales activas (p.ej. 'dormant+softcancel') — así no se pierde traza:
        el valor dice qué se agregó. Las series no afectadas reciben
        'no_agrupa' y CONSERVAN sus columnas originales.
        Un solo movimiento, a prueba: se mide el soporte antes/después.
        El signo viene de cfg.structural_timevarying_dims (config, no del dato).
        """
        self._require("step_2_collapse_signal_support", ["step_2_build_support"])
        cfg, df = self.cfg, self.df
        pu = cfg.pipeline_units_col
        tv = cfg.structural_timevarying_dims or {}
        tv_cols = [c for c in tv if c in df.columns]
        if not tv_cols:
            df["step_2_grupo_colapso"] = "no_agrupa"
            df["step_2_fs_id_L1"] = df.get("step_2_fs_id")
            self.df = df
            self._mark("step_2_collapse_signal_support")
            self._done("step_2_collapse_signal_support",
                       "sin dims time-varying declaradas: no aplica")
            return {"colapsadas": 0}
        pos_vals = set(cfg.timevarying_positive_values)
        def activa(s):  # serie booleana: señal activa por fila
            return s.isin(pos_vals)
        # signo activo por fila: concatena las señales activas (orden estable)
        signo_por_col = {c: tv[c] for c in tv_cols}
        act = pd.DataFrame({c: activa(df[c]) for c in tv_cols})
        # etiqueta = "col1+col2" de las activas; "" si ninguna activa
        def etiqueta(row):
            on = [c for c in tv_cols if row[c]]
            return "+".join(sorted(on)) if on else ""
        df["_sig_label"] = act.apply(etiqueta, axis=1)
        # soporte por FS (mediana de pipeline en historia real)
        sop = (df[df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0]
               .groupby("step_2_fs_id")[pu].median()
               if "step_2_fs_id" in df.columns else pd.Series(dtype=float))
        df["_sop_fs"] = df["step_2_fs_id"].map(sop) if "step_2_fs_id" in df.columns else np.nan
        # candidata: tiene señal activa Y soporte bajo
        cand = (df["_sig_label"] != "") & (df["_sop_fs"] < support_floor)
        df["step_2_grupo_colapso"] = np.where(cand, df["_sig_label"], "no_agrupa")
        # id de NIVEL 1: si agrupa, reemplaza las columnas de señal por la etiqueta;
        # si no, idéntico al fs_id fino (idempotente)
        dims = self.dimension_cols
        otras = [d for d in dims if d not in tv_cols]
        def id_l1(row):
            if row["step_2_grupo_colapso"] == "no_agrupa":
                return row.get("step_2_fs_id")
            base = "|".join(str(row[d]) for d in otras)
            return f"{base}|SIG={row['step_2_grupo_colapso']}"
        df["step_2_fs_id_L1"] = df.apply(id_l1, axis=1)
        # medición antes/después: soporte mediano de las candidatas
        n_cand = int(df.loc[cand, "step_2_fs_id"].nunique()) if "step_2_fs_id" in df.columns else 0
        sop_antes = float(df.loc[cand, "_sop_fs"].median()) if cand.any() else float("nan")
        sop_desp = float(df[cand].groupby("step_2_fs_id_L1")[pu].median().median()) if cand.any() else float("nan")
        n_grupos = int(df.loc[cand, "step_2_fs_id_L1"].nunique()) if cand.any() else 0
        t = pd.DataFrame({
            "métrica": ["FS candidatas (señal+soporte bajo)", "grupos tras unir",
                        "soporte mediano ANTES", "soporte mediano DESPUÉS"],
            "valor": [n_cand, n_grupos, round(sop_antes, 1), round(sop_desp, 1)]
        }).set_index("métrica")
        self._table(t, "Colapso Nivel 1 (señales del mismo signo):", col_defs={
            "valor": "conteo o soporte mediano (unidades/mes)"})
        if n_cand:
            ej = df.loc[cand, "step_2_grupo_colapso"].value_counts().head(4)
            self._log("    etiquetas de unión (traza conservada):")
            for k, v in ej.items():
                self._log(f"      {k}: {v:,} filas")
        df.drop(columns=["_sig_label", "_sop_fs"], inplace=True)
        self.df = df
        mejora = (sop_desp - sop_antes) if (cand.any() and np.isfinite(sop_antes)) else 0
        self._mark("step_2_collapse_signal_support",
                   metadata={"n_candidatas": n_cand, "n_grupos": n_grupos,
                             "soporte_antes": sop_antes, "soporte_despues": sop_desp})
        self._done("step_2_collapse_signal_support",
                   f"{n_cand:,} FS candidatas → {n_grupos:,} grupos; "
                   f"soporte mediano {sop_antes:.0f}→{sop_desp:.0f}")
        return {"n_candidatas": n_cand, "n_grupos": n_grupos}

    def step_2_report_density_money(self) -> dict:
        """TABLA CANÓNICA dinero × soporte (medición única; la foto del HITO 2
        solo COMPARA antes/después con estos mismos cortes). Reparte el $ a
        predecir por intervalos de soporte mediano de su serie, mide la
        concentración (Lorenz/Gini), cuántas series concentran el 50% del $
        y el dólar mediano (soporte donde cruza el 50% del $ acumulado)."""
        self._require("step_2_report_density_money",
                      ["step_2_build_support", "step_1_report_density"])
        df = self.df
        fs = (df[df["step_1_fs_density_median"].notna() & df["step_2_fs_id"].notna()]
              .drop_duplicates("step_2_fs_id")
              .loc[:, ["step_2_fs_id", "step_2_usd_proj", "step_1_fs_density_median"]]
              .copy())
        fs["step_2_usd_proj"] = fs["step_2_usd_proj"].fillna(0)
        total = fs["step_2_usd_proj"].sum()
        cortes = [0, 30, 100, 200, 500, np.inf]
        et = ["<30", "30-99", "100-199", "200-499", ">=500"]
        fs["_b"] = pd.cut(fs["step_1_fs_density_median"], cortes, right=False, labels=et)
        t = fs.groupby("_b", observed=False).agg(
            n_series=("step_2_fs_id", "count"), usd_proj=("step_2_usd_proj", "sum"))
        t["pct_usd"] = 100*t["usd_proj"]/max(total, 1)
        t["pct_usd_acum"] = t["pct_usd"].cumsum()
        self._table(t.round(1), "Dinero a predecir por intervalo de SOPORTE (canónica):",
                    col_defs={"n_series": "nº de forecast series en el intervalo de soporte",
                              "usd_proj": "$ de projection que vive en esas series",
                              "pct_usd / pct_usd_acum": "% del $ total y acumulado (soporte bajo a alto)"})
        x = np.sort(fs["step_2_usd_proj"].values)
        n = len(x)
        gini = float((2*np.sum(np.arange(1, n+1)*x)/(n*max(x.sum(), 1))) - (n+1)/n) if n else 0.0
        orden = fs.sort_values("step_2_usd_proj", ascending=False)
        n50 = int((orden["step_2_usd_proj"].cumsum() < total*.5).sum()) + 1
        s = fs.sort_values("step_1_fs_density_median")
        cruz = s.loc[s["step_2_usd_proj"].cumsum() >= total/2, "step_1_fs_density_median"]
        usd_med = float(cruz.iloc[0]) if len(cruz) else float("nan")
        self._log(f"    Gini={gini:.3f} | 50% del $ en {n50:,} series | "
                  f"dólar mediano en soporte {usd_med:,.0f}")
        self._mark("step_2_report_density_money",
                   metadata={"gini": gini, "n50": n50, "usd_mediano_soporte": usd_med})
        self._done("step_2_report_density_money",
                   f"Gini {gini:.2f}; 50% del $ en {n50:,} series; dólar mediano soporte {usd_med:,.0f}")
        return {"gini": gini, "n50": n50, "usd_mediano_soporte": usd_med}

    def step_2_report_gap_density_money(self) -> dict:
        """¿Cuánto DINERO vive en series intermitentes? Tabla de $ por nº
        EXACTO de meses de hueco de la serie (0, 1, 2, ...), ordenada por nº
        de meses: se ve de un vistazo dónde se acaba el dinero limpio. Hueco
        = mes sintético (cero legítimo bajo el contrato vigente)."""
        self._require("step_2_report_gap_density_money",
                      ["step_2_build_support", "step_1_fill_gaps"])
        cfg, df = self.cfg, self.df
        rc = cfg.dataset_role_col
        hist = df[(df["step_1_forecast_route"] == "trainable")
                  & (df["step_1_universe"] == "normal")
                  & df[rc].isin(cfg.history_roles) & df["step_2_fs_id"].notna()]
        g = hist.groupby("step_2_fs_id")
        gaps = (g["step_1_synthetic"].sum() if "step_1_synthetic" in hist.columns
                else g.size()*0).astype(int)
        usd = (hist.drop_duplicates("step_2_fs_id")
               .set_index("step_2_fs_id")["step_2_usd_proj"].fillna(0))
        t = pd.DataFrame({"n_gaps": gaps,
                          "usd_proj": usd.reindex(gaps.index).fillna(0)})
        total = max(t["usd_proj"].sum(), 1)
        r = t.groupby("n_gaps").agg(n_series=("usd_proj", "count"),
                                    usd_proj=("usd_proj", "sum")).sort_index()
        r["pct_usd"] = 100*r["usd_proj"]/total
        r["pct_usd_acum"] = r["pct_usd"].cumsum()
        self._table(r.round(1), "Dinero por nº de meses de hueco de la serie:",
                    col_defs={"n_gaps (índice)": "nº exacto de meses de hueco en la historia",
                              "n_series": "series con ese nº de huecos",
                              "usd_proj": "$ a predecir en esas series",
                              "pct_usd_acum": "% del $ acumulado de menos a más huecos"},
                    max_rows=25)
        limpio = float(r.loc[r.index == 0, "pct_usd"].sum()) if 0 in r.index else 0.0
        self._mark("step_2_report_gap_density_money",
                   metadata={"pct_usd_sin_huecos": limpio})
        self._done("step_2_report_gap_density_money",
                   f"{limpio:.1f}% del $ en series SIN huecos")
        return {"pct_usd_sin_huecos": limpio}

    def step_2_report_only_projection_top(self, top=3) -> dict:
        """FORECAST SERIES que SOLO existen en projection (ni train ni test en
        todo el histórico): necesitan heurística pura (no hay nada que
        aprender). Cuenta cuántas SERIES son (no FU) y lista el TOP por $ con
        la clave completa dim=valor — para revisar en 3 líneas si una variable
        concreta no está calculada a futuro y corta la continuidad de esos
        productos. Solo informa; el framework las cubre heredando del grupo."""
        self._require("step_2_report_only_projection_top", ["step_1_derive_roles_from_period"])
        cfg, df = self.cfg, self.df
        rc = cfg.dataset_role_col
        # id de serie AL VUELO (no requiere build_identity): basta el group-by
        # por dimensiones. Si fs_id ya existe persistido, se reutiliza.
        if "step_2_fs_id" in df.columns and df["step_2_fs_id"].notna().any():
            key = df["step_2_fs_id"]
        else:
            key = df[self.dimension_cols].astype(str).agg("|".join, axis=1)
        df = df.assign(_fsk=key)
        real = (df[df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0]
                if "step_1_synthetic" in df.columns else df)
        con_hist = set(real[real[rc].isin(("train", "test"))
                            & (real[cfg.pipeline_units_col] > 0)]["_fsk"])
        prj = df[(df[rc] == "projection") & df["_fsk"].notna()]
        usd = prj.groupby("_fsk")[cfg.pipeline_usd_col].sum()
        solo = usd[~usd.index.isin(con_hist)].sort_values(ascending=False)
        total = max(usd.sum(), 1)
        n_series = int(len(solo))
        if n_series == 0:
            self._log("    ninguna forecast series vive solo en projection.")
        else:
            self._log(f"    forecast series SOLO-projection: {n_series:,} "
                      f"({100*solo.sum()/total:.1f}% del $ a predecir)")
            self._log(f"    TOP {top} por $ (revisar si falta una variable a futuro):")
            ej = (prj[prj["_fsk"].isin(solo.head(top).index)]
                  .drop_duplicates("_fsk").set_index("_fsk"))
            for fsid in solo.head(top).index:
                clave = " | ".join(f"{d}={ej.loc[fsid, d]}" for d in self.dimension_cols) \
                    if fsid in ej.index else str(fsid)
                self._log(f"      ${solo[fsid]:>12,.0f}  {clave}")
        self._mark("step_2_report_only_projection_top",
                   metadata={"n_series": n_series,
                             "pct_usd": 100*float(solo.sum())/total})
        self._done("step_2_report_only_projection_top",
                   f"{n_series:,} forecast series solo-projection "
                   f"({100*solo.sum()/total:.1f}% del $)")
        return {"n_series": n_series, "pct_usd": 100*float(solo.sum())/total}

    def step_2_report_support_profile(self) -> dict:
        """DESCRIPCIÓN del raw por TRAMOS DE SOPORTE + REFERENCIA de error
        binomial. Dos tablas:
          (1) forecast series por tramo de soporte mediano: nº de series, $ a
              predecir, y de la TASA del tramo: promedio, mediana y dispersión
              (desv. típica). Describe la materia prima sin predecir nada.
          (2) referencia de error binomial por nivel de soporte: ±z·√(p(1−p)/n)
              con p=0,5 (peor caso) al 95% — el error que arrastras SÓLO por
              tamaño de muestra, aunque acertaras la probabilidad. Es la regla
              de la moneda hecha tabla: motiva el HITO 2 (aumentar soporte)."""
        self._require("step_2_report_support_profile",
                      ["step_2_build_support", "step_1_report_density", "step_1_add_rates"])
        cfg, df = self.cfg, self.df
        pu, rc = cfg.pipeline_units_col, cfg.dataset_role_col
        # tasa observada por FS (Σren/Σpipe en train+test reales)
        real = df[(df["step_1_forecast_route"] == "trainable")
                  & df[rc].isin(cfg.history_roles)
                  & (df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0)
                  & (df[pu] > 0)]
        g = real.groupby("step_2_fs_id")
        rate_fs = (g[cfg.renewed_units_col].sum()/g[pu].sum()).rename("rate")
        fs = (df[df["step_2_fs_id"].notna()].drop_duplicates("step_2_fs_id")
              .set_index("step_2_fs_id")
              .loc[:, ["step_1_fs_density_median", "step_2_usd_proj"]].copy())
        fs["rate"] = rate_fs.reindex(fs.index)
        fs["sop"] = fs["step_1_fs_density_median"].fillna(0)
        fs["usd"] = fs["step_2_usd_proj"].fillna(0)
        cortes = [0, 30, 100, 200, 500, np.inf]
        et = ["0-29", "30-99", "100-199", "200-499", ">=500"]
        fs["_b"] = pd.cut(fs["sop"], cortes, right=False, labels=et)
        t1 = fs.groupby("_b", observed=False).agg(
            n_series=("rate", "size"), usd_proj=("usd", "sum"),
            tasa_media=("rate", "mean"), tasa_mediana=("rate", "median"),
            tasa_desv=("rate", "std"))
        t1["tasa_media"] = (t1["tasa_media"]*100).round(1)
        t1["tasa_mediana"] = (t1["tasa_mediana"]*100).round(1)
        t1["tasa_desv"] = (t1["tasa_desv"]*100).round(1)
        t1["usd_proj"] = t1["usd_proj"].round(0)
        self._table(t1, "(1) Forecast series por tramo de SOPORTE:", col_defs={
            "(índice)": "tramo de soporte mediano de la serie (unidades/mes)",
            "n_series": "nº de forecast series en el tramo",
            "usd_proj": "$ a predecir que vive en el tramo",
            "tasa_media/mediana": "tasa de renovación del tramo, % (centro)",
            "tasa_desv": "desviación típica de la tasa entre series del tramo, pp (dispersión)"})
        # (2) referencia de error binomial, independiente de los datos
        z = 1.96
        ref = pd.DataFrame({"soporte_n": [30, 100, 200, 500, 1000, 5000]})
        ref["error_pp_95"] = (100*z*np.sqrt(0.5*0.5/ref["soporte_n"])).round(1)
        ref = ref.set_index("soporte_n")
        self._table(ref, "(2) REFERENCIA error binomial (p=0,5, 95% confianza):", col_defs={
            "(índice) soporte_n": "nº de muestras (unidades de pipeline)",
            "error_pp_95": "± error de la tasa en puntos %, SOLO por tamaño de muestra: z·√(p(1−p)/n)"})
        self._log(f"    lectura: con 30 muestras la tasa ya tiene ±{ref.loc[30,'error_pp_95']:.0f}pp "
                  f"de error irreducible; con 500, ±{ref.loc[500,'error_pp_95']:.0f}pp. "
                  f"Por eso el HITO 2 busca aumentar el soporte.")
        # veredicto: cuánto $ vive bajo soporte ruidoso (<100)
        usd_total = max(fs["usd"].sum(), 1)
        usd_ruido = float(fs.loc[fs["sop"] < 100, "usd"].sum())
        pct = 100*usd_ruido/usd_total
        self._mark("step_2_report_support_profile",
                   metadata={"pct_usd_sop_lt100": round(pct, 1)})
        self._done("step_2_report_support_profile",
                   f"{pct:.1f}% del $ en series con soporte<100 (error binomial material)")
        return {"pct_usd_sop_lt100": pct}


    def step_2_report_history_length(self, bands=(6, 12, 18, 24, 36)) -> dict:
        """HISTOGRAMA doble de LONGITUD de historia: nº de series y $ a
        predecir por tramo de meses, con barra proporcional al $. El corte
        >=18 marca quién puede optar a la técnica de serie temporal (EWM) del
        HITO 2; <12 no permite observar un ciclo estacional completo."""
        self._require("step_2_report_history_length", ["step_2_build_support"])
        df = self.df
        fs = (df[(df["step_1_forecast_route"] == "trainable") & df["step_2_fs_id"].notna()]
              .drop_duplicates("step_2_fs_id")
              .loc[:, ["step_2_fs_id", "step_2_history_months", "step_2_usd_proj"]].copy())
        if len(fs) == 0:
            self._mark("step_2_report_history_length", metadata={"empty": True})
            self._done("step_2_report_history_length", "sin FS trainable")
            return {}
        fs["step_2_usd_proj"] = fs["step_2_usd_proj"].fillna(0)
        total = max(fs["step_2_usd_proj"].sum(), 1)
        cortes = [0, *bands, np.inf]
        et = [f"{a}-{b-1}" for a, b in zip(cortes[:-1], cortes[1:-1])] + [f">={bands[-1]}"]
        fs["_b"] = pd.cut(fs["step_2_history_months"], cortes, right=False, labels=et)
        t = fs.groupby("_b", observed=False).agg(
            n_series=("step_2_fs_id", "count"), usd_proj=("step_2_usd_proj", "sum"))
        t["pct_usd"] = 100*t["usd_proj"]/total
        mx = max(t["usd_proj"].max(), 1)
        t["hist_$"] = t["usd_proj"].map(lambda v: "#"*int(30*v/mx))
        self._table(t.round(1), "Histograma de longitud de historia (series y $):",
                    col_defs={"(índice)": "tramo de meses de historia de la serie",
                              "n_series": "nº de series en el tramo",
                              "usd_proj / pct_usd": "$ a predecir y % del total",
                              "hist_$": "barra proporcional al $ del tramo"})
        ts_ok = float(fs.loc[fs["step_2_history_months"] >= 18, "step_2_usd_proj"].sum())
        self._log(f"    elegible para técnica de serie temporal (>=18m): "
                  f"{100*ts_ok/total:.1f}% del $")
        self._mark("step_2_report_history_length")
        self._done("step_2_report_history_length",
                   f"{100*ts_ok/total:.0f}% del $ con >=18m de historia (apto TS)")
        return {"pct_usd_ts": 100*ts_ok/total}

    def _coarse_dims(self, coarse_dims):
        coarse = list(coarse_dims) if coarse_dims else list(self.cfg.business_mandatory_dims)
        coarse = [d for d in coarse if d in self.dimension_cols]
        if not coarse:
            raise RuntimeError(
                "dims gruesas/mandatory vacías: pásalas en coarse_dims o repuebla "
                "cfg.business_mandatory_dims con las dims de reporting tradicionales.")
        return coarse

    def step_2_report_density_mandatory_vs_full(self, coarse_dims=None) -> dict:
        """Compara el soporte mensual (densidad) en dos granos: la agregación
        MANDATORY (gruesa, reporting tradicional) vs el grano FINO (FS). Muestra
        cómo, al desagregar, el dinero se mueve de celdas con mucho soporte a
        celdas con poco — el precio en muestra de evitar Simpson. Ponderado por
        usd_proj. El soporte de cada celda = unidades sumadas por mes (aditivo)."""
        self._require("step_2_report_density_mandatory_vs_full",
                      ["step_2_build_support", "step_1_report_density"])
        coarse = self._coarse_dims(coarse_dims)
        cfg, df = self.cfg, self.df
        pu, pc, rc = cfg.pipeline_units_col, cfg.period_col, cfg.dataset_role_col
        hist = df[(df["step_1_forecast_route"] == "trainable")
                  & (df["step_1_universe"] == "normal")
                  & df[rc].isin(cfg.history_roles)].copy()
        bands = [(">=500", 500, np.inf), ("100-500", 100, 500),
                 ("30-100", 30, 100), ("<30", -np.inf, 30)]

        def _money_by_band(support, usd):
            support = np.asarray(support, float); usd = np.asarray(usd, float)
            tot = float(usd.sum())
            rows = {}
            for lab, lo, hi in bands:
                m = (support >= lo) & (support < hi)
                u = float(usd[m].sum())
                rows[lab] = (u, round(100 * u / tot, 1) if tot else 0.0, int(m.sum()))
            return rows, tot

        # FINO: por FS (soporte mediano ya calculado) y su usd_proj
        fsf = (df[(df["step_1_forecast_route"] == "trainable") & df["step_2_fs_id"].notna()
                  & df["step_1_fs_density_median"].notna()].drop_duplicates("step_2_fs_id"))
        fine_rows, fine_tot = _money_by_band(fsf["step_1_fs_density_median"].values,
                                             fsf["step_2_usd_proj"].values)
        # GRUESO: por celda mandatory, soporte = mediana del soporte mensual (unidades sumadas/mes)
        hist["_ck"] = hist[coarse].astype(str).agg("|".join, axis=1)
        cell_month = hist.groupby(["_ck", pc])[pu].sum()
        coarse_sup = cell_month.groupby(level=0).median()
        fsf2 = fsf.copy(); fsf2["_ck"] = fsf2[coarse].astype(str).agg("|".join, axis=1)
        coarse_usd = fsf2.groupby("_ck")["step_2_usd_proj"].sum()
        idx = coarse_sup.index.union(coarse_usd.index)
        coarse_rows, coarse_tot = _money_by_band(
            coarse_sup.reindex(idx).fillna(0).values,
            coarse_usd.reindex(idx).fillna(0).values)
        self._log(f"  step_2_report_density_mandatory_vs_full: soporte por grano "
                  f"(gruesa={coarse} → {len(idx):,} celdas | fina={len(fsf):,} FS)")
        self._log(f"    {'soporte':>9} | {'%$ GRUESA':>10} | {'%$ FINA':>9}")
        for lab, _, _ in bands:
            self._log(f"    {lab:>9} | {coarse_rows[lab][1]:9.1f}% | {fine_rows[lab][1]:8.1f}%")
        low_fine = fine_rows["<30"][1] + fine_rows["30-100"][1]
        low_coarse = coarse_rows["<30"][1] + coarse_rows["30-100"][1]
        self._log(f"    → dinero en soporte <100: gruesa {low_coarse:.1f}% vs fina {low_fine:.1f}% "
                  f"(el grano fino fragmenta el soporte: ese es el precio).")
        out = {"coarse_dims": coarse, "n_coarse_cells": int(len(idx)), "n_fine_fs": int(len(fsf)),
               "by_band_coarse": {k: {"usd": round(v[0], 0), "pct": v[1], "n": v[2]} for k, v in coarse_rows.items()},
               "by_band_fine": {k: {"usd": round(v[0], 0), "pct": v[1], "n": v[2]} for k, v in fine_rows.items()}}
        self._mark("step_2_report_density_mandatory_vs_full", metadata=out)
        self._done("step_2_report_density_mandatory_vs_full", "densidad gruesa vs fina")
        return out


    def _example_dormant_split(self, min_share=0.15, min_support=100):
        """Busca la celda mandatory donde estratificar por UNA variable muy
        explicativa (dormant) más cambia el forecast, con AMBOS lados
        materiales (cada uno ≥min_share del pipeline de la celda y ≥min_support
        de soporte). Devuelve dict con el cálculo tradicional vs estratificado
        y la diferencia en $, o None si no hay caso limpio."""
        cfg, df = self.cfg, self.df
        pu, ru, pc, rc = (cfg.pipeline_units_col, cfg.renewed_units_col,
                          cfg.period_col, cfg.dataset_role_col)
        split = "dormant" if "dormant" in self.dimension_cols else (
            self.dimension_cols[0] if self.dimension_cols else None)
        if split is None:
            return None
        mdims = list(cfg.business_mandatory_dims)
        real = df[(df["step_1_universe"] == "normal")
                  & df[rc].isin(cfg.history_roles)
                  & (df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0)
                  & (df[pu] > 0)].copy()
        cy = _derive_current_year(df, cfg)
        prj = df[(df[rc] == "projection")
                 & (df[pc].apply(lambda x: pd.notna(x) and x.year == cy))] if cy else df.iloc[:0]
        real["_mk"] = real[mdims].astype(str).agg("|".join, axis=1)
        prj = prj.assign(_mk=prj[mdims].astype(str).agg("|".join, axis=1))
        best = None
        for mk, sub in real.groupby("_mk"):
            lados = sub.groupby(split).agg(pipe=(pu, "sum"), ren=(ru, "sum"))
            if len(lados) < 2:
                continue
            tot = lados["pipe"].sum()
            lados["rate"] = lados["ren"]/lados["pipe"]
            lados["share"] = lados["pipe"]/tot
            mat = lados[(lados["share"] >= min_share) & (lados["pipe"] >= min_support)]
            if len(mat) < 2:                       # AMBOS lados deben pesar
                continue
            gap = float(mat["rate"].max() - mat["rate"].min())
            pj = prj[prj["_mk"] == mk]
            if not len(pj):
                continue
            usd_proj = float(pj[cfg.pipeline_usd_col].sum())
            score = gap * usd_proj                 # diferencia real × dinero
            if best is None or score > best["score"]:
                rbar = float(sub[ru].sum()/sub[pu].sum())
                pj_l = pj.groupby(split).agg(npipe=(pu, "sum"),
                                             usd=(cfg.pipeline_usd_col, "sum"))
                auv = pj[cfg.pipeline_usd_col].sum()/max(pj[pu].sum(), 1)
                trad = float(pj[pu].sum()*rbar*auv)
                strat = float(sum(pj_l.loc[g, "npipe"]*lados.loc[g, "rate"]*auv
                                  for g in pj_l.index if g in lados.index))
                best = dict(score=score, mk=mk, split=split, gap_pp=100*gap,
                            rbar=rbar, lados=lados, pj_l=pj_l, auv=auv,
                            trad=trad, strat=strat, diff=strat-trad,
                            usd_proj=usd_proj)
        return best

    def _print_dormant_example(self):
        ex = self._example_dormant_split()
        if ex is None:
            self._log("    (sin celda con ambos lados materiales para el ejemplo)")
            return
        self._log("")
        self._log(f"    ★ EJEMPLO REPRESENTATIVO — estratificar por '{ex['split']}' "
                  f"(ambos lados con soporte y dinero):")
        self._log(f"      celda: {ex['mk']}  |  $ a predecir ${ex['usd_proj']:,.0f}")
        self._log(f"      FÓRMULA tradicional : pipeline_proj × tasa_celda × AUV")
        self._log(f"      FÓRMULA estratificada: Σ_lado pipeline_lado × tasa_lado × AUV")
        tab = ex["lados"].join(ex["pj_l"], how="outer")
        tab = tab.assign(
            tasa_pct=(tab["rate"]*100).round(1),
            share_hist_pct=(tab["share"]*100).round(1),
            pipe_proj=tab["npipe"].round(0),
            usd_proj=tab["usd"].round(0))[
            ["tasa_pct", "share_hist_pct", "pipe_proj", "usd_proj"]]
        self._table(tab, f"    sub-poblaciones por '{ex['split']}':", col_defs={
            "tasa_pct": "tasa de renovación histórica del lado (%)",
            "share_hist_pct": "% del pipeline histórico de la celda que es ese lado",
            "pipe_proj": "unidades de pipeline a predecir del lado",
            "usd_proj": "$ de pipeline a predecir del lado"})
        self._log(f"      tasa única de la celda (tradicional): {100*ex['rbar']:.1f}%  "
                  f"| diferencia entre lados: {ex['gap_pp']:.0f}pp")
        self._log(f"      → forecast TRADICIONAL  : ${ex['trad']:,.0f}")
        self._log(f"      → forecast ESTRATIFICADO: ${ex['strat']:,.0f}")
        self._log(f"      → DIFERENCIA por incluir '{ex['split']}': ${ex['diff']:+,.0f} "
                  f"({100*ex['diff']/max(ex['trad'],1):+.1f}% del tradicional)")

    def step_2_report_aggregation_cost(self, coarse_dims=None,
                                       tolerance_usd: float = 100_000.0,
                                       top_n: int = 5) -> dict:
        """Coste en DINERO de promediar la tasa al grano mandatory vs predecir
        por hija y sumar atribuciones. Por celda gruesa, tres números:
          - $/pp: lo que vale 1pp de error de tasa = Σᵢ Nᵢ_proj × AUVᵢ × 0.01.
          - coste realizado: e = Σᵢ Nᵢ_proj × AUVᵢ × (rᵢ − r̄), con r̄ la tasa
            agregada de la celda (Σren/Σpipe, aditiva). Es EXACTAMENTE 0 si el
            mix de proyección replica el histórico: promediar solo sesga cuando
            el mix se mueve (heterogeneidad = exposición; mix-shift = gatillo).
            Signo: e>0 → la media SUBestima; e<0 → SOBREestima.
          - exposición: rango_pp × $/pp — techo si el mix derivase del todo.
        Flag 'coste material': (|e| ≥ tolerance_usd O exposición ≥ tolerance_usd)
        Y dispersión > ruido binomial. Cartera: NETO Σe (lo que pegaría al número
        final) y BRUTO Σ|e| (la magnitud real del problema; neto pequeño con
        bruto grande = compensación entre celdas, no precisión). Mide solo el
        SESGO de agregar (toma las tasas hijas como verdad); la varianza por n
        pequeño es la historia de densidad — juntas son el trade completo.
        tolerance_usd = presupuesto de error en USD, misma filosofía que el
        freeze por deciles (freeze_decile_tolerances_usd)."""
        self._require("step_2_report_aggregation_cost",
                      ["step_2_build_support", "step_1_add_rates"])
        coarse = self._coarse_dims(coarse_dims)
        cfg, df = self.cfg, self.df
        pu, ru, rusd, pc, rc = (cfg.pipeline_units_col, cfg.renewed_units_col,
                                cfg.renewed_usd_col, cfg.period_col, cfg.dataset_role_col)
        base = df[(df["step_1_forecast_route"] == "trainable")
                  & (df["step_1_universe"] == "normal")]
        real = base[base[rc].isin(cfg.history_roles)
                    & (base.get("step_1_synthetic", 0).fillna(0).astype(int) == 0)]
        g = real.groupby("step_2_fs_id")
        child = pd.DataFrame({"pipe_h": g[pu].sum().astype(float),
                              "ren_h": g[ru].sum().astype(float),
                              "ren_usd_h": g[rusd].sum().astype(float)})
        child = child[child["pipe_h"] > 0]
        child["rate"] = child["ren_h"] / child["pipe_h"]
        # pipeline a proyectar por hija (mismo filtro que usd_proj: projection del año actual)
        cy = _derive_current_year(df, cfg)
        if cy is not None:
            pj = df[(df[rc] == "projection")
                    & (df[pc].apply(lambda x: pd.notna(x) and x.year == cy))]
            n_proj = pj.groupby("step_2_fs_id")[pu].sum()
        else:
            n_proj = pd.Series(dtype=float)
        child["n_proj"] = n_proj.reindex(child.index).fillna(0.0).astype(float)
        fsmeta = df.drop_duplicates("step_2_fs_id").set_index("step_2_fs_id")
        child["usd_proj"] = fsmeta["step_2_usd_proj"].astype(float).reindex(child.index).fillna(0.0)
        # AUV de renovación por hija; sin renovaciones, proxy = AUV de pipeline en proyección
        auv = pd.Series(0.0, index=child.index)
        m_ren = child["ren_h"] > 0
        auv[m_ren] = child.loc[m_ren, "ren_usd_h"] / child.loc[m_ren, "ren_h"]
        m_pj = (~m_ren) & (child["n_proj"] > 0)
        auv[m_pj] = child.loc[m_pj, "usd_proj"] / child.loc[m_pj, "n_proj"]
        child["auv"] = auv
        ckey_map = real.drop_duplicates("step_2_fs_id").set_index("step_2_fs_id")[coarse] \
            .astype(str).agg("|".join, axis=1)
        child["_ck"] = ckey_map.reindex(child.index)
        # trainable con dinero a proyectar pero sin tasa propia (no atribuible aquí → shrink)
        tr_ids = pd.Index(base["step_2_fs_id"].dropna().unique())
        usd_no_rate = float(fsmeta["step_2_usd_proj"]
                            .reindex(tr_ids.difference(child.index)).fillna(0.0).sum())

        cells, single_usd = [], 0.0
        for ckey, sub in child.groupby("_ck"):
            if len(sub) < 2:
                single_usd += float(sub["usd_proj"].sum())
                continue
            rbar, _spread, rng, disp, k = _cell_rate_heterogeneity(
                sub["rate"].values, sub["pipe_h"].values)
            sens = sub["n_proj"].values * sub["auv"].values
            dpp = 0.01 * float(sens.sum())                    # $ por 1pp
            attr = sens * (sub["rate"].values - rbar)         # atribución por hija
            e = float(attr.sum())                             # coste realizado (mix actual)
            exposure = float(rng) * dpp                       # techo por deriva total del mix
            cells.append({"_ck": ckey, "range_pp": float(rng), "dispersion": disp,
                          "k": k, "usd": float(sub["usd_proj"].sum()), "dpp": dpp,
                          "e": e, "exposure": exposure,
                          "attr": sorted(zip(list(sub.index), attr),
                                         key=lambda t: -abs(t[1]))})
        if not cells:
            self._mark("step_2_report_aggregation_cost", metadata={"empty": True})
            self._done("step_2_report_aggregation_cost", "ninguna celda mandatory con ≥2 FS")
            return {}
        total_usd = sum(c["usd"] for c in cells)
        dpp_total = sum(c["dpp"] for c in cells)
        net = sum(c["e"] for c in cells)
        gross = sum(abs(c["e"]) for c in cells)
        flagged = [c for c in cells
                   if (abs(c["e"]) >= tolerance_usd or c["exposure"] >= tolerance_usd)
                   and (c["dispersion"] is not None and not np.isnan(c["dispersion"])
                        and c["dispersion"] > 1.0)]
        usd_flag = sum(c["usd"] for c in flagged)
        pct_flag = 100 * usd_flag / total_usd if total_usd else 0.0
        lect = "SUBestimaría" if net > 0 else ("SOBREestimaría" if net < 0 else "no movería")
        self._log(f"  step_2_report_aggregation_cost: ¿cuánto dinero cuesta predecir")
        self._log(f"    al grano GRUESO (mandatory) en vez del FINO? (gruesa={coarse})")
        self._log(f"    ── MARCO DEL CÁLCULO ──")
        self._log(f"    1) Dispersión (la pipeline que entra): dentro de cada celda mandatory las")
        self._log(f"       sub-poblaciones tienen tasas distintas (rango en pp).")
        self._log(f"    2) Coste en $: e = Σ_hija  N_proj × AUV_pipeline × (tasa_hija − tasa_media_celda)")
        self._log(f"       = lo que SUB/SOBRE-estima usar la tasa media en vez de la fina.")
        self._log(f"       Es 0 si el mix de proyección replica el histórico; crece si el mix se mueve.")
        self._log(f"    ⚠ AISLA el efecto de la TASA, valorada a precio de pipeline (AUV).")
        self._log(f"      NO incluye revalorización (uplift) ni el efecto del soporte → es una")
        self._log(f"      COTA INFERIOR del valor de bajar de grano; el efecto real es ≥ este número.")
        # dispersión (la pipeline a renovar) como contexto, ponderada por $
        cdf0 = pd.DataFrame(cells)
        tot0 = max(float(cdf0["usd"].sum()), 1) if "usd" in cdf0 else 1
        if "range_pp" in cdf0:
            disp20 = 100*float(cdf0.loc[cdf0["range_pp"] >= 20, "usd"].sum())/tot0 if "usd" in cdf0 else 0
            self._log(f"    DISPERSIÓN (pipeline a renovar): {disp20:.0f}% del $ vive en celdas cuyas hijas "
                      f"difieren ≥20pp en tasa → el promedio mezcla comportamientos.")
        self._log(f"    {len(cells):,} celdas con ≥2 FS | dinero ${total_usd:,.0f} | "
                  f"sensibilidad cartera: 1pp de tasa ≈ ${dpp_total:,.0f}")
        self._log(f"    coste realizado con el mix de proyección actual: NETO "
                  f"${net:+,.0f} (promediar {lect} el total) | BRUTO ${gross:,.0f} "
                  f"(si neto<<bruto, las celdas se compensan por suerte de mix).")
        banda = float(np.sqrt(sum(c["e"]**2 for c in cells)))   # cuadratura por celda
        self._log(f"    NÚMERO-BANDERA → bajar de grano vale ${gross:,.0f} (bruto) "
                  f"[banda agregada ±${banda:,.0f}]")
        self._log(f"    lectura negocio: si grueso y fino NO se solapan en banda, "
                  f"el método tradicional vive en otro número.")
        self._log(f"    → COSTE MATERIAL (|e| o exposición ≥ tolerancia, y más allá del "
                  f"ruido): {pct_flag:.1f}% del $ ({len(flagged)} celdas).")
        if single_usd > 0:
            self._log(f"    nota: ${single_usd:,.0f} en celdas de 1 sola FS — sin riesgo "
                      f"de agregación por construcción.")
        if usd_no_rate > 0:
            self._log(f"    nota: ${usd_no_rate:,.0f} de FS sin tasa histórica propia, "
                      f"no atribuible aquí (irán a shrink).")
        worst = sorted(cells, key=lambda c: -max(abs(c["e"]), c["exposure"]))[:top_n]
        self._log(f"    peores celdas (realizado / exposición):")
        for c in worst:
            self._log(f"      {c['_ck']}: e=${c['e']:+,.0f} | exposición=${c['exposure']:,.0f} "
                      f"| $/pp=${c['dpp']:,.0f} (rango {c['range_pp']:.0f}pp, {c['k']} FS, "
                      f"${c['usd']:,.0f})")
        self._print_dormant_example()
        w0 = worst[0]
        out = {"coarse_dims": coarse, "tolerance_usd": tolerance_usd,
               "n_cells": len(cells), "usd_total": round(total_usd, 0),
               "dollar_per_pp_total": round(dpp_total, 0),
               "net_bias_usd": round(net, 0), "gross_bias_usd": round(gross, 0),
               "pct_usd_material_cost": round(pct_flag, 1),
               "usd_single_child_cells": round(single_usd, 0),
               "usd_no_own_rate": round(usd_no_rate, 0),
               "worst": [{"cell": c["_ck"], "e_usd": round(c["e"], 0),
                          "exposure_usd": round(c["exposure"], 0),
                          "dollar_per_pp": round(c["dpp"], 0),
                          "range_pp": round(c["range_pp"], 1), "k": c["k"],
                          "usd": round(c["usd"], 0)} for c in worst],
               "worst_cell_attributions": [
                   {"fs_id": fid, "usd": round(a, 0)} for fid, a in w0["attr"][:3]]}
        self._mark("step_2_report_aggregation_cost", metadata=out)
        self._done("step_2_report_aggregation_cost",
                   f"coste material en {pct_flag:.0f}% del $ (neto ${net:+,.0f})")
        return out

    # ========================================================
    # FASE B2 — step_3 (ANOVA: qué dimensión separa la tasa)
    # ========================================================
    # ================== STORYTELLING (parte del framework) ==================
    def step_6_add_story_columns(self, coarse_dims=None, year_a=None, year_b=None,
                                 eval_months=6, n_sano=30, ruido_max_support=5,
                                 ruido_pct_usd=50.0, min_meses=12, max_pct_hueco=25.0,
                                 min_amplitude_pp=5.0, min_slope_pp_year=5.0,
                                 min_mix_pp=1.0, cota_mult=1.0):
        """LA ESCALERA: etiqueta cada celda mandatory con su modo de fallo y sus mediciones (cota binomial = escala del error aceptable).
        Regla de lectura explícita: la cota binomial es el error mínimo del caso
        MÁS FAVORABLE (mundo plano); error ≤ cota → ruido de muestra; error ≫
        cota → la situación estudiada (tendencia/mezcla/estacionalidad) AÑADE
        error real. Estampa columnas story_* en TODAS las filas de cada celda.
        Ventanas: A/B = dos últimos años naturales completos; WHAT-IF = último
        año natural completo → meses siguientes (mimetiza el forecast)."""
        self._require("step_6_add_story_columns",
                      ["step_2_build_support", "step_1_fill_gaps"])
        cfg, df = self.cfg, self.df
        pu, ru, pc, rc = (cfg.pipeline_units_col, cfg.renewed_units_col,
                          cfg.period_col, cfg.dataset_role_col)
        pus = cfg.pipeline_usd_col
        coarse = list(coarse_dims or cfg.business_mandatory_dims)
        base = ((df["step_1_forecast_route"] == "trainable")
                & (df["step_1_universe"] == "normal")
                & df[rc].isin(cfg.history_roles) & df["step_2_fs_id"].notna())
        syn = df.get("step_1_synthetic", 0).fillna(0).astype(int)
        hist_all = df[base].copy()
        hist_all["_syn"] = syn[base].values
        hist = hist_all[hist_all["_syn"] == 0].copy()
        for h in (hist_all, hist):
            h["_ck"] = h[coarse].astype(str).agg("|".join, axis=1)
        usd_fs = (df.drop_duplicates("step_2_fs_id").set_index("step_2_fs_id")
                  ["step_2_usd_proj"].astype(float))
        hist["_usd"] = hist["step_2_fs_id"].map(usd_fs).fillna(0)
        meses = sorted(hist[pc].unique())
        anios = pd.Series([p.year for p in meses])
        completos = [a for a, c in anios.value_counts().sort_index().items() if c >= 12]
        if year_a is None and len(completos) >= 2:
            year_a, year_b = completos[-2], completos[-1]
        if year_a is not None:
            py = {p: p.year for p in meses}
            mask_a = hist[pc].map(py) == year_a
            mask_b = hist[pc].map(py) == year_b
            et_a, et_b = str(year_a), str(year_b)
            modo = f"años naturales {year_a} vs {year_b} (estacionalidad cancelada)"
        else:
            corte = meses[len(meses) // 2]
            mask_a, mask_b = hist[pc] < corte, hist[pc] >= corte
            et_a, et_b = f"H1(<{corte})", f"H2(>={corte})"
            modo = "mitades H1/H2 (fallback: sin 2 años completos; caveat estacional)"
        if year_b is not None and any(p.year > year_b for p in meses):
            ref_meses = [p for p in meses if p.year == year_b]
            eval_meses = [p for p in meses if p.year > year_b]
        else:
            eval_meses = meses[-eval_months:]
            ref_meses = meses[max(0, len(meses) - eval_months - 12):-eval_months] or meses[:1]
        mask_ref, mask_eval = hist[pc].isin(ref_meses), hist[pc].isin(eval_meses)
        et_ref = f"{ref_meses[0]}..{ref_meses[-1]}"
        et_eval = f"{eval_meses[0]}..{eval_meses[-1]} ({len(eval_meses)}m)"

        def _bennet(g1, g2):
            idx = g1.index.union(g2.index)
            P1 = g1[pu].reindex(idx).fillna(0.0); R1 = g1[ru].reindex(idx).fillna(0.0)
            P2 = g2[pu].reindex(idx).fillna(0.0); R2 = g2[ru].reindex(idx).fillna(0.0)
            if P1.sum() <= 0 or P2.sum() <= 0 or len(idx) < 2:
                return (np.nan,) * 5
            w1, w2 = P1 / P1.sum(), P2 / P2.sum()
            r1 = R1 / P1.replace(0, np.nan); r2 = R2 / P2.replace(0, np.nan)
            r1f, r2f = r1.fillna(r2), r2.fillna(r1)
            within = float((((w1 + w2) / 2) * (r2f - r1f)).sum()) * 100
            mix = float(((w2 - w1) * ((r1f + r2f) / 2)).sum()) * 100
            return (100 * float(R1.sum() / P1.sum()), 100 * float(R2.sum() / P2.sum()),
                    within, mix, 100 * float((w1 * r2f).sum()))

        rows = []
        for ck, sub in hist.groupby("_ck"):
            sub_all = hist_all[hist_all["_ck"] == ck]
            m = sub.groupby(pc)[[pu, ru]].sum().sort_index()
            rate_m = (m[ru] / m[pu].clip(lower=1)).astype(float)
            u_cell = float(sub.drop_duplicates("step_2_fs_id")["_usd"].sum())
            p_cell = float(sub[ru].sum() / max(sub[pu].sum(), 1e-9))
            n_med = float(m[pu].median()) if len(m) else np.nan
            se_cell_pp = 100 * np.sqrt(max(p_cell * (1 - p_cell), 1e-9) / max(n_med, 1))
            med_h = sub.groupby("step_2_fs_id")[pu].median()
            u_h = sub.drop_duplicates("step_2_fs_id").set_index("step_2_fs_id")["_usd"]
            se_h = np.sqrt(np.maximum(p_cell * (1 - p_cell), 1e-9)
                           / np.maximum(med_h.reindex(u_h.index).fillna(1), 1)).clip(0, 0.5)
            err_binom_usd = float((se_h * u_h).sum())
            ruido_pct = 100 * float(u_h[med_h.reindex(u_h.index) <= ruido_max_support].sum()) \
                / max(u_cell, 1e-9)
            peor_hija = str((se_h * u_h).idxmax()) if len(u_h) else ""
            meses_hist = int(len(m))
            pct_hueco = 100 * float(sub_all["_syn"].mean()) if len(sub_all) else np.nan
            slope = np.nan
            if meses_hist >= 12:
                t = pd.PeriodIndex(m.index).asi8.astype(float); y = rate_m.values
                dt = t[:, None] - t[None, :]; dy = y[:, None] - y[None, :]
                mk = dt > 0
                slope = float(np.median(dy[mk] / dt[mk])) * 12 * 100
            ma = mask_a.reindex(sub.index, fill_value=False)
            mb = mask_b.reindex(sub.index, fill_value=False)
            ra, rb, w_ab, x_ab, frozen = _bennet(
                sub[ma].groupby("step_2_fs_id")[[pu, ru]].sum(),
                sub[mb].groupby("step_2_fs_id")[[pu, ru]].sum())
            mr = mask_ref.reindex(sub.index, fill_value=False)
            me = mask_eval.reindex(sub.index, fill_value=False)
            wi_ref, wi_real, wi_re, wi_mz, _f = _bennet(
                sub[mr].groupby("step_2_fs_id")[[pu, ru]].sum(),
                sub[me].groupby("step_2_fs_id")[[pu, ru]].sum())
            wi_err = wi_real - wi_ref if np.isfinite(wi_ref) else np.nan
            pipe_eval_usd = float(sub.loc[me, pus].sum())
            n_eval = float(sub.loc[me, pu].sum())
            wi_imp = wi_err / 100 * pipe_eval_usd if np.isfinite(wi_err) else np.nan
            p0 = wi_ref / 100 if np.isfinite(wi_ref) else p_cell
            cota_pp = 100 * np.sqrt(max(p0 * (1 - p0), 1e-9) / max(n_eval, 1)) \
                if n_eval > 0 else np.nan
            cota_usd = cota_pp / 100 * pipe_eval_usd if np.isfinite(cota_pp) else np.nan
            ratio = abs(wi_err) / max(cota_pp, 1e-9) \
                if np.isfinite(wi_err) and np.isfinite(cota_pp) else np.nan
            mas_alla = int(np.isfinite(ratio) and ratio > cota_mult)
            amp, coste_est = np.nan, np.nan
            if meses_hist >= 24:
                mm2 = m.copy(); mm2["_m"] = pd.PeriodIndex(mm2.index).month
                bym = mm2.groupby("_m").apply(
                    lambda s: s[ru].sum() / max(s[pu].sum(), 1e-9), include_groups=False)
                if len(bym) >= 10:
                    amp = 100 * float(bym.max() - bym.min())
            if np.isfinite(wi_ref):
                refm = sub[mr].groupby(sub.loc[mr, pc].map(lambda p: p.month)) \
                    .apply(lambda s: s[ru].sum() / max(s[pu].sum(), 1e-9),
                           include_groups=False)
                c = 0.0
                for p in eval_meses:
                    rows_m = sub[me & (sub[pc] == p)]
                    if not len(rows_m):
                        continue
                    seas = float(refm.get(p.month, p0))
                    c += abs(seas - p0) * float(rows_m[pus].sum())
                coste_est = c
            f_ruido = int(ruido_pct >= ruido_pct_usd)
            f_temp = int(meses_hist < min_meses or
                         (np.isfinite(pct_hueco) and pct_hueco >= max_pct_hueco))
            f_est = int(np.isfinite(amp) and amp >= min_amplitude_pp)
            f_tend = int(np.isfinite(slope) and abs(slope) >= min_slope_pp_year
                         and np.isfinite(w_ab) and np.isfinite(x_ab)
                         and abs(w_ab) >= abs(x_ab) and mas_alla == 1)
            f_simp = int(np.isfinite(x_ab) and abs(x_ab) >= min_mix_pp
                         and np.isfinite(w_ab) and abs(x_ab) > abs(w_ab))
            gate = ("soporte" if f_ruido else "temporal" if f_temp else
                    "estacionalidad" if f_est else "tendencia" if f_tend else
                    "mezcla" if f_simp else "apto_promedio")
            rows.append({
                "_ck": ck, "story_eval_gate": gate,
                "story_cell_usd_proj": round(u_cell, 0),
                "story_soporte_mediano_n": round(n_med, 0) if np.isfinite(n_med) else np.nan,
                "story_se_binomial_pp": round(se_cell_pp, 1),
                "story_error_binomial_usd": round(err_binom_usd, 0),
                "story_pct_usd_en_ruido": round(ruido_pct, 1),
                "story_fs_peor_hija": peor_hija,
                "story_meses_historia": meses_hist,
                "story_pct_meses_hueco": round(pct_hueco, 1) if np.isfinite(pct_hueco) else np.nan,
                "story_estacionalidad_amplitud_pp": round(amp, 1) if np.isfinite(amp) else np.nan,
                "story_coste_ignorar_estacionalidad_usd": round(coste_est, 0) if np.isfinite(coste_est) else np.nan,
                "story_trend_slope_pp_year": round(slope, 1) if np.isfinite(slope) else np.nan,
                "story_whatif_rate_ref_pct": round(wi_ref, 1) if np.isfinite(wi_ref) else np.nan,
                "story_whatif_rate_real_pct": round(wi_real, 1) if np.isfinite(wi_real) else np.nan,
                "story_whatif_error_pp": round(wi_err, 1) if np.isfinite(wi_err) else np.nan,
                "story_whatif_real_pp": round(wi_re, 1) if np.isfinite(wi_re) else np.nan,
                "story_whatif_mezcla_pp": round(wi_mz, 1) if np.isfinite(wi_mz) else np.nan,
                "story_whatif_impacto_usd": round(wi_imp, 0) if np.isfinite(wi_imp) else np.nan,
                "story_whatif_cota_binomial_pp": round(cota_pp, 1) if np.isfinite(cota_pp) else np.nan,
                "story_whatif_cota_binomial_usd": round(cota_usd, 0) if np.isfinite(cota_usd) else np.nan,
                "story_whatif_error_vs_cota": round(ratio, 1) if np.isfinite(ratio) else np.nan,
                "story_whatif_mas_alla_ruido": mas_alla,
                "story_cell_rate_a_pct": round(ra, 1) if np.isfinite(ra) else np.nan,
                "story_cell_rate_b_pct": round(rb, 1) if np.isfinite(rb) else np.nan,
                "story_cell_delta_pp": round(rb - ra, 1) if np.isfinite(ra) else np.nan,
                "story_cell_real_pp": round(w_ab, 1) if np.isfinite(w_ab) else np.nan,
                "story_cell_mezcla_pp": round(x_ab, 1) if np.isfinite(x_ab) else np.nan,
                "story_cell_rate_b_mix_congelado_pct": round(frozen, 1) if np.isfinite(frozen) else np.nan,
                "story_flag_ruido": f_ruido, "story_flag_temporal": f_temp,
                "story_flag_estacionalidad": f_est, "story_flag_tendencia": f_tend,
                "story_flag_simpson": f_simp,
            })
        cells = pd.DataFrame(rows)
        cells["story_cell_periodo_a"], cells["story_cell_periodo_b"] = et_a, et_b
        cells["story_whatif_periodo_ref"], cells["story_whatif_periodo_eval"] = et_ref, et_eval
        key_all = df[coarse].astype(str).agg("|".join, axis=1)
        scols = [c for c in cells.columns if c != "_ck"]
        self.df = df.drop(columns=[c for c in scols if c in df.columns]) \
            .assign(_ck=key_all.values).merge(cells, on="_ck", how="left").drop(columns="_ck")
        self._story_cells = cells
        self._log(f"  step_6_add_story_columns: A/B = {modo}")
        self._log(f"    WHAT-IF forecast-símil: referencia {et_ref} → evalúa {et_eval}")
        self._log("    REGLA: cota binomial = error mínimo del caso más favorable (plano); "
                  "error ≤ cota → ruido; error ≫ cota → error añadido real.")
        self._log(f"    celdas por story_eval_gate (primer peldaño fallado):")
        g = cells.groupby("story_eval_gate")["story_cell_usd_proj"].agg(["count", "sum"])
        tot = max(cells["story_cell_usd_proj"].sum(), 1e-9)
        for k, r in g.sort_values("sum", ascending=False).iterrows():
            self._log(f"      {k:>15}: {int(r['count']):>5,} celdas | ${r['sum']:>14,.0f} "
                      f"({100*r['sum']/tot:.1f}% del $)")
        top = cells.reindex(cells["story_whatif_impacto_usd"].abs()
                            .sort_values(ascending=False).index).head(5)
        self._log("    TOP 5 'dónde más dinero se equivoca el promedio' (con su cota):")
        for _, r in top.iterrows():
            self._log(f"      {str(r['_ck'])[:60]}: err={r['story_whatif_error_pp']}pp | "
                      f"impacto=${r['story_whatif_impacto_usd']:,.0f} | "
                      f"cota=${r['story_whatif_cota_binomial_usd']:,.0f} | "
                      f"ratio={r['story_whatif_error_vs_cota']} | gate={r['story_eval_gate']}")
        apto = float(g.loc["apto_promedio", "sum"]) if "apto_promedio" in g.index else 0.0
        self._mark("step_6_add_story_columns", metadata={"n_cells": len(cells)})
        self._done("step_6_add_story_columns",
                   f"escalera etiquetada en {len(cells):,} celdas; apto_promedio="
                   f"{100*apto/tot:.0f}% del $, resto con modo de fallo asignado")
        return cells

    def step_6_report_story_figures(self, out_dir="."):
        """Gráficas del storytelling (PNG): dinero por gate, error vs cota binomial, mezcla congelada del top Simpson y tendencia-vs-plano del top impacto.
        Requiere step_6_add_story_columns y matplotlib."""
        self._require("step_6_report_story_figures", ["step_6_add_story_columns"])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self._mark("step_6_report_story_figures")
            self._done("step_6_report_story_figures", "matplotlib no disponible: sin gráficas")
            return []
        import os
        cells = self._story_cells
        cfg, df = self.cfg, self.df
        pu, ru, pc, rc = (cfg.pipeline_units_col, cfg.renewed_units_col,
                          cfg.period_col, cfg.dataset_role_col)
        coarse = list(cfg.business_mandatory_dims)
        base = ((df["step_1_forecast_route"] == "trainable")
                & (df["step_1_universe"] == "normal")
                & df[rc].isin(cfg.history_roles) & df["step_2_fs_id"].notna()
                & (df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0))
        hist = df[base].copy()
        hist["_ck"] = hist[coarse].astype(str).agg("|".join, axis=1)
        hecho = []
        # 1) dinero por gate
        fig, ax = plt.subplots(figsize=(8, 4.5))
        g = (cells.groupby("story_eval_gate")["story_cell_usd_proj"].sum()
             .sort_values(ascending=False))
        ax.bar(g.index, g.values / 1e6, color="#4878a8")
        ax.set_ylabel("$M a predecir"); ax.set_title("Dinero por modo de fallo (story_eval_gate)")
        plt.xticks(rotation=20); plt.tight_layout()
        p1 = os.path.join(out_dir, "story_gates_money.png"); fig.savefig(p1, dpi=130); plt.close(fig)
        hecho.append(p1)
        # 2) error vs cota (la regla de lectura, dibujada)
        d = cells.dropna(subset=["story_whatif_cota_binomial_usd", "story_whatif_impacto_usd"])
        d = d[(d["story_whatif_cota_binomial_usd"] > 0)]
        if len(d):
            fig, ax = plt.subplots(figsize=(6.5, 6))
            x = d["story_whatif_cota_binomial_usd"].abs().clip(lower=1)
            y = d["story_whatif_impacto_usd"].abs().clip(lower=1)
            ax.scatter(x, y, s=18, alpha=0.5)
            lim = [1, max(float(x.max()), float(y.max())) * 1.3]
            ax.plot(lim, lim, "r--", label="error = cota (suelo binomial)")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("cota binomial del what-if ($, caso plano)")
            ax.set_ylabel("|error del promedio| en el what-if ($)")
            ax.set_title("Encima de la diagonal = error MÁS ALLÁ del ruido"); ax.legend()
            plt.tight_layout()
            p2 = os.path.join(out_dir, "story_error_vs_cota.png"); fig.savefig(p2, dpi=130); plt.close(fig)
            hecho.append(p2)
        # 3) top Simpson: 3 barras (A, B real, B con mezcla congelada)
        s = cells[cells["story_flag_simpson"] == 1].sort_values("story_cell_usd_proj", ascending=False)
        if len(s):
            r = s.iloc[0]
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            vals = [r["story_cell_rate_a_pct"], r["story_cell_rate_b_pct"],
                    r["story_cell_rate_b_mix_congelado_pct"]]
            labs = [f"tasa {r['story_cell_periodo_a']}", f"tasa {r['story_cell_periodo_b']} real",
                    f"{r['story_cell_periodo_b']} mezcla congelada"]
            ax.bar(labs, vals, color=["#888", "#c25b4e", "#4878a8"])
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v}%", ha="center", va="bottom")
            ax.set_ylabel("tasa de renovación (%)")
            ax.set_title(f"Simpson/mezcla: {str(r['_ck'])[:55]}\n"
                         f"real {r['story_cell_real_pp']}pp | mezcla {r['story_cell_mezcla_pp']}pp")
            plt.xticks(rotation=10); plt.tight_layout()
            p3 = os.path.join(out_dir, "story_simpson_top.png"); fig.savefig(p3, dpi=130); plt.close(fig)
            hecho.append(p3)
        # 4) top impacto: tasa mensual de la celda + plano de referencia + ventana eval
        t5 = cells.reindex(cells["story_whatif_impacto_usd"].abs()
                           .sort_values(ascending=False).index)
        if len(t5):
            r = t5.iloc[0]
            sub = hist[hist["_ck"] == r["_ck"]]
            m = sub.groupby(pc)[[pu, ru]].sum().sort_index()
            rate = 100 * m[ru] / m[pu].clip(lower=1)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            xs = [str(p) for p in rate.index]
            ax.plot(xs, rate.values, marker="o", ms=3, label="tasa mensual de la celda")
            if np.isfinite(r["story_whatif_rate_ref_pct"]):
                ax.axhline(r["story_whatif_rate_ref_pct"], color="#888", ls="--",
                           label=f"promedio ref {r['story_whatif_periodo_ref']}")
            n_ev = int(str(r["story_whatif_periodo_eval"]).split("(")[-1].rstrip("m)"))
            if n_ev and n_ev < len(xs):
                ax.axvspan(len(xs) - n_ev - 0.5, len(xs) - 0.5, color="#f4d35e", alpha=0.3,
                           label="ventana evaluada")
            ax.set_ylabel("tasa (%)"); ax.legend(fontsize=8)
            ax.set_title(f"El promedio vs la realidad: {str(r['_ck'])[:55]}\n"
                         f"error {r['story_whatif_error_pp']}pp = "
                         f"${r['story_whatif_impacto_usd']:,.0f} (cota ${r['story_whatif_cota_binomial_usd']:,.0f})")
            step = max(1, len(xs) // 12)
            ax.set_xticks(range(0, len(xs), step))
            ax.set_xticklabels(xs[::step], rotation=45, fontsize=7)
            plt.tight_layout()
            p4 = os.path.join(out_dir, "story_trend_top.png"); fig.savefig(p4, dpi=130); plt.close(fig)
            hecho.append(p4)
        for p in hecho:
            self._log(f"    gráfica: {p}")
        self._mark("step_6_report_story_figures")
        self._done("step_6_report_story_figures", f"{len(hecho)} gráficas PNG generadas")
        return hecho

    def step_3_report_level_coverage(self, min_share_pct=0.5, max_levels_shown=5) -> dict:
        """¿Qué NIVELES de cada dimensión faltan en algunos meses? El test va
        MES A MES: si en un mes falta alguno de los valores posibles de una dim,
        ese valor genera huecos GARANTIZADOS ese mes en todas las series vivas
        que lo contienen — sí o sí, antes de mirar región/dormant/nada. No hace
        falta que falte en toda la historia: basta un mes. Cota inferior barata;
        la presencia en el mes NO garantiza nada al grano fino (asimetría).
        Por (dim, nivel): % de meses con presencia, % de unidades y % de $hist.
        Candidato a AGRUPAR ('Other'): presencia <100% de los meses o cuota
        < min_share_pct. Veredicto por dim: nº de niveles intermitentes."""
        self._require("step_3_report_level_coverage", ["step_1_fill_gaps"])
        cfg, df = self.cfg, self.df
        pu, pus, pc, rc = (cfg.pipeline_units_col, cfg.pipeline_usd_col,
                           cfg.period_col, cfg.dataset_role_col)
        dc = self.dimension_cols
        hist = df[(df["step_1_forecast_route"] == "trainable")
                  & (df["step_1_universe"] == "normal")
                  & df[rc].isin(cfg.history_roles)
                  & (df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0)]
        n_meses = hist[pc].nunique()
        tot_u = float(hist[pu].sum()); tot_usd = float(hist[pus].sum())
        self._log(f"  step_3_report_level_coverage: presencia mensual de niveles "
                  f"({n_meses} meses, {len(dc)} dims) — un nivel que falta en un mes = "
                  f"huecos garantizados ese mes en todas las series vivas con ese nivel.")
        out, ranking = {}, []
        for d in dc:
            g = hist.groupby(d, observed=True)
            pres = g[pc].nunique()
            u_share = 100 * g[pu].sum() / max(tot_u, 1e-9)
            usd_share = 100 * g[pus].sum() / max(tot_usd, 1e-9)
            tabla = pd.DataFrame({"meses": pres, "pct_meses": 100 * pres / max(n_meses, 1),
                                  "pct_units": u_share, "pct_usd": usd_share})
            inter = tabla[(tabla["pct_meses"] < 100) | (tabla["pct_units"] < min_share_pct)]
            out[d] = {"n_niveles": len(tabla), "intermitentes": len(inter),
                      "pct_units_intermitentes": round(float(inter["pct_units"].sum()), 2)}
            ranking.append((d, len(inter), len(tabla)))
            if len(inter):
                self._log(f"    {d} ({len(tabla)} niveles): {len(inter)} intermitentes/raros "
                          f"({out[d]['pct_units_intermitentes']:.1f}% de las unidades):")
                peor = inter.sort_values(["pct_meses", "pct_units"]).head(max_levels_shown)
                for lvl, r in peor.iterrows():
                    falta = n_meses - int(r["meses"])
                    self._log(f"      {str(lvl)[:42]:<42} presente {int(r['meses'])}/{n_meses} meses "
                              f"(falta {falta}) | {r['pct_units']:.2f}% units | {r['pct_usd']:.2f}% $ "
                              f"→ candidato a AGRUPAR")
                if len(inter) > max_levels_shown:
                    self._log(f"      ... y {len(inter) - max_levels_shown} niveles más")
        limpias = [d for d, i, _ in ranking if i == 0]
        if limpias:
            self._log(f"    dims con TODOS los niveles presentes todos los meses: {limpias}")
        peores = sorted([r for r in ranking if r[1] > 0], key=lambda x: -x[1])
        resumen = ", ".join(f"{d} ({i}/{n})" for d, i, n in peores[:4]) if peores \
            else "ninguna dim con niveles intermitentes"
        self._mark("step_3_report_level_coverage", metadata=out)
        self._done("step_3_report_level_coverage", f"niveles intermitentes: {resumen}")
        return out

    def step_3_anova_rate(self, eta_low: float = 0.02, eta_high: float = 0.10) -> dict:
        """ANOVA de la TASA DE RENOVACIÓN por dimensión (η² marginal, ponderado
        por soporte). Para cada dim, qué fracción de la varianza de la tasa
        explican sus niveles:
          - η² ALTO  → la dim separa las tasas: lleva señal, NO colapsar (y alto
            riesgo de Simpson si se colapsa).
          - η² BAJO  → no separa: es la variable ANULABLE, la que define hermanas
            sanas y por la que conviene agrupar primero.
        Marginal a propósito (directo y robusto): NO captura interacciones —
        anular una dim de η² bajo puede ser malo si interactúa con otra—; esa
        versión condicional se difiere (ver NOTAS bloque E). Se mide sobre la
        historia observada real (trainable, no sintéticas, rate válido)."""
        self._require("step_3_anova_rate", ["step_1_add_rates", "step_1_add_forecast_route"])
        cfg, df = self.cfg, self.df
        pu, rcol = cfg.pipeline_units_col, "step_1_rate_renewal"
        mask = ((df["step_1_universe"] == "normal")
                & (df["step_1_forecast_route"] == "trainable")
                & (df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0)
                & (df[cfg.dataset_role_col].isin(cfg.history_roles))
                & (df[rcol].notna()) & (df[pu].astype(float) > 0))
        sub = df.loc[mask]
        if len(sub) < 2:
            self._mark("step_3_anova_rate", metadata={"empty": True})
            self._done("step_3_anova_rate", "sin filas reales con tasa válida")
            return {}
        x = sub[rcol].astype(float).values
        w = sub[pu].astype(float).values
        ss_total = _weighted_ss_total(x, w)
        rows = []
        for d in self.dimension_cols:
            eta2 = _weighted_eta2(x, w, sub[d].values, ss_total)
            n_levels = int(pd.Series(sub[d].values).nunique())
            rows.append((d, eta2, n_levels))
        rows.sort(key=lambda r: (r[1] if r[1] is not None else -1), reverse=True)
        out = {"eta_low": eta_low, "eta_high": eta_high, "n_obs": int(len(sub)), "by_dim": {}}
        self._log("  step_3_anova_rate: η² de la tasa de renovación por dimensión "
                  "(alto=separa/NO colapsar; bajo=anulable)")
        self._log(f"    {'dimensión':<24} {'η²':>7}  niveles  veredicto")
        nullable, keep = [], []
        for d, eta2, nl in rows:
            if eta2 is None:
                verdict = "s/datos"
            elif eta2 >= eta_high:
                verdict = "SEPARA (no colapsar)"; keep.append(d)
            elif eta2 <= eta_low:
                verdict = "anulable (colapsar)"; nullable.append(d)
            else:
                verdict = "intermedia"
            e = f"{eta2:.3f}" if eta2 is not None else "  -  "
            self._log(f"    {d:<24} {e:>7}  {nl:>7}  {verdict}")
            out["by_dim"][d] = {"eta2": round(eta2, 4) if eta2 is not None else None,
                                "n_levels": nl, "verdict": verdict}
        out["nullable_dims"] = nullable
        out["keep_dims"] = keep
        self._log(f"    → anulables (colapsar primero): {nullable or '—'}")
        self._log(f"    → separan la tasa (conservar): {keep or '—'}")
        self._mark("step_3_anova_rate", metadata=out)
        self._done("step_3_anova_rate", f"η² por dim ({len(nullable)} anulables, {len(keep)} a conservar)")
        return out

    def step_3_collapse_anova(self, eta_low=0.02, support_floor=30) -> dict:
        """COLAPSO NIVEL 2 — ejecuta lo que el ANOVA solo informaba: para las
        series que SIGUEN pobres tras el Nivel 1, colapsa (pone '*') la
        dimensión NO-mandatory que menos separa la tasa (η² < eta_low). No
        toca geografía/producto (mandatory) — eso es Nivel 3. Genera
        step_2_fs_id_L2 (idempotente: = L1 si no se toca)."""
        self._require("step_3_collapse_anova",
                      ["step_3_anova_rate", "step_2_collapse_signal_support"])
        cfg, df = self.cfg, self.df
        pu = cfg.pipeline_units_col
        anova = self.step_metadata.get("step_3_anova_rate", {})
        bydim = anova.get("by_dim", {})
        mand = set(cfg.business_mandatory_dims)
        # dim no-mandatory con menor η² por debajo del umbral
        cand_dims = [(d, v.get("eta2")) for d, v in bydim.items()
                     if d not in mand and d in self.dimension_cols
                     and v.get("eta2") is not None and v["eta2"] < eta_low]
        base_id = "step_2_fs_id_L1" if "step_2_fs_id_L1" in df.columns else "step_2_fs_id"
        if not cand_dims:
            df["step_2_fs_id_L2"] = df[base_id]
            self.df = df
            self._mark("step_3_collapse_anova")
            self._done("step_3_collapse_anova",
                       "ninguna dim no-mandatory anulable (η²<umbral): N2 no actúa")
            return {"colapsada": None}
        drop_dim = sorted(cand_dims, key=lambda x: x[1])[0][0]
        # soporte por grupo actual (L1)
        sop = (df[df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0]
               .groupby(base_id)[pu].median())
        df["_sop"] = df[base_id].map(sop)
        pobre = df["_sop"] < support_floor
        dims = self.dimension_cols
        pos = {d: i for i, d in enumerate(dims)}
        def id_l2(row):
            if not pobre.loc[row.name]:
                return row[base_id]
            parts = str(row[base_id]).split("|")
            # colapsar drop_dim solo si el id aún tiene esa posición (no SIG=)
            if "SIG=" in str(row[base_id]) or pos[drop_dim] >= len(parts):
                return row[base_id]
            parts[pos[drop_dim]] = "*"
            return "|".join(parts)
        df["step_2_fs_id_L2"] = df.apply(id_l2, axis=1)
        n_antes = df.loc[pobre, base_id].nunique()
        n_desp = df.loc[pobre, "step_2_fs_id_L2"].nunique()
        sop_desp = float(df[pobre].groupby("step_2_fs_id_L2")[pu].median().median()) if pobre.any() else float("nan")
        df.drop(columns=["_sop"], inplace=True)
        self.df = df
        t = pd.DataFrame({"métrica": ["dim colapsada (η² más bajo)",
                                      "grupos pobres antes", "grupos pobres después",
                                      "soporte mediano después"],
                          "valor": [drop_dim, n_antes, n_desp, round(sop_desp, 1)]}).set_index("métrica")
        self._table(t, "Colapso Nivel 2 (ANOVA: dim que no separa):", col_defs={
            "valor": "dim colapsada / conteo de grupos / soporte mediano"})
        self._mark("step_3_collapse_anova", metadata={"drop_dim": drop_dim})
        self._done("step_3_collapse_anova",
                   f"colapsada '{drop_dim}' (η²<{eta_low}); {n_antes}→{n_desp} grupos pobres")
        return {"colapsada": drop_dim}

    def step_3_classify_small_series(self, support_small=30, support_dust=10) -> dict:
        """CLASIFICA las series de bajo soporte en tres DESTINOS DE GESTIÓN
        (no tres métodos de forecast — el motor sigue siendo uno):
          - FUSIONABLE: soporte en [dust, small) Y su celda tiene alguna
            dimensión SIN señal (anulable según ANOVA) que, quitada, la une a
            hermanas → puede ganar grano real. Candidata a subir medio nivel.
          - ENCOGIBLE: pequeña, pero su celda padre (mandatory) tiene MASA →
            el evidence weight ya la trata bien (z bajo, hereda del padre).
          - POLVO: soporte < dust Y padre también pobre → no hay señal que
            extraer; hereda y se REPORTA AGREGADA. Pretender más es inventar.
        Da conteo y $ por cubo: la información para decidir el grano, sin
        tocar el forecast. La acción (fusionar/subir grano) la decides tú."""
        self._require("step_3_classify_small_series",
                      ["step_2_build_support", "step_3_anova_rate"])
        cfg, df = self.cfg, self.df
        anova = self.step_metadata.get("step_3_anova_rate", {})
        nullable = [d for d, v in anova.get("by_dim", {}).items()
                    if v.get("eta2") is not None and v["eta2"] <= anova.get("eta_low", 0.02)]
        # soporte mediano por serie y $ a predecir
        fs = (df[df["step_2_fs_id"].notna()].drop_duplicates("step_2_fs_id")
              .loc[:, ["step_2_fs_id", "step_1_fs_density_median", "step_2_usd_proj"]]
              .rename(columns={"step_1_fs_density_median": "sop"}).copy())
        fs["sop"] = fs["sop"].fillna(0)
        fs["usd"] = fs["step_2_usd_proj"].fillna(0)
        # pipeline del padre mandatory de cada serie (clave = dims obligatorias)
        mdims = list(cfg.business_mandatory_dims)
        df_k = df.assign(_mk=df[mdims].astype(str).agg("|".join, axis=1))
        parent_mass = df_k.groupby("_mk")[cfg.pipeline_units_col].sum()
        mk = (df_k.drop_duplicates("step_2_fs_id")
              .set_index("step_2_fs_id")["_mk"])
        fs["parent_mass"] = fs["step_2_fs_id"].map(mk).map(parent_mass).fillna(0)
        big_parent = parent_mass.median()
        def cubo(r):
            if r["sop"] >= support_small:
                return "grande_ok"
            if r["sop"] < support_dust and r["parent_mass"] < big_parent:
                return "polvo"
            if (support_dust <= r["sop"] < support_small) and nullable:
                return "fusionable"
            return "encogible"
        fs["destino"] = fs.apply(cubo, axis=1)
        tot_usd = max(fs["usd"].sum(), 1)
        t = fs.groupby("destino").agg(n_series=("step_2_fs_id", "count"),
                                      usd=("usd", "sum"))
        t["pct_usd"] = 100*t["usd"]/tot_usd
        t["pct_series"] = 100*t["n_series"]/len(fs)
        orden = ["grande_ok", "fusionable", "encogible", "polvo"]
        t = t.reindex([o for o in orden if o in t.index])
        self._table(t.round(1), "Destino de gestión de las series por soporte:",
                    col_defs={"n_series": "nº de forecast series en el cubo",
                              "usd": "$ a predecir que vive en ellas",
                              "pct_usd": "% del $ total (lo que de verdad importa)",
                              "pct_series": "% del censo de series (incomodidad de gestión, no de $)"})
        if nullable:
            self._log(f"    dimensiones SIN señal que podrían fusionar (ANOVA): {nullable}")
        else:
            self._log("    ninguna dimensión anulable → no hay fusión gratis; "
                      "pequeñas van a encogible/polvo")
        self.small_class = fs
        d_usd = float(t.loc["polvo", "usd"]) if "polvo" in t.index else 0.0
        f_usd = float(t.loc["fusionable", "usd"]) if "fusionable" in t.index else 0.0
        self._mark("step_3_classify_small_series",
                   metadata={"pct_usd_polvo": 100*d_usd/tot_usd,
                             "pct_usd_fusionable": 100*f_usd/tot_usd,
                             "nullable_dims": nullable})
        self._done("step_3_classify_small_series",
                   f"polvo {100*d_usd/tot_usd:.1f}% del $ | "
                   f"fusionable {100*f_usd/tot_usd:.1f}% del $ "
                   f"({'hay' if nullable else 'no hay'} dim sin señal)")
        return {"pct_usd_polvo": 100*d_usd/tot_usd,
                "pct_usd_fusionable": 100*f_usd/tot_usd}

    def step_3_report_dim_fragmentation(self) -> dict:
        """¿Qué dimensión FABRICA el grano? Responde a 'no me creo la mediana 1'
        separando las dos hipótesis: cola larga real vs una dim que multiplica
        combos artificialmente. Por cada dim, leave-one-out sobre el universo
        trainable:
          - factor_grano = nº FS con todas las dims / nº FS quitando esta
            (cuántas veces multiplica el grano esa dim, dadas las demás;
            ~1 = redundante/correlada, ~nº niveles = cruza con todo).
          - soporte mediano SIN ella: re-agrupando hermanas a ese grano,
            ¿cuánto soporte mensual recupera el portfolio? (la dim que al
            quitarla dispara el soporte es la que lo fragmenta).
          - η² (si step_3_anova_rate corrió): cuánta tasa explica = señal.
        Veredicto coste×señal×mandatory: factor alto + η² bajo + no mandatory
        = CARA SIN SEÑAL (candidata a ignore/anular primero). Cabecera con
        reconciliación: unidades totales/mes del portfolio — si son millones,
        el dato es correcto y la mediana 1 es fragmentación, no error."""
        self._require("step_3_report_dim_fragmentation",
                      ["step_2_build_identity", "step_1_fill_gaps"])
        cfg, df = self.cfg, self.df
        pu, pc, rc = cfg.pipeline_units_col, cfg.period_col, cfg.dataset_role_col
        dims = list(self.dimension_cols)
        tr = (df["step_1_forecast_route"] == "trainable") & (df["step_1_universe"] == "normal")
        combos = df.loc[tr & df["step_2_fs_id"].notna(), dims].drop_duplicates()
        n_full = len(combos)
        hist = df.loc[tr & df[rc].isin(cfg.history_roles), dims + [pc, pu]].copy()
        # Claves de TEXTO por combinación OBSERVADA: el groupby multinivel con
        # categóricas materializaba el producto cartesiano de niveles en el
        # MultiIndex (MemoryError de 12.8 GiB a escala real). Con una clave
        # concatenada, la memoria es lineal en combinaciones observadas.
        scol = {d: hist[d].astype(str) for d in dims}

        def _sup_median(cols):
            k = (scol[cols[0]] if len(cols) == 1
                 else scol[cols[0]].str.cat([scol[d] for d in cols[1:]], sep="|"))
            s = hist.assign(_k=k.values).groupby(["_k", pc])[pu].sum()
            return s.groupby(level=0).median()
        # reconciliación: ¿cuántas unidades mueve el portfolio al mes?
        units_month = hist.groupby(pc)[pu].sum()
        med_units_month = float(units_month.median()) if len(units_month) else 0.0
        full_sup = _sup_median(dims)
        med_full = float(full_sup.median()) if len(full_sup) else 0.0
        anova_meta = self.step_metadata.get("step_3_anova_rate", {})
        anova = anova_meta.get("by_dim", {})
        n_obs = anova_meta.get("n_obs")
        mand = set(cfg.business_mandatory_dims or [])
        rows = []
        for d in dims:
            rest = [c for c in dims if c != d]
            n_wo = len(combos[rest].drop_duplicates()) if rest else 1
            factor = n_full / n_wo if n_wo else float("nan")
            sup = _sup_median(rest)
            med_wo = float(sup.median()) if len(sup) else 0.0
            eta2 = (anova.get(d) or {}).get("eta2")
            nlev = int(combos[d].nunique())
            # η² ajustado por niveles: el marginal infla con la cardinalidad
            # (más niveles = más df para explicar varianza espuria). Aproximado
            # con n filas sin ponderar; suficiente para el veredicto.
            if eta2 is not None and n_obs and n_obs > nlev:
                eta2 = max(0.0, 1.0 - (1.0 - eta2) * (n_obs - 1) / (n_obs - nlev))
            if factor < 1.2:
                verdict = "barata"
            elif eta2 is None:
                verdict = "cara (corre ANOVA para señal)"
            elif eta2 <= 0.02:
                verdict = "CARA SIN SEÑAL" + ("" if d in mand else " → candidata a anular/ignore")
            elif eta2 >= 0.10:
                verdict = "cara CON señal (shrink, no quitar)"
            else:
                verdict = "cara, señal intermedia"
            rows.append({"dim": d, "n_levels": nlev, "factor": factor,
                         "median_support_without": med_wo, "eta2": eta2,
                         "mandatory": d in mand, "verdict": verdict})
        rows.sort(key=lambda r: -r["factor"])
        self._log(f"  step_3_report_dim_fragmentation: qué dim fabrica el grano "
                  f"({n_full:,} FS trainable, {len(dims)} dims)")
        self._log(f"    reconciliación: unidades/mes del portfolio (mediana): "
                  f"{med_units_month:,.0f} | soporte mediano por FS al grano completo: {med_full:,.0f}")
        self._log(f"    {'dimensión':<24} {'niveles':>7} {'factor':>7} {'sop.med sin ella':>16} "
                  f"{'η²aj':>6} {'mand':>5}  veredicto")
        for r in rows:
            e = f"{r['eta2']:.3f}" if r["eta2"] is not None else "  -  "
            self._log(f"    {r['dim']:<24} {r['n_levels']:>7,} {r['factor']:>7.2f} "
                      f"{r['median_support_without']:>16,.0f} {e:>6} "
                      f"{'sí' if r['mandatory'] else 'no':>5}  {r['verdict']}")
        candidatas = [r["dim"] for r in rows
                      if r["factor"] >= 1.2 and not r["mandatory"]
                      and r["eta2"] is not None and r["eta2"] <= 0.02]
        self._log(f"    → caras sin señal y no mandatory: {candidatas or '—'}")
        out = {"n_fs_full": n_full, "median_units_month_portfolio": round(med_units_month, 0),
               "median_support_full_grain": round(med_full, 1), "by_dim": rows,
               "candidatas_anular": candidatas}
        self._mark("step_3_report_dim_fragmentation", metadata=out)
        self._done("step_3_report_dim_fragmentation",
                   f"factor de grano por dim ({len(candidatas)} candidatas a anular)")
        return out

    def step_2_report_distributions(self, out_path: str = "hito1_distribuciones.png",
                                    money_targets=(50, 80, 90, 95, 99)) -> dict:
        """Los resúmenes numéricos (mediana) son limitados: aquí, DISTRIBUCIONES.
        Percentiles del soporte por FS en dos pesos — por nº de series (cuántas
        líneas son astillas) y por DINERO a predecir (dónde vive el dólar
        mediano) —, concentración tipo Lorenz ('¿cuántas líneas necesito para
        cubrir el X% del dinero?', la respuesta directa a 'demasiadas líneas'),
        y un PNG con 4 paneles: (a) distribución del soporte, series vs dinero;
        (b) curva de Lorenz del dinero por FS; (c) dispersión soporte×dinero por
        FS; (d) longitud de historia, %FS vs %$. Si matplotlib no está, deja
        solo los resúmenes."""
        self._require("step_2_report_distributions",
                      ["step_2_build_support", "step_1_report_density"])
        df = self.df
        fs = df[(df["step_1_forecast_route"] == "trainable")
                & df["step_2_fs_id"].notna()
                & df["step_1_fs_density_median"].notna()].drop_duplicates("step_2_fs_id")
        sup = fs["step_1_fs_density_median"].astype(float).values
        usd = fs["step_2_usd_proj"].astype(float).clip(lower=0).values
        hm = fs["step_2_history_months"].astype(float).values
        qs = (5, 25, 50, 75, 90, 95, 99)
        q_n = {q: float(np.percentile(sup, q)) for q in qs}
        q_usd = _weighted_quantiles(sup, usd, qs)
        self._log(f"  step_2_report_distributions: distribución del soporte por FS "
                  f"({len(fs):,} series, ${usd.sum():,.0f})")
        self._log(f"    {'percentil':>9} | {'por nº series':>13} | {'por $ a predecir':>16}")
        for q in qs:
            self._log(f"    {'p'+str(q):>9} | {q_n[q]:>13,.0f} | {q_usd[q]:>16,.0f}")
        self._log(f"    lectura: la serie mediana tiene soporte {q_n[50]:,.0f}, "
                  f"pero el DÓLAR mediano vive en una serie de soporte {q_usd[50]:,.0f}.")
        # Lorenz: cuántas líneas cubren el X% del dinero
        order = np.argsort(-usd)
        cum = np.cumsum(usd[order]) / max(usd.sum(), 1e-9)
        lines_for = {}
        for t in money_targets:
            n_t = int(np.searchsorted(cum, t / 100.0) + 1)
            lines_for[t] = n_t
            self._log(f"    para cubrir el {t}% del $: {n_t:,} FS "
                      f"({100*n_t/len(fs):.1f}% de las series)")
        x_lor = np.arange(1, len(usd) + 1) / len(usd)
        v_asc = np.sort(usd); n_g = len(v_asc); s_g = max(v_asc.sum(), 1e-9)
        gini = float((2.0 * np.sum(np.arange(1, n_g + 1) * v_asc)) / (n_g * s_g) - (n_g + 1) / n_g)
        self._log(f"    concentración (Gini del $ por FS): {gini:.3f}")
        out = {"n_fs": int(len(fs)), "support_quantiles_by_series": {f"p{q}": round(q_n[q], 1) for q in qs},
               "support_quantiles_by_usd": {f"p{q}": round(q_usd[q], 1) for q in qs},
               "fs_needed_for_money_pct": {str(t): lines_for[t] for t in money_targets},
               "gini_usd": round(gini, 3), "chart": None}
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))
            # (a) soporte: % series vs % dinero, bins log
            s1 = np.clip(sup, 0.5, None)
            bins = np.logspace(np.log10(0.5), np.log10(max(s1.max(), 600)), 40)
            hn, _ = np.histogram(s1, bins=bins); hu, _ = np.histogram(s1, bins=bins, weights=usd)
            ctr = np.sqrt(bins[:-1] * bins[1:])
            ax[0, 0].step(ctr, 100 * hn / max(hn.sum(), 1), where="mid", label="% de series")
            ax[0, 0].step(ctr, 100 * hu / max(hu.sum(), 1e-9), where="mid", label="% del $ a predecir")
            for v in (30, 100, 500):
                ax[0, 0].axvline(v, ls=":", lw=0.8, color="gray")
            ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("soporte mensual mediano de la FS (log)")
            ax[0, 0].set_ylabel("%"); ax[0, 0].legend()
            ax[0, 0].set_title("Soporte: muchas series-astilla, el dinero en las gordas")
            # (b) Lorenz del dinero por FS
            ax[0, 1].plot(100 * x_lor, 100 * cum)
            for t in money_targets:
                n_t = lines_for[t]
                ax[0, 1].plot(100 * n_t / len(fs), t, "o", ms=4, color="firebrick")
                ax[0, 1].annotate(f"{t}%→{n_t:,} FS", (100 * n_t / len(fs), t),
                                  textcoords="offset points", xytext=(6, -4), fontsize=8)
            ax[0, 1].set_xlabel("% de series (ordenadas por $ desc)"); ax[0, 1].set_ylabel("% del $ acumulado")
            ax[0, 1].set_title(f"¿Cuántas líneas necesito? (Gini {gini:.2f})")
            # (c) soporte × dinero por FS
            ax[1, 0].scatter(np.clip(sup, 0.5, None), np.clip(usd, 0.5, None), s=4, alpha=0.15)
            for v in (30, 100, 500):
                ax[1, 0].axvline(v, ls=":", lw=0.8, color="gray")
            ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log")
            ax[1, 0].set_xlabel("soporte mensual mediano (log)"); ax[1, 0].set_ylabel("usd_proj de la FS (log)")
            ax[1, 0].set_title("Dónde vive el dinero en el espectro de soporte")
            # (d) longitud de historia: % series vs % dinero
            hb = [0, 6, 12, 24, max(36, float(np.nanmax(hm)) + 1)]
            hn2, _ = np.histogram(hm, bins=hb); hu2, _ = np.histogram(hm, bins=hb, weights=usd)
            xb = np.arange(len(hn2)); lbl = ["<6m", "6-11m", "12-23m", ">=24m"]
            ax[1, 1].bar(xb - 0.2, 100 * hn2 / max(hn2.sum(), 1), width=0.4, label="% de series")
            ax[1, 1].bar(xb + 0.2, 100 * hu2 / max(hu2.sum(), 1e-9), width=0.4, label="% del $")
            ax[1, 1].set_xticks(xb); ax[1, 1].set_xticklabels(lbl)
            ax[1, 1].set_title("Longitud de historia"); ax[1, 1].legend()
            fig.tight_layout(); fig.savefig(out_path, dpi=130)
            try:
                plt.show()
            except Exception:
                pass
            plt.close(fig)
            out["chart"] = out_path
            self._log(f"    gráfico guardado en: {out_path}")
        except ImportError:
            self._log("    (matplotlib no disponible: solo resúmenes numéricos)")
        self._mark("step_2_report_distributions", metadata=out)
        self._done("step_2_report_distributions", f"percentiles + Lorenz + gráfico ({out['chart']})")
        return out


    def step_2b_describe(self, **kwargs) -> dict:
        """Describe las FS (foto/fingerprint): homogeneidades, soporte, economía,
        riesgo. No modifica self.df. La lógica está integrada al final de este fichero."""
        self._require("step_2b_describe", ["step_2_build_support"])
        # integrado: run_raw_diagnosis es función de módulo al final de este fichero
        result = run_raw_diagnosis(self, **kwargs)
        print(result["report"])
        self._mark("step_2b_describe", metadata={"fingerprint": result["fingerprint"],
                                                 "signals": result["signals"]})
        self._done("step_2b_describe", "foto generada")
        return result


# ============================================================
# Helpers de módulo (aislables)
# ============================================================
def _weighted_ss_total(x, w) -> float:
    """Suma de cuadrados total ponderada de x con pesos w."""
    w = np.asarray(w, float); x = np.asarray(x, float)
    wsum = w.sum()
    if wsum <= 0:
        return 0.0
    gm = float((w * x).sum() / wsum)
    return float((w * (x - gm) ** 2).sum())


def _weighted_eta2(x, w, groups, ss_total: float) -> Optional[float]:
    """η² ponderado de x explicado por 'groups' (niveles de una dimensión).
    η² = SS_between / SS_total. None si no hay varianza o <2 niveles."""
    if ss_total <= 0:
        return None
    x = np.asarray(x, float); w = np.asarray(w, float)
    g = pd.Series(groups).astype(str).values
    levels = pd.unique(g)
    if len(levels) < 2:
        return 0.0
    wsum = w.sum()
    gm = float((w * x).sum() / wsum)
    ss_between = 0.0
    for lv in levels:
        m = (g == lv)
        wl = w[m].sum()
        if wl <= 0:
            continue
        ml = float((w[m] * x[m]).sum() / wl)
        ss_between += wl * (ml - gm) ** 2
    return float(min(max(ss_between / ss_total, 0.0), 1.0))


def _weighted_quantiles(values, weights, qs):
    """Cuantiles ponderados: en qué valor cae el q% del PESO (p.ej. del dinero),
    no de las filas. v y w alineados; qs en [0,100]."""
    v = np.asarray(values, float); w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    if len(v) == 0:
        return {q: np.nan for q in qs}
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()
    return {q: float(np.interp(q / 100.0, cw, v)) for q in qs}


def _cell_rate_heterogeneity(rates, ns):
    """Heterogeneidad de las tasas hijas dentro de una celda. ns = n efectivo
    (unidades de pipeline). Devuelve (tasa_celda, spread_pp, rango_pp, dispersión, k):
      - tasa_celda = Σ(n·rate)/Σn (aditiva, no promedio de tasas).
      - spread_pp  = std ponderada de las tasas hijas (legible, comunicación).
      - dispersión = Q de Cochran / (k-1) con pesos inverso-varianza binomial.
        >1 → las hijas difieren MÁS de lo que explica el ruido binomial (real)."""
    rates = np.asarray(rates, float); ns = np.asarray(ns, float)
    m = ns > 0
    rates, ns = rates[m], ns[m]
    k = int(len(rates))
    if k < 2:
        return (float(rates[0]) if k == 1 else np.nan, 0.0, 0.0, np.nan, k)
    wsum = ns.sum()
    p = float((ns * rates).sum() / wsum)
    var_w = float((ns * (rates - p) ** 2).sum() / wsum)
    spread_pp = 100.0 * float(np.sqrt(max(var_w, 0.0)))
    range_pp = 100.0 * float(rates.max() - rates.min())
    pv = p * (1.0 - p)
    if pv <= 0:
        disp = np.nan
    else:
        Q = float(((ns / pv) * (rates - p) ** 2).sum())  # pesos inverso-varianza
        disp = Q / (k - 1)
    return (p, spread_pp, range_pp, disp, k)


def _resolve_fs_dims(cfg, dims, extra_dims=None) -> list:
    """Ordena el conjunto de dimensiones (ya resuelto por complemento en step_0)
    para componer el fs_id, en el ORDEN DEL RAW (el que el analista ve en su
    vista), seguido de extra_dims. Sin duplicados. El orden del id es cosmético
    —los parent_fs_ids colapsan por nombre, no por posición—; usar el orden del
    raw lo hace predecible y evita que un flag (time-varying) lidere el id. Los
    tiers NO ordenan el id: son semántica aguas abajo (dirección, backstop)."""
    ordered = []
    for c in list(dims):
        if c and c not in ordered:
            ordered.append(c)
    for c in (extra_dims or []):
        if c and c not in ordered:
            ordered.append(c)
    return ordered


def _derive_current_year(df: pd.DataFrame, cfg) -> Optional[int]:
    """Año a predecir: el primer año de la projection con pipeline_usd > 0."""
    proj = df[df[cfg.dataset_role_col] == "projection"]
    if len(proj) == 0:
        return None
    pc = proj.copy()
    pc["_yr"] = pc[cfg.period_col].apply(lambda x: x.year if pd.notna(x) and hasattr(x, "year") else None)
    for yr, usd in pc.groupby("_yr")[cfg.pipeline_usd_col].sum().sort_index().items():
        if usd > 0 and yr is not None:
            return int(yr)
    return None


def mean_rate_excluding_synthetic(df, rate_col="step_1_rate_renewal", weight_col=None) -> float:
    """Media de una tasa EXCLUYENDO sintéticas (proporción → excluye)."""
    real = (df[df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0]
            if "step_1_synthetic" in df.columns else df)
    vals = real[rate_col].dropna()
    if len(vals) == 0:
        return np.nan
    if weight_col and weight_col in real.columns:
        w = real.loc[vals.index, weight_col].astype(float)
        if (w > 0).any():
            return float(np.average(vals[w > 0], weights=w[w > 0]))
    return float(vals.mean())


# ============================================================
# step_2b_raw_diagnosis — integrado (sin fichero externo)
# ============================================================
"""
step_2b_raw_diagnosis.py — Etapa de diagnóstico del raw (HITO 1)
=================================================================

Funde el diagnóstico baseline en una etapa del framework. Mira la pipeline
raw —ya en FS (series) por step_2, SIN modificarla ni imponer buckets todavía— y
cuantifica sus problemas estructurales. Es informacional: mide y reporta, NO
añade columnas a self.df (igual que step_3).

DOBLE SALIDA:
  - 'fingerprint': dict estructurado COMPLETO con todo lo medido. Es la pieza
    importante: alimenta al generador y es lo que se audita. Sin límite de
    columnas: itera sobre todas las dims declaradas.
  - 'report': str compacto (formateo mínimo; lo importante es el dict).

SECCIONES:
  1. Columnas: cardinalidad, concentración, degeneración (todas las dims).
  1b. Rate media por valor de dim (ANOVA descriptivo, sin buckets).
  2. FS — economía: concentración (Pareto), distribución de soporte.
  3. FS — trayectoria: history, recencia, gaps, fracción sintética.
  3b. Migración de cuartil train→proj (¿el ranking de train representa proj?).
  4. Rate: CV temporal, SE binomial, regime, patológicas.
  5. Uplift: CV, outliers, regime, inter/intra (n_renewals>=30).
  5b. UPLIFT — DIAGNÓSTICO DE LOS CUATRO MUNDOS (el bloqueo del η² bajo).
  5c. ELASTICIDAD POSIBLE al precio (con aviso de sesgo de supervivencia).
  6. Separabilidad rate↔uplift (justifica las dos rondas).
  7. Simpson z-score jerárquico (naive vs ajustado por SE binomial).
  8. Cruce projection × historia útil (filtro previo).
  9. Perfil de intensidad: proporción de meses por umbral de soporte × dinero.
  10. Señales accionables.

NO HARDCODEA COLUMNAS: todo por cfg; dims de step_metadata. El marcador de
"sin información de descuento" es configurable (default 'Without_Discount_info'),
y aplica tanto a la columna de estado de descuento como a la de incremento.

Integración: ver INTEGRATION_NOTE al final.
"""


try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------- formato mínimo ----------
def _pct(x, total):
    return (x / total * 100) if total else 0.0


def _supp_bucket(n):
    if n < 5:    return "<5"
    if n < 20:   return "5-20"
    if n < 100:  return "20-100"
    if n < 500:  return "100-500"
    return "500+"


_SUPP_ORDER = ["<5", "5-20", "20-100", "100-500", "500+"]


def _pareto_count(values, total, pct):
    if total <= 0:
        return 0
    s = np.sort(values)[::-1]
    cum = np.cumsum(s)
    idx = np.searchsorted(cum, total * pct / 100.0)
    return int(idx) + 1


def _w_avg(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not m.any():
        return np.nan
    return float(np.average(values[m], weights=weights[m]))


def _w_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not m.any():
        return np.nan
    v, w = values[m], weights[m]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.nan
    cutoff = q * cw[-1]
    idx = np.searchsorted(cw, cutoff)
    idx = min(idx, len(v) - 1)
    return float(v[idx])


class _Progress:
    def __init__(self, total, desc="Diagnóstico raw"):
        self.bar = tqdm(total=total, desc=desc) if _HAS_TQDM else None

    def step(self, desc):
        if self.bar:
            self.bar.set_postfix_str(desc)
            self.bar.update(1)

    def close(self):
        if self.bar:
            self.bar.close()


# ============================================================
# Núcleo
# ============================================================
def run_raw_diagnosis(sf, df=None, *,
                        no_discount_marker: str = "Without_Discount_info",
                        discount_state_col: str | None = None,
                        price_increase_col: str | None = None,
                        recency_stale_months: int = 6,
                        regime_shift_pp: float = 0.10,
                        cv_high: float = 0.10,
                        se_low: float = 0.02,
                        seasonal_min_months: int = 24,
                        uplift_min_renewals: int = 30) -> dict:
    """
    Diagnóstico del raw sobre df post-step_2. Devuelve dict con 'report',
    'fingerprint' (huella completa) y 'signals'. No modifica sf.df.

    Args clave:
        no_discount_marker: valor que marca "sin info de descuento" en las
            columnas de pricing (mismo marcador para descuento e incremento).
        discount_state_col / price_increase_col: nombres de las dos columnas
            categóricas de pricing. Si None, se intentan detectar entre las
            dims por contener el marcador; si no aparecen, las secciones 5b/5c
            se omiten con aviso.
    """
    if df is None:
        df = sf.df.copy()
    if "step_2_build_identity" not in sf.step_metadata:
        raise ValueError("run_raw_diagnosis: step_2_build_identity debe ejecutarse antes.")

    cfg = sf.config.cfg
    fs_dims = sf.step_metadata["step_2_build_identity"]["fs_dims"]
    period_col = cfg.period_col
    role_col = cfg.dataset_role_col
    pu_col = cfg.pipeline_units_col
    pusd_col = cfg.pipeline_usd_col
    ru_col = cfg.renewed_units_col
    rusd_col = cfg.renewed_usd_col
    uplift_col = "step_1_auv_uplift_ratio"
    rate_col = "step_1_rate_renewal"
    synth_col = "step_1_synthetic"

    lines = []
    def P(s=""):
        lines.append(s)

    fp: dict = {"columns": {}, "fu": {}, "fu_series": {}, "uplift_worlds": {},
                "elasticity": {}, "cross": {}, "simpson": {}}
    signals: list[str] = []
    prog = _Progress(total=12)
    prog.step("setup")

    # detectar columnas de pricing si no se pasaron. Buscamos el marcador en
    # TODAS las columnas del df (no solo en fs_dims): las columnas de pricing
    # pueden existir en el raw sin ser dimensiones de la FS.
    if discount_state_col is None or price_increase_col is None:
        candidates = [c for c in df.columns
                      if df[c].astype(str).eq(no_discount_marker).any()]
        if discount_state_col is None and len(candidates) >= 1:
            discount_state_col = candidates[0]
        if price_increase_col is None and len(candidates) >= 2:
            price_increase_col = candidates[1]

    # ---- current year ----
    proj_df = df[df[role_col] == "projection"]
    current_year = None
    if len(proj_df) > 0:
        pc = proj_df.copy()
        pc["_yr"] = pc[period_col].apply(
            lambda x: x.year if pd.notna(x) and hasattr(x, "year") else None)
        for yr, usd in pc.groupby("_yr")[pusd_col].sum().sort_index().items():
            if usd > 0:
                current_year = int(yr)
                break

    normal = df[df.get("step_1_universe", "normal") == "normal"]
    fu_table = (normal.dropna(subset=["step_2_fs_id"])
                .drop_duplicates("step_2_fs_id").copy())
    n_fs = len(fu_table)
    hist = normal[normal[role_col].isin(cfg.history_roles)]  # historia observada: train ∪ test
    n_history_periods = float(
        sf.step_metadata["step_2_build_identity"].get("n_history_periods", 1)) or 1.0

    raw_path = getattr(sf.config, "raw_data_path", None) or "(df directo)"
    P(f"DIAGNÓSTICO DEL RAW — {os.path.basename(str(raw_path))}")
    P(f"dims: {len(fs_dims)} | FS: {n_fs:,} | current_year: {current_year}")
    P(f"pricing cols: discount={discount_state_col} increase={price_increase_col}")
    P("")

    # ============================================================
    # S1 — Columnas
    # ============================================================
    prog.step("S1 columnas")
    cart = 1
    for d in fs_dims:
        cart *= max(len(df[d].dropna().unique()), 1)
    cols_fp = {}
    P("[S1] Columnas (cardinalidad / concentración / degeneración):")
    for d in fs_dims:
        usd_per_val = (df.groupby(d)[pusd_col].sum() / n_history_periods
                       ).sort_values(ascending=False)
        if len(usd_per_val) == 0:
            continue
        total = usd_per_val.sum()
        top_share = (usd_per_val.iloc[0] / total * 100) if total else 0.0
        cum = (usd_per_val.cumsum() / total * 100) if total else usd_per_val * 0
        n_cover = int((cum >= 80).values.argmax()) + 1 if (cum >= 80).any() else len(cum)
        degenerate = top_share >= 75.0
        cols_fp[d] = {
            "n_vals": int(len(usd_per_val)),
            "top1_usd_share_pct": round(float(top_share), 1),
            "n_vals_cover_80pct": int(n_cover),
            "degenerate": bool(degenerate),
            "usd_share_by_value_pct": {
                str(k): round(float(v / total * 100), 2)
                for k, v in usd_per_val.items()} if total else {},
        }
        P(f"  {d}: {len(usd_per_val)} vals, top1={top_share:.0f}%, "
          f"80% en {n_cover} vals{' [DEGENERADA]' if degenerate else ''}")
        if degenerate:
            signals.append(f"Dim '{d}' degenerada: top1={top_share:.0f}% USD.")
    fp["columns"] = {"n_dims": len(fs_dims), "cartesian_product": int(cart),
                     "fus_built": int(n_fs),
                     "populated_pct": round(_pct(n_fs, cart), 3),
                     "per_dim": cols_fp}

    # ---- S1b rate por dim ----
    prog.step("S1b rate-por-dim")
    train_pos = hist[(hist[pu_col] > 0) & (hist[rate_col].notna())]
    rate_by_dim_fp = {}
    P("[S1b] Spread de rate entre valores de cada dim (criterio Distinta):")
    if len(train_pos) > 0:
        for d in fs_dims:
            g = train_pos.groupby(d).apply(
                lambda x: _w_avg(x[rate_col].values, x[pu_col].values)).dropna()
            if len(g) < 2:
                continue
            spread = (g.max() - g.min()) * 100
            rate_by_dim_fp[d] = {"rate_min": round(float(g.min()), 4),
                                 "rate_max": round(float(g.max()), 4),
                                 "spread_pp": round(float(spread), 2)}
            P(f"  {d}: {g.min()*100:.1f}%–{g.max()*100:.1f}% (spread {spread:.1f}pp)")
    fp["columns"]["rate_spread_by_dim"] = rate_by_dim_fp

    # ============================================================
    # S2 — FS economía + soporte
    # ============================================================
    prog.step("S2 economía")
    if "step_2_usd_proj" in normal.columns:
        # reutilizar el dinero a predecir ya calculado en step_2 (coherencia)
        usd_proj_by_fu = (normal.dropna(subset=["step_2_fs_id"])
                          .drop_duplicates("step_2_fs_id")
                          .set_index("step_2_fs_id")["step_2_usd_proj"])
    elif current_year is not None:
        proj_curr = normal[(normal[role_col] == "projection") &
                           (normal[period_col].apply(
                               lambda x: x.year == current_year if pd.notna(x) else False))]
        usd_proj_by_fu = proj_curr.groupby("step_2_fs_id")[pusd_col].sum()
    else:
        usd_proj_by_fu = pd.Series(dtype=float)
    fu_table["usd_proj"] = fu_table["step_2_fs_id"].map(usd_proj_by_fu).fillna(0)
    fu_table["usd_train_annual"] = fu_table["step_2_usd_avg"] * 12
    total_usd_proj = fu_table["usd_proj"].sum()
    total_usd_train = fu_table["usd_train_annual"].sum()

    supp = fu_table["step_2_n_avg"].dropna()
    supp_fp = {}
    for b in _SUPP_ORDER:
        cnt = int((supp.apply(_supp_bucket) == b).sum())
        usd_b = fu_table[fu_table["step_2_n_avg"].apply(_supp_bucket) == b]["usd_proj"].sum()
        supp_fp[b] = {"n_fs": cnt, "usd_proj": round(float(usd_b), 0)}
    tiny = fu_table[fu_table["step_2_n_avg"] < 5]
    tiny_usd = tiny["usd_proj"].sum()
    P(f"[S2] FS={n_fs}, USD_proj=${total_usd_proj:,.0f}. "
      f"n_avg<5: {len(tiny)} FS ({_pct(tiny_usd, total_usd_proj):.0f}% USD).")
    if _pct(tiny_usd, total_usd_proj) > 10:
        signals.append(f"Soporte escaso material: n_avg<5 = "
                       f"{_pct(tiny_usd, total_usd_proj):.0f}% USD proj.")
    fp["fu"] = {
        "n_fs": int(n_fs), "total_usd_proj": round(float(total_usd_proj), 0),
        "total_usd_train_annual": round(float(total_usd_train), 0),
        "pareto": {f"n_fs_to_{p}pct": _pareto_count(fu_table["usd_proj"].values, total_usd_proj, p)
                   for p in (50, 80, 90, 95)} if total_usd_proj > 0 else {},
        "support_distribution": supp_fp,
        "n_avg_percentiles": {q: round(float(supp.quantile(v)), 1) if len(supp) else None
                              for q, v in [("p25", .25), ("p50", .5), ("p75", .75)]},
        "tiny_fus_n_avg_lt5": {"n_fs": int(len(tiny)),
                               "usd_proj_pct": round(_pct(tiny_usd, total_usd_proj), 1)},
    }

    # ============================================================
    # S2b — DINERO EN RIESGO: deciles de dinero × información disponible
    # ============================================================
    # El error tolerable del forecast está acotado por cuánto dinero está en FS
    # que NO podemos predecir bien. No importa fallar en calderilla; importa el
    # dinero concentrado en FS con poca información (poco soporte, poca historia,
    # o sin cobertura entrenable). Esta sección cuantifica ese riesgo.
    prog.step("S2b dinero en riesgo")
    risk_fp = {}

    # (0) DINERO POR RUTA DE FORECAST (forecast_route): cuánto USD proyectado hay
    #     en cada veredicto de cobertura. En no_impact DEBE ser 0 (sin projection);
    #     si no lo es, hay una incoherencia → WARNING (no bloquea). El desglose
    #     por ruta ya se pinta en el log de step_1; aquí solo se guarda al fingerprint.
    if "step_1_forecast_route" in normal.columns:
        route_by_fu = (normal.dropna(subset=["step_2_fs_id"])
                       .drop_duplicates("step_2_fs_id")
                       .set_index("step_2_fs_id")["step_1_forecast_route"])
        ft_route = fu_table.copy()
        ft_route["route"] = ft_route["step_2_fs_id"].map(route_by_fu).fillna("unknown")
        route_money = {}
        for r in ["no_impact", "heuristic", "trainable", "unknown"]:
            grp = ft_route[ft_route["route"] == r]
            if len(grp) == 0:
                continue
            usd_r = float(grp["usd_proj"].sum())
            route_money[r] = {"n_fs": int(len(grp)), "usd_proj": round(usd_r, 0),
                              "usd_pct": round(_pct(usd_r, total_usd_proj), 1)}
        risk_fp["money_by_route"] = route_money
        ni_usd = route_money.get("no_impact", {}).get("usd_proj", 0)
        if ni_usd and ni_usd > 0:
            signals.append(f"Incoherencia: no_impact con ${ni_usd:,.0f} usd_proj (debería ser 0).")

    if total_usd_proj > 0:
        ft = fu_table.copy()
        # añadir historia y cobertura (de step_2 / step_1) si están en el df
        hist_by_fu = (normal.dropna(subset=["step_2_fs_id"])
                      .drop_duplicates("step_2_fs_id")
                      .set_index("step_2_fs_id"))
        for col, default in [("step_2_history_months", np.nan),
                              ("step_1_coverage_pattern", "unknown")]:
            if col in hist_by_fu.columns:
                ft[col] = ft["step_2_fs_id"].map(hist_by_fu[col]).fillna(default)

        # (1) DECILES DE DINERO: ordena las FS por usd_proj desc y reparte el USD
        #     en 10 tramos iguales de DINERO; mira qué información tiene cada tramo.
        ft_sorted = ft.sort_values("usd_proj", ascending=False).reset_index(drop=True)
        ft_sorted["cum_usd"] = ft_sorted["usd_proj"].cumsum()
        # decil por dinero acumulado; una FS grande puede ocupar su decil entera.
        # se asigna por el borde superior del USD acumulado (1..10), con el cuidado
        # de que el primer FS caiga en el decil 1 aunque supere el 10% él solo.
        prev_cum = ft_sorted["cum_usd"].shift(1).fillna(0)
        ft_sorted["usd_decile"] = np.clip(
            np.ceil(((prev_cum + ft_sorted["usd_proj"] / 2) / total_usd_proj) * 10).astype(int),
            1, 10)
        P("[S2b] Deciles de DINERO (cada uno = 10% del USD proyectado):")
        P("      decil | nFS | n_avg_med | hist_med | %FS n_avg<5")
        decile_rows = []
        for d in range(1, 11):
            grp = ft_sorted[ft_sorted["usd_decile"] == d]
            if len(grp) == 0:
                continue
            n = len(grp)
            nam = float(grp["step_2_n_avg"].median()) if grp["step_2_n_avg"].notna().any() else np.nan
            hm = (float(grp["step_2_history_months"].median())
                  if "step_2_history_months" in grp.columns and grp["step_2_history_months"].notna().any() else np.nan)
            pct_micro = _pct(int((grp["step_2_n_avg"] < 5).sum()), n)
            decile_rows.append({"decile": d, "n_fs": n, "n_avg_med": round(nam, 1),
                                "hist_med": round(hm, 1) if np.isfinite(hm) else None,
                                "pct_micro": round(pct_micro, 0)})
            P(f"      {d:>5} | {n:>4} | {nam:>9.1f} | "
              f"{hm:>8.1f} | {pct_micro:>4.0f}%" if np.isfinite(hm)
              else f"      {d:>5} | {n:>4} | {nam:>9.1f} |      n/a | {pct_micro:>4.0f}%")
        risk_fp["money_deciles"] = decile_rows

        # (2) DINERO EN RIESGO: USD en FS poco predecibles, por criterio.
        #     Tres cortes de "poca información" (no excluyentes), con su % de USD.
        def _usd_pct(mask):
            return round(_pct(ft.loc[mask, "usd_proj"].sum(), total_usd_proj), 1)
        risk = {
            "usd_pct_en_soporte_bajo_n_avg_lt5": _usd_pct(ft["step_2_n_avg"] < 5),
            "usd_pct_en_soporte_medio_5_30": _usd_pct((ft["step_2_n_avg"] >= 5) & (ft["step_2_n_avg"] < 30)),
        }
        if "step_2_history_months" in ft.columns:
            risk["usd_pct_en_historia_corta_lt12m"] = _usd_pct(ft["step_2_history_months"] < 12)
            risk["usd_pct_en_historia_muy_corta_lt6m"] = _usd_pct(ft["step_2_history_months"] < 6)
        if "step_1_coverage_pattern" in ft.columns:
            no_train = ft["step_1_coverage_pattern"].isin(["projection_only", "test_only", "projection_test"])
            risk["usd_pct_sin_train_entrenable"] = _usd_pct(no_train)
        risk_fp["money_at_risk"] = risk
        P("[S2b] DINERO EN RIESGO (% del USD proyectado en FS poco predecibles):")
        for k, v in risk.items():
            P(f"      {k}: {v}%")
        # señal si hay mucho dinero con poca información
        worst = max(risk.values()) if risk else 0
        if risk.get("usd_pct_en_soporte_bajo_n_avg_lt5", 0) >= 10:
            signals.append(f"Dinero en riesgo por soporte: "
                           f"{risk['usd_pct_en_soporte_bajo_n_avg_lt5']}% del USD en FS n_avg<5.")
        if risk.get("usd_pct_en_historia_corta_lt12m", 0) >= 15:
            signals.append(f"Dinero en riesgo por historia: "
                           f"{risk['usd_pct_en_historia_corta_lt12m']}% del USD en FS con <12m.")
    fp["fu"]["risk"] = risk_fp

    # ============================================================
    # S3 — temporal
    # ============================================================
    prog.step("S3 temporal")
    hm = fu_table["step_2_history_months"].dropna()
    if synth_col in hist.columns:
        is_synth = hist[synth_col].fillna(False).astype(bool)
        real_train = hist[(hist[pu_col] > 0) & (~is_synth)]
    else:
        real_train = hist[hist[pu_col] > 0]
    last_real = real_train.groupby("step_2_fs_id")[period_col].max()
    last_tp = hist[period_col].max() if len(hist) > 0 else None
    recency = ((last_tp - last_real).apply(lambda x: x.n)
               if (last_tp is not None and len(last_real) > 0) else pd.Series(dtype=float))
    seas = {"lt12": int((hm < 12).sum()),
            "mid": int(((hm >= 12) & (hm < seasonal_min_months)).sum()),
            "ge_seasonal": int((hm >= seasonal_min_months).sum())} if len(hm) else {}
    n_stale = int((recency > recency_stale_months).sum()) if len(recency) else 0
    P(f"[S3] history med={int(hm.median()) if len(hm) else 0}m, "
      f"≥{seasonal_min_months}m(estacional): {seas.get('ge_seasonal', 0)}, "
      f"recencia>{recency_stale_months}m: {n_stale} FS.")
    if len(recency) and _pct(n_stale, n_fs) > 10:
        signals.append(f"{_pct(n_stale, n_fs):.0f}% FS con recencia>"
                       f"{recency_stale_months}m (forecast sobre datos viejos).")
    fp["fu_series"]["history_months"] = {
        "p50": int(hm.median()) if len(hm) else None, "seasonal_buckets": seas}
    fp["fu_series"]["recency"] = {
        "p50": int(recency.median()) if len(recency) else None,
        "max": int(recency.max()) if len(recency) else None, "n_stale": n_stale}

    # ---- S3b migración de cuartil ----
    prog.step("S3b migración")
    mig_fp = {}
    if total_usd_train > 0 and total_usd_proj > 0 and n_fs >= 8:
        ft = fu_table.copy()
        try:
            ft["q_tr"] = pd.qcut(ft["usd_train_annual"].rank(method="first"),
                                 4, labels=["Q1", "Q2", "Q3", "Q4"])
            ft["q_pr"] = pd.qcut(ft["usd_proj"].rank(method="first"),
                                 4, labels=["Q1", "Q2", "Q3", "Q4"])
            diag = int((ft["q_tr"] == ft["q_pr"]).sum())
            mig_fp = {"pct_same_quartile": round(_pct(diag, n_fs), 1)}
            P(f"[S3b] {_pct(diag, n_fs):.0f}% FS mantienen cuartil train→proj.")
            if _pct(diag, n_fs) < 50:
                signals.append(f"Solo {_pct(diag, n_fs):.0f}% FS mantienen "
                               f"cuartil train→proj (ranking train no representa futuro).")
        except (ValueError, IndexError):
            pass
    fp["cross"]["quartile_migration"] = mig_fp

    # ============================================================
    # S4 — Rate
    # ============================================================
    prog.step("S4 rate")
    rate_stats = _series_stats(hist, period_col, pu_col, ru_col, rate_col, "rate",
                                regime_shift_pp)
    rate_stats = rate_stats.merge(fu_table[["step_2_fs_id", "usd_proj"]],
                                   on="step_2_fs_id", how="left")
    rate_valid = rate_stats[rate_stats["mean"].notna()]
    n_rate_valid = len(rate_valid)
    rate_fp = {}
    if n_rate_valid > 0:
        rcv = rate_valid["cv_temp"].dropna()
        patho = rate_valid[(rate_valid["cv_temp"] > cv_high) & (rate_valid["se_binom"] < se_low)]
        regime = rate_valid[rate_valid["regime_shift"].abs() > regime_shift_pp]
        rate_fp = {
            "mean_usd_weighted": round(_w_avg(rate_valid["mean"], rate_valid["usd_proj"]), 4),
            "cv_temp_p50": round(float(rcv.median()), 4) if len(rcv) else None,
            "cv_temp_p95": round(float(rcv.quantile(0.95)), 4) if len(rcv) else None,
            "n_pathological": int(len(patho)),
            "n_regime_change": int(len(regime)),
            "pct_regime_change": round(_pct(len(regime), n_rate_valid), 1)}
        # tendencia formal (para el shrinkage: no agrupar tendencias opuestas)
        if "trend_direction" in rate_valid.columns:
            dirs = rate_valid["trend_direction"].value_counts().to_dict()
            n_up = int(dirs.get("up", 0))
            n_down = int(dirs.get("down", 0))
            n_stable = int(dirs.get("stable", 0))
            rate_fp["trend_up"] = n_up
            rate_fp["trend_down"] = n_down
            rate_fp["trend_stable"] = n_stable
            rate_fp["slope_pp_month_p50"] = (round(float(rate_valid["slope_pp_month"].median()), 3)
                                              if rate_valid["slope_pp_month"].notna().any() else None)
        P(f"[S4] rate={rate_fp['mean_usd_weighted']:.3f}, CV_med={rate_fp['cv_temp_p50']}, "
          f"patológicas={len(patho)}, regime>{regime_shift_pp:.0%}={len(regime)} "
          f"({_pct(len(regime), n_rate_valid):.0f}%).")
        if "trend_up" in rate_fp:
            P(f"[S4] tendencia: ↑{rate_fp['trend_up']} ↓{rate_fp['trend_down']} "
              f"→{rate_fp['trend_stable']} estables (slope_med="
              f"{rate_fp['slope_pp_month_p50']}pp/mes). Mezclar ↑ y ↓ en un grupo de "
              f"shrinkage sería error.")
            if n_up > 0 and n_down > 0:
                signals.append(f"Tendencias opuestas coexisten (↑{n_up}/↓{n_down}): "
                               f"el shrinkage no debe agruparlas.")
        if _pct(len(regime), n_rate_valid) > 10:
            signals.append(f"Regime change rate material: {_pct(len(regime), n_rate_valid):.0f}%.")
    fp["fu_series"]["rate"] = rate_fp

    # ============================================================
    # S5 — Uplift básico
    # ============================================================
    prog.step("S5 uplift")
    upl_fp = {}
    upl_valid = pd.DataFrame()
    if uplift_col in hist.columns:
        upl_stats = _uplift_stats(hist, period_col, pu_col, pusd_col, ru_col,
                                   rusd_col, regime_shift_pp)
        upl_stats = upl_stats.merge(fu_table[["step_2_fs_id", "usd_proj"]],
                                     on="step_2_fs_id", how="left")
        upl_valid = upl_stats[upl_stats["n_renewals"] >= uplift_min_renewals]
        n_upl_valid = len(upl_valid)
        if n_upl_valid > 0:
            w = upl_valid["usd_proj"].values
            if np.nansum(w) == 0:
                w = np.ones(len(upl_valid))
            ucv = upl_valid["uplift_cv_temp"].dropna()
            u_iqr = (_w_quantile(upl_valid["uplift_obs"].values, w, 0.75) -
                     _w_quantile(upl_valid["uplift_obs"].values, w, 0.25))
            intra_med = float(ucv.median()) if len(ucv) else np.nan
            ratio = (u_iqr / intra_med) if (intra_med and intra_med > 0) else float("inf")
            extreme = upl_valid[(upl_valid["uplift_obs"] < 0.1) | (upl_valid["uplift_obs"] > 5)]
            upl_regime = upl_valid[upl_valid["uplift_regime"].abs() > regime_shift_pp]
            upl_fp = {
                "n_evaluable": int(n_upl_valid),
                "min_renewals_threshold": uplift_min_renewals,
                "uplift_p50_usd_weighted": round(_w_quantile(upl_valid["uplift_obs"].values, w, 0.5), 4),
                "uplift_cv_temp_p50": round(intra_med, 4) if np.isfinite(intra_med) else None,
                "inter_iqr": round(float(u_iqr), 4),
                "intra_cv_med": round(intra_med, 4) if np.isfinite(intra_med) else None,
                "inter_over_intra_ratio": round(float(ratio), 2) if np.isfinite(ratio) else None,
                "n_extreme_outliers": int(len(extreme)),
                "n_regime_change": int(len(upl_regime))}
            P(f"[S5] uplift={upl_fp['uplift_p50_usd_weighted']:.3f}, "
              f"inter/intra={upl_fp['inter_over_intra_ratio']}x "
              f"({'dims NO explican' if ratio < 2 else 'señal estructural'}).")
            if ratio < 2:
                signals.append("Dims actuales no explican uplift (inter/intra<2x) "
                               "— ver diagnóstico de los 4 mundos (S5b).")
    fp["fu_series"]["uplift"] = upl_fp

    # ============================================================
    # S5b — UPLIFT: DIAGNÓSTICO DE LOS CUATRO MUNDOS
    #   Opción A: ANOVA partido por completitud de pricing.
    #   Distingue: contractual / idiosincrático / faltan dims / datos.
    # ============================================================
    prog.step("S5b 4 mundos")
    worlds = _uplift_four_worlds(
        hist, fs_dims, period_col, pu_col, pusd_col, ru_col, rusd_col,
        rate_col, uplift_col, discount_state_col, price_increase_col,
        no_discount_marker, P, signals)
    fp["uplift_worlds"] = worlds

    # ============================================================
    # S5c — ELASTICIDAD POSIBLE (con aviso de sesgo de supervivencia)
    # ============================================================
    prog.step("S5c elasticidad")
    elast = _price_elasticity(
        hist, period_col, pu_col, pusd_col, ru_col, rusd_col, rate_col,
        uplift_col, discount_state_col, price_increase_col, no_discount_marker, P)
    fp["elasticity"] = elast

    # ============================================================
    # S6 — Separabilidad rate↔uplift
    # ============================================================
    prog.step("S6 separabilidad")
    corr = np.nan
    if len(upl_valid) > 5 and n_rate_valid > 0:
        m = rate_valid[["step_2_fs_id", "mean", "usd_proj"]].rename(
            columns={"mean": "r"}).merge(
            upl_valid[["step_2_fs_id", "uplift_obs"]].rename(columns={"uplift_obs": "u"}),
            on="step_2_fs_id", how="inner").dropna()
        if len(m) > 5:
            w = np.where(m["usd_proj"].values > 0, m["usd_proj"].values, 0).astype(float)
            if w.sum() > 0:
                rm, um = np.average(m["r"], weights=w), np.average(m["u"], weights=w)
                cov = np.average((m["r"]-rm)*(m["u"]-um), weights=w)
                vr, vu = np.average((m["r"]-rm)**2, weights=w), np.average((m["u"]-um)**2, weights=w)
                if vr > 0 and vu > 0:
                    corr = float(cov/np.sqrt(vr*vu))
    if np.isfinite(corr):
        P(f"[S6] corr rate↔uplift={corr:+.3f} "
          f"({'redundantes' if abs(corr)>0.7 else 'necesarias' if abs(corr)<0.3 else 'media'}).")
    fp["cross"]["rate_uplift_corr"] = round(corr, 3) if np.isfinite(corr) else None

    # ============================================================
    # S7 — Simpson z-score jerárquico
    # ============================================================
    prog.step("S7 Simpson")
    simpson_fp = _simpson_zscore(rate_valid, fu_table, fs_dims, P)
    fp["simpson"] = simpson_fp

    # ============================================================
    # S8 — Cruce projection × historia
    # ============================================================
    prog.step("S8 cruce")
    in_proj = fu_table["usd_proj"] > 0
    has_hist = fu_table["step_2_n_periods"].fillna(0) >= 6
    usd_pn = fu_table[in_proj & ~has_hist]["usd_proj"].sum()
    quad = {"in_proj_with_hist": int((in_proj & has_hist).sum()),
            "in_proj_no_hist": int((in_proj & ~has_hist).sum()),
            "no_proj_with_hist": int((~in_proj & has_hist).sum()),
            "no_proj_no_hist": int((~in_proj & ~has_hist).sum()),
            "usd_in_proj_no_hist_pct": round(_pct(usd_pn, total_usd_proj), 1) if total_usd_proj else 0}
    P(f"[S8] proj×hist: con_hist={quad['in_proj_with_hist']}, "
      f"sin_hist={quad['in_proj_no_hist']} ({quad['usd_in_proj_no_hist_pct']}% USD→shrink).")
    if quad["usd_in_proj_no_hist_pct"] > 15:
        signals.append(f"{quad['usd_in_proj_no_hist_pct']}% USD proj sin historia "
                       f"útil (caso de poco soporte material).")
    fp["cross"]["projection_x_history"] = quad

    # (S9 'Perfil de intensidad' eliminada: la densidad por sample size se mide
    #  ahora en step_1_report_density y su cruce con el dinero de projection en
    #  step_2_report_density_money. step_2b ya no recalcula densidad.)

    # ---- señales ----
    prog.step("señales")
    P("")
    P("[SEÑALES]")
    for s in (signals or ["(sin señales material)"]):
        P(f"  • {s}")
    prog.close()

    return {"report": "\n".join(lines), "fingerprint": fp, "signals": signals}


# ============================================================
# Helpers de series
# ============================================================
def _weighted_linear_trend(y, weights=None):
    """Regresión y ~ a + b*t (t=0..n-1). Devuelve (slope, r_squared, n)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return np.nan, np.nan, n
    t = np.arange(n, dtype=float)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return np.nan, np.nan, n
    tbar = np.average(t, weights=w)
    ybar = np.average(y, weights=w)
    cov = np.average((t - tbar) * (y - ybar), weights=w)
    var_t = np.average((t - tbar) ** 2, weights=w)
    if var_t <= 0:
        return np.nan, np.nan, n
    slope = cov / var_t
    yhat = ybar + slope * (t - tbar)
    ss_res = np.average((y - yhat) ** 2, weights=w)
    ss_tot = np.average((y - ybar) ** 2, weights=w)
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return slope, r2, n


def _series_stats(train, period_col, pu_col, ru_col, value_col, kind, regime_pp):
    _cols = ["step_2_fs_id", "mean", "cv_temp", "se_binom", "regime_shift",
             "slope_pp_month", "trend_r2", "trend_direction"]
    df = train[(train[pu_col] > 0) & (train[value_col].notna())].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=_cols)
    df = df.sort_values(["step_2_fs_id", period_col])
    out = []
    for fu, g in df.groupby("step_2_fs_id"):
        if len(g) < 6:
            continue
        vals = g[value_col].values.astype(float)
        w = g[pu_col].values.astype(float)
        if w.sum() <= 0:
            continue
        mean = np.average(vals, weights=w)
        sd = np.sqrt(np.average((vals - mean) ** 2, weights=w))
        cv = (sd / mean) if mean else np.nan
        n_avg = w.mean()
        se = np.sqrt(mean*(1-mean)/n_avg) if (kind == "rate" and n_avg > 0 and 0 <= mean <= 1) else np.nan
        t = len(g) // 3
        shift = np.nan
        if t >= 1:
            old = np.average(vals[:t], weights=w[:t]) if w[:t].sum() > 0 else np.nan
            new = np.average(vals[-t:], weights=w[-t:]) if w[-t:].sum() > 0 else np.nan
            shift = (new - old) if (np.isfinite(old) and np.isfinite(new)) else np.nan
        # tendencia formal (regresión ponderada por soporte)
        slope, r2, _ = _weighted_linear_trend(vals, w)
        slope_pp = slope * 100 if np.isfinite(slope) else np.nan  # pp/mes para rate
        direction = "unknown"
        if np.isfinite(slope_pp) and np.isfinite(r2):
            if abs(slope_pp) <= 0.5 or r2 < 0.3:
                direction = "stable"
            elif slope_pp > 0.5:
                direction = "up"
            elif slope_pp < -0.5:
                direction = "down"
        out.append({"step_2_fs_id": fu, "mean": mean, "cv_temp": cv,
                    "se_binom": se, "regime_shift": shift,
                    "slope_pp_month": slope_pp, "trend_r2": r2,
                    "trend_direction": direction})
    return pd.DataFrame(out, columns=_cols)


def _uplift_stats(train, period_col, pu_col, pusd_col, ru_col, rusd_col, regime_pp):
    df = train[(train[pu_col] > 0)].copy().sort_values(["step_2_fs_id", period_col])
    df["_u"] = np.where((df[ru_col] > 0) & (df[pu_col] > 0) & (df[pusd_col] > 0),
                        (df[rusd_col]/df[ru_col]) / (df[pusd_col]/df[pu_col]), np.nan)
    out = []
    for fu, g in df.groupby("step_2_fs_id"):
        ru_sum = g[ru_col].sum()
        pu_sum, pusd_sum, rusd_sum = g[pu_col].sum(), g[pusd_col].sum(), g[rusd_col].sum()
        if ru_sum <= 0 or pu_sum <= 0 or pusd_sum <= 0:
            continue
        uplift_obs = (rusd_sum/ru_sum) / (pusd_sum/pu_sum)
        std_temp = g["_u"].std()
        cv = (std_temp/uplift_obs) if uplift_obs else np.nan
        t = len(g) // 3
        shift = np.nan
        if t >= 1:
            g1, g3 = g.iloc[:t], g.iloc[-t:]
            def _u(gg):
                r, p, ru_, pu_ = gg[rusd_col].sum(), gg[pusd_col].sum(), gg[ru_col].sum(), gg[pu_col].sum()
                return (r/ru_)/(p/pu_) if (ru_ > 0 and pu_ > 0 and p > 0) else np.nan
            u1, u3 = _u(g1), _u(g3)
            shift = (u3-u1) if (np.isfinite(u1) and np.isfinite(u3)) else np.nan
        out.append({"step_2_fs_id": fu, "uplift_obs": uplift_obs,
                    "uplift_cv_temp": cv, "uplift_regime": shift,
                    "n_renewals": int(ru_sum)})
    return pd.DataFrame(out, columns=["step_2_fs_id", "uplift_obs",
                                      "uplift_cv_temp", "uplift_regime", "n_renewals"])


def _row_uplift(d, ru_col, pu_col, rusd_col, pusd_col):
    return np.where((d[ru_col] > 0) & (d[pu_col] > 0) & (d[pusd_col] > 0),
                    (d[rusd_col]/d[ru_col]) / (d[pusd_col]/d[pu_col]), np.nan)


# ============================================================
# S5b — los cuatro mundos del uplift (opción A)
# ============================================================
def _uplift_four_worlds(train, fs_dims, period_col, pu_col, pusd_col, ru_col,
                          rusd_col, rate_col, uplift_col, disc_col, inc_col,
                          marker, P, signals):
    """
    Distingue por qué el η² del uplift es bajo:
      Mundo 0 (datos): cobertura de pricing incompleta / sin varianza.
      Mundo 1 (contractual): uplift uniforme, std intra baja, η² bajo.
      Mundo 2 (idiosincrático): std intra alta, η² bajo.
      Mundo 3 (faltan dims): señal existe en parte con datos.
    Opción A: ANOVA partido por completitud (conocido vs sin info).
    """
    res = {}
    d = train[(train[pu_col] > 0) & (train[ru_col] > 0) & (train[pusd_col] > 0)].copy()
    if len(d) == 0:
        P("[S5b] sin filas con renovación para diagnóstico de uplift.")
        return {"available": False}
    d["_u"] = _row_uplift(d, ru_col, pu_col, rusd_col, pusd_col)
    d = d[d["_u"].notna()].copy()

    # ---- cobertura de pricing ----
    cov = {}
    if disc_col and disc_col in d.columns:
        known = d[d[disc_col].astype(str) != marker]
        unknown = d[d[disc_col].astype(str) == marker]
        cov = {
            "discount_col": disc_col,
            "pct_rows_known": round(_pct(len(known), len(d)), 1),
            "pct_renewals_known": round(_pct(known[ru_col].sum(), d[ru_col].sum()), 1),
            "uplift_mean_known": round(float(_w_avg(known["_u"], known[ru_col])), 4) if len(known) else None,
            "uplift_mean_unknown": round(float(_w_avg(unknown["_u"], unknown[ru_col])), 4) if len(unknown) else None,
            "uplift_std_known": round(float(known["_u"].std()), 4) if len(known) else None,
            "uplift_std_unknown": round(float(unknown["_u"].std()), 4) if len(unknown) else None,
        }
        P(f"[S5b] cobertura descuento: {cov['pct_rows_known']}% filas con dato. "
          f"uplift conocido={cov['uplift_mean_known']} (std {cov['uplift_std_known']}) "
          f"vs sin info={cov['uplift_mean_unknown']} (std {cov['uplift_std_unknown']}).")
        # missing informativo
        if (cov["uplift_mean_known"] and cov["uplift_mean_unknown"] and
                abs(cov["uplift_mean_known"] - cov["uplift_mean_unknown"]) > 0.05):
            signals.append("El 'sin info de descuento' tiene uplift medio distinto "
                           "→ el missing es informativo (no aleatorio).")
    res["coverage"] = cov

    # ---- ANOVA partido por completitud (opción A) ----
    pricing_dims = [c for c in [disc_col, inc_col] if c and c in d.columns]
    other_dims = [c for c in fs_dims if c not in pricing_dims]
    split = {}
    if pricing_dims:
        known_mask = np.logical_and.reduce(
            [d[c].astype(str) != marker for c in pricing_dims])
        d_known = d[known_mask]
        d_unknown = d[~known_mask]
        eta_known = _quick_eta2(d_known, fs_dims, "_u", ru_col)
        eta_unknown = _quick_eta2(d_unknown, other_dims or fs_dims, "_u", ru_col)
        split = {
            "eta2_known_pricing": round(eta_known, 4) if eta_known is not None else None,
            "eta2_unknown_pricing": round(eta_unknown, 4) if eta_unknown is not None else None,
            "n_known": int(len(d_known)), "n_unknown": int(len(d_unknown))}
        P(f"[S5b] η² uplift (opción A): con pricing conocido="
          f"{split['eta2_known_pricing']} (n={len(d_known)}) vs "
          f"sin info={split['eta2_unknown_pricing']} (n={len(d_unknown)}).")
        if (eta_known is not None and eta_unknown is not None and
                eta_known > eta_unknown + 0.05):
            signals.append(f"Señal de uplift VIVE en la parte con datos "
                           f"(η² {eta_known:.2f} vs {eta_unknown:.2f}): el η² global "
                           f"bajo es por cobertura, no por falta de elasticidad.")
    res["anova_split_by_completeness"] = split

    # ---- veredicto de los 4 mundos ----
    verdict = _classify_uplift_world(res, P)
    res["verdict"] = verdict
    return res


def _quick_eta2(d, dims, value_col, weight_col):
    """η² del modelo WLS (rápido) de value_col ~ dims categóricas."""
    if len(d) < 30 or not dims:
        return None
    try:
        from statsmodels.formula.api import wls
        dd = d.copy()
        rename = {c: f"x{i}" for i, c in enumerate(dims)}
        for o, s in rename.items():
            dd[s] = dd[o].astype(str)
        dd["_y"] = dd[value_col].astype(float)
        dd["_w"] = pd.to_numeric(dd[weight_col], errors="coerce").fillna(0)
        dd = dd[dd["_w"] > 0]
        if len(dd) < 30:
            return None
        formula = "_y ~ " + " + ".join(f"C({s})" for s in rename.values())
        return float(wls(formula, data=dd, weights=dd["_w"]).fit().rsquared)
    except Exception:
        return None


def _classify_uplift_world(res, P):
    cov = res.get("coverage", {})
    std_known = cov.get("uplift_std_known")
    split = res.get("anova_split_by_completeness", {})
    eta_known = split.get("eta2_known_pricing")

    # Mundo 0: cobertura baja
    if cov.get("pct_rows_known") is not None and cov["pct_rows_known"] < 70:
        v = ("MUNDO 0 (datos): cobertura de pricing baja "
             f"({cov['pct_rows_known']}%). El η² global no mide la realidad del "
             "uplift sino la falta de dato. Cerrar cobertura antes de concluir.")
        P(f"[S5b] VEREDICTO: {v}")
        return v
    if eta_known is not None and eta_known > 0.20:
        v = (f"MUNDO 3→resuelto: con datos, las dims SÍ explican (η²={eta_known:.2f}). "
             "El problema era cobertura/cajón, no elasticidad.")
        P(f"[S5b] VEREDICTO: {v}")
        return v
    if std_known is not None and std_known < 0.10:
        v = ("MUNDO 1 (contractual): uplift uniforme dentro de grupos con datos "
             f"(std={std_known:.3f}). No necesita dims: modela con un factor.")
        P(f"[S5b] VEREDICTO: {v}")
        return v
    if std_known is not None and std_known >= 0.10:
        v = ("MUNDO 2 (idiosincrático o faltan dims): dispersión intra alta "
             f"(std={std_known:.3f}) y η² bajo aún con datos. Banda ancha, o "
             "buscar una dimensión de pricing nueva (p.ej. nº de renovación).")
        P(f"[S5b] VEREDICTO: {v}")
        return v
    P("[S5b] VEREDICTO: indeterminado (faltan columnas de pricing para diagnosticar).")
    return "indeterminado"


# ============================================================
# S5c — elasticidad posible
# ============================================================
def _price_elasticity(train, period_col, pu_col, pusd_col, ru_col, rusd_col,
                        rate_col, uplift_col, disc_col, inc_col, marker, P):
    """
    Lo medible sin caer en sesgo de supervivencia:
      - descuento(estado) → uplift (normalización de precio).
      - estado pricing → rate (señal indirecta de sensibilidad).
      - cruce de los dos estados con rate Y uplift.
    NO es elasticidad causal: el uplift solo se ve en los que renovaron.
    """
    out = {"survivorship_warning":
           "El uplift solo se observa en renovados; no hay elasticidad causal "
           "(los que se fueron por precio no tienen uplift). Esto son asociaciones."}
    if not disc_col or disc_col not in train.columns:
        P("[S5c] sin columna de estado de descuento: elasticidad omitida.")
        out["available"] = False
        return out

    d = train[(train[pu_col] > 0)].copy()
    d["_u"] = _row_uplift(d, ru_col, pu_col, rusd_col, pusd_col)

    # estado de descuento → rate y uplift
    by_state = {}
    for state, g in d.groupby(disc_col):
        rate_m = _w_avg(g[rate_col].values, g[pu_col].values) if rate_col in g else np.nan
        upl_m = _w_avg(g["_u"].values, g[ru_col].values)
        by_state[str(state)] = {
            "rate_mean": round(float(rate_m), 4) if np.isfinite(rate_m) else None,
            "uplift_mean": round(float(upl_m), 4) if np.isfinite(upl_m) else None,
            "n_renewals": int(g[ru_col].sum())}
    out["by_discount_state"] = by_state
    P(f"[S5c] estado descuento → rate/uplift: {len(by_state)} estados medidos.")

    # cruce de los dos estados
    if inc_col and inc_col in d.columns:
        cross = {}
        for (ds, ic), g in d.groupby([disc_col, inc_col]):
            key = f"{ds}|{ic}"
            rate_m = _w_avg(g[rate_col].values, g[pu_col].values) if rate_col in g else np.nan
            upl_m = _w_avg(g["_u"].values, g[ru_col].values)
            cross[key] = {
                "rate_mean": round(float(rate_m), 4) if np.isfinite(rate_m) else None,
                "uplift_mean": round(float(upl_m), 4) if np.isfinite(upl_m) else None,
                "n_renewals": int(g[ru_col].sum())}
        out["by_state_cross"] = cross
        P(f"[S5c] cruce descuento×incremento: {len(cross)} combinaciones.")
    out["available"] = True
    return out


# ============================================================
# S7 — Simpson z-score jerárquico
# ============================================================
def _simpson_zscore(rate_valid, fu_table, fs_dims, P):
    if len(rate_valid) == 0:
        return {"available": False}
    merged = rate_valid.merge(fu_table[["step_2_fs_id", "usd_train_annual"] + fs_dims],
                               on="step_2_fs_id", how="left")
    # SE binomial por FS (de n_avg)
    merged = merged.merge(fu_table[["step_2_fs_id", "step_2_n_avg"]],
                           on="step_2_fs_id", how="left")
    merged["se_b"] = np.where(
        (merged["step_2_n_avg"] > 0) & merged["mean"].between(0, 1),
        np.sqrt(merged["mean"]*(1-merged["mean"])/merged["step_2_n_avg"]), np.nan)
    results = []
    for axis in [d for d in ["regional_level_1", "product_level_1"] if d in merged.columns]:
        for gval, g in merged.groupby(axis):
            if len(g) < 2:
                continue
            usd = g["usd_train_annual"]
            if usd.sum() <= 0:
                continue
            w = usd / usd.sum()
            r_group = (g["mean"] * w).sum()
            naive = (w * (g["mean"] - r_group).abs()).sum() * 100
            se_bucket = np.sqrt((w**2 * g["se_b"].fillna(0)**2).sum())
            adj = np.maximum(0, (g["mean"]-r_group).abs()
                             - 1.96*np.sqrt(g["se_b"].fillna(0)**2 + se_bucket**2)).fillna(0)
            adj_score = (w * adj).sum() * 100
            results.append({"axis": axis, "naive": naive, "adjusted": adj_score})
    if not results:
        return {"available": False}
    naive_mean = float(np.mean([r["naive"] for r in results]))
    adj_mean = float(np.mean([r["adjusted"] for r in results]))
    reduction = (naive_mean - adj_mean) / max(naive_mean, 0.01) * 100
    P(f"[S7] Simpson: naive={naive_mean:.2f}pp, ajustado={adj_mean:.2f}pp "
      f"(reducción {reduction:.0f}% es ruido binomial; resto es Simpson real).")
    return {"available": True, "naive_pp": round(naive_mean, 3),
            "adjusted_pp": round(adj_mean, 3), "reduction_pct": round(reduction, 1)}


# ============================================================
# Cobertura de fenómenos §2.4
# ============================================================
PHENOMENA_COVERAGE = {
    "A1_trend_rate": "S4 (regime_shift); trend formal en step_7",
    "A2_regime_change": "S4 (|t1-t3|>umbral con USD)",
    "A3_volatility_rate": "S4 (CV + patológicas: CV alto Y SE bajo = real)",
    "A4_seasonality_rate": "S3 (history_months: detectabilidad)",
    "A5_outliers_rate": "Parcial — distribución CV; outliers en step_6",
    "B1_trend_volume": "Parcial — soporte/recencia; trend en step_7/8",
    "B2_volatility_volume": "Parcial — soporte y n_avg",
    "B3_volume_jump": "Parcial — recencia y migración cuartil (S3b)",
    "B4_seasonality_volume": "S3 (history_months)",
    "C1_trend_auv": "S5 (regime uplift)",
    "C2_auv_shock": "S5 (CV uplift)",
    "C3_uplift_drift": "S5 (regime uplift)",
    "C4_outliers_auv": "S5 (outliers extremos)",
    "D1_mix_shift": "S1 (concentración) + S8 (nacen/mueren FS)",
    "D2_directional_growth": "Universo time-varying (step_1)",
    "D3_acq_vs_renewal": "S1 (si purchase_type es dim)",
    "D4_simpson": "S7 (z-score naive vs ajustado) + S1b (rate por dim)",
    "E1_generalization": "Fuera de alcance (HITO 2)",
    "E2_band_coverage": "Fuera de alcance (HITO 2/3)",
    "E3_method_bias": "Fuera de alcance (HITO 2)",
    "E4_portfolio_reliability": "S2 (% USD en FS fiables)",
    "F1_attribution": "Fuera de alcance (HITO 3)",
    "F2_small_buckets_risk": "S2 (n_avg<5 y su USD) + S8",
    "F3_concentration": "S2 (Pareto USD proj)",
    "F4_aggregate_bands": "Fuera de alcance (HITO 3)",
    # nuevo foco
    "UPLIFT_low_eta2": "S5b (4 mundos: datos/contractual/idiosincrático/faltan dims)",
    "PRICE_elasticity": "S5c (asociaciones; con aviso de sesgo de supervivencia)",
}


if __name__ == "__main__":
    import sys
    from forecast_configuration import ForecastConfiguration, Step1Config
    from stratified_forecast import StratifiedForecast
    csv = sys.argv[1] if len(sys.argv) > 1 else "synthetic_min.csv"
    config = ForecastConfiguration(cfg=Step1Config(
        dataset_role_col="forecast_dataset", reacq_units_col=None, reacq_usd_col=None,
        auv_pipeline_col=None, auv_renewed_col=None, auv_reacq_col=None,
        business_mandatory_dims=["regional_level_1", "regional_level_2",
                                   "product_level_1", "product_level_2"],
        structural_stable_dims=[], structural_timevarying_dims={}),
        raw_data_path=csv, verbosity="execution")
    sf = StratifiedForecast(config)
    sf.step_0_validate_input(); sf.step_1_prepare(); sf.step_2_build_fus()
    result = run_raw_diagnosis(sf)
    print(result["report"])

# ============================== 4. HITO 2 =================================


class StratifiedForecastHito2(StratifiedForecastHito1):

    @classmethod
    def from_hito1(cls, sf1):
        o = cls(sf1.config if hasattr(sf1, "config") else sf1._config) \
            if False else cls.__new__(cls)
        # copia ligera del estado del hito 1
        o.__dict__.update(sf1.__dict__)
        return o

    # ---------- helpers ----------
    def _mand_key(self, df):
        md = self.cfg.business_mandatory_dims
        return df[md].astype(str).agg("|".join, axis=1)

    def _real_hist(self, roles):
        cfg, df = self.cfg, self.df
        m = (df["step_1_forecast_route"] == "trainable") \
            & (df["step_1_universe"] == "normal") \
            & df[cfg.dataset_role_col].isin(roles)
        if "step_1_synthetic" in df.columns:
            m &= df["step_1_synthetic"].fillna(0).astype(int) == 0
        return df[m]

    # ---------- pasos ----------
    def step_h2_fit_baseline_mandatory(self) -> None:
        """Método ANTES (tradicional): tasa promedio plana por celda MANDATORY
        sobre train (Σ renovadas / Σ pipeline del agregado de la celda)."""
        self._require("step_h2_fit_baseline_mandatory", ["step_2_build_support"])
        cfg = self.cfg
        tr = self._real_hist(("train",)).copy()
        tr["_mk"] = self._mand_key(tr)
        g = tr.groupby("_mk")
        tab = pd.DataFrame({
            "n": g[cfg.pipeline_units_col].sum(),
            "ren": g[cfg.renewed_units_col].sum(),
            "ren_usd": g[cfg.renewed_usd_col].sum(),
        })
        tab["rate"] = np.where(tab["n"] > 0, tab["ren"] / tab["n"], np.nan)
        tab["auv_ren"] = np.where(tab["ren"] > 0, tab["ren_usd"] / tab["ren"], np.nan)
        self.h2_mandatory = tab
        self._log(f"  step_h2_fit_baseline_mandatory: {len(tab):,} celdas mandatory | "
                  f"tasa global train: {tab['ren'].sum()/max(tab['n'].sum(),1):.1%}")
        self._mark("step_h2_fit_baseline_mandatory")
        self._done("step_h2_fit_baseline_mandatory",
                   f"baseline plano en {len(tab):,} celdas")

    def step_h2_fit_shrunk(self) -> None:
        """Método DESPUÉS (framework): tasa por FS fina con EVIDENCE WEIGHT (peso por evidencia) hacia
        su celda mandatory: rate = z·rate_fs + (1−z)·rate_celda, z = n/(n+k).
        k estimado por momentos (Bühlmann simplificado) y logueado."""
        self._require("step_h2_fit_shrunk", ["step_h2_fit_baseline_mandatory"])
        cfg = self.cfg
        tr = self._real_hist(("train",)).copy()
        tr["_mk"] = self._mand_key(tr)
        g = tr.groupby("step_2_fs_id")
        fs = pd.DataFrame({
            "n": g[cfg.pipeline_units_col].sum(),
            "ren": g[cfg.renewed_units_col].sum(),
            "_mk": g["_mk"].first(),
        })
        fs["rate_fs"] = np.where(fs["n"] > 0, fs["ren"] / fs["n"], np.nan)
        fs["rate_parent"] = fs["_mk"].map(self.h2_mandatory["rate"])
        # k por momentos: E[p(1-p)] / Var_entre(p) ponderado
        ok = fs["n"] >= 5
        p = fs.loc[ok, "rate_fs"].clip(0.001, 0.999)
        w = fs.loc[ok, "n"]
        var_entre = float(np.average((p - np.average(p, weights=w)) ** 2, weights=w))
        ev_dentro = float(np.average(p * (1 - p), weights=w))
        k = ev_dentro / var_entre if var_entre > 1e-9 else 1000.0
        k = float(np.clip(k, 5, 5000))
        fs["z"] = fs["n"] / (fs["n"] + k)
        fs["rate_shrunk"] = (fs["z"] * fs["rate_fs"].fillna(fs["rate_parent"])
                             + (1 - fs["z"]) * fs["rate_parent"])
        self.h2_k = k
        self.h2_fs = fs
        z_usd = None
        if "step_2_usd_proj" in self.df.columns:
            u = (self.df.drop_duplicates("step_2_fs_id")
                 .set_index("step_2_fs_id")["step_2_usd_proj"])
            fs["_usd"] = u.reindex(fs.index).fillna(0)
            tot = fs["_usd"].sum()
            z_usd = float((fs["z"] * fs["_usd"]).sum() / tot) if tot > 0 else None
        self._log(f"  step_h2_fit_shrunk: k={k:,.0f} (credibilidad) | "
                  f"z mediano={fs['z'].median():.2f}"
                  + (f" | z ponderado por $={z_usd:.2f}" if z_usd is not None else ""))
        self._log(f"    lectura: z=1 habla el dato propio; z=0 habla la celda. "
                  f"El dinero gordo tiene voz propia; la astilla hereda.")
        self._mark("step_h2_fit_shrunk")
        self._done("step_h2_fit_shrunk", f"{len(fs):,} FS con tasa shrunk (k={k:,.0f})")

    def step_h2_fit_uplift_covariates(self) -> None:
        """UPLIFT con covariables (Fase 2): T3 = factores por COMBINACIÓN
        estimados en train sobre la tabla fina (uplift_comb = AUV_ren/AUV_pipe
        del agregado; fallback global si n<min). uplift_ajustado por (FU,mes)
        = Σ share_comb × factor_comb, con no_info → uplift base de la celda
        mandatory. Sin df_fine: uplift base para todo (declarado)."""
        self._require("step_h2_fit_uplift_covariates", ["step_h2_fit_shrunk"])
        cfg = self.cfg
        pu, pus = cfg.pipeline_units_col, cfg.pipeline_usd_col
        ru, rus = cfg.renewed_units_col, cfg.renewed_usd_col
        pc, rc = cfg.period_col, cfg.dataset_role_col
        # uplift base por celda: AUV_ren / AUV_pipe del train de la celda
        tr = self._real_hist(("train",)).copy(); tr["_mk"] = self._mand_key(tr)
        g = tr.groupby("_mk")
        auvp = g[pus].sum()/g[pu].sum().clip(lower=1)
        auvr = g[rus].sum()/g[ru].sum().clip(lower=1)
        self.h2_uplift_base = (auvr/auvp).replace([np.inf,-np.inf],np.nan).fillna(1.0)
        gl_up = float(np.clip((tr[rus].sum()/max(tr[ru].sum(),1))
                              / (tr[pus].sum()/max(tr[pu].sum(),1)), .5, 3))
        if self.df_fine is None:
            self.h2_T3, self.h2_uplift_adj = None, None
            self._mark("step_h2_fit_uplift_covariates")
            self._done("step_h2_fit_uplift_covariates",
                       f"sin covariables: uplift base por celda (global {gl_up:.2f})")
            return
        f = self.df_fine
        ftr = f[(f[rc] == "train")]
        gt = ftr.groupby("comb_id")
        T3 = pd.DataFrame({"n": gt[ru].sum(),
                           "up": (gt[rus].sum()/gt[ru].sum().clip(lower=1))
                                 / (gt[pus].sum()/gt[pu].sum().clip(lower=1))})
        T3["up"] = T3["up"].replace([np.inf,-np.inf],np.nan)
        T3.loc[T3["n"] < 200, "up"] = np.nan          # fallback
        T3["up"] = T3["up"].fillna(gl_up)
        self.h2_T3 = T3
        ni = f["comb_id"].str.contains("no_info")
        f = f.assign(_up=f["comb_id"].map(T3["up"]))
        f.loc[ni, "_up"] = np.nan                      # no_info → base luego
        w = f.groupby(["fu_id", pc]).apply(
            lambda d: pd.Series({
                "w_info": d.loc[~d["_up"].isna(), pu].sum(),
                "w_tot": d[pu].sum(),
                "up_info": np.average(d.loc[~d["_up"].isna(), "_up"],
                                      weights=d.loc[~d["_up"].isna(), pu])
                           if d.loc[~d["_up"].isna(), pu].sum() > 0 else np.nan}),
            include_groups=False)
        self.h2_uplift_adj = w
        self._log(f"  step_h2_fit_uplift_covariates: T3={len(T3)} combinaciones "
                  f"(global {gl_up:.2f}) | uplift_ajustado en {len(w):,} (FU,mes)")
        for cid, r in T3.sort_values("n", ascending=False).head(5).iterrows():
            self._log(f"    {str(cid)[:30]:<30} n={int(r['n']):>8,}  ×{r['up']:.2f}")
        self._mark("step_h2_fit_uplift_covariates")
        self._done("step_h2_fit_uplift_covariates", f"{len(T3)} factores estimados")

    def _uplift_for(self, dfp):
        """uplift por fila: mezcla info (T3×shares) + no_info (base celda)."""
        cfg = self.cfg
        base = self._mand_key(dfp).map(self.h2_uplift_base).fillna(1.0)
        if self.h2_uplift_adj is None:
            return base
        key = list(zip(dfp[self.dimension_cols].astype(str).agg("|".join, axis=1),
                       dfp[cfg.period_col]))
        w = self.h2_uplift_adj.reindex(key)
        wi = w["w_info"].fillna(0).values; wt = w["w_tot"].fillna(0).values
        ui = w["up_info"].values
        share = np.where(wt > 0, wi/np.maximum(wt, 1), 0)
        return pd.Series(np.where((share > 0) & np.isfinite(ui),
                                  share*ui + (1-share)*base.values, base.values),
                         index=dfp.index)

    def step_h2_fit_ts(self, min_hist=18, halflife=6) -> None:
        """Técnica de SERIE TEMPORAL para FS con historial suficiente (≥18
        meses reales): nivel EWM (half-life 6) de la tasa mensual de la FS,
        mezclado con la credibilidad (z de la FS). Series cortas → shrunk."""
        self._require("step_h2_fit_ts", ["step_h2_fit_shrunk"])
        cfg = self.cfg
        tr = self._real_hist(("train",)).copy()
        tr["_r"] = np.where(tr[cfg.pipeline_units_col] > 0,
                            tr[cfg.renewed_units_col]/tr[cfg.pipeline_units_col], np.nan)
        def ew(d):
            d = d.sort_values(cfg.period_col)["_r"].dropna()
            return d.ewm(halflife=halflife).mean().iloc[-1] if len(d) >= min_hist else np.nan
        ts = tr.groupby("step_2_fs_id").apply(ew, include_groups=False)
        fs = self.h2_fs
        fs["rate_ts"] = ts.reindex(fs.index)
        eligible = fs["rate_ts"].notna()
        fs["rate_shrunk_ts"] = np.where(
            eligible, fs["z"]*fs["rate_ts"] + (1-fs["z"])*fs["rate_parent"],
            fs["rate_shrunk"])
        self._log(f"  step_h2_fit_ts: {int(eligible.sum()):,} FS con ≥{min_hist}m "
                  f"reales usan EWM(hl={halflife}); el resto, shrunk plano.")
        self._mark("step_h2_fit_ts")
        self._done("step_h2_fit_ts", f"TS en {int(eligible.sum()):,} series largas")

    def step_h2_reassess_support(self, support_floor=100, dust_floor=10) -> dict:
        """PASO A del cierre — re-evalúa el SOPORTE tras los colapsos del árbol
        (N1 sustrato, N2 ANOVA) y etiqueta el estado final de cada serie:
          RESUELTA       : soporte del grupo >= support_floor → tasa propia.
          EVIDENCE_WEIGHT: pobre pero con padre → tasa mezclada (z).
          HEURISTICA     : soporte < dust_floor o sin train → regla/horizonte.
        Usa el id de grupo más colapsado disponible (L2>L1>fino). No cambia el
        forecast: ETIQUETA para que la elección de técnica y el reporte sepan
        de qué se fían. KPI: deja trazable cuánto $ es de cada estado."""
        self._require("step_h2_reassess_support", ["step_h2_fit_shrunk"])
        cfg, df = self.cfg, self.df
        pu = cfg.pipeline_units_col
        gcol = ("step_2_fs_id_L2" if "step_2_fs_id_L2" in df.columns else
                "step_2_fs_id_L1" if "step_2_fs_id_L1" in df.columns else "step_2_fs_id")
        df["step_h2_fs_group"] = df[gcol]
        real = df[df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0]
        sop = real.groupby("step_h2_fs_group")[pu].median()
        tr = real[real[cfg.dataset_role_col] == "train"].groupby("step_h2_fs_group")[pu].sum()
        g = df["step_h2_fs_group"]
        s = g.map(sop).fillna(0)
        has_train = g.map(tr).fillna(0) > 0
        estado = np.where((s < dust_floor) | (~has_train), "HEURISTICA",
                  np.where(s >= support_floor, "RESUELTA", "EVIDENCE_WEIGHT"))
        df["step_h2_estado_final"] = estado
        self.df = df
        usd = df.groupby("step_h2_estado_final")[cfg.pipeline_usd_col].sum()
        tot = max(usd.sum(), 1)
        t = pd.DataFrame({"usd": usd, "pct_usd": (100*usd/tot).round(1)})
        self._table(t, "Estado final de las series tras el colapso:", col_defs={
            "usd": "$ de pipeline en series de ese estado",
            "pct_usd": "% del $ total a predecir"})
        self._mark("step_h2_reassess_support",
                   metadata={"pct_resuelta": float(100*usd.get("RESUELTA", 0)/tot)})
        self._done("step_h2_reassess_support",
                   f"{100*usd.get('RESUELTA',0)/tot:.0f}% del $ RESUELTA, "
                   f"{100*usd.get('HEURISTICA',0)/tot:.0f}% heurística")
        return {"estados": usd.to_dict()}

    def step_h2_improvement_summary(self) -> dict:
        """PASO B del cierre — resumen ANTES/DESPUÉS del colapso, en las varas
        del HITO 1: soporte mediano y % del $ bajo soporte<100. Muestra qué
        compró el árbol de colapso. KPI: soporte, error binomial efectivo."""
        self._require("step_h2_improvement_summary", ["step_h2_reassess_support"])
        cfg, df = self.cfg, self.df
        pu = cfg.pipeline_units_col
        real = df[df.get("step_1_synthetic", 0).fillna(0).astype(int) == 0]
        def perfil(col):
            sop = real.groupby(col)[pu].median()
            usd = df.groupby(col)[cfg.pipeline_usd_col].sum()
            tot = max(usd.sum(), 1)
            pct_lt100 = 100*usd[sop.reindex(usd.index).fillna(0) < 100].sum()/tot
            return float(sop.median()), float(pct_lt100)
        s_fino, p_fino = perfil("step_2_fs_id")
        s_grp, p_grp = perfil("step_h2_fs_group")
        t = pd.DataFrame({
            "métrica": ["soporte mediano de serie", "% del $ en soporte<100"],
            "ANTES (fino)": [round(s_fino, 1), round(p_fino, 1)],
            "DESPUÉS (colapso)": [round(s_grp, 1), round(p_grp, 1)]}).set_index("métrica")
        self._table(t, "Mejora del colapso (varas del HITO 1):", col_defs={
            "ANTES (fino)": "grano fino original",
            "DESPUÉS (colapso)": "tras N1+N2 del árbol"})
        self._mark("step_h2_improvement_summary")
        self._done("step_h2_improvement_summary",
                   f"soporte mediano {s_fino:.0f}→{s_grp:.0f}; "
                   f"$ bajo soporte<100 {p_fino:.0f}%→{p_grp:.0f}%")
        return {"soporte_antes": s_fino, "soporte_despues": s_grp}

    def step_h2_quality_photo(self) -> None:
        """LA FOTO: calidad de las forecast series ANTES (grano mandatory,
        promedio plano) vs DESPUÉS (grano fino + credibilidad). Estructura:
        nº series, soporte mediano, % del $ en soporte débil, y el ERROR DE
        MUESTREO PRESUPUESTADO en $ (Σ SE binomial × exposición)."""
        self._require("step_h2_quality_photo", ["step_h2_fit_shrunk"])
        cfg = self.cfg
        fs = self.h2_fs.copy()
        proj = self.df[self.df[cfg.dataset_role_col] == "projection"]
        usd_m = proj.assign(_mk=self._mand_key(proj)).groupby("_mk")[cfg.pipeline_usd_col].sum()
        mand = self.h2_mandatory.join(usd_m.rename("usd_proj")).fillna({"usd_proj": 0})
        if "_usd" not in fs.columns:
            fs["_usd"] = 0.0
        def foto(n, rate, usd, se_extra=None):
            p = rate.clip(0.02, 0.98).fillna(0.5)
            se = np.sqrt(p * (1 - p) / n.clip(lower=1))
            if se_extra is not None:
                se = se * se_extra          # shrink: SE efectiva ≈ z·SE_propia
            return {
                "series": int(len(n)),
                "sop_med": float(n.median()),
                "pct_usd_n30": float(100 * usd[n < 30].sum() / max(usd.sum(), 1)),
                "pct_usd_n100": float(100 * usd[n < 100].sum() / max(usd.sum(), 1)),
                "err_usd": float((se * usd).sum()),
            }
        antes = foto(mand["n"], mand["rate"], mand["usd_proj"])
        despues = foto(fs["n"], fs["rate_fs"], fs["_usd"], se_extra=fs["z"])
        self.h2_photo = {"antes": antes, "despues": despues}
        t = pd.DataFrame({"ANTES (mandatory)": antes, "DESPUÉS (framework)": despues}).round(1)
        self._table(t, "FOTO de calidad de las forecast series:", col_defs={
            "series": "nº de forecast series del esquema",
            "sop_med": "soporte mediano por serie-mes en train (n de la binomial)",
            "pct_usd_n30/100": "% del $ a predecir en series con soporte débil",
            "err_usd": "error de muestreo presupuestado: Σ SE_binomial×$ (DESPUÉS: SE×z, credibilidad)"})
        self._log(f"    lectura: el grano fino multiplica series y fragmenta soporte; "
                  f"la credibilidad (z) devuelve el error presupuestado a escala — "
                  f"esa es la compra: sesgo↓ (Simpson) pagando varianza CONTROLADA.")
        self._mark("step_h2_quality_photo", metadata=self.h2_photo)
        self._done("step_h2_quality_photo", "foto antes/después registrada")

    def step_h2_backtest_test_months(self) -> None:
        """Backtest ESCALONADO en los meses de TEST, $ reales (renewed_usd):
        M1 mandatory (tasa celda × uplift base) · M2 +covariables (uplift
        ajustado) · M3 +dims finas/time-varying (shrunk) · M4 +serie temporal
        (EWM en series largas). La escalera muestra qué aporta CADA bloque de
        variables."""
        self._require("step_h2_backtest_test_months",
                      ["step_h2_quality_photo", "step_h2_fit_ts",
                       "step_h2_fit_uplift_covariates"])
        cfg = self.cfg
        te = self._real_hist(("test",)).copy()
        if not len(te):
            self._mark("step_h2_backtest_test_months")
            self._done("step_h2_backtest_test_months", "sin test")
            return
        te["_mk"] = self._mand_key(te)
        gl = self.h2_mandatory["ren"].sum()/max(self.h2_mandatory["n"].sum(), 1)
        r_m = te["_mk"].map(self.h2_mandatory["rate"]).fillna(gl)
        r_s = te["step_2_fs_id"].map(self.h2_fs["rate_shrunk"]).fillna(r_m)
        r_t = te["step_2_fs_id"].map(self.h2_fs["rate_shrunk_ts"]).fillna(r_s)
        up_b = te["_mk"].map(self.h2_uplift_base).fillna(1.0)
        up_a = self._uplift_for(te)
        pipe_u, pipe_usd = te[cfg.pipeline_units_col], te[cfg.pipeline_usd_col]
        auvp = np.where(pipe_u > 0, pipe_usd/pipe_u, 0)
        real_usd = te[cfg.renewed_usd_col]
        metodos = {"M1 mandatory": (r_m, up_b),
                   "M2 +covariables": (r_m, up_a),
                   "M3 +dims finas (shrunk)": (r_s, up_a),
                   "M4 +serie temporal": (r_t, up_a)}
        res = {}
        self._log(f"  step_h2_backtest_test_months (ESCALONADO, $ reales): "
                  f"{len(te):,} filas test")
        for nom, (r, u) in metodos.items():
            pred = r*pipe_u*auvp*u
            err = pred - real_usd
            res[nom] = {"WAPE_$": float(err.abs().sum()/max(real_usd.sum(), 1)),
                        "sesgo_$": float(err.sum()/max(real_usd.sum(), 1))}
            te[f"err_{nom[:2]}"] = err
        tt = (pd.DataFrame(res).T*100).round(1)
        self._table(tt, "ESCALERA de mejora por bloque de variables (test, $ reales):",
                    col_defs={"WAPE_$": "Σ|pred−real|/Σreal en $, %",
                              "sesgo_$": "(Σpred−Σreal)/Σreal, %: ± = sobre/infra-estima"})
        res = {k: {"wape": v["WAPE_$"], "sesgo": v["sesgo_$"]} for k, v in res.items()}
        self.h2_backtest = res
        gana = min(res, key=lambda k: res[k]["wape"])
        self._mark("step_h2_backtest_test_months", metadata=res)
        self._done("step_h2_backtest_test_months",
                   f"gana {gana} (WAPE $ {res[gana]['wape']:.1%})")

    def step_h2_forecast_projection(self) -> None:
        """Predicción final para PROJECTION con ambos métodos (columnas
        h2_pred_units_mandatory / h2_pred_units_shrunk en el df), lista para
        PBI y para la back-annotation futura a la tabla fina."""
        self._require("step_h2_forecast_projection", ["step_h2_backtest_test_months"])
        cfg, df = self.cfg, self.df
        pr = df[cfg.dataset_role_col] == "projection"
        mk = self._mand_key(df[pr])
        gl = self.h2_mandatory["ren"].sum() / max(self.h2_mandatory["n"].sum(), 1)
        rm = mk.map(self.h2_mandatory["rate"]).fillna(gl)
        rs = df.loc[pr, "step_2_fs_id"].map(self.h2_fs["rate_shrunk"]).fillna(rm)
        df.loc[pr, "h2_pred_units_mandatory"] = rm * df.loc[pr, cfg.pipeline_units_col]
        df.loc[pr, "h2_pred_units_shrunk"] = rs * df.loc[pr, cfg.pipeline_units_col]
        tm = float(df.loc[pr, "h2_pred_units_mandatory"].sum())
        ts = float(df.loc[pr, "h2_pred_units_shrunk"].sum())
        self.df = df
        self._log(f"  step_h2_forecast_projection: unidades previstas — "
                  f"mandatory {tm:,.0f} | shrunk {ts:,.0f} (Δ {ts-tm:+,.0f})")
        self._mark("step_h2_forecast_projection")
        self._done("step_h2_forecast_projection", "predicciones escritas en el df")


    def step_h2_forecast_bands(self, z_score=1.96, trend_skew=0.5) -> dict:
        """BANDA DE CONFIANZA del forecast 2026 (projection), por celda
        mandatory y total. Tres ingredientes, todos declarados:
          1) ERROR RELATIVO empírico = WAPE en $ del método ganador del
             backtest (error ya demostrado fuera de muestra en el test).
          2) SUELO binomial por serie: aunque el backtest fuese perfecto,
             una tasa p sobre n unidades tiene error de muestreo irreducible
             SE=raíz(p(1-p)/n); se agrega en $ por celda.
          3) ASIMETRÍA por tendencia: en celdas con pendiente reciente, el
             lado 'contra' la pendiente se ensancha trend_skew·|pendiente|
             (una serie que sube puede quedarse corta más que larga).
        Banda = pred ± z·error, con los dos lados escalados por la asimetría.
        NO es una banda frecuentista exacta: es una cuantificación honesta de
        incertidumbre para comunicar a negocio (declarado)."""
        self._require("step_h2_forecast_bands",
                      ["step_h2_forecast_projection", "step_h2_backtest_test_months"])
        cfg, df = self.cfg, self.df
        pc = cfg.period_col
        gana = min(self.h2_backtest, key=lambda k: self.h2_backtest[k]["WAPE_$"]
                   if "WAPE_$" in self.h2_backtest[k] else self.h2_backtest[k]["wape"])
        wb = self.h2_backtest[gana]
        wape = wb.get("WAPE_$", wb.get("wape"))
        wape = wape/100 if wape > 1.5 else wape           # acepta % o fracción
        pr = df[df[cfg.dataset_role_col] == "projection"].copy()
        pr["_mk"] = self._mand_key(pr)
        auvp = np.where(pr[cfg.pipeline_units_col] > 0,
                        pr[cfg.pipeline_usd_col]/pr[cfg.pipeline_units_col], 0)
        up = self._uplift_for(pr); up = up.values if hasattr(up, "values") else up
        pr["_usd"] = pr["h2_pred_units_shrunk"].fillna(0).values*auvp*up
        # suelo binomial por fila: SE_unidades = raíz(n p (1-p)) → $ = ·AUV·uplift
        rate = (pr["h2_pred_units_shrunk"].fillna(0)
                / pr[cfg.pipeline_units_col].replace(0, np.nan)).clip(0, 1).fillna(0)
        se_u = np.sqrt(pr[cfg.pipeline_units_col]*rate*(1-rate)).fillna(0)
        pr["_se_usd"] = se_u.values*auvp*up
        # pendiente reciente por celda (si existe el radar)
        skew = pd.Series(0.0, index=pr.index)
        if getattr(self, "recent_slope", None) is not None and len(self.recent_slope):
            sl = self.recent_slope["pp_año"]/100.0
            skew = pr["step_2_fs_id"].map(sl).fillna(0.0).clip(-1, 1)
        g = pr.groupby("_mk")
        rows = []
        for mk, d in g:
            pred = d["_usd"].sum()
            err_model = wape*pred                          # error relativo del backtest
            err_floor = float(np.sqrt((d["_se_usd"]**2).sum()))  # suelo en cuadratura
            half = z_score*max(err_model, err_floor)
            sk = float(np.average(skew.loc[d.index], weights=d["_usd"].clip(lower=0)+1))
            lo = pred - half*(1 + trend_skew*max(-sk, 0))  # baja: más si pendiente <0
            hi = pred + half*(1 + trend_skew*max(sk, 0))   # sube: más si pendiente >0
            rows.append([mk, pred, max(lo, 0), hi, half, sk])
        t = pd.DataFrame(rows, columns=["celda", "pred_usd", "lo", "hi",
                                        "half_width", "skew"]).set_index("celda")
        tot_pred = t["pred_usd"].sum()
        tot_lo, tot_hi = t["lo"].sum(), t["hi"].sum()
        self.h2_bands = t
        self._table(t.sort_values("pred_usd", ascending=False).head(12).round(0),
                    "Forecast 2026 con banda por celda (top 12 por $):",
                    col_defs={"pred_usd": "forecast central de la celda, $",
                              "lo / hi": "extremos inferior y superior de la banda",
                              "half_width": "semi-anchura base (antes de asimetría), $",
                              "skew": "pendiente reciente ponderada (+sube/−baja): abre el lado correspondiente"})
        self._log(f"    método ganador del backtest: {gana} (WAPE $ {wape:.1%}) "
                  f"= error relativo de la banda")
        self._log(f"    TOTAL 2026: ${tot_pred:,.0f}  "
                  f"[banda {z_score}σ: ${tot_lo:,.0f} … ${tot_hi:,.0f}]  "
                  f"(±{100*(tot_hi-tot_pred)/max(tot_pred,1):.0f}%/"
                  f"−{100*(tot_pred-tot_lo)/max(tot_pred,1):.0f}%)")
        self._mark("step_h2_forecast_bands",
                   metadata={"total": tot_pred, "lo": tot_lo, "hi": tot_hi,
                             "metodo": gana, "wape": wape})
        self._done("step_h2_forecast_bands",
                   f"2026: ${tot_pred:,.0f} [${tot_lo:,.0f}…${tot_hi:,.0f}] {z_score}σ")
        return {"total": tot_pred, "lo": tot_lo, "hi": tot_hi}

    def step_h2_forecast_next_year(self) -> None:
        """FORECAST COMPLETO del año siguiente al pending (p.ej. 2027), donde
        el pipeline AÚN NO EXISTE. Cadena de regeneración a grano MANDATORY:
        pipeline(año+1, mes) = renovadas(año, mes) × regen_celda, con
        renovadas(año) = reales hasta pending + previstas shrunk después, y
        regen = mediana histórica de pipeline_t / renovadas_{t-12} (captura
        new business implícito; cap [0.2, 5]; fallback global). Supuesto v1
        declarado: lag 12 (term 1y dominante; term 2y apuntado para v2)."""
        self._require("step_h2_forecast_next_year", ["step_h2_forecast_projection"])
        cfg, df = self.cfg, self.df
        pc, rc = cfg.period_col, cfg.dataset_role_col
        pu, ru = cfg.pipeline_units_col, cfg.renewed_units_col
        pend = pd.Period(str(self.step_metadata
                  ["step_1_derive_roles_from_period"]["pending_date"])[:7], freq="M")
        y0, y1 = pend.year, pend.year + 1
        d = df[(df["step_1_universe"] == "normal")
               & (df["step_1_forecast_route"] == "trainable")].copy()
        d["_mk"] = self._mand_key(d)
        real = d[d[rc].isin(cfg.history_roles)
                 & (d.get("step_1_synthetic", 0).fillna(0).astype(int) == 0)]
        # regen histórico: pipeline_t / renovadas_{t-12} por celda
        m_pipe = real.groupby(["_mk", pc])[pu].sum()
        m_ren = real.groupby(["_mk", pc])[ru].sum()
        lag = m_ren.copy(); lag.index = lag.index.set_levels(
            lag.index.levels[1] + 12, level=1)
        ratio = (m_pipe / lag).replace([np.inf, -np.inf], np.nan).dropna()
        regen = ratio.groupby(level=0).median().clip(0.2, 5.0)
        regen_gl = float(np.clip(ratio.median(), 0.2, 5.0)) if len(ratio) else 1.0
        # renovadas del año base: reales (< pending) + previstas shrunk (>=)
        ren_real = (real[real[pc].dt.year == y0].groupby(["_mk", pc])[ru].sum())
        prj = d[(d[rc] == "projection") & (d[pc].dt.year == y0)]
        ren_pred = prj.groupby(["_mk", pc])["h2_pred_units_shrunk"].sum()
        ren_y0 = pd.concat([ren_real, ren_pred]).groupby(level=[0, 1]).sum()
        # tasa por celda: la implícita del método ganador en projection
        tasa = (prj.groupby("_mk")["h2_pred_units_shrunk"].sum()
                / prj.groupby("_mk")[pu].sum()).clip(0.01, 0.99)
        gl_rate = self.h2_mandatory["ren"].sum() / max(self.h2_mandatory["n"].sum(), 1)
        rows = []
        for (mk, p), v in ren_y0.items():
            p1 = p + 12
            if p1.year != y1:
                continue
            pipe1 = float(v) * float(regen.get(mk, regen_gl))
            r = float(tasa.get(mk, gl_rate))
            auv = float(self.h2_mandatory["auv_ren"].get(mk, 0) or 0)
            rows.append({"celda_mandatory": mk, "mes": str(p1),
                         "pipeline_units_proy": pipe1,
                         "pred_renovadas_units": pipe1 * r,
                         "pred_renovadas_usd": pipe1 * r * auv})
        out = pd.DataFrame(rows).sort_values(["mes", "celda_mandatory"])
        self.h2_fcst_next = out
        tot = out.groupby("mes")[["pipeline_units_proy",
                                  "pred_renovadas_units",
                                  "pred_renovadas_usd"]].sum()
        self._log(f"  step_h2_forecast_next_year: forecast {y1} completo "
                  f"(grano mandatory, {out['celda_mandatory'].nunique()} celdas) | "
                  f"regen mediana={float(regen.median()) if len(regen) else regen_gl:.2f}")
        self._log(f"    {'mes':>8} {'pipe_proy':>10} {'ren_pred':>10} {'$_pred':>14}")
        for mes, r in tot.iterrows():
            self._log(f"    {mes:>8} {r['pipeline_units_proy']:>10,.0f} "
                      f"{r['pred_renovadas_units']:>10,.0f} "
                      f"{r['pred_renovadas_usd']:>14,.0f}")
        self._log(f"    TOTAL {y1}: {tot['pred_renovadas_units'].sum():,.0f} unidades | "
                  f"${tot['pred_renovadas_usd'].sum():,.0f} | "
                  f"SUPUESTO v1: lag 12 (term 1y); term 2y y new explícito → v2")
        self._mark("step_h2_forecast_next_year",
                   metadata={"year": y1, "total_usd": float(tot['pred_renovadas_usd'].sum())})
        self._done("step_h2_forecast_next_year",
                   f"forecast {y1}: ${tot['pred_renovadas_usd'].sum():,.0f}")


    def step_h2_extend_AB(self, term_col="term", term_1y="1Y",
                          ptype_col="purchase_type", acq_value="Acquisition",
                          acq_factor=1.0) -> None:
        """2027 A+B TERM-AWARE (diseño consensuado): solo el tramo 1Y se
        simula — A = renovadas 1Y del mismo mes año anterior (reales+pred);
        B = pipeline de Acquisition 1Y (M−12) × acq_factor (altas que
        expirarán). 2Y–3Y ya están en el pipeline real. Filas etiquetadas
        rol projection_extended en h2_fcst_next (no se mezclan con projection)."""
        self._require("step_h2_extend_AB", ["step_h2_forecast_projection"])
        cfg, df = self.cfg, self.df
        pc, rc = cfg.period_col, cfg.dataset_role_col
        pu, ru = cfg.pipeline_units_col, cfg.renewed_units_col
        if term_col not in df.columns:
            self._mark("step_h2_extend_AB")
            self._done("step_h2_extend_AB", f"sin columna {term_col}: no aplica")
            return
        pend = pd.Period(str(self.step_metadata
                ["step_1_derive_roles_from_period"]["pending_date"])[:7], "M")
        y1 = pend.year + 1
        d = df[(df["step_1_universe"] == "normal")
               & (df["step_1_forecast_route"] == "trainable")].copy()
        d["_mk"] = self._mand_key(d)
        es1y = d[term_col].astype(str) == term_1y
        ret = (d[ptype_col].astype(str) != acq_value) if ptype_col in d.columns \
              else pd.Series(True, index=d.index)
        # A: renovadas 1Y por celda-mes del año base (real + pred shrunk)
        hist = d[es1y & ret & d[rc].isin(cfg.history_roles)
                 & (d.get("step_1_synthetic", 0).fillna(0).astype(int) == 0)]
        a_real = hist[hist[pc].dt.year == pend.year].groupby(["_mk", pc])[ru].sum()
        prj = d[es1y & ret & (d[rc] == "projection") & (d[pc].dt.year == pend.year)]
        a_pred = prj.groupby(["_mk", pc])["h2_pred_units_shrunk"].sum()
        A = pd.concat([a_real, a_pred]).groupby(level=[0, 1]).sum()
        # B: pipeline Acquisition 1Y del año base
        if ptype_col in d.columns:
            acq = d[es1y & (d[ptype_col].astype(str) == acq_value)
                    & (d[pc].dt.year == pend.year)]
            B = acq.groupby(["_mk", pc])[pu].sum() * acq_factor
        else:
            B = pd.Series(dtype=float)
            self._log("    ⚠ sin purchase_type: B=0 (solo re-renewals)")
        gl = self.h2_mandatory["ren"].sum()/max(self.h2_mandatory["n"].sum(), 1)
        tasa = self.h2_mandatory["rate"]
        rows = []
        for (mk, p), va in A.items():
            p1 = p + 12
            if p1.year != y1:
                continue
            vb = float(B.get((mk, p), 0.0))
            pipe = float(va) + vb
            r = float(tasa.get(mk, gl))
            auv = float(self.h2_mandatory["auv_ren"].get(mk, 0) or 0)
            up = float(self.h2_uplift_base.get(mk, 1.0))
            rows.append({"rol": "projection_extended", "celda": mk,
                         "mes": str(p1), "A_rerenew": float(va), "B_altas": vb,
                         "pipeline": pipe, "pred_units": pipe*r,
                         "pred_usd": pipe*r*auv*up})
        ext = pd.DataFrame(rows)
        self.h2_extended = ext
        tot = ext.groupby("mes")[["A_rerenew", "B_altas", "pred_usd"]].sum()
        self._log(f"  step_h2_extend_AB: {y1} tramo 1Y simulado "
                  f"({ext['celda'].nunique()} celdas, {ext['mes'].nunique()} meses) "
                  f"| A={ext['A_rerenew'].sum():,.0f}u B={ext['B_altas'].sum():,.0f}u")
        self._log(f"    TOTAL extendido {y1}: ${ext['pred_usd'].sum():,.0f} "
                  f"(banda: MAYOR que projection — estimación sobre estimación)")
        self._mark("step_h2_extend_AB")
        self._done("step_h2_extend_AB",
                   f"{y1} 1Y: ${ext['pred_usd'].sum():,.0f} (A+B, rol projection_extended)")

    def step_h2_assemble_2027(self, term_col="term") -> dict:
        """ENSAMBLA EL AÑO COMPLETO siguiente al pending (p.ej. 2027) con sus
        TRES aportaciones visibles y sumadas:
          C1 multi-año ya firmado: renovaciones de contratos 2Y/3Y que están
             en projection real y vencen en el año objetivo (NO se simula);
          C2 = componente A de extend_AB: re-renovación del tramo 1Y simulada;
          C3 = componente B de extend_AB: adquisición 1Y simulada que vencerá.
        Total 2027 = C1 + C2 + C3. Devuelve la tabla con cada parte y su % de
        aportación. Banda: C1 firme; C2/C3 estimación sobre estimación."""
        self._require("step_h2_assemble_2027",
                      ["step_h2_extend_AB", "step_h2_forecast_projection"])
        cfg, df = self.cfg, self.df
        pc, rc = cfg.period_col, cfg.dataset_role_col
        pend = pd.Period(str(self.step_metadata
                ["step_1_derive_roles_from_period"]["pending_date"])[:7], "M")
        y1 = pend.year + 1
        # C1: projection real cuyo mes cae en el año objetivo (multi-año 2Y/3Y)
        pr = df[(df[rc] == "projection") & (df[pc].dt.year == y1)].copy()
        if term_col in pr.columns:
            pr = pr[pr[term_col].astype(str) != "1Y"]   # 1Y del año objetivo lo da A+B
        if len(pr) == 0:
            c1 = 0.0
            self._log(f"    C1=0: no hay projection multi-año (2Y/3Y) que venza en {y1}")
        else:
            auvp = np.where(pr[cfg.pipeline_units_col] > 0,
                            pr[cfg.pipeline_usd_col]/pr[cfg.pipeline_units_col], 0)
            up = self._uplift_for(pr) if hasattr(self, "_uplift_for") else 1.0
            up = up.values if hasattr(up, "values") else up
            c1 = float((pr["h2_pred_units_shrunk"].fillna(0).values*auvp*up).sum())
        # C2, C3 desde extend_AB
        ext = getattr(self, "h2_extended", None)
        if ext is not None and len(ext):
            auv = ext["pipeline"].replace(0, np.nan)
            share_a = (ext["A_rerenew"]/ext["pipeline"].replace(0, np.nan)).fillna(0)
            c2 = float((ext["pred_usd"]*share_a).sum())
            c3 = float((ext["pred_usd"]*(1-share_a)).sum())
        else:
            c2 = c3 = 0.0
        total = c1 + c2 + c3
        t = pd.DataFrame({
            "aportacion": ["C1 multi-año 2Y/3Y (firme, ya en pipeline)",
                           "C2 re-renovación 1Y (simulada)",
                           "C3 adquisición 1Y (simulada)", "TOTAL " + str(y1)],
            "usd": [c1, c2, c3, total],
            "pct": [100*c1/max(total,1), 100*c2/max(total,1),
                    100*c3/max(total,1), 100.0],
            "banda": ["firme", "ancha", "ancha (la mayor)", "—"]}).set_index("aportacion")
        self._table(t.round(1), f"AÑO COMPLETO {y1}: aportación de cada parte:",
                    col_defs={"usd": "$ previsto de la parte",
                              "pct": "% del total del año objetivo",
                              "banda": "firmeza: C1 ya firmado; C2/C3 estimación sobre estimación"})
        self.assembled_2027 = t
        self._mark("step_h2_assemble_2027",
                   metadata={"c1": c1, "c2": c2, "c3": c3, "total": total, "year": y1})
        self._done("step_h2_assemble_2027",
                   f"{y1} total ${total:,.0f} = C1 ${c1:,.0f} + C2 ${c2:,.0f} + C3 ${c3:,.0f}")
        return {"c1": c1, "c2": c2, "c3": c3, "total": total}

    def step_h2_export_final_table(self, path="/mnt/user-data/outputs/tabla_final_forecast.csv") -> None:
        """TABLA FINAL: dato raw fino + predicciones + columnas explicativas
        (z, tasa padre, uplift ajustado) — back-annotation por fu_id+mes."""
        self._require("step_h2_export_final_table", ["step_h2_forecast_projection"])
        cfg = self.cfg
        pc = cfg.period_col
        base = self.df_fine if self.df_fine is not None else self.df.copy()
        if "fu_id" not in base.columns:
            base = base.copy()
            base["fu_id"] = base[self.dimension_cols].astype(str).agg("|".join, axis=1)
        v = self.df.copy()
        v["fu_id"] = v[self.dimension_cols].astype(str).agg("|".join, axis=1)
        v["_mk"] = self._mand_key(v)
        v["expl_z"] = v["step_2_fs_id"].map(self.h2_fs["z"])
        v["expl_rate_celda"] = v["_mk"].map(self.h2_mandatory["rate"])
        v["expl_uplift_adj"] = self._uplift_for(v)
        keep = ["fu_id", pc, "h2_pred_units_mandatory", "h2_pred_units_shrunk",
                "expl_z", "expl_rate_celda", "expl_uplift_adj"]
        out = base.merge(v[keep], on=["fu_id", pc], how="left")
        out["pred_usd_shrunk"] = (out["h2_pred_units_shrunk"]
                                  * np.where(out[cfg.pipeline_units_col] > 0,
                                             out[cfg.pipeline_usd_col]
                                             / out[cfg.pipeline_units_col], 0)
                                  * out["expl_uplift_adj"].fillna(1))
        out.to_csv(path, index=False)
        self._log(f"  step_h2_export_final_table: {len(out):,} filas → {path}")
        self._mark("step_h2_export_final_table")
        self._done("step_h2_export_final_table", f"tabla final exportada ({len(out):,} filas)")


def run_hito_2(sf1) -> StratifiedForecastHito2:
    """Encadena HITO 2 sobre un HITO 1 ya corrido."""
    sf = StratifiedForecastHito2.from_hito1(sf1)
    sf.step_h2_fit_baseline_mandatory()
    sf.step_h2_fit_shrunk()
    sf.step_h2_fit_uplift_covariates()
    sf.step_h2_fit_ts()
    sf.step_h2_reassess_support()
    sf.step_h2_improvement_summary()
    sf.step_h2_quality_photo()
    sf.step_h2_backtest_test_months()
    sf.step_h2_forecast_projection()
    sf.step_h2_forecast_bands()
    sf.step_h2_forecast_next_year()
    sf.step_h2_extend_AB()
    sf.step_h2_assemble_2027()
    sf.step_h2_export_final_table()
    return sf

# ============================= 5. RUNNERS =================================
def run_hito_1(config, df_raw=None) -> StratifiedForecastHito1:
    """df_raw: DataFrame precargado opcional — evita releer el dato en cada
    corrida (apaño de iteración; step_0 trabaja sobre una copia)."""
    sf = StratifiedForecastHito1(config)
    if df_raw is not None:
        sf.df_raw = df_raw
    # FASE A — validar y preparar fila a fila
    sf.step_0_validate_input()
    sf.step_1_normalize_period()
    sf.step_1_collapse_covariates()               # covariables → vista FU interna (fina en df_fine)
    sf.step_1_derive_roles_from_period()  # audita roles del raw; con el flag, los deriva del periodo
    sf.step_2_report_only_projection_top()  # cobertura: $ solo en projection (id al vuelo)
    sf.step_1_add_universe()
    sf.step_1_add_coverage_pattern()
    sf.step_1_add_forecast_route()
    sf.step_1_report_money_by_route()   # magnitud del problema
    sf.step_1_drop_no_impact()          # las que no impactan se borran
    sf.step_1_fill_gaps()
    sf.step_1_report_density()       # VALORACIÓN 2 — densidad de las FU por sample size
    sf.step_1_add_rates()
    sf.step_1_add_auv()
    sf.step_1_add_synthetic_flag()
    sf.step_1_assert_coherence()
    # FASE B — construir Forecast Units
    sf.step_2_build_identity()
    sf.step_2_build_support()
    sf.step_2_collapse_signal_support()
    sf.step_2_report_gap_examples()              # ¿POR QUÉ tienen huecos las series GRANDES? (timeline + meses)
    sf.step_2_report_density_money()       # VALORACIÓN 2 (dinero): densidad × usd_proj
    sf.step_2_report_gap_density_money()
    sf.step_2_report_no_training_top()
    sf.step_2_report_support_profile()   # COMPOSICIÓN: sano / ralo-rescatable / ralo-solitaria
    sf.step_2_report_history_length()      # TOPOLOGÍA: eje longitud (ortogonal a densidad)
    # FASE B1 — comparar agregación mandatory (gruesa) vs grano fino
    sf.step_2_report_density_mandatory_vs_full()   # soporte: gruesa vs fina
    sf.step_2_report_aggregation_cost()            # COSTE en $ de promediar vs atribuir
    sf.step_2_report_distributions()               # DISTRIBUCIONES: percentiles + Lorenz + gráfico
    # FASE B2 — qué dimensión hace qué: ANOVA activado (llegó su caso de uso:
    # cruzar señal con coste de grano para detectar dims caras sin información).
    sf.step_3_anova_rate()
    sf.step_3_collapse_anova()                 # SEÑAL: qué dim separa la tasa (η²)
    sf.step_3_report_dim_fragmentation()
    sf.step_3_classify_small_series()   # COSTE: qué dim fabrica el grano + cruce
    sf.step_3_report_level_coverage()             # ¿qué NIVELES faltan en meses? (certificado de huecos al grano fino)
    sf.step_3_report_covariate_value()            # FASE 0: cobertura + factores empíricos por combinación
    sf.step_6_add_story_columns()                 # ESCALERA: etiqueta modos de fallo (cota binomial como escala)
    sf.step_6_report_story_figures()              # gráficas PNG del storytelling
    # FASE C — describir (foto). Opcional para una corrida rápida de magnitud.
    sf.step_2b_describe()
    return sf




def run_all(config, df_raw=None):
    """Ejecuta el framework completo (HITO 1 + HITO 2) y devuelve (sf1, sf2)."""
    sf1 = run_hito_1(config, df_raw=df_raw)
    sf2 = run_hito_2(sf1)
    return sf1, sf2
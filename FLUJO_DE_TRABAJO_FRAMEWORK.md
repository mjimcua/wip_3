# FLUJO DE TRABAJO DEL FRAMEWORK — guía de métodos (introspección de stratified_forecast.py)

## Correspondencia hitos conceptuales ↔ código

Los 4 HITOS (GUIA_CAP3..6) son la arquitectura CONCEPTUAL; el código los
empaqueta en 2 clases por razones de estado compartido (decisión de
implementación, no conceptual):

| Hito conceptual | Capítulo | Dónde vive |
|---|---|---|
| H1 construir FUs y valorar | CAP3 | clase Hito1 → §1.* |
| H2 elegir el método | CAP4 | clase Hito2 → §2.1–§2.7 |
| H3 valor, bandas, trazabilidad | CAP5 | §2.10 (tabla final) · bandas/walk-forward PENDIENTES (diseño en CAP5) |
| H4 horizonte extendido | CAP6 | §2.8–§2.9 (next_year referencia + A+B que manda) |

Hilo argumental: 1) PREPARAR el dato → 2) VALORAR el terreno → 3) MÉTODO
→ 4) VALOR y trazabilidad → 5) HORIZONTE.

## Glosario
- **FU**: Forecast Unit: una celda (combinación de dims) en UN mes.
- **FS**: Forecast Series: la serie temporal de una combinación de dims.
- **soporte**: unidades de pipeline de la FU/FS (n de la binomial).
- **cota binomial**: error mínimo del caso más favorable (mundo plano): sqrt(p(1-p)/n). Error≤cota=ruido; error>>cota=error real añadido.
- **z (evidence weight)**: peso por evidencia del dato propio: z=n/(n+k). z→1 habla la serie; z→0 hereda su celda.
- **uplift**: revalorización: AUV_renovado / AUV_pipeline.
- **WAPE**: suma de |error| / suma de real (ponderado por tamaño).
- **eta2**: η²: % de la varianza de la tasa que explica una dimensión.
- **celda mandatory**: agregado a las dims obligatorias de negocio.
- **gate**: primer criterio de la escalera que la celda NO supera.

## El concepto central: PESO POR EVIDENCIA (evidence weight) — explicación completa\n\n*Término oficial del proyecto: **evidence weight / peso por evidencia**. La literatura lo llama credibilidad (actuarios) o shrinkage; evitamos «credibilidad» para no sugerir que otras fuentes no sean creíbles. **Uplift** se mantiene siempre en inglés como concepto.*

### El problema que z resuelve
Tras estratificar tenemos decenas de miles de series finas. Para cada una
necesitamos una tasa de renovación, y hay dos fuentes: su PROPIA historia y
la de su GRUPO (su celda mandatory). Los dos extremos fallan. Creer solo al
dato propio: una serie con 3 unidades da tasas de 0%, 33% o 100% según caiga
la moneda — forecasts absurdos serie a serie. Creer solo al grupo: ignoras
que dentro conviven poblaciones genuinamente distintas (dormant vs activo) —
el sesgo de mezcla/Simpson que motivó el framework. La credibilidad es el
compromiso: una mezcla cuyo peso depende de cuánta evidencia propia hay.

### La fórmula y sus piezas
    tasa_publicada = z · tasa_propia + (1−z) · tasa_de_su_celda,  z = n/(n+k)
- **z** ES la credibilidad: el peso (0 a 1) del dato propio.
- **n** = soporte propio en train (unidades de pipeline observadas).
- **k** = parámetro intermedio: el soporte en el que ambas fuentes pesan
  igual (n=k → z=0,5). Léelo como "cuántas observaciones propias vale lo que
  ya sé por pertenecer al grupo".
- **Ejemplo** (k=200): propia 40%, celda 70%. Con n=50 → z=0,2 → publicada
  0,2·40+0,8·70 = 64%. Con n=2.000 → z=0,91 (habla casi sola). Con n=0 →
  hereda la celda entera.

### De dónde sale k, y qué es el "clip 5–5.000"
k NO se elige a mano: se calcula de los datos comparando dos variabilidades
— cuánto rebota cada serie alrededor de su propia verdad (ruido) frente a
cuánto difieren de verdad las series del grupo entre sí (señal). Series del
grupo muy parecidas entre sí → k grande → mezcla fuerte hacia el grupo
(correcto). Series genuinamente distintas → k pequeño → el dato propio manda
antes (correcto). El portfolio decide cuánto pooling necesita. El **clip**
es una barandilla sobre ese cálculo: si k sale por debajo de 5 se fuerza a
5, y por encima de 5.000 se fuerza a 5.000 — un mes atípico o una serie
disparatada pueden producir un k absurdo (casi 0 = "créete cualquier
astilla"; casi infinito = "ignora hasta 50.000 unidades propias") y el
recorte impide que un accidente del cálculo produzca mezclas sin sentido.
Los topes 5 y 5.000 son elegidos y revisables.

### ¿Para qué se usa z? ¿Solo informa o decide algo?
**Decide: entra directamente en el número del forecast.** No hay un paso
posterior donde alguien mire z y decida — z ES la decisión, tomada en
automático y en continuo (no un sí/no), una vez por serie: cuánto del
forecast viene de su historia y cuánto hereda del grupo. Condiciona en
cuatro sitios: (1) la tasa del método M3 (shrunk); (2) la tasa temporal de
M4 (la media móvil EWM también se mezcla con z); (3) el error presupuestado
de la foto de calidad (SE×z: cuantifica en $ lo que la credibilidad compra);
(4) el componente A del horizonte (usa predicciones shrunk). Lo que NO
decide: no colapsa dimensiones (eso es de ANOVA/fragmentación), no filtra
series, no es umbral de aceptación. La única pieza que es SOLO vigilancia es
el **z ponderado por $** del log: dice si el dinero importante habla con voz
propia o hereda el grupo — información para el analista, no una palanca.

### Tasa vs uplift: por qué no comparten k
La tasa es un porcentaje acotado (cada unidad renueva o no): su variabilidad
tiene techo. El uplift es un cociente de dinero (puede valer 0,9, 1,1 o 3,0)
y unas pocas operaciones raras mueven mucho la media de una serie pequeña.
Consecuencia: para fiarte del uplift propio necesitas MÁS datos que para
fiarte de la tasa propia → el k del uplift debe ser mayor que el de la tasa.
Misma fórmula, más exigencia de evidencia. (Hoy el uplift por covariables
usa factores por combinación con mínimo n=200 y fallback global — el k
propio del uplift por serie es extensión futura.)

### Equivalencias (por si quieres leer más)
La misma fórmula con otros nombres: credibilidad actuarial (Bühlmann, 1967),
Fay–Herriot/EBLUP en estadística oficial, media posterior beta-binomial.
Tres campos llegaron a ella porque es la mezcla lineal de mínimo error.

*(Nota: trim_history y trim_pilot fueron ELIMINADOS — la limpieza del arranque se hace en ORIGEN. hidden_heterogeneity se fusionó en aggregation_cost.)*

## Densidad, huecos y longitud — explicación extendida (acordada en repaso)

**Dólar mediano.** Ordena cada dólar a predecir según el soporte de la serie
donde vive y mira el del medio: "el 50% del dinero está en series con soporte
>= X". Es el contrapunto de la SERIE mediana: las series son pequeñas; el
dinero, no. Dos medianas de la misma población, pesadas distinto.

**Lorenz y Gini.** Ordena las series de mayor a menor dinero; la curva de
Lorenz dibuja x = % acumulado de series, y = % acumulado de $. Gini = 2 x
área entre la curva y la diagonal de igualdad (0 = reparto uniforme; 1 =
todo en una serie). Implementación: Gini = 2*Suma(i*x_i)/(n*Suma x) - (n+1)/n
con x ordenado ascendente. Esperado en suscripciones B2C: alto (0,8-0,95).
Lectura operativa: precisión artesanal en la cabeza (las series top son
auditables una a una), evidence weight industrial en la cola.

**¿El Gini decide algo? (informar vs condicionar).** El Gini NO entra en
ningún cálculo del framework (a diferencia de z, que SÍ es el número). Sus
tres usos legítimos: (1) decidir dónde va la ATENCIÓN HUMANA — con Gini alto,
la cabeza (las pocas series que concentran el 50% del $) se audita y explica
serie a serie, y la cola se delega al evidence weight; con Gini bajo no
existiría esa cabeza auditable y el esfuerzo se repartiría distinto;
(2) REFERENCIA COMPARATIVA: un solo número que resume la forma del dinero —
al comparar granos (mandatory vs fino) o pasadas sucesivas (tras limpiar
origen, tras agrupar niveles), un Gini que se mueve mucho avisa de que la
estructura cambió y hay que revisar; (3) SANITY CHECK: en este negocio se
espera 0,8-0,95; un valor fuera de rango en una pasada real sugiere problema
de datos (duplicados, filtros). Posible extensión no implementada: alerta
automática si el Gini varía más de un umbral entre pasadas.

**Una medición, dos vistas (decisión).** La tabla canónica de dinero por
intervalos de soporte (cortes 30/100/200/500) vive en
step_2_report_density_money — es diagnóstico del terreno. La foto del HITO 2
(step_h2_quality_photo) NO re-mide: solo COMPARA antes/después (mandatory vs
framework) con esos mismos cortes, para que sean comparables.

**Dinero por nº de meses de hueco.** step_2_report_gap_density_money agrupa
el $ por el nº EXACTO de meses de hueco de cada serie (0, 1, 2, ...) con %
acumulado: se ve de un vistazo dónde se acaba el dinero limpio. Bajo el
contrato vigente un hueco es un cero legítimo (mes sin negocio) y solo
debería aparecer en series de poco valor — si carga $ material, es señal de
revisar el origen.

**Histograma de longitud de historia.** step_2_report_history_length:
nº de series y $ por tramo de meses, con barra proporcional al $. El corte
>=18 meses marca el dinero que puede optar a la técnica de serie temporal
(EWM) del HITO 2; <12 meses no permite observar un ciclo estacional completo.

## §1 — clase Hito1 (= HITO conceptual 1)

### §1.1 `step_0_validate_input`
Valida el raw (columnas, dims, NaN en dims, flag_time_series, rol) y
  normaliza el rol a minúsculas. No transforma nada más.

### §1.2 `step_1_normalize_period`
Convierte la columna de periodo a Period[M] (in-place).

### §1.3 `step_1_collapse_covariates`
Colapsa el raw multiplicado por covariables a la VISTA FU interna
  (grano dims_estables × mes): Σ unidades/USD, AUVs RECALCULADAS del
  agregado (jamás promediadas). La tabla FINA original se conserva en
  self.df_fine con fu_id (dims estables) y comb_id (covariables) para
  Fase 0, auditoría y back-annotation futura. Asserts: conservación de
  masa. Sin covariables declaradas = no hace nada.

### §1.4 `step_1_derive_roles_from_period`
Asigna roles train/test/projection a partir de pending_date.
    projection = desde pending_date en adelante (mes incluido)
    test       = los test_months meses inmediatamente anteriores
    train      = todo lo anterior
  Con use_raw_roles=True solo DIAGNOSTICA (respeta la columna del raw).

### §1.5 `step_1_add_universe`
Añade step_1_universe (normal | time_series): cómo se predice la fila.

### §1.6 `step_1_add_coverage_pattern`
Añade step_1_coverage_pattern (7 categorías por roles cubiertos).

### §1.7 `step_1_add_forecast_route`
Añade step_1_forecast_route (veredicto de cobertura):
  no_impact (sin projection) | heuristic (projection sin train/test) |
  trainable (projection con historia).

### §1.8 `step_1_report_money_by_route`
Pinta el dinero a predecir por ruta (pipeline_usd de projection del año
  en curso, agregado por forecast_route). La magnitud del problema.

### §1.9 `step_1_drop_no_impact`
Borra del df las FS no_impact (sin projection): ya no se usan.

### §1.10 `step_1_fill_gaps`
Rellena huecos con filas sintéticas SOLO en series trainable y SOLO
  dentro del tramo de historia (train/test), no hasta projection.

### §1.11 `step_1_report_density` — *¿Qué calidad de MUESTRA tienen las FU/FS?*
VALORACIÓN 2 — densidad de las FU por sample size.

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
  series trainable (heurísticas y demás quedan fuera, como en el resto).
**Criterio:** soporte<30=ruidoso; ≥100=utilizable; mediana del portfolio como termómetro.

### §1.12 `step_1_add_rates`
Añade step_1_rate_renewal y step_1_rate_reacq (NaN en projection,
  0 en sintéticas).

### §1.13 `step_1_add_auv`
Añade step_1_auv_pipeline, step_1_auv_renewed, step_1_auv_uplift_ratio
  (uplift=1 si no renovó nadie, 0 prohibido; NaN en projection).

### §1.14 `step_1_add_synthetic_flag`
Añade step_1_fs_with_synthetic_months (1 si la serie tiene imputación).

### §1.15 `step_1_assert_coherence`
Valida la fórmula de 4 factores en filas reales train/test; si falla,
  lista las filas a corregir y lanza DataNotReadyError.

### §1.16 `step_2_build_identity`
Añade step_2_fs_id y step_2_parent_fs_ids (identidad y caminos de
  agregación) sobre el universo normal.

### §1.17 `step_2_build_support`
Añade soporte y economía de cada FS sobre la HISTORIA OBSERVADA
  (cfg.history_roles = train ∪ test): n_avg, usd_avg, usd_proj (dinero a
  predecir, este sí sobre projection), n_periods, history_months,
  first_period, k_share_usd. n_avg/usd_avg incluyen sintéticas.

### §1.18 `step_2_collapse_signal_support` — *¿Ganamos soporte uniendo las series de señal negativa (o positiva) del mismo grupo?*
COLAPSO NIVEL 1 — ganar soporte uniendo señales del MISMO SIGNO.
  Cuando una forecast series con señal time-varying activa (dormant,
  softcancel...) tiene soporte bajo, se une con sus pares del mismo
  signo que COMPARTEN el resto de dimensiones. La unión se marca en una
  columna nueva `step_2_grupo_colapso` con la CONCATENACIÓN de las
  señales activas (p.ej. 'dormant+softcancel') — así no se pierde traza:
  el valor dice qué se agregó. Las series no afectadas reciben
  'no_agrupa' y CONSERVAN sus columnas originales.
  Un solo movimiento, a prueba: se mide el soporte antes/después.
  El signo viene de cfg.structural_timevarying_dims (config, no del dato).
**Fórmula:** candidata = señal activa Y soporte_FS < floor; grupo = concat(señales activas) + resto de dims iguales
**Criterio:** NIVEL 1 del árbol de colapso: une por signo (concatena señales activas, conserva traza) solo si soporte bajo; un movimiento a prueba, mide antes/después.

### §1.19 `step_2_report_gap_examples`
¿POR QUÉ tienen huecos las series grandes? Timeline mensual de las
  top_n series por $ a predecir con algún mes-hueco. El parámetro dims
  fija el GRANO de la serie y se DECLARA en el log: None = grano FU
  COMPLETO (todas las dimensiones, donde viven los huecos); 'mandatory'
  = business_mandatory_dims (agrega el resto: comparable con tu SQL de
  reporting); o lista explícita. A grano agregado, un mes es HUECO solo
  si TODAS sus hijas son sintéticas ese mes.

### §1.20 `step_2_report_density_money` — *¿Cuánto dinero vive en cada nivel de soporte y cómo de concentrado está?*
TABLA CANÓNICA dinero × soporte (medición única; la foto del HITO 2
  solo COMPARA antes/después con estos mismos cortes). Reparte el $ a
  predecir por intervalos de soporte mediano de su serie, mide la
  concentración (Lorenz/Gini), cuántas series concentran el 50% del $
  y el dólar mediano (soporte donde cruza el 50% del $ acumulado).
**Fórmula:** Gini = 2*Suma(i*x_i)/(n*Suma(x)) - (n+1)/n con x = $ por serie ordenado ASC (0=uniforme, 1=todo en una serie); dolar mediano = soporte donde el $ acumulado (soporte asc) cruza el 50%
**Criterio:** tabla CANÓNICA por cortes 30/100/200/500 (la foto del HITO 2 compara antes/después con estos mismos cortes); esperado Gini alto (0,8-0,95). El Gini SOLO INFORMA: no condiciona ningún cálculo; decide dónde va la atención HUMANA (cabeza auditable serie a serie vs cola industrial) y es referencia comparativa entre granos y entre pasadas (si cambia mucho, la forma del dinero cambió: revisar).

### §1.21 `step_2_report_gap_density_money` — *¿Cuánto DINERO vive en series intermitentes?*
¿Cuánto DINERO vive en series intermitentes? Tabla de $ por nº
  EXACTO de meses de hueco de la serie (0, 1, 2, ...), ordenada por nº
  de meses: se ve de un vistazo dónde se acaba el dinero limpio. Hueco
  = mes sintético (cero legítimo bajo el contrato vigente).
**Fórmula:** huecos de la serie = nº de meses sintéticos en su historia; tabla = $ agrupado por nº EXACTO de meses de hueco
**Criterio:** huecos=ceros legítimos (contrato vigente); preocupa solo si cargan $ material.

### §1.22 `step_2_report_no_training_top` — *¿Qué dinero hay que predecir SIN ninguna evidencia de entrenamiento?*
RIESGO: dinero a predecir SIN NINGUNA evidencia de entrenamiento.
  Series con $ en projection y CERO meses reales en train. En un
  negocio continuo esto es anómalo: candidatas a revisión en origen
  (migración de población entre celdas por dims time-varying, producto
  nuevo, o clave mal informada). Lista el top por $ con la clave
  completa dim=valor para poder buscarlas en el origen.
**Fórmula:** sin train = 0 meses reales con pipeline>0 en rol train; ranking por $ de projection
**Criterio:** en un negocio continuo es ANÓMALO: revisar en origen (migración entre celdas por dims time-varying, producto nuevo, clave mal informada). El framework las cubre con la tasa del grupo (z=0), pero la revisión manda.

### §1.23 `step_2_report_only_projection_top` — *¿Qué forecast series viven SOLO en projection (heurística pura) y cuáles son las 3 de más $?*
FORECAST SERIES que SOLO existen en projection (ni train ni test en
  todo el histórico): necesitan heurística pura (no hay nada que
  aprender). Cuenta cuántas SERIES son (no FU) y lista el TOP por $ con
  la clave completa dim=valor — para revisar en 3 líneas si una variable
  concreta no está calculada a futuro y corta la continuidad de esos
  productos. Solo informa; el framework las cubre heredando del grupo.
**Fórmula:** solo-projection = FS con pipeline>0 en projection y 0 en train+test
**Criterio:** ni train ni test en todo el histórico → no hay nada que aprender; top 3 con clave completa para revisar si falta una variable a futuro. Solo informa.

### §1.24 `step_2_report_support_profile` — *¿Cómo es la materia prima por tramos de soporte y cuánto error arrastramos solo por tamaño?*
DESCRIPCIÓN del raw por TRAMOS DE SOPORTE + REFERENCIA de error
  binomial. Dos tablas:
    (1) forecast series por tramo de soporte mediano: nº de series, $ a
        predecir, y de la TASA del tramo: promedio, mediana y dispersión
        (desv. típica). Describe la materia prima sin predecir nada.
    (2) referencia de error binomial por nivel de soporte: ±z·√(p(1−p)/n)
        con p=0,5 (peor caso) al 95% — el error que arrastras SÓLO por
        tamaño de muestra, aunque acertaras la probabilidad. Es la regla
        de la moneda hecha tabla: motiva el HITO 2 (aumentar soporte).
**Fórmula:** error_pp = 1.96·√(0.5·0.5/n)·100 (p=0,5 peor caso, 95%)
**Criterio:** describe sin predecir; el error binomial (±z·√(p(1−p)/n)) es el error irreducible por muestra → motiva el HITO 2.

### §1.25 `step_2_report_history_length` — *¿Cuánta historia tiene cada serie y cuánto $ puede optar a la técnica de serie temporal?*
HISTOGRAMA doble de LONGITUD de historia: nº de series y $ a
  predecir por tramo de meses, con barra proporcional al $. El corte
  >=18 marca quién puede optar a la técnica de serie temporal (EWM) del
  HITO 2; <12 no permite observar un ciclo estacional completo.
**Fórmula:** longitud = meses entre nacimiento y fin de historia de la serie
**Criterio:** <12m: sin ciclo estacional observable; >=18m: elegible para EWM (HITO 2).

### §1.26 `step_2_report_density_mandatory_vs_full`
Compara el soporte mensual (densidad) en dos granos: la agregación
  MANDATORY (gruesa, reporting tradicional) vs el grano FINO (FS). Muestra
  cómo, al desagregar, el dinero se mueve de celdas con mucho soporte a
  celdas con poco — el precio en muestra de evitar Simpson. Ponderado por
  usd_proj. El soporte de cada celda = unidades sumadas por mes (aditivo).

### §1.27 `step_2_report_aggregation_cost`
Coste en DINERO de promediar la tasa al grano mandatory vs predecir
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
  freeze por deciles (freeze_decile_tolerances_usd).

### §1.28 `step_2_report_distributions`
Los resúmenes numéricos (mediana) son limitados: aquí, DISTRIBUCIONES.
  Percentiles del soporte por FS en dos pesos — por nº de series (cuántas
  líneas son astillas) y por DINERO a predecir (dónde vive el dólar
  mediano) —, concentración tipo Lorenz ('¿cuántas líneas necesito para
  cubrir el X% del dinero?', la respuesta directa a 'demasiadas líneas'),
  y un PNG con 4 paneles: (a) distribución del soporte, series vs dinero;
  (b) curva de Lorenz del dinero por FS; (c) dispersión soporte×dinero por
  FS; (d) longitud de historia, %FS vs %$. Si matplotlib no está, deja
  solo los resúmenes.

### §1.29 `step_3_anova_rate` — *¿Qué dimensiones SEPARAN la tasa (no colapsar)?*
ANOVA de la TASA DE RENOVACIÓN por dimensión (η² marginal, ponderado
  por soporte). Para cada dim, qué fracción de la varianza de la tasa
  explican sus niveles:
    - η² ALTO  → la dim separa las tasas: lleva señal, NO colapsar (y alto
      riesgo de Simpson si se colapsa).
    - η² BAJO  → no separa: es la variable ANULABLE, la que define hermanas
      sanas y por la que conviene agrupar primero.
  Marginal a propósito (directo y robusto): NO captura interacciones —
  anular una dim de η² bajo puede ser malo si interactúa con otra—; esa
  versión condicional se difiere (ver NOTAS bloque E). Se mide sobre la
  historia observada real (trainable, no sintéticas, rate válido).
**Fórmula:** eta2 = varianza-entre-niveles / varianza-total de la tasa, ponderada por soporte
**Criterio:** η²≥0.10 separa; 0.03–0.10 intermedia; <0.03 anulable si además fragmenta.

### §1.30 `step_3_collapse_anova` — *¿Ganamos soporte colapsando la dimensión no-mandatory que no separa la tasa?*
COLAPSO NIVEL 2 — ejecuta lo que el ANOVA solo informaba: para las
  series que SIGUEN pobres tras el Nivel 1, colapsa (pone '*') la
  dimensión NO-mandatory que menos separa la tasa (η² < eta_low). No
  toca geografía/producto (mandatory) — eso es Nivel 3. Genera
  step_2_fs_id_L2 (idempotente: = L1 si no se toca).
**Fórmula:** drop_dim = argmin η² entre no-mandatory con η²<eta_low; '*' en su posición si soporte<floor
**Criterio:** NIVEL 2 del árbol: ejecuta lo que el ANOVA informaba. Colapsa con '*' la dim no-mandatory de menor η² para las series aún pobres. No toca geografía/producto.

### §1.31 `step_3_report_dim_fragmentation` — *¿Qué dimensión FABRICA el grano fino?*
¿Qué dimensión FABRICA el grano? Responde a 'no me creo la mediana 1'
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
  el dato es correcto y la mediana 1 es fragmentación, no error.
**Criterio:** factor alto SIN señal = quitar; CON señal = shrink, no quitar.

### §1.32 `step_3_classify_small_series` — *¿Cuántas series pequeñas hay y cuánto $ son: fusionables, encogibles o polvo?*
CLASIFICA las series de bajo soporte en tres DESTINOS DE GESTIÓN
  (no tres métodos de forecast — el motor sigue siendo uno):
    - FUSIONABLE: soporte en [dust, small) Y su celda tiene alguna
      dimensión SIN señal (anulable según ANOVA) que, quitada, la une a
      hermanas → puede ganar grano real. Candidata a subir medio nivel.
    - ENCOGIBLE: pequeña, pero su celda padre (mandatory) tiene MASA →
      el evidence weight ya la trata bien (z bajo, hereda del padre).
    - POLVO: soporte < dust Y padre también pobre → no hay señal que
      extraer; hereda y se REPORTA AGREGADA. Pretender más es inventar.
  Da conteo y $ por cubo: la información para decidir el grano, sin
  tocar el forecast. La acción (fusionar/subir grano) la decides tú.
**Fórmula:** fusionable: dust<=sop<small Y existe dim anulable; polvo: sop<dust Y padre<mediana; resto encogible
**Criterio:** 3 destinos de GESTIÓN, no 3 métodos: fusionable (subir grano si hay dim sin señal), encogible (evidence weight ya resuelve), polvo (heredar y reportar agregado). El nº de series es incomodidad; el $ es lo que decide.

### §1.33 `step_3_report_level_coverage`
¿Qué NIVELES de cada dimensión faltan en algunos meses? El test va
  MES A MES: si en un mes falta alguno de los valores posibles de una dim,
  ese valor genera huecos GARANTIZADOS ese mes en todas las series vivas
  que lo contienen — sí o sí, antes de mirar región/dormant/nada. No hace
  falta que falte en toda la historia: basta un mes. Cota inferior barata;
  la presencia en el mes NO garantiza nada al grano fino (asimetría).
  Por (dim, nivel): % de meses con presencia, % de unidades y % de $hist.
  Candidato a AGRUPAR ('Other'): presencia <100% de los meses o cuota
  < min_share_pct. Veredicto por dim: nº de niveles intermitentes.

### §1.34 `step_3_report_covariate_value`
FASE 0 (solo informa): cobertura mensual de las covariables (% de
  unidades con dato vs no_info) y tabla de FACTORES EMPÍRICOS de uplift
  por COMBINACIÓN observada en la historia: uplift = AUV_renovado /
  AUV_pipeline del agregado de la combinación. Con soporte; las
  combinaciones con n<min_n se marcan (candidatas a fallback marginal).

### §1.35 `step_6_add_story_columns` — *¿Qué calidad tiene cada celda y cuál es su MODO DE FALLO?*
LA ESCALERA: etiqueta cada celda mandatory con su modo de fallo y sus mediciones (cota binomial = escala del error aceptable).
  Regla de lectura explícita: la cota binomial es el error mínimo del caso
  MÁS FAVORABLE (mundo plano); error ≤ cota → ruido de muestra; error ≫
  cota → la situación estudiada (tendencia/mezcla/estacionalidad) AÑADE
  error real. Estampa columnas story_* en TODAS las filas de cada celda.
  Ventanas: A/B = dos últimos años naturales completos; WHAT-IF = último
  año natural completo → meses siguientes (mimetiza el forecast).
**Criterio:** escalera: soporte→temporal→estacionalidad→tendencia→mezcla→apto_promedio; el gate es el PRIMER fallo, los flags story_* guardan todos.

### §1.36 `step_6_report_story_figures`
Gráficas del storytelling (PNG): dinero por gate, error vs cota binomial, mezcla congelada del top Simpson y tendencia-vs-plano del top impacto.
  Requiere step_6_add_story_columns y matplotlib.

### §1.37 `step_2b_describe`
Describe las FS (foto/fingerprint): homogeneidades, soporte, economía,
  riesgo. No modifica self.df. La lógica está integrada al final de este fichero.

## §2 — clase Hito2 (= HITOS conceptuales 2, 3 y 4)

--- *bloque del HITO conceptual 2 (CAP4)* ---

### §2.1 `step_h2_fit_baseline_mandatory`
Método ANTES (tradicional): tasa promedio plana por celda MANDATORY
  sobre train (Σ renovadas / Σ pipeline del agregado de la celda).

### §2.2 `step_h2_fit_shrunk` — *¿Cuánta voz tiene el dato propio de cada serie?*
Método DESPUÉS (framework): tasa por FS fina con EVIDENCE WEIGHT (peso por evidencia) hacia
  su celda mandatory: rate = z·rate_fs + (1−z)·rate_celda, z = n/(n+k).
  k estimado por momentos (Bühlmann simplificado) y logueado.
**Fórmula:** z = n/(n+k); k = ruido-dentro/varianza-entre (momentos), recortado a [5, 5000]; tasa publicada = z*propia + (1-z)*celda
**Criterio:** k por momentos; vigilar z ponderado por $.

### §2.3 `step_h2_fit_uplift_covariates` — *¿Cómo mueve el descuento/destope la REVALORIZACIÓN?*
UPLIFT con covariables (Fase 2): T3 = factores por COMBINACIÓN
  estimados en train sobre la tabla fina (uplift_comb = AUV_ren/AUV_pipe
  del agregado; fallback global si n<min). uplift_ajustado por (FU,mes)
  = Σ share_comb × factor_comb, con no_info → uplift base de la celda
  mandatory. Sin df_fine: uplift base para todo (declarado).
**Fórmula:** uplift_comb = (Suma $ren/Suma u_ren)/(Suma $pipe/Suma u_pipe) del train de la combinación; ajustado = share_info*factor + (1-share_info)*base de la celda
**Criterio:** factor por combinación con n≥200; si no, fallback global; no_info usa el base de la celda.

### §2.4 `step_h2_fit_ts`
Técnica de SERIE TEMPORAL para FS con historial suficiente (≥18
  meses reales): nivel EWM (half-life 6) de la tasa mensual de la FS,
  mezclado con la credibilidad (z de la FS). Series cortas → shrunk.

### §2.5 `step_h2_reassess_support` — *Tras el colapso, ¿cuánto $ tiene tasa propia, cuánto hereda y cuánto es heurística?*
PASO A del cierre — re-evalúa el SOPORTE tras los colapsos del árbol
  (N1 sustrato, N2 ANOVA) y etiqueta el estado final de cada serie:
    RESUELTA       : soporte del grupo >= support_floor → tasa propia.
    EVIDENCE_WEIGHT: pobre pero con padre → tasa mezclada (z).
    HEURISTICA     : soporte < dust_floor o sin train → regla/horizonte.
  Usa el id de grupo más colapsado disponible (L2>L1>fino). No cambia el
  forecast: ETIQUETA para que la elección de técnica y el reporte sepan
  de qué se fían. KPI: deja trazable cuánto $ es de cada estado.
**Fórmula:** RESUELTA si soporte_grupo>=floor; HEURISTICA si <dust o sin train; resto EVIDENCE_WEIGHT
**Criterio:** PASO A del cierre: re-evalúa soporte del grupo colapsado y etiqueta estado_final.

### §2.6 `step_h2_improvement_summary` — *¿Qué soporte hemos ganado con el colapso (antes/después)?*
PASO B del cierre — resumen ANTES/DESPUÉS del colapso, en las varas
  del HITO 1: soporte mediano y % del $ bajo soporte<100. Muestra qué
  compró el árbol de colapso. KPI: soporte, error binomial efectivo.
**Fórmula:** soporte mediano y %$ bajo soporte<100, fino vs grupo
**Criterio:** PASO B: compara grano fino vs colapsado en las varas del HITO 1.

### §2.7 `step_h2_quality_photo` — *¿Qué calidad tienen las series ANTES (mandatory) vs DESPUÉS (framework)?*
LA FOTO: calidad de las forecast series ANTES (grano mandatory,
  promedio plano) vs DESPUÉS (grano fino + credibilidad). Estructura:
  nº series, soporte mediano, % del $ en soporte débil, y el ERROR DE
  MUESTREO PRESUPUESTADO en $ (Σ SE binomial × exposición).
**Fórmula:** error presupuestado = Suma_series SE*$, con SE = raiz(p(1-p)/n) (en DESPUÉS, SE*z)
**Criterio:** el fino fragmenta; la credibilidad devuelve el error presupuestado a escala.

### §2.8 `step_h2_backtest_test_months` — *¿Cuánto MEJORA cada bloque de variables, en $ reales?*
Backtest ESCALONADO en los meses de TEST, $ reales (renewed_usd):
  M1 mandatory (tasa celda × uplift base) · M2 +covariables (uplift
  ajustado) · M3 +dims finas/time-varying (shrunk) · M4 +serie temporal
  (EWM en series largas). La escalera muestra qué aporta CADA bloque de
  variables.
**Fórmula:** WAPE = Suma|pred-real|/Suma(real) en $; sesgo = (Suma pred - Suma real)/Suma real; pred$ = tasa*pipeline*AUV_pipe*uplift
**Criterio:** escalonado M1→M4 sobre test reservado; gana el WAPE $ mínimo (validación out-of-sample pendiente, ver doc).

--- *bloque del HITO conceptual 3 (CAP5)* ---

### §2.9 `step_h2_forecast_projection`
Predicción final para PROJECTION con ambos métodos (columnas
  h2_pred_units_mandatory / h2_pred_units_shrunk en el df), lista para
  PBI y para la back-annotation futura a la tabla fina.

--- *bloque del HITO conceptual 2 (CAP4)* ---

### §2.10 `step_h2_forecast_bands` — *¿Cuál es el forecast 2026 y su banda de confianza?*
BANDA DE CONFIANZA del forecast 2026 (projection), por celda
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
  incertidumbre para comunicar a negocio (declarado).
**Fórmula:** half = z·max(WAPE·pred, raíz(Σ SE_usd²)); lo/hi escalados por 1+trend_skew·|pendiente| en el lado correspondiente
**Criterio:** banda anclada en el error REAL del backtest (no inventada); suelo binomial irreducible por serie; asimétrica donde hay tendencia reciente. Declarada como cuantificación honesta, no IC frecuentista exacto.

--- *bloque del HITO conceptual 4 (CAP6)* ---

### §2.11 `step_h2_forecast_next_year`
FORECAST COMPLETO del año siguiente al pending (p.ej. 2027), donde
  el pipeline AÚN NO EXISTE. Cadena de regeneración a grano MANDATORY:
  pipeline(año+1, mes) = renovadas(año, mes) × regen_celda, con
  renovadas(año) = reales hasta pending + previstas shrunk después, y
  regen = mediana histórica de pipeline_t / renovadas_{t-12} (captura
  new business implícito; cap [0.2, 5]; fallback global). Supuesto v1
  declarado: lag 12 (term 1y dominante; term 2y apuntado para v2).

--- *bloque del HITO conceptual 2 (CAP4)* ---

### §2.12 `step_h2_assemble_2027` — *¿Cuánto saldrá el año completo siguiente y cuánto aporta cada parte?*
ENSAMBLA EL AÑO COMPLETO siguiente al pending (p.ej. 2027) con sus
  TRES aportaciones visibles y sumadas:
    C1 multi-año ya firmado: renovaciones de contratos 2Y/3Y que están
       en projection real y vencen en el año objetivo (NO se simula);
    C2 = componente A de extend_AB: re-renovación del tramo 1Y simulada;
    C3 = componente B de extend_AB: adquisición 1Y simulada que vencerá.
  Total 2027 = C1 + C2 + C3. Devuelve la tabla con cada parte y su % de
  aportación. Banda: C1 firme; C2/C3 estimación sobre estimación.
**Fórmula:** C1 = pred_shrunk×AUV×uplift de projection 2Y/3Y del año objetivo; C2/C3 = pred_usd de extend_AB partido por share de A vs B
**Criterio:** C1 multi-año ya firmado (firme) + C2 re-renovación 1Y + C3 adquisición 1Y (ambas simuladas, banda ancha); total = C1+C2+C3.

--- *bloque del HITO conceptual 3 (CAP5)* ---

### §2.13 `step_h2_export_final_table`
TABLA FINAL: dato raw fino + predicciones + columnas explicativas
  (z, tasa padre, uplift ajustado) — back-annotation por fu_id+mes.

## Utilidades no-step

### `describe_portfolio()`
Describe pipeline, FUs y FSs del df actual: totales mensuales,
  nº de series, soporte, terms/purchase_type/covariables y % no_info.
  Información de referencia (p.ej. para regenerar sintéticos).

### `step_2_report_fu_audit(match)` (a demanda)
AUDITORÍA de una combinación PARCIAL de dims (las de tu consulta SQL).
  Compara el agregado mensual del RAW intacto vs el procesado (¿se borró
  algo?), lista cuántas FUs FINAS conviven bajo ese paraguas con su
  presencia mes a mes, y señala qué dims difieren entre las dos mayores
  (la dim que mueve población entre celdas). Para investigar huecos.

### `run_all(config, df_raw)`
Ejecuta el framework completo (HITO 1 + HITO 2) y devuelve (sf1, sf2).
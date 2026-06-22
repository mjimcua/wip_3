# El colapso multinivel — documento canónico (2026-06-13)
*Consolida ARBOL + EJEMPLOS + IDENTIFICADORES + MONOGRAFICO + PLAN. Sustituye a
esos cinco. Estado: NIVEL 1 implementado; niveles 2-3 diseñados, no construidos.*

## BLUF
El framework gana soporte para la tasa de renovación bajando por un ÁRBOL de
colapso ordenado de la decisión MÁS informada (menos error) a la de EMERGENCIA.
Cada salto cuesta detalle: solo se da si el soporte obliga, se elige el que
menos información destruye, y se para al alcanzar el umbral. Lo que sobrevive
con soporte conserva su grano fino. Todo en una tabla, con un id por nivel.

## El árbol (orden por nivel de error, de menos a más)

```
NIVEL 0  serie fina (todas las dims)     ¿soporte >= umbral? → SE QUEDA FINA
│
NIVEL 1  SUSTRATO (signo time-varying)   [IMPLEMENTADO: step_2_collapse_signal_support]
│        une señales del MISMO signo que comparten el resto de dims.
│        explica el PORQUÉ de la tasa → primera opción, error casi nulo.
│
NIVEL 2  ANOVA sobre dims NO-mandatory    [diseñado]
│        colapsa (asterisco) la dim que no separa la tasa (η² bajo),
│        sin tocar geografía/producto todavía.
│
NIVEL 3  ruta semántica + mandatory       [parcial: parent_fs_ids existe]
│        device level_2→level_1; en último término, unir mandatory
│        (geografía/producto) — el grano más grueso, el que solo ORGANIZA.
│
NIVEL 4  evidence weight al padre alcanzado  [IMPLEMENTADO: step_h2_fit_shrunk,
│        pero hoy salta directo a mandatory; pendiente que use el padre del árbol]
│
NIVEL 5  heurística (siguiente etapa)
```
**Por qué este orden:** las mandatory (región, producto) solo ORGANIZAN — unir
dos países dice "no tengo info para distinguirlos", no "son iguales". El
sustrato (dormant, softcancel) EXPLICA la tasa. Por eso lo que explica va antes
que lo que solo organiza.

**Regla de oro de toda unión:** se colapsa UN bloque; todas las demás dims
deben coincidir. Nunca se une dormant con softcancel si difieren en otra dim.

## Las TRES mecánicas (distintas, una por tipo de unión)

| Operación | Cómo | Nivel |
|---|---|---|
| Ignorar 1 dim que no separa | `*` en su posición | 2 (ANOVA) |
| Unir señales del mismo signo | columna `step_2_grupo_colapso` con CONCATENACIÓN de las señales activas | 1 (sustrato) |
| Subir jerarquía device | level_2 → level_1 | 3 |

**Clave (monográfico):** unir varias columnas NO es poner asteriscos. El doble
`*` borraría el signo y fundiría positivas con negativas. Por eso el Nivel 1
deriva una columna que concatena las señales activas y conserva la traza.

## NIVEL 1 implementado — cómo funciona (step_2_collapse_signal_support)
- **Disparador:** una FS con señal time-varying activa Y soporte mediano < floor.
- **Acción:** columna `step_2_grupo_colapso` = concatenación de las señales
  activas: `"softcancel"`, `"dormant"`, `"dormant+softcancel"`. Las no afectadas
  → `"no_agrupa"` (centinela) y conservan sus columnas originales.
- **id de nivel:** `step_2_fs_id_L1` = resto de dims + `SIG=<etiqueta>`; si
  `no_agrupa`, idéntico al fs_id fino (idempotente).
- **Un movimiento a prueba:** mide soporte mediano antes/después.
- **Traza:** el valor dice QUÉ se unió (no un opaco "negativo"). Generaliza a
  cualquier nº de señales del mismo signo (config structural_timevarying_dims).
- **OJO config:** el valor "activo" del dato debe estar en
  timevarying_positive_values (incluir "si" si el extract usa español, o 1/0).

## Identificadores por nivel (todo en una tabla)
```
step_2_fs_id        identidad fina (NIVEL 0)
step_2_fs_id_L1     tras sustrato (implementado)
[futuro] _L2, _L3   tras ANOVA / levels
[futuro] fs_id_group = id del último nivel = grupo con el que se ESTIMA la tasa
[futuro] nivel_resuelto, colapsos_aplicados (auditoría legible)
```
Idempotencia: si un nivel no toca una serie, su id = el del nivel anterior.
La tasa se estima sobre el grupo final; el revenue vuelve a la línea fina (L0):
estimar agregado, reportar en origen.

## Tres ejemplos motivadores
- **A (resuelve en N1):** dos series negativas (una dormant, otra softcancel)
  de misma región×producto×purchase_type, ambas pobres → se unen en
  `...|SIG=dormant+softcancel`, ganan soporte.
- **B (N1 no aplica, resuelve en N2):** serie positiva pobre; el ANOVA dice que
  purchase_type no separa → se colapsa con `*`, gana soporte sin mezclar países.
- **C (llega a N3):** serie con device_level_2 y país pequeño, sin no-mandatory
  anulable → sube device a level_1 y, en último término, agrega geografía.

## Pendiente de construir (niveles 2-5)
1. NIVEL 2: step que ejecuta el colapso por ANOVA (hoy el ANOVA solo informa).
2. NIVEL 3: usar parent_fs_ids (ya existe) para device + mandatory.
3. NIVEL 4: que fit_shrunk encoja hacia el padre del ÁRBOL, no al mandatory crudo.
4. fs_id_group, nivel_resuelto, colapsos_aplicados como columnas de salida.
5. Decisión abierta: umbral de soporte único o por nivel (liga con umbrales auto).

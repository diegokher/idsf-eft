# Framework de selección inteligente de datos para el ajuste fino eficiente de modelos de lenguaje

Código, resultados y artefactos del trabajo de grado.

## Descripción

El framework reduce el consumo energético del ajuste fino de DistilBERT en clasificación de texto. Para ello decide de forma conjunta **qué ejemplos conservar** y **hasta qué longitud truncar las secuencias**:

1. **GPT-2 small** (modelo proxy) calcula para cada ejemplo su perplejidad, su longitud y su diversidad.
2. Esas señales se resumen en un **descriptor de 18 dimensiones** por dataset.
3. Una **política PPO compartida** recibe el descriptor y produce una acción de 5 dimensiones: los pesos de selección, la proporción de datos retenidos (ρ) y el percentil del límite de longitud (q).
4. **DistilBERT** (modelo objetivo) se entrena con el subconjunto seleccionado y truncado. **CodeCarbon** mide la energía del entrenamiento.

La recompensa combina el desempeño relativo frente a Full (Accuracy y Macro-F1) con el ahorro energético medido, y penaliza las acciones cuyo desempeño relativo cae por debajo de τ = 0.95.

## Resultados principales

Evaluación sobre el conjunto de prueba oficial: media de 5 semillas, con la política congelada. La reducción de energía y el speedup se calculan sobre el ajuste fino monitorizado de DistilBERT.

| Dataset | Reducción de energía vs. Full | Speedup | ΔAccuracy vs. Full | P medio |
|---|---|---|---|---|
| AG News | 60.03 % | 2.48× | -1.56 pp | 0.982 |
| IMDb | 59.40 % | 2.51× | -2.05 pp | 0.964 |
| TREC | 36.99 % | 1.56× | -2.68 pp | 0.971 |

- **Ablaciones:** agregar el límite de longitud adaptativo a una selección aleatoria del 70 % aumenta el ahorro entre 8 y 23 puntos, con Accuracy media igual o mayor. Con el mismo ρ y el mismo límite de longitud, la selección aprendida supera a la selección aleatoria en IMDb (+11.41 pp, en 5 de 5 semillas) y en TREC (+6.32 pp, en 4 de 5).
- **Leave-one-dataset-out:** el ahorro energético se conserva en los datasets no observados (entre 42 % y 75 %). Hacia AG News la política se transfiere sin adaptación (-0.32 pp). Hacia IMDb y TREC requiere calibración (-5.31 y -17.04 pp).

Los detalles y la discusión completa están en el documento de la tesis.

## Estructura del repositorio

```
.
├── notebooks/
│   ├── pipeline_multidataset.ipynb                 # 1. Datos, características GPT-2 y descriptor 18D
│   ├── entrenamiento_politica_compartida.ipynb     # 2. Entrenamiento de la política PPO (600 episodios)
│   ├── evaluacion_multidataset.ipynb               # 3. Evaluación Full / Random-70 / Agent y ablaciones
│   └── generalizacion_leave_one_dataset_out.ipynb  # 4. Experimento leave-one-dataset-out
├── results/            # CSV con los que se construyen las tablas y figuras de la tesis
├── models/             # Políticas PPO entrenadas
├── requirements.txt
└── README.md
```

## Entorno

| Elemento | Valor |
|---|---|
| Plataforma | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Python | 3.13 |
| CUDA | 12.8 (`torch 2.11.0+cu128`) |
| Librerías | Ver `requirements.txt` |
| Medición de energía | CodeCarbon 3.3.1 |
| Dispositivo de PPO | CPU |

Instalación fuera de Colab:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Los notebooks verifican las versiones al iniciar y se detienen si no coinciden con las esperadas.

## Datos

Los datasets se descargan automáticamente desde Hugging Face y no se incluyen en el repositorio:

| Dataset | Identificador | Clases |
|---|---|---|
| IMDb | `stanfordnlp/imdb` | 2 |
| AG News | `fancyzhx/ag_news` | 4 |
| TREC | `CogComp/trec` | 6 |

Se usan 800 ejemplos de entrenamiento y 300 de validación por dataset, tomados por muestreo estratificado. El conjunto de prueba oficial se reserva para la evaluación final. Antes de evaluar se aplica una descontaminación exacta por texto con SHA-256.

## Reproducción

Ejecuta los notebooks en orden en Google Colab con GPU T4. Todos guardan sus resultados en Google Drive, bajo la ruta:

```python
DRIVE_ROOT = Path("/content/drive/MyDrive/idsf-eft/framework")
```

Si quieres usar otra carpeta, cambia `DRIVE_ROOT` en la celda de configuración de cada notebook.

| Paso | Notebook | Depende de | Genera |
|---|---|---|---|
| 1 | `pipeline_multidataset.ipynb` | — | `artifacts/` (particiones, características, descriptor 18D) |
| 2 | `entrenamiento_politica_compartida.ipynb` | 1 | `ppo_training/ppo_shared_final_v2/` (política e historial). |
| 3 | `evaluacion_multidataset.ipynb` | 1 y 2 | `final_evaluation_v2/` (evaluación principal y ablaciones) |
| 4 | `generalizacion_leave_one_dataset_out.ipynb` | 1 | `loo_training_v2/` (3 políticas LOO y su evaluación). |

**Reanudación y caché.** Los notebooks guardan checkpoints cada 12 episodios y reutilizan las ejecuciones ya completadas. Si una sesión de Colab se interrumpe, basta con volver a ejecutar el notebook y continúa donde quedó. Para recalcular todo desde cero, borra la carpeta de resultados correspondiente o cambia `EXPERIMENT_NAME`.

### Semillas y configuración

| Parámetro | Valor |
|---|---|
| Semilla global / PPO | 42 |
| Semillas de la recompensa | 11, 22, 33 |
| Semillas de la evaluación final | 101, 202, 303, 404, 505 |
| Episodios PPO | 600 (200 por dataset; 300 por dataset fuente en LOO) |
| PPO | MLP 18→64→64, lr 3e-4, n_steps 12, batch 12, 10 épocas, γ = 1, λ = 1, clip 0.2, entropía 0.01 |
| DistilBERT | `distilbert/distilbert-base-uncased`, 1 época, AdamW, lr 5e-5, batch 16, weight decay 0 |
| Longitud | Cota superior de 256 tokens, mínimo técnico de 8, padding dinámico por batch |
| Recompensa | τ = 0.95, β = 2, recorte a [-1, 1] |

## Correspondencia entre resultados y tablas de la tesis

| Tabla / Figura | Archivo(s) en `results/` |
|---|---|
| Tabla 4.1 | `decontamination_report.csv` |
| Tabla 4.2, Figura 4.1 | `dataset_descriptors_18d.csv` |
| Tabla 4.3, Figura 4.2 | `ppo_training_summary.csv`, `ppo_final_history.csv` |
| Tabla 4.4, Figura 4.9 | `agent_frozen_actions.csv` |
| Tablas 4.5 y 4.6, Figuras 4.3, 4.7 y 4.8 | `final_evaluation_history.csv`, `final_evaluation_with_derived_metrics.csv`, `final_summary_by_dataset_condition.csv` |
| Tablas 4.7 y 4.8 | `paired_agent_vs_random70.csv`, `paired_agent_vs_random70_tests_exploratory.csv` |
| Tabla 4.9, Figuras 4.4 y 4.5 | `ablation_controls_history.csv`, `ablation_controls_summary.csv`, `ablation_controls_paired_vs_agent.csv` |
| Tablas 4.10 y 4.11, Figura 4.6 | `loo_summary_by_fold_condition.csv`, `loo_agent_action_*.csv` (un archivo por partición) |

## Alcance

Los resultados corresponden al ajuste fino de DistilBERT sobre IMDb, AG News y TREC, a la escala experimental descrita (800/300 ejemplos, una época). Los porcentajes de ahorro se refieren al entrenamiento monitorizado del modelo objetivo. La extracción de características con GPT-2 y el entrenamiento de la política son un costo previo que se reporta por separado en la tesis (sección 4.6.7).

## Cita

```
D. A. Kher Sánchez, "Framework de selección inteligente de datos para el ajuste fino eficiente de
modelos de lenguaje en clasificación de texto mediante aprendizaje por refuerzo y estimación del
consumo energético", Trabajo de grado, Ingeniería Electrónica, Universidad Francisco de Paula
Santander, 2026.
```

## Licencia

El código se distribuye bajo licencia MIT. Los datasets conservan sus licencias originales.

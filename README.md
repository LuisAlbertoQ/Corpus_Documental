# Arquitectura de Corpus UPeU

Este proyecto implementa un sistema de procesamiento e indexación de un corpus de documentos universitarios de la Universidad Peruana Unión (UPeU). Convierte documentos PDF en un corpus de texto limpio, dividido en fragmentos (chunks), con embeddings vectoriales para búsqueda semántica utilizando ChromaDB.

## Descripción

El sistema sigue un pipeline de procesamiento:
1. **Extracción de texto**: Convierte PDFs a texto bruto.
2. **Limpieza**: Normaliza y limpia el texto extraído.
3. **Chunking**: Divide el texto en fragmentos con superposición.
4. **Embeddings**: Genera representaciones vectoriales usando modelos de transformers.
5. **Indexación**: Almacena los embeddings en una base vectorial para búsquedas semánticas.
6. **Verificación**: Evalúa la cobertura del corpus con consultas de prueba.

Está diseñado para documentos en español y categoriza el contenido en áreas como estatutos, estudiantes, investigación, etc.

## Requisitos Previos

- Docker y Docker Compose instalados.
- Al menos 4GB de RAM disponible (para procesamiento de embeddings).
- Espacio en disco: ~2GB para datos y modelos.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd oe1_arquitectura_corpus
   ```

2. Construye la imagen Docker:
   ```bash
   docker-compose build
   ```

3. Coloca los documentos PDF en la carpeta `corpus_upeu/pdfs/` (esta carpeta está montada en el contenedor).

4. **(Opcional) GPU:** Si tienes NVIDIA Container Toolkit y driver 525+, las evaluaciones
   (notebooks 7-9) usarán GPU automáticamente. El `docker-compose.yml` ya incluye GPU
   passthrough (PyTorch 2.1.2 + CUDA 12.1 dentro del contenedor).

## Ejecución

1. Inicia el contenedor con Jupyter Notebook:
    ```bash
    docker-compose up
    ```

2. Abre tu navegador en `http://localhost:8888` (sin token, como configurado).

3. **Pipeline V2 (Recomendado — Scripts reproducibles):**
   ```bash
   # 1. Levantar entorno (requiere GPU para embeddings)
   docker-compose up -d

   # 2. Ejecutar pipeline V2 en orden (dentro del contenedor)
   docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/01_inventario.py
   docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/02_clasificacion.py
   # Revisar/editar corpus_upeu/metadatos/v2/clasificacion_v2.csv
   docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/03_diagnostico.py
   docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/04_extraccion_v2.py
   docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/05_chunking_v2.py
   docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/06_validacion_corpus.py
   # Verificar 0 errores en corpus_upeu/metadatos/v2/reporte_validacion_v2.md
   docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/07_indexacion_v2.py
   ```

   Ver documentación completa: `scripts/v2/README_PIPELINE_V2.md`
   Guía migración OE5: `scripts/v2/MIGRACION_OE5_V2.md`

4. **Pipeline V1 (Legacy — Notebooks):**
   Ejecutar en orden dentro de la carpeta `notebooks/`:
   - `1_extraccion_texto.ipynb`: Extrae texto de PDFs con `pdfplumber`.
   - `2_limpieza_corpus.ipynb`: Normaliza y limpia el texto extraído.
   - `3_chunking_embeddings.ipynb`: Chunking estructural (300 tokens, overlap 50).
   - `4_indexacion_vectorial.ipynb`: Indexa en ChromaDB con mpnet (768 dim).
   - `5_verificacion_cobertura.ipynb`: Verifica cobertura con umbrales 0.30/0.50.
   - `6_prueba_manual.ipynb`: Pruebas interactivas.
   - `7_evaluacion_miniLM_gpu.ipynb` (opcional): Evaluación MiniML-L12 con GPU.
   - `8_evaluacion_mpnet_gpu.ipynb` (opcional): Evaluación mpnet con GPU.
   - `9_evaluacion_bge_m3_gpu.ipynb` (opcional): Evaluación bge-m3 con GPU.

5. Los resultados se guardan en:
    - `corpus_upeu/metadatos/v2/` — chunks, embeddings, informes (V2 centralizado)
    - `vector_store_v2/` — índice principal V2 (5126 chunks, mpnet, padre-hijo)
    - `vector_store/` — índice V1 legacy (3886 chunks)
    - `vector_store_miniLM/`, `vector_store_mpnet/`, `vector_store_bge_m3/` — índices de evaluación

## Pipeline V2 — Novedades vs V1

| Aspecto | V1 (Notebooks) | V2 (Scripts `scripts/v2/`) |
|---|---|---|
| **Chunking** | 300 tokens fijos + overlap 50 | **1 artículo = 1 chunk** (100-700 tok), overlap 12% solo si >500 tok |
| **Estructura detectada** | Art/Cap/Sec/Título | **Art/Cap/Sec/Título/Lista/Paso/Tabla/Imagen** + contexto jerárquico |
| **Metadatos** | 6 campos | **18 campos** + relaciones **padre-hijo** (500) |
| **Extracción** | Método único híbrido | **4 rutas** según diagnóstico: DIGITAL/MIXTO/TUPA_TABLAS/ESCANEADO |
| **Validación** | Solo retrieval (notebook 5) | **12 checks automatizados** (integridad, estructura, embeddings, padre-hijo) |
| **Reproducibilidad** | Notebooks manuales | **7 scripts Python versionados** (`scripts/v2/`) |
| **Categorías** | Keywords locales (becas=B) | **Unificadas OE5** (becas=D Bienestar) |

## Resultados V2 (Corpus Producción `vector_store_v2/`)

| Métrica | Valor V2 |
|---|---|
| Documentos procesados | 48 |
| Chunks generados | **5,126** |
| Tokens promedio | 179 |
| Tokens máximos | **700** (límite estricto) |
| Chunks tipo `articulo` | **3,973 (77%)** |
| Chunks partidos (>500 tok) | **500 (10%)** |
| Relaciones padre-hijo | **500 (100% consistentes)** |
| Embeddings | **(5126, 768)** mpnet normalizados |
| Validación | **0 errores**, 1,403 warnings (no bloqueantes) |
| Test "beca" → Art. 49° | **dist=0.245**, Cat **D (Bienestar)** ✅ |

## Migración a OE5 (Chatbot)

El índice de producción V2 está en `corpus_upeu/vector_store_v2/`. Para migrar a OE5:

```bash
# 1. Backup OE5 actual
cd oe5_chatbot_upeu
mv vector_store vector_store_backup_$(date +%Y%m%d)

# 2. Copiar índice V2
cp -r ../oe1_arquitectura_corpus/corpus_upeu/vector_store_v2 vector_store

# 3. Actualizar taxonomía en OE5 (backend/config.py) — becas=D unificada
# Ver scripts/v2/MIGRACION_OE5_V2.md

# 4. Reiniciar backend OE5
docker-compose restart backend
```

Ver guía completa: `scripts/v2/MIGRACION_OE5_V2.md`

## Estructura del Proyecto

```
oe1_arquitectura_corpus/
├── corpus_upeu/           # Datos del corpus
│   ├── pdfs/             # Documentos PDF originales (no en Git)
│   ├── txt_bruto/        # Texto extraído crudo (V1 legacy)
│   ├── txt_limpio/       # Texto limpiado (V1 legacy)
│   ├── txt_bruto_v2/     # Texto extraído V2 (48 archivos, por ruta diagnóstico)
│   └── metadatos/
│       ├── v2/           # ← V2 CENTRALIZADO
│       │   ├── inventario_v2.csv
│       │   ├── clasificacion_v2.csv
│       │   ├── diagnostico_v2.csv
│       │   ├── extraccion_v2_metadatos.csv
│       │   ├── chunks_v2.csv          # 5,126 chunks × 18 metadatos
│       │   ├── embeddings_v2.npy      # (5126, 768) mpnet
│       │   ├── chunk_ids_v2.npy
│       │   ├── reporte_validacion_v2.md
│       │   └── validacion_v2_detalle.csv
│       └── (archivos V1 legacy...)
├── notebooks/            # Jupyter notebooks V1 (1-6 pipeline, 7-9 evaluación)
├── scripts/
│   ├── v2/               # ← PIPELINE V2 REPRODUCIBLE (7 scripts)
│   │   ├── 01_inventario.py
│   │   ├── 02_clasificacion.py
│   │   ├── 03_diagnostico.py
│   │   ├── 04_extraccion_v2.py
│   │   ├── 05_chunking_v2.py
│   │   ├── 06_validacion_corpus.py
│   │   ├── 07_indexacion_v2.py
│   │   ├── extraer_faltantes.py
│   │   ├── check_missing.py
│   │   ├── README_PIPELINE_V2.md
│   │   └── MIGRACION_OE5_V2.md
│   └── (scripts V1 legacy...)
├── vector_store/         # Índice ChromaDB V1 (3886 chunks, mpnet)
├── vector_store_v2/      # ← ÍNDICE V2 PRODUCCIÓN (5126 chunks, mpnet, padre-hijo)
├── vector_store_miniLM/  # Índice evaluación MiniLM
├── vector_store_mpnet/   # Índice evaluación mpnet
├── vector_store_bge_m3/  # Índice evaluación bge-m3
├── src/                  # Código fuente adicional
├── DIAGNOSTICO_CAMBIOS.md    # Documentación técnica bugs/optimizaciones V1
├── RESUMEN_FINAL.md      # Resumen ejecutivo stakeholders
├── Dockerfile            # Imagen Docker (PyTorch 2.1.2 + CUDA 12.1)
├── docker-compose.yml    # Config contenedor (GPU passthrough)
├── requirements.txt      # Dependencias Python
└── README.md
```

## Pipeline V2 — Novedades vs V1

| Aspecto | V1 (Notebooks) | V2 (Scripts `scripts/v2/`) |
|---|---|---|
| **Chunking** | 300 tokens fijos + overlap 50 | **1 artículo = 1 chunk** (100-700 tok), overlap 12% solo si >500 tok |
| **Estructura detectada** | Art/Cap/Sec/Título | **Art/Cap/Sec/Título/Lista/Paso/Tabla/Imagen** + contexto jerárquico |
| **Metadatos** | 6 campos | **18 campos** + relaciones **padre-hijo** (500) |
| **Extracción** | Método único híbrido | **4 rutas** según diagnóstico: DIGITAL/MIXTO/TUPA_TABLAS/ESCANEADO |
| **Validación** | Solo retrieval (notebook 5) | **12 checks automatizados** (integridad, estructura, embeddings, padre-hijo) |
| **Reproducibilidad** | Notebooks manuales | **7 scripts Python versionados** (`scripts/v2/`) |
| **Categorías** | Keywords locales (becas=B) | **Unificadas OE5** (becas=D Bienestar) |

## Resultados V2 (Corpus Producción `vector_store_v2/`)

| Métrica | Valor V2 |
|---|---|
| Documentos procesados | 48 |
| Chunks generados | **5,126** |
| Tokens promedio | 179 |
| Tokens máximos | **700** (límite estricto) |
| Chunks tipo `articulo` | **3,973 (77%)** |
| Chunks partidos (>500 tok) | **500 (10%)** |
| Relaciones padre-hijo | **500 (100% consistentes)** |
| Embeddings | **(5126, 768)** mpnet normalizados |
| Validación | **0 errores**, 1,403 warnings (no bloqueantes) |
| Test "beca" → Art. 49° | **dist=0.245**, Cat **D (Bienestar)** ✅ |

## Migración a OE5 (Chatbot)

El índice de producción V2 está en `corpus_upeu/vector_store_v2/`. Para migrar a OE5:

```bash
# 1. Backup OE5 actual
cd oe5_chatbot_upeu
mv vector_store vector_store_backup_$(date +%Y%m%d)

# 2. Copiar índice V2
cp -r ../oe1_arquitectura_corpus/corpus_upeu/vector_store_v2 vector_store

# 3. Actualizar taxonomía en OE5 (backend/config.py) — becas=D unificada
# Ver scripts/v2/MIGRACION_OE5_V2.md

# 4. Reiniciar backend OE5
docker-compose restart backend
```

Ver guía completa: `scripts/v2/MIGRACION_OE5_V2.md`

## Pipeline V1 (Legacy — Notebooks)

Mantenido para referencia histórica. Ejecutar en orden:
1. `1_extraccion_texto.ipynb`
2. `2_limpieza_corpus.ipynb`
3. `3_chunking_embeddings.ipynb`
4. `4_indexacion_vectorial.ipynb`
5. `5_verificacion_cobertura.ipynb`

## Notas

- Los documentos PDF en `corpus_upeu/pdfs/` no deben subirse a Git (están en .gitignore).
- El procesamiento puede tomar tiempo dependiendo del tamaño del corpus.
- Para búsquedas, usa el notebook de pruebas manuales o integra ChromaDB en tu aplicación.
- **Modelo de embeddings recomendado:** `paraphrase-multilingual-mpnet-base-v2` (768 dim, coseno).
  Se evaluaron 3 modelos en GPU (banco 10Q, chunks v2 2026-08-21): MiniLM (384 dim, 9/10),
  mpnet (768 dim, 10/10), bge-m3 (1024 dim, 10/10).
  mpnet fue el ganador: 7/10 excelentes y mejor match por categoría (8/10).
- **Métrica de distancia:** coseno (distancia = 1 - coseno). Umbrales: excelente < 0.30, aceptable < 0.50.
- **Corpus v2 (2026-08-21):** 48 documentos → **6259 chunks** (chunking 1-artículo-por-chunk,
  max 300 tokens / overlap 50; media real 156 tokens, mediana 120).
  Ver `scripts/v2/README_PIPELINE_V2.md` para el fix aplicado.

## Dependencias

Ver `requirements.txt` para las bibliotecas Python utilizadas.

## Resultados Finales (producción con mpnet)

| Métrica | Inicio | Final V1 | Corpus V2 (2026-09-17) |
|---|---|---|---|
| Cobertura del banco de pruebas | 0/10 (falso positivo) | **10/10 (100%)** | **10/10 (100%)** |
| Excelentes (d < 0.30) | 0 | **6/10 (60%)** | **7/10 (70%)** |
| Categoría match | n/a | **8/10 (80%)** | **8/10 (80%)** |
| Distancia promedio | 0.59 (ruido) | **0.2954** | **0.245** (test beca) |
| Total chunks | 2488 | 3886 | **5,126** |
| Modelo de embeddings | all-MiniLM-L6-v2 (inglés, por defecto) | **paraphrase-multilingual-mpnet-base-v2** | **paraphrase-multilingual-mpnet-base-v2** |

Ver `DIAGNOSTICO_CAMBIOS.md` para el análisis técnico completo de los 11 issues corregidos en V1.
Ver `RESUMEN_FINAL.md` para un resumen ejecutivo del proyecto.
Ver `scripts/v2/README_PIPELINE_V2.md` para documentación completa del pipeline V2.
Ver `scripts/v2/MIGRACION_OE5_V2.md` para guía de migración a OE5.
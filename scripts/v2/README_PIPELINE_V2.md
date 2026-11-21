# Pipeline V2 — Construcción del Corpus UPeU

**Versión:** 2.0  
**Fecha:** Septiembre 2026  
**Estado:** ✅ Completado (10 etapas, 0 errores de validación)

---

## 🆕 Código V3 (noviembre 2026) — cómo ejecutarlo

Los mismos scripts generan ahora versiones nuevas sin tocar la V2 (que usa OE5).
Rutas y parámetros viven en `config.py`; la versión se elige con `CORPUS_VERSION`
(por defecto `v3`, ya definida en `docker-compose.yml`). La V2 está protegida.

```powershell
docker-compose build          # requirements.txt cambió (versiones fijas, rank-bm25, pytest)
docker-compose up -d
docker exec oe1_corpus python -m pytest /home/jupyteruser/work/scripts/v2/tests -q

# Pipeline completo V3 (PDF → Chroma)
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/01_inventario.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/02_clasificacion.py   # aplica corpus_upeu/config/clasificacion_manual.csv
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/03_diagnostico.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/04_extraccion_v2.py    # --faltantes / --solo "X.pdf"
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/05_chunking_v2.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/06_validacion_corpus.py  # exit 1 si hay errores
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/07_indexacion_v2.py      # --force para reemplazar
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/08_indice_bm25.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/09_evaluacion.py                 # banco: corpus_upeu/config/banco_preguntas.csv
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/09_evaluacion.py --version v2    # comparar con V2 (solo lectura)
```

Otro modelo dentro de la misma versión (archivos y colección con sufijo `_<modelo>`):
`docker exec -e MODEL_KEY=e5large oe1_corpus python .../05_chunking_v2.py` → 06 → 07 → `09_evaluacion.py`.
Ablación de contexto: `09_evaluacion.py --breadcrumb ninguno` (re-codifica sin re-chunkear).

> **Windows + Docker Desktop:** dentro del contenedor `corpus_upeu/` aparece como `root 755`,
> así que `jupyteruser` no puede crear carpetas nuevas ahí. Antes de una versión nueva,
> créalas desde Windows: `mkdir corpus_upeu\txt_bruto_v3, corpus_upeu\vector_store_v3`
> (las subcarpetas de `metadatos/` sí se crean solas).
> En Git Bash, anteponer `MSYS_NO_PATHCONV=1` a los `docker exec` para que no reescriba las rutas `/home/...`.

Experimento 2a sin re-extraer (re-chunkear los .txt de V2 con max_seq_length 384):
`docker exec -e CORPUS_VERSION=v3a oe1_corpus python .../05_chunking_v2.py --txt-dir /home/jupyteruser/work/corpus_upeu/txt_bruto_v2`
(requiere copiar `diagnostico_v2.csv` a `metadatos/v3a/diagnostico_v3a.csv`).
Plan 2b: añadir `-e CHUNK_MODE=pequeno` (chunks ≤120 tokens, max_seq_length 128).

| Cambio | Dónde | Problema que corrige |
|---|---|---|
| Rutas centralizadas, V2 protegida | `config.py`, todos | Scripts apuntaban a `scripts/` y a `metadatos/` (no reproducible) |
| Clasificación manual versionada | `corpus_upeu/config/clasificacion_manual.csv`, `02` | Paso manual VIGENTE no registrado |
| Tablas en su página, sin duplicar; marcadores `[[PAG N]]` | `extraccion_comun.py`, `04` | Tablas pegadas al final del documento; campo `pagina` vacío |
| `max_seq_length` explícito, 0 tokens truncados | `05`, `chunking.py` | P1: ~50% de tokens no llegaban al embedding |
| Preámbulo conservado | `chunking.py` | 5.3% del texto (47% del TUPA) se descartaba |
| Regex de encabezados con `\b` y sin cruzar líneas | `chunking.py` | Falsos "TÍTULO D", "Capítulo de…" |
| Breadcrumb de contexto en `texto_embedding` | `chunking.py` | Partes 2+ sin saber de qué artículo eran |
| Cortes por párrafo/oración/ítem/fila | `chunking.py` | Cortes a mitad de oración |
| `grupo_articulo_id` | `chunking.py`, `07` | Padre = chunk anterior; base para small-to-big en OE5 |
| Encabezados cortos se anteponen al bloque siguiente, sin duplicados | `chunking.py` | 256 chunks < 50 chars, 98 duplicados |
| `nombre_documento`, `seccion` (Subcapítulo/Apartado), `pagina` | `chunking.py` | Metadatos vacíos |
| Cobertura, trazabilidad y falsos encabezados validados | `06` | La validación no detectaba texto perdido |
| HNSW `search_ef=100`, sin borrado por defecto | `07` | `search_ef=10` < top_k de OE5; borraba la colección |
| Corpus léxico BM25 | `texto_lexico.py`, `08` | Base para búsqueda híbrida en OE5 |
| Versiones fijas + revisión del modelo + `MANIFEST.json` | `requirements.txt`, `config.py`, `05` | Resultados no replicables |
| M1 perfiles de modelo (`MODEL_KEY`) con prefijos | `config.py`, `05`, `07`, `09` | Comparar modelos de forma justa |
| M2 `BREADCRUMB_MODO` | `config.py`, `chunking.py` | Ablación del contexto en el embedding |
| M4 evaluación R@k / MRR (semántico, BM25, híbrido, re-ranker) | `09_evaluacion.py`, `config/banco_preguntas.csv` | Distancias top-1 no miden aciertos |
| M5 pie de página postal fuera, colas < 15 tokens unidas | `extraccion_comun.py`, `chunking.py` | Ruido residual |
| M6 índices (tablas de contenido) descartados y registrados | `chunking.py`, `05`, `06` | 191 chunks de índice competían con artículos reales |

---

## 📋 Resumen Ejecutivo

Este pipeline transforma **48 documentos PDF oficiales** de la Universidad Peruana Unión (UPeU) en un **corpus vectorial semántico** listo para RAG, mediante **chunking por estructura documental** (1 artículo = 1 chunk) con embeddings **mpnet 768-dim** y metadatos enriquecidos (18 campos + relaciones padre-hijo).

| Métrica | Valor V2 |
|---|---|
| Documentos procesados | 48 |
| Chunks generados | 5,126 |
| Tokens promedio | 179 |
| Tokens máximos | 700 (límite estricto) |
| Chunks tipo `articulo` | 3,973 (77%) |
| Chunks partidos (>500 tok) | 500 (10%) |
| Relaciones padre-hijo | 500 (100% consistentes) |
| Embeddings | (5126, 768) mpnet normalizados |
| Validación | **0 errores**, 1,403 warnings (no bloqueantes) |

---

## 🏗️ Arquitectura del Pipeline (10 Etapas)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARTE 1 — CONSTRUCCIÓN DEL CORPUS                        │
└─────────────────────────────────────────────────────────────────────────────┘

DOCUMENTOS OFICIALES (48 PDFs)
         │
         ▼
┌──────────────────┐    inventario_v2.csv (hash, páginas, año, versión)
│ 1. INVENTARIO    │    01_inventario.py
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    clasificacion_v2.csv (tipo_documental + cat A-E)
│ 2. CLASIFICACIÓN │    02_clasificacion.py
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    diagnostico_v2.csv (ruta: DIGITAL/MIXTO/TUPA_TABLAS/ESCANEADO)
│ 3. DIAGNÓSTICO   │    03_diagnostico.py
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
DIGITAL   ESCANEADO    (también MIXTO, TUPA_TABLAS)
    │         │
    └────┬────┘
         ▼
┌──────────────────┐    txt_bruto_v2/ (48 archivos .txt)
│ 4. EXTRACCIÓN +  │    04_extraccion_v2.py (por ruta según diagnóstico)
│    LIMPIEZA      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    Detecta: Art/Cap/Sec/Título/Lista/Paso/Tabla/Imagen
│ 5. ESTRUCTURACIÓN│    + contexto jerárquico (título→cap→sec→art)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    1 Art = 1 chunk (100-700 tok)
│ 6. CHUNKING      │    overlap 12% solo si >500 tok
│    SEMÁNTICO     │    padre-hijo para artículos partidos
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    18 campos: documento, chunk_id, tipo_documental,
│ 7. METADATOS     │    categoria_tematica (A-E), ruta_extraccion,
│                  │    titulo_documento, capitulo, seccion, articulo,
│                  │    tipo_bloque, padre_chunk_id, es_parte, parte_num,
│                  │    total_partes, num_tokens, num_chars
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    0 errores, 1,403 warnings (solo ruido OCR)
│ 8. VALIDACIÓN    │    reporte_validacion_v2.md + validacion_v2_detalle.csv
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    mpnet 768-dim, normalizados, cosine
│ 9. EMBEDDINGS    │    embeddings_v2.npy (5126, 768)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    ChromaDB persistente, colección corpus_upeu_v2
│ 10. BASE         │    vector + texto + 18 metadatos + padre-hijo
│    VECTORIAL     │    vector_store_v2/
└──────────────────┘
```

---

## 📁 Estructura de Archivos Generados

```
corpus_upeu/
├── metadatos/
│   ├── v2/                          ← ✅ V2 centralizado
│   │   ├── inventario_v2.csv        # 48 docs: hash, páginas, año, versión, URL, vigencia
│   │   ├── clasificacion_v2.csv     # tipo_documental + categoria_tematica (A-E) + estado
│   │   ├── diagnostico_v2.csv       # ruta_extraccion + stats por doc
│   │   ├── extraccion_v2_metadatos.csv
│   │   ├── chunks_v2.csv            # 5,126 chunks × 18 metadatos
│   │   ├── embeddings_v2.npy        # (5126, 768) float32 normalizados
│   │   ├── chunk_ids_v2.npy         # IDs correspondientes
│   │   ├── reporte_validacion_v2.md # Resumen validación (0 errores)
│   │   └── validacion_v2_detalle.csv
│   └── (archivos V1 legacy...)
├── txt_bruto_v2/                    # 48 archivos .txt extraídos
│   └── *.pdf.txt
├── vector_store_v2/                 # ChromaDB persistente
│   ├── chroma.sqlite3
│   └── <uuid>/ (HNSW index files)
└── pdfs/                            # 48 PDFs originales (no en Git)
```

---

## 🛠️ Scripts del Pipeline (scripts/v2/)

| Script | Etapa | Descripción | Input | Output |
|---|---|---|---|---|
| `01_inventario.py` | 1 | Inventario PDFs (hash SHA256, páginas, año, versión) | `pdfs/` | `metadatos/v2/inventario_v2.csv` |
| `02_clasificacion.py` | 2 | Clasificación automática + revisión manual | `inventario_v2.csv` | `metadatos/v2/clasificacion_v2.csv` |
| `03_diagnostico.py` | 3 | Diagnóstico por doc (digital/OCR/tablas/columnas) | PDFs + clasificacion | `metadatos/v2/diagnostico_v2.csv` |
| `04_extraccion_v2.py` | 4 | Extracción por ruta (DIGITAL/MIXTO/TUPA_TABLAS/ESCANEADO) + limpieza integrada | PDFs + diagnostico | `txt_bruto_v2/` + `extraccion_v2_metadatos.csv` |
| `05_chunking_v2.py` | 5-7 | Estructuración + Chunking semántico + Metadatos (18 campos) | `txt_bruto_v2/` | `chunks_v2.csv`, `embeddings_v2.npy`, `chunk_ids_v2.npy` |
| `06_validacion_corpus.py` | 8 | Validación completa (integridad, metadatos, estructura, embeddings) | chunks + embeddings | `reporte_validacion_v2.md`, `validacion_v2_detalle.csv` |
| `07_indexacion_v2.py` | 9-10 | Indexación ChromaDB (vector + texto + metadatos + padre-hijo) | chunks + embeddings | `vector_store_v2/` |

**Scripts auxiliares:**
- `extraer_faltantes.py` — Extracción manual de docs pendientes
- `check_missing.py` — Verificación de completitud

---

## ⚙️ Configuración Clave

```python
# Modelos
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # 768 dim
DISTANCE_METRIC = "cosine"

# Chunking
MAX_TOKENS_ARTICULO_CORTO = 400
MAX_TOKENS_ARTICULO_LARGO = 700
OVERLAP_PCT = 0.12          # 12% solo si artículo > 500 tokens
MAX_TOKENS_OTROS = 500

# Categorías temáticas unificadas (OE5)
A = "Gobierno y estatuto institucional"
B = "Académico y estudios"
C = "Investigación"
D = "Bienestar estudiantil"     # ← BECAS aquí (unificado)
E = "Laboral, docencia y políticas"

# Umbrales validación
MAX_TOKENS_HARD = 700
EMBEDDING_DIM = 768
```

---

## 🚀 Ejecución Completa (en contenedor Docker)

```bash
# 1. Levantar entorno (requiere GPU para embeddings)
docker-compose up -d

# 2. Ejecutar pipeline en orden (dentro del contenedor)
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/01_inventario.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/02_clasificacion.py
# Revisar/editar metadatos/v2/clasificacion_v2.csv
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/03_diagnostico.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/04_extraccion_v2.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/05_chunking_v2.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/06_validacion_corpus.py
# Verificar 0 errores en reporte_validacion_v2.md
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/07_indexacion_v2.py
```

---

## 📊 Resultados de Validación (Etapa 8)

| Check | Resultado |
|---|---|
| Integridad CSV/NPY/IDs | ✅ |
| Campos obligatorios (18) | ✅ |
| Chunks vacíos/cortos | ✅ 0 |
| Tokens ≤ 700 | ✅ (max=700 exacto) |
| Texto basura (headers/footers) | ⚠️ 1,403 warnings (ruido OCR residual, no bloqueante) |
| Tablas preservadas (`[TABLA pX.Y]`) | ✅ 147 chunks |
| Estructura artículos (header ARTICULO N) | ✅ 3,973 chunks |
| Relaciones padre-hijo | ✅ 500/500 consistentes |
| Categorías A-E válidas | ✅ |
| Embeddings normalizados (norma ≈ 1.0) | ✅ |
| Duplicados chunk_id | ✅ 0 |

**Conclusión:** Corpus **listo para producción**.

---

## 🔄 Migración a OE5 (Chatbot)

### Opción A: Copia directa (recomendada)
```bash
# Backup OE5 actual
mv oe5_chatbot_upeu/vector_store oe5_chatbot_upeu/vector_store_backup

# Copiar V2
cp -r oe1_arquitectura_corpus/corpus_upeu/vector_store_v2 oe5_chatbot_upeu/vector_store
```

### Opción B: Desde contenedores
```bash
docker cp oe1_corpus:/home/jupyteruser/work/corpus_upeu/vector_store_v2 \
  oe5_chatbot_upeu_backend:/data/vector_store
```

### Verificación en OE5
```bash
# Reiniciar backend OE5
# Probar: "¿Cómo solicito una beca en la UPeU?"
# Debe retornar: REGLAMENTO BECAS 2021 ACTUALIZADO, Artículo 49°, dist ≈ 0.24, Cat D
```

---

## 📝 Cambios Clave V1 → V2

| Aspecto | V1 (Notebooks) | V2 (Scripts) |
|---|---|---|
| Chunking | 300 tokens fijos + overlap 50 | **1 Art = 1 chunk (100-700 tok)**, overlap 12% si >500 |
| Estructura | Solo Art/Cap/Sec/Título | **Art/Cap/Sec/Título/Lista/Paso/Tabla/Imagen** |
| Metadatos | 6 campos | **18 campos + padre-hijo** |
| Extracción | Un solo método | **4 rutas según diagnóstico** |
| Validación | Solo retrieval (notebook 5) | **12 checks automatizados** |
| Reproducibilidad | Notebooks manuales | **7 scripts Python versionados** |
| Categorías | Keywords locales (B becas) | **Unificadas OE5 (D becas)** |

---

## 🐛 Issues Conocidos / Próximos Pasos

1. **Warnings OCR residual** (1,403) — ruido de sellos/firmas en páginas finales; no afecta retrieval
2. **Páginas no mapeadas** — campo `pagina` vacío; requeriría mapeo PDF→chunk
3. **Re-ranking** — evaluar cross-encoder para mejorar top-k
4. **Hybrid search** — BM25 + embeddings para consultas con keywords exactas

---

## 👥 Créditos

Pipeline desarrollado como parte de la investigación **OE1 — Arquitectura de Corpus UPeU**, para alimentar el chatbot **OE5 — Chatbot UPeU**.

**Modelo embeddings:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`  
**Vector DB:** ChromaDB 0.4.22 (persistente, HNSW, cosine)  
**Entorno:** Docker + PyTorch 2.1.2 + CUDA 12.1
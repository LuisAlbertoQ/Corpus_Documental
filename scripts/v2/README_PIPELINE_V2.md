# Pipeline V2/V3 — Construcción del Corpus UPeU

**Versión del código:** V3 (los scripts conservan el nombre `*_v2.py`)  
**Última actualización:** 2026-11-21  
**Estado:** ✅ V3 + bge-m3 generada e indexada (`corpus_upeu/vector_store_v3_bgem3/`, 5,117 chunks, 0 errores).
La V2 (septiembre 2026, mpnet, 5,126 chunks) sigue siendo la que usa OE5 hasta la migración.

> Las secciones desde "Resumen Ejecutivo" hacia abajo describen la **V2 de septiembre 2026** y se conservan
> como registro histórico. Lo vigente está en esta primera parte.

---

## 🏁 Resultado V3 (2026-11-21)

| Fecha | Paso | Resultado |
|---|---|---|
| 2026-11-20 | Auditoría de V2 con los archivos en disco | mpnet trunca a 128 tokens: **50.6% de los tokens no llegaba al embedding**; validación con ~1,400 falsos positivos; `pagina` y `seccion` vacíos |
| 2026-11-20 | Código V3 (`config.py`, `chunking.py`, `extraccion_comun.py`, tests) + primera corrida | 5,260 chunks, 100% tokens embebidos, 100% cobertura, 0 errores / 11 warnings |
| 2026-11-20 | Experimento 2a (mpnet con max_seq 384) | No mejora el retrieval semántico (mpnet se entrenó a 128) |
| 2026-11-21 | M1–M6 (perfiles de modelo, breadcrumb, evaluación, limpieza de pies e índices) | 5,053 chunks; 84 índices descartados; re-ranker MRR 0.82 en banco de 14 preguntas |
| 2026-11-21 | Banco ampliado | 154 positivas (2–5 por documento, 127 con artículo esperado) + 8 negativas |
| 2026-11-21 | Imagen Docker con transformers 4.51.3 | mpnet idéntico (coseno mín 0.9999998, top-10 igual en 162/162) |
| 2026-11-21 | Comparación de 5 modelos | bge-m3 MRR semántico 0.724 vs mpnet 0.358; los 4 modelos nuevos sin diferencia significativa entre sí |
| 2026-11-21 | Corrección NBSP (`\xa0` tras "Artículo") | +62 artículos (Jefes de prácticas 0 → 24, Auditoría interna 4 → 42); bge-m3 MRR **0.756**, re-ranker 0.776 |
| 2026-11-21 | Indexación del ganador | `vector_store_v3_bgem3/` (81 MB), HNSW vs búsqueda exacta: top-1 igual en 161/162 |

### Métricas de bge-m3 (banco 154+8, nivel artículo)

| Modo | R@1 | R@3 | R@5 | R@10 | MRR@10 | Rdoc@1 | Rdoc@5 |
|---|---|---|---|---|---|---|---|
| Semántico | 0.636 | 0.851 | 0.916 | 0.948 | **0.756** | 0.786 | 0.955 |
| BM25 | 0.409 | 0.565 | 0.656 | 0.766 | 0.510 | 0.688 | 0.903 |
| Híbrido (RRF k=60) | 0.552 | 0.799 | 0.838 | 0.890 | 0.681 | 0.779 | 0.974 |
| Re-ranker (bge-reranker-v2-m3) | 0.675 | 0.851 | 0.896 | 0.968 | **0.776** | 0.779 | 0.974 |

Coseno top-1: positivas 0.690 vs negativas 0.484 (base para calibrar el umbral "fuera de dominio" en OE5).

### Corpus V3 + bge-m3 (`MANIFEST_bgem3.json`)

| Métrica | Valor |
|---|---|
| Chunks / documentos | 5,117 / 48 |
| Tokens de texto (media / mediana / máx) | 169.2 / 131 / 350 |
| Tokens de embedding máx (límite 382) | 382 → 0 chunks truncados |
| Artículos partidos | 1,791 |
| tipo_bloque | articulo 4,632 · texto_libre 211 · capitulo 120 · preambulo 67 · seccion 46 · titulo 41 |
| Índices descartados | 84 (`chunks_descartados_v3_bgem3.csv`) |
| Validación | 0 errores · 5 warnings (chunks cortos legítimos) |
| Modelo | `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`, 1024 dim, sin prefijos |
| Parámetros | articulo · max_seq 384 · chunk 350/40 · breadcrumb completo · HNSW coseno M16 / ef 200 / search_ef 100 |

> Nota: `scripts/README.md` (agosto) advertía que chromadb 0.4.22 corrompía índices de 1024 dim en lotes de 500.
> En V3 se indexó bge-m3 con lotes de 500 sin problema: `count()` = 5,117 y el top-1 coincide con la búsqueda exacta en 161/162.

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
Para comparar modelos no hace falta 07: `09_evaluacion.py` busca exacto sobre el `.npy`.
Qwen3: añadir `-e EMBED_BATCH_SIZE=4` (con 16 usa 5.9 GB de GPU).

Modelo ganador en una carpeta limpia para desplegar (la que se copia a OE5):

```powershell
docker exec -e MODEL_KEY=bgem3 -e VSTORE_DIR=vector_store_v3_bgem3 oe1_corpus python /home/jupyteruser/work/scripts/v2/07_indexacion_v2.py --force
```
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
| transformers 4.51.3 / tokenizers 0.21.1 | `requirements.txt` | Qwen3-Embedding exige transformers ≥ 4.51 |
| Revisión de bge-m3 fijada | `config.py` | Había dos snapshots en caché; se fijó la que reproduce los embeddings |
| `normalizar_espacios()` (NBSP y espacios Unicode → espacio, 1 a 1) | `chunking.py`, `extraccion_comun.py`, test nuevo | 62 artículos sin detectar en 2 reglamentos |
| `VSTORE_DIR` opcional | `config.py`, `07` | Indexar solo el ganador en una carpeta limpia para OE5 |
| Descripción de colección con el modelo real | `07` | Decía "mpnet" fijo |

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

Estado en V3 (2026-11-21) de los puntos que quedaron abiertos en V2:

1. ~~Warnings OCR residual (1,403)~~ — eran casi todos falsos positivos de la validación; en V3: 5 warnings.
2. ~~Páginas no mapeadas~~ — resuelto con marcadores `[[PAG N]]`: `pagina` lleno al 100%.
3. **Re-ranking** — evaluado: `bge-reranker-v2-m3` sube bge-m3 de 0.756 a 0.776 MRR. Falta integrarlo en OE5.
4. **Hybrid search** — evaluado: RRF con BM25 empeora a bge-m3 (0.681). BM25 sigue disponible (`08`) para casos de cifras exactas.

Pendientes: revisión humana del banco (`revisado = no`); soporte multi-documento (`alt:`) y normalización NFC en `09`;
migración de OE5.

---

## 👥 Créditos

Pipeline desarrollado como parte de la investigación **OE1 — Arquitectura de Corpus UPeU**, para alimentar el chatbot **OE5 — Chatbot UPeU**.

**Modelo embeddings:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`  
**Vector DB:** ChromaDB 0.4.22 (persistente, HNSW, cosine)  
**Entorno:** Docker + PyTorch 2.1.2 + CUDA 12.1
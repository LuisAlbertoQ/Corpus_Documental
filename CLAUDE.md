# CLAUDE.md — Estado actual del proyecto `oe1_arquitectura_corpus`

> Auditoría realizada el **2026-11-20** leyendo los archivos reales del repo (no solo la documentación).
> Último commit: `d591bc2` (`V2 Pipeline: semantic chunking, unified taxonomy, OE5 migration ready`).
> **Actualización 2026-11-21:** V3 regenerada con las mejoras M1–M6 (ver §9): 5,053 chunks. Con re-ranker, V3 supera claramente a V2 en el banco provisional (MRR 0.82 vs 0.57), pero el banco **aún no está revisado**. **V2 sigue siendo la canónica** hasta revisar el banco y probar en OE5.
> Los números de §2 siguen siendo los de la V2 en disco.

---

## 1. Qué es este proyecto

Convierte **48 PDFs oficiales de la Universidad Peruana Unión (UPeU)** en un corpus vectorial para **RAG**, consumido por el chatbot **OE5** (repo hermano `../oe5_chatbot_upeu`).

Pipeline: `PDF → texto bruto → diagnóstico/clasificación → chunking semántico → embeddings mpnet → ChromaDB → OE5`.

- Idioma: español.
- Categorías temáticas: **A** gobierno, **B** académico, **C** investigación, **D** bienestar, **E** políticas.
- Modelo de embeddings: **`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`** (768 dim, distancia coseno).
- OCR: **Tesseract** vía `pytesseract`, con `lang="spa"` y 250 DPI. No hay OCR propio ni modelos de deep learning.

---

## 2. Estado real del corpus V2 (medido en disco)

| Métrica | Valor |
|---|---|
| PDFs / archivos en `txt_bruto_v2/` | **48 / 48** |
| Chunks en `corpus_upeu/metadatos/v2/chunks_v2.csv` | **5,126** (18 columnas) |
| Tokens totales | **917,929** |
| Tokens por chunk | media **179.1**, mediana **99**, mín **5**, máx **700** |
| Caracteres totales | **3,967,833** |
| Chars por chunk | media **774.1**, mediana **447**, mín **21**, máx **4,496** |
| `embeddings_v2.npy` | **(5,126 × 768)** `float32`, norma 1.0, sin NaN/Inf |
| `chunk_ids_v2.npy` | 5,126 IDs, alineados al 100% con el CSV |
| Índice Chroma `corpus_upeu_v2` | **5,126 embeddings** en `corpus_upeu/vector_store_v2/` |
| Copia desplegada en OE5 | **5,126 embeddings**, colección `corpus_upeu_v2` ✅ |

### Distribución

| `tipo_bloque` | N | `ruta_extraccion` | N | Categoría | N |
|---|---|---|---|---|---|
| `articulo` | 3,973 (77.5%) | DIGITAL | 4,000 | A | 1,497 |
| `capitulo` | 739 | MIXTO | 1,017 | B | 1,778 |
| `titulo` | 366 | TUPA_TABLAS | 108 | C | 536 |
| `texto_libre` | 37 | ESCANEADO | 1 | D | 611 |
| `paso` | 11 | | | E | 704 |

### Validación y retrieval (reportes existentes)

- `reporte_validacion_v2.md`: **0 errores**, **1,403 warnings** (padre-hijo 685, tokens 432, contenido 257, basura 23, metadatos 5, estructura 1).
- Banco 10Q: **10/10 cubiertas**, **7/10 excelentes** (d<0.30), **8/10 category match**.
- OE5 end-to-end (17Q): **16/17 M02**, tiempo medio 8.32 s.
- Caso "beca" → Art. 49°, d≈0.245, categoría D.

---

## 3. Estructura y scripts

### Pipeline V2 (activo, en `scripts/v2/`)

| # | Script | Salida |
|---|---|---|
| 1 | `01_inventario.py` | `metadatos/v2/inventario_v2.csv` |
| 2 | `02_clasificacion.py` | `metadatos/v2/clasificacion_v2.csv` |
| 3 | `03_diagnostico.py` | `metadatos/v2/diagnostico_v2.csv` |
| 4 | `04_extraccion_v2.py` | `txt_bruto_v2/` + `extraccion_v2_metadatos.csv` |
| 5 | `05_chunking_v2.py` | `chunks_v2.csv`, `embeddings_v2.npy`, `chunk_ids_v2.npy` |
| 6 | `06_validacion_corpus.py` | `reporte_validacion_v2.md`, `validacion_v2_detalle.csv` |
| 7 | `07_indexacion_v2.py` | `corpus_upeu/vector_store_v2/` |

Auxiliares: `extraer_faltantes.py`, `check_missing.py`.
Documentación: `scripts/v2/README_PIPELINE_V2.md`, `scripts/v2/MIGRACION_OE5_V2.md`.

### Legacy (no activo, pero presente)

- `notebooks/1..9_*.ipynb`: pipeline V1 original.
- `scripts/rechunk_fixed.py`, `reembed_index.py`, `evaluar_modelos.py`, `rebuild_*.py`: generación de agosto 2026 (**6,259 chunks**).
- `vector_store/`, `vector_store_mpnet/`, `vector_store_miniLM/`, `vector_store_bge_m3/`: **6,259 embeddings cada uno**.
- `corpus_upeu/metadatos/chunks*.csv` y `embeddings*.npy` sin sufijo `v2`: artefactos V1.

> ⚠️ El README dice que `vector_store/` tiene 3,886 chunks; en disco tiene **6,259**. Dato desactualizado.

---

## 4. Cómo se ejecutó / cómo reproducir

Artefactos V2 generados el **17/09/2026**. Procedimiento según `README.md`:

```powershell
# 1. Entorno
docker-compose build
docker-compose up -d
# contexto: contenedor oe1_corpus, workdir /home/jupyteruser/work

# 2. Pipeline V2, en orden
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/01_inventario.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/02_clasificacion.py
# revisar/editar corpus_upeu/metadatos/v2/clasificacion_v2.csv
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/03_diagnostico.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/04_extraccion_v2.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/05_chunking_v2.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/06_validacion_corpus.py
# verificar 0 errores en reporte_validacion_v2.md
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/07_indexacion_v2.py
```

Requisitos: Docker + Compose, GPU NVIDIA (CUDA 12.1 en el Dockerfile), modelos de HuggingFace descargables.

### Migración a OE5 (ya aplicada)

```bash
cd ../oe5_chatbot_upeu
mv vector_store vector_store_backup_YYYYMMDD
cp -r ../oe1_arquitectura_corpus/corpus_upeu/vector_store_v2 vector_store
# backend/config.py: becas → categoría D (taxonomía unificada)
docker-compose restart backend
```

OE5 usa `paraphrase-multilingual-mpnet-base-v2` en `backend/rag_pipeline.py:138` y `EMBEDDING_MODEL` en `backend/config.py:199`. **No cambiar el modelo sin re-indexar.**

---

## 5. Problemas conocidos (pendientes de mejora)

### P1 — 🔴 Truncamiento real a 128 tokens (lo más importante)

- `sentence_bert_config.json` del modelo indica **`max_seq_length: 128`**.
- `05_chunking_v2.py:299` usa `model.encode()` sin modificarlo.
- `06_validacion_corpus.py:88-90` asume 512 → no detecta el problema.
- Medido: **2,000 chunks > 128 tokens** y **~50.6% de los tokens (464,632) no llegan al embedding**.
- Ejemplo: `Politica-ambiental` (único doc ESCANEADO) es 1 chunk de 488 tokens.

### P2 — 🟡 Validación con falsos positivos

- `06_validacion_corpus.py:134`: regex `ART[IÍ]CULO` sensible a mayúsculas → 3,931 falsos "sin header" (3,688 sí lo tienen).
- `06_validacion_corpus.py:153`: 185 "padre con artículo distinto" y ~499 "partes no secuenciales" por `padre_chunk_id = chunks[-1]` en vez de un padre explícito.
- Umbral de 512 hardcodeado en vez de leerse desde `model.max_seq_length`.

### P3 — 🟡 Metadatos y ruido

- `seccion`: **100% nulos** (regex no reconoce `Subcapítulo`/`Apartado`).
- `titulo_documento`: **1,626 nulos (32%)**; hace falta fallback al nombre del archivo.
- **256 chunks < 50 chars** y **98 textos duplicados** (79 grupos, headers tipo "TÍTULO I...").
- **5 chunks** con marcadores `--- Pág N (OCR) ---`; 23 warnings de basura.
- Campo `pagina` siempre vacío (`TODO` en `05_chunking_v2.py:217`) aunque los marcadores OCR contienen el número.

### P4 — 🟡 Dos generaciones conviviendo

- V2 = 5,126 (canónica, la que usa OE5) vs generación agosto = 6,259.
- README desactualizado. Conviene un `MANIFEST.json` con versión, hash, modelo y conteos.

### P5 — 🟡 Evaluación pequeña

- Solo 10 preguntas para decidir el modelo y 17 end-to-end. Insuficiente para afirmar "100% de cobertura".

---

## 6. Plan de mejora propuesto (aún no ejecutado)

| Fase | Acción | ¿Toca corpus? | Resultado esperado |
|---|---|---|---|
| 1 | Corregir validación, README y MANIFEST | No | Warnings reales, sin falsos positivos |
| 2a | Experimento: subir `max_seq_length` a 384-512 en copia de V2 y re-evaluar | Sí, temporal | Sin truncamiento manteniendo 5,126 chunks; descartar si cae la calidad |
| 2b | Si 2a falla: chunks ≤120 tokens con padre-hijo real (~9-11k chunks) | Sí, genera V3 | Todo el texto embebido; OE5 debe recuperar el padre |
| 3 | Fallback de títulos, secciones, duplicados, limpieza OCR, campo `pagina` | Sí, va en V3 | Menos ruido y metadatos completos |
| 4 | Banco de 30-50 preguntas estratificado + negativas; comparar V2 vs V3 | No | Evidencia real de mejora |
| 5 | Migrar OE5 a V3 con rollback | Deploy | Producción actualizada |

**Regla de decisión:** hacer primero la Fase 1 (gratis), luego el experimento 2a. No mezclar fases ni regenerar embeddings sin antes guardar V2.

---

## 7. Reglas para el asistente

1. **V2 (5,126 chunks) es la versión canónica**; OE5 ya la consume. No reemplazarla sin crear una versión nueva y comparar.
2. No borrar `vector_store*` ni `corpus_upeu/metadatos/v2/*` sin confirmación explícita; están en git pero son la evidencia del trabajo.
3. Antes de afirmar una mejora, medir: `% de tokens embebidos`, 0 errores de validación, banco de preguntas y end-to-end.
4. Cualquier cambio en el modelo de embeddings exige re-generar embeddings, re-indexar Chroma **y** ajustar OE5.
5. Los números de esta auditoría son de los archivos en disco; si se regenera el pipeline, actualizar este `CLAUDE.md`.

---

## 8. Cambios de código V3 y primera ejecución (2026-11-20)

- **Rutas y parámetros** en `scripts/v2/config.py`. Versión por `CORPUS_VERSION` (default `v3`; `docker-compose.yml` la define). `asegurar_version_escribible()` bloquea escribir en `v2`.
- **Nuevos módulos:** `chunking.py` (lógica pura, testeable), `extraccion_comun.py`, `texto_lexico.py`, `08_indice_bm25.py`, `tests/` (31 tests, pasan).
- **Eliminados:** `check_missing.py`, `extraer_faltantes.py` → reemplazados por `04_extraccion_v2.py --faltantes / --solo`.
- **Clasificación manual** versionada en `corpus_upeu/config/clasificacion_manual.csv` (copia de la `clasificacion_v2.csv` revisada).
- **Esquema de chunks V3:** añade `nombre_documento`, `texto_embedding`, `num_tokens_embedding`, `titulo_estructura`, `pagina_fin`, `grupo_articulo_id`, `char_inicio`, `char_fin`. Mantiene los campos de V2 (`titulo_documento` = alias de `titulo_estructura`).
- **`.Trash-1000`** fuera del índice de git (`git rm --cached`, los archivos siguen en disco). `requirements.txt` con versiones fijas → requiere `docker-compose build`.

### Prueba en seco sobre `txt_bruto_v2` (tokenizer real, sin embeddings)

| Modo | Chunks | Tokens texto media / mediana / máx | Embeddings que exceden max_seq | Cobertura mínima por documento |
|---|---|---|---|---|
| `articulo` (2a, max_seq 384) | 5,515 | 175.9 / 138 / 350 | 0 | 100% |
| `pequeno` (2b, max_seq 128) | 12,166 | 81.1 / 87 / 120 | 0 | 99.33% |

`06_validacion_corpus.py` corrido sobre el modo `articulo` (embeddings aleatorios, solo para probar el script): **0 errores, 14 warnings** (11 URL de pie, 19 folios sueltos heredados de los .txt de V2, 1 chunk < 50 chars), 100% de tokens embebidos. V2 tenía 1,403 warnings, la mayoría falsos positivos.
BM25 (`08`) de prueba: "costo pre-ingreso S/. 120.00" → TUPA; "artículo 53 nota mínima admisión" → Admisión Art. 53°.

### Ejecución real en Docker (CORPUS_VERSION=v3, modo articulo, max_seq 384)

Docker: el build fallaba (torch desde el índice de PyTorch sin PyPI) → arreglado en `Dockerfile` con `--extra-index-url`. Versiones fijadas a las de la imagen que generó V2 (sentence-transformers 3.1.1, hub 0.36.2, transformers 4.39.3, torch 2.1.2+cu121). Modelo en `.hf_cache/`. En Windows, `txt_bruto_vN/` y `vector_store_vN/` deben crearse desde el host (permiso 755 en `corpus_upeu/` dentro del contenedor).

| Métrica | V2 | V3 |
|---|---|---|
| Chunks | 5,126 | 5,260 |
| Tokens embebidos | 49.4% | **100%** |
| Cobertura de texto | ~94.7% (TUPA ~53%) | **100%** |
| `pagina` lleno | 0% | **100%** |
| Validación (errores / warnings) | 0 / 1,403 | 0 / 11 |
| TUPA | texto revuelto 320k chars | filas limpias 164k chars (`1.3 \| Pre-Ingreso … \| S/. 120.00`) |

**Retrieval semántico, banco 10Q (beca = D):**

| | Cubiertas | Excelentes (<0.30) | Cat match | Dist media |
|---|---|---|---|---|
| V2 | 10/10 | **7/10** | 6/10 | **0.274** |
| V3 (2a) | 10/10 | 5/10 | 6/10 | 0.285 |

- V3 mejora "nota mínima admisión" (Admisión rank 2 → 1) y baja distancias en preguntas de costos, pero **el TUPA no aparece en top-5 semántico** para "cuánto cuesta pre-ingreso" ni en V2 ni en V3; **BM25 sí lo encuentra en el puesto 1**.
- "¿Cómo solicito una beca?" empeora (top-1 pasa de REGLAMENTO BECAS a REGLAMENTO DE ESTUDIOS).
- Conclusión: el experimento 2a (subir max_seq a 384) **no mejora** el retrieval semántico; coherente con que mpnet se entrenó a 128. Siguiente paso según la regla de decisión: **2b** (`CHUNK_MODE=pequeno`, max_seq 128) sobre `txt_bruto_v3`, y banco de 30–50 preguntas con artículo correcto esperado (Fase 4), además de búsqueda híbrida en OE5 para costos/TUPA.
- La distancia top-1 no es lo mismo que acertar el artículo: con 10 preguntas la diferencia no es concluyente (P5).


---

## 9. Mejoras M1–M6 y evaluación con banco provisional (2026-11-21)

- **M1** `config.py`: perfiles de modelo por `MODEL_KEY` (`mpnet` | `e5base` | `e5large` | `bgem3` | `qwen3`) con prefijos de pregunta/documento. `mpnet` conserva los nombres V2/OE5; los demás usan sufijo `_<modelo>` en archivos y colección.
- **M2** `BREADCRUMB_MODO` (`completo` | `documento` | `ninguno`). Se mantiene `completo`.
- **M4** `09_evaluacion.py` + `corpus_upeu/config/banco_preguntas.csv` (14 positivas con artículo verificado en el texto + 2 negativas, **revisado = no**). Mide R@k/MRR en modos semántico, BM25, híbrido (RRF) y re-ranker (`BAAI/bge-reranker-v2-m3`). Salida en `metadatos/v3/evaluacion/`.
- **M5** filtro del pie de página postal ("Casilla 3564…", 18 documentos; URLs útiles se conservan) y unión de colas < 15 tokens.
- **M6** índices / tablas de contenido: sus líneas ya no cuentan como encabezado y los chunks de índice se descartan (84) registrados en `chunks_descartados_v3.csv` (cuentan para la cobertura).
- Tests: 38, pasan en el contenedor.

### V3 regenerada

5,053 chunks · 100% tokens embebidos · cobertura 100% · validación 0 errores / 4 warnings (artículos cortos legítimos) · 84 índices descartados.

### Banco provisional (14 positivas, nivel artículo; NO revisado)

| Versión | Modo | R@1 | R@5 | MRR@10 |
|---|---|---|---|---|
| V2 | semántico | 0.14 | 0.43 | 0.25 |
| V2 | híbrido | 0.29 | 0.50 | 0.40 |
| V2 | re-ranker | 0.43 | 0.71 | 0.57 |
| V3 | semántico | 0.14 | 0.36 | 0.24 |
| V3 | BM25 | 0.29 | 0.50 | 0.40 |
| V3 | híbrido | 0.21 | 0.50 | 0.33 |
| **V3** | **re-ranker** | **0.71** | **0.93** | **0.82** |

- Breadcrumb (V3 + re-ranker): completo 0.82 = documento 0.82 > ninguno 0.79 (MRR).
- V3 encuentra el TUPA (pre-ingreso, duplicado de diploma) y la Política ambiental; V2 no.
- **mpnet solo (semántico) es débil en ambas versiones** (MRR ≈ 0.24): el cuello de botella es el modelo, lo que recupera el re-ranker.
- Coseno top-1: positivas ≈ 0.72 vs negativas ≈ 0.50 → hay margen para calibrar el umbral "fuera de dominio".
- Compatibilidad para Qwen3 (`transformers 4.51.3`): exige `tokenizers>=0.21,<0.22` y `huggingface-hub>=0.30` (ya hay 0.36.2). chromadb 0.4.22 (`tokenizers>=0.13.2`) y sentence-transformers 3.1.1 (`transformers>=4.38,<5`) son compatibles; solo choca el pin `tokenizers==0.15.0`.

Pendiente: revisar/ampliar el banco a 30–50 preguntas, comparar modelos (`MODEL_KEY`), decidir modelo y luego OE5 (re-ranker + híbrido + fixes de recorte).

### Banco de preguntas ampliado (2026-11-21)

`corpus_upeu/config/banco_preguntas.csv`: **154 positivas (2–5 por cada uno de los 48 documentos, 127 con artículo esperado) + 8 negativas** (4 cercanas al dominio, 4 fuera). Por categoría: A 20 · B 47 · C 19 · D 25 · E 43. Todas con `revisado = no`.
- Documentos pequeños leídos completos; en los grandes se leyó la estructura completa y el artículo esperado completo. Cada artículo se verificó contra los chunks V3.
- Preguntas cuya respuesta existe también en otro reglamento: anotadas como `alt:` en `notas` (09 aún evalúa un solo documento esperado).
- 6 preguntas de Jefes de prácticas y Auditoría interna apuntan a artículos que el chunking no detecta (espacio no separable ` ` tras "Artículo"): hoy solo pueden acertar a nivel documento.
- Nombres de archivo con tilde descompuesta (NFD): usar subcadenas sin la letra acentuada (p.ej. `Acoso y Hostigamiento Sexual`).

### Imagen Docker actualizada para comparar modelos (2026-11-21)

- `requirements.txt`: `transformers 4.39.3 → 4.51.3`, `tokenizers 0.15.0 → 0.21.1` (Qwen3). Imagen anterior respaldada como `oe1_arquitectura_corpus-jupyter:tf439`.
- Diferencias con la imagen anterior (`pip freeze`): solo esos 2 paquetes + `decorator`/`wcwidth` (menores). `pip check` limpio; 38 tests pasan; ChromaDB V3 se lee y consulta.
- **mpnet idéntico:** 5,053/5,053 conteos de tokens iguales, coseno mínimo vs `embeddings_v3.npy` = 0.9999998, top-10 idéntico en las 162 preguntas. V2/V3 siguen siendo reproducibles.
- Cargan y codifican en GPU: e5base (1.2 GB pico), e5large (3.4 GB), bgem3 (3.1 GB), **qwen3 (5.9 GB con batch 16 → usar `EMBED_BATCH_SIZE=4`)**. Re-ranker bge-v2-m3 OK. Qwen3 en lote da el mismo vector que solo (pooling last-token correcto con padding a la derecha).
- El aviso `Failed to send telemetry event … capture()` de chromadb ya existía antes (posthog), es inofensivo.

Primera corrida V3 + mpnet con este banco: semántico MRR 0.358 · BM25 0.496 · híbrido 0.448 · **re-ranker 0.709 (R@1 0.64, R@5 0.81, Rdoc@5 0.95)**. Coseno top-1: positivas 0.71 vs negativas 0.49.

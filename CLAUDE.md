# CLAUDE.md — Estado actual del proyecto `oe1_arquitectura_corpus`

> Auditoría realizada el **2026-11-20** leyendo los archivos reales del repo (no solo la documentación).
> Commits: `73d1f58` (pipeline V3), `3da49fe` (dependencias para comparar modelos), `b872e1a` (normalización de espacios Unicode), todos del 2026-11-21.
> **Estado al cierre de OE1 (2026-11-21):** corpus **V3 + `BAAI/bge-m3`** generado, validado e indexado en
> **`corpus_upeu/vector_store_v3_bgem3/`** (colección `corpus_upeu_v3_bgem3`, 5,117 chunks, MRR semántico 0.756 vs 0.358 de mpnet).
> **OE5 sigue usando la V2** (mpnet) hasta su migración; el banco de preguntas aún no tiene revisión humana.
> Los números de §2 son los de la V2 en disco; lo nuevo está en §8–§10.

---

## 0. Línea de tiempo

| Fecha | Hito | Resultado clave |
|---|---|---|
| 2026-06 | V1 en notebooks: 11 bugs corregidos, mpnet elegido (banco 10Q) | 0/10 → 10/10 cubiertas |
| 2026-08-21 / 09-10 | "corpus v2": 1 artículo = 1 chunk | 6,259 chunks; OE5 16/17 M02 |
| 2026-09-17 | V2 en scripts reproducibles, migrada a OE5 | 5,126 chunks, 0 errores |
| 2026-11-20 | Auditoría (§2–§5) y código V3 (§8) | V2 trunca el 50.6% de tokens; V3: 100% embebido |
| 2026-11-21 | M1–M6 y banco 154+8 (§9) | mpnet V3 + re-ranker MRR 0.709 |
| 2026-11-21 | Imagen con transformers 4.51.3 y comparación de 5 modelos (§9) | bge-m3 0.724, sin diferencia significativa con e5/qwen3; todos ≫ mpnet |
| 2026-11-21 | Fix NBSP + indexación del ganador (§9, §10) | +62 artículos; bge-m3 0.756 / re-ranker 0.776; `vector_store_v3_bgem3/` |

---

## 1. Qué es este proyecto

Convierte **48 PDFs oficiales de la Universidad Peruana Unión (UPeU)** en un corpus vectorial para **RAG**, consumido por el chatbot **OE5** (repo hermano `../oe5_chatbot_upeu`).

Pipeline: `PDF → texto bruto → diagnóstico/clasificación → chunking por artículo → embeddings → validación → ChromaDB (+ BM25) → evaluación → OE5`.

- Idioma: español.
- Categorías temáticas: **A** gobierno, **B** académico, **C** investigación, **D** bienestar, **E** políticas.
- Modelo de embeddings: **V3 → `BAAI/bge-m3`** (1024 dim, coseno, revisión `5617a9f6…`, sin prefijos).
  V2/OE5 actual → `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 dim).
- OCR: **Tesseract** vía `pytesseract`, con `lang="spa"`; 250 DPI en V2, **300 DPI en V3**. No hay OCR propio ni modelos de deep learning.

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

## 5. Problemas conocidos de la V2 (resueltos en V3, ver §6)

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

## 6. Plan de mejora (estado al 2026-11-21)

> Fases 1, 3 y 4 hechas; 2a probada y descartada (mpnet no mejora con 384); 2b innecesaria porque se cambió de
> modelo (bge-m3 lee hasta 8,192 tokens, se usa 384 sin truncar). **Fase 5 (OE5) pendiente.**
> Estado de P1–P5: P1 ✅ (0 tokens truncados) · P2 ✅ (0 errores / 5 warnings reales) · P3 ✅ (`pagina` 100%, `seccion`, títulos,
> sin duplicados ni marcadores OCR) · P4 ✅ (`MANIFEST*.json` con hashes) · P5 ✅ parcial (banco 154+8, falta revisión humana).

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

1. **V2 (5,126 chunks) es la versión que OE5 consume hoy**; no tocarla. La versión a desplegar es V3 + bge-m3 (`vector_store_v3_bgem3/`), solo junto con los cambios de código de OE5.
   - **No hacer commits ni operaciones de GitHub**: el usuario los hace; solo indicarle archivos y comandos.
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

### Comparación de modelos de embeddings (V3, banco 154+8, 2026-11-21)

Mismos parámetros para todos (articulo, max_seq 384, chunk 350/40, breadcrumb completo). Solo se generaron `chunks/embeddings/MANIFEST_<modelo>` en `metadatos/v3/`; **no se indexó ChromaDB** (09 busca exacto sobre el `.npy`). Los 5 corpus: 0 errores, 100% tokens embebidos, 100% cobertura. Chunks: mpnet/e5large 5,053 · e5base 5,061 · bgem3 5,072 · qwen3 5,723 (su tokenizer cuenta más tokens en español).

| Modelo | Sem. R@1 | Sem. R@5 | **Sem. MRR** (IC95%) | Rerank MRR | AUC pos/neg |
|---|---|---|---|---|---|
| mpnet | 0.28 | 0.47 | 0.358 (0.29–0.43) | 0.709 | 0.981 |
| e5base | 0.60 | 0.81 | 0.689 (0.62–0.75) | 0.743 | 0.994 |
| e5large | 0.60 | 0.86 | 0.700 (0.64–0.76) | 0.747 | 0.982 |
| **bgem3** | **0.60** | **0.88** | **0.724** (0.67–0.78) | 0.746 | 0.994 |
| qwen3 | 0.58 | 0.82 | 0.678 (0.62–0.74) | 0.735 | 0.997 |

- Los 4 modelos nuevos duplican el MRR semántico de mpnet (+0.32 a +0.37, significativo; bgem3 mejora 95 preguntas y empeora 11).
- Entre los 4 nuevos las diferencias **no son significativas** (bgem3 vs e5base ΔMRR +0.035, IC [-0.016, +0.084]).
- Con re-ranker todos convergen (0.735–0.747); aun así superan a mpnet (+0.03–0.04).
- El híbrido (RRF con BM25) **empeora** a los modelos nuevos frente al semántico solo (bgem3 0.724 → 0.657): en OE5 conviene re-rankear sobre candidatos semánticos o ponderar menos BM25.
- Ojo: la escala de coseno cambia por modelo (e5 deja las negativas en ~0.80–0.87), por eso el umbral de OE5 debe recalibrarse con el modelo elegido.

### Ganador indexado: bge-m3 (2026-11-21)

- Colección **`corpus_upeu_v3_bgem3`** en `corpus_upeu/vector_store_v3/` (5,072 chunks al indexarse; 5,117 tras el fix NBSP; 1024 dim, coseno, HNSW M16/ef200/ef_search100). Convive con `corpus_upeu_v3` (mpnet, 5,053). Carpeta: 289 MB (incluye carpetas huérfanas de colecciones recreadas).
- Revisión fijada en `config.py`: `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181` (reproduce los embeddings guardados, coseno ≥ 0.9999999). Sin prefijos de pregunta/documento.
- HNSW vs búsqueda exacta (162 preguntas): top-1 igual 161/162, solapamiento top-10 99.4%.
- `07_indexacion_v2.py`: la descripción de la colección ahora usa el nombre real del modelo (antes decía "mpnet" fijo).
- **Corrección NBSP (2026-11-21):** `chunking.normalizar_espacios()` convierte espacios Unicode (` ` etc.) a espacio normal en `separar_paginas` (05 y 06) y en `extraccion_comun.limpiar_lineas`; reemplazo 1 a 1, offsets intactos. Test nuevo (39 tests). Recupera **62 artículos**: Jefes de prácticas 0 → 24, Auditoría interna 4 → 42; los otros 46 documentos quedan idénticos chunk a chunk. Solo se regeneró **bge-m3**: 5,117 chunks, 0 errores / 5 warnings (chunks cortos). Los archivos de mpnet/e5/qwen3 en `metadatos/v3/` siguen con el chunking anterior (la comparación fue justa: todos con el mismo fallo).
- **bge-m3 tras la corrección:** semántico R@1 0.636 · R@5 0.916 · **MRR 0.756** (antes 0.724) · re-ranker MRR 0.776. Las 6 preguntas de esos 2 reglamentos: 5 en el puesto 1.
- **Carpeta para OE5: `corpus_upeu/vector_store_v3_bgem3/`** (81 MB, solo la colección `corpus_upeu_v3_bgem3`, 5,117 chunks; HNSW vs exacto top-1 161/162). Creada con `VSTORE_DIR=vector_store_v3_bgem3` (nueva variable opcional en `config.py`). `vector_store_v3/` también tiene la colección (más la de mpnet y carpetas huérfanas de colecciones recreadas; no se usa para desplegar). OE5 tiene chromadb 0.4.22 (igual).
- **Para OE5:** modelo `BAAI/bge-m3` (misma revisión) para codificar preguntas, colección `corpus_upeu_v3_bgem3`, recalibrar el umbral de distancia (las distancias cambian de escala respecto a mpnet) y re-rankear sobre candidatos semánticos (el RRF con BM25 empeora a bge-m3). V2 sigue siendo la canónica hasta probar en OE5.

Primera corrida V3 + mpnet con este banco: semántico MRR 0.358 · BM25 0.496 · híbrido 0.448 · **re-ranker 0.709 (R@1 0.64, R@5 0.81, Rdoc@5 0.95)**. Coseno top-1: positivas 0.71 vs negativas 0.49.

---

## 10. Cierre de OE1 (2026-11-21)

**Entregable:** `corpus_upeu/vector_store_v3_bgem3/` (81 MB: `chroma.sqlite3` + 1 carpeta de segmento HNSW).

| Dato | Valor |
|---|---|
| Colección | `corpus_upeu_v3_bgem3` · 5,117 chunks · 48 documentos |
| Modelo | `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`, 1024 dim, sin prefijos, `normalize_embeddings=True` |
| Chunking | articulo · max_seq 384 · chunk 350/40 · breadcrumb completo · 0 tokens truncados · cobertura 100% |
| Índice | coseno · HNSW M16 / construction_ef 200 / search_ef 100 · top-1 = búsqueda exacta en 161/162 |
| Validación | 0 errores · 5 warnings · 84 índices descartados |
| Retrieval (banco 154+8) | semántico R@1 0.636 · R@5 0.916 · MRR 0.756 · re-ranker MRR 0.776 · híbrido 0.681 · BM25 0.510 |
| Coseno top-1 | positivas 0.690 vs negativas 0.484 |
| Trazabilidad | `metadatos/v3/MANIFEST_bgem3.json` (config, estadísticas, sha256) |

Documentación actualizada el mismo día: `README.md`, `RESUMEN_FINAL.md`, `DIAGNOSTICO_CAMBIOS.md` (§10),
`scripts/README.md`, `scripts/v2/README_PIPELINE_V2.md`, `scripts/v2/MIGRACION_OE5_V2.md` (pasos de la migración V3).

Pendientes fuera de OE1: migración de OE5 (colección, modelo, umbral, re-ranker, recortes de contexto).
Opcionales en OE1: revisión humana del banco; multi-documento (`alt:`) y NFC en `09`; regenerar los archivos de
mpnet/e5/qwen3 con el fix NBSP si se quisiera repetir la comparación; carpetas huérfanas de `vector_store_v3/`
(no borrar sin confirmación).

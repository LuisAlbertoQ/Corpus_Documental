# Arquitectura de Corpus UPeU (OE1)

Convierte **48 PDFs oficiales de la Universidad Peruana Unión (UPeU)** en un corpus vectorial para RAG
(Retrieval-Augmented Generation). Lo consume el chatbot **OE5** (`../oe5_chatbot_upeu`).

```
PDF → inventario → clasificación → diagnóstico → extracción (4 rutas + OCR)
    → chunking por artículo → embeddings → validación → ChromaDB (+ BM25) → evaluación → OE5
```

> **Estado al 2026-11-21**
> - **Versión entregable: V3 + `BAAI/bge-m3`** → `corpus_upeu/vector_store_v3_bgem3/`
>   (colección `corpus_upeu_v3_bgem3`, 5,117 chunks, 1024 dim, coseno).
> - **OE5 todavía usa la V2** (mpnet, colección `corpus_upeu_v2`, 5,126 chunks). Pasar a V3 exige
>   cambios en el código de OE5 (ver [Migración a OE5](#4-migración-a-oe5)).
> - El banco de 162 preguntas usado para elegir es **provisional** (`revisado = no`).

---

## 1. Resultados principales

### Evolución del corpus

| Fecha | Versión | Chunks | Modelo | Qué cambió | Resultado |
|---|---|---|---|---|---|
| 2026-06 | V1 (notebooks) | 2,488 → 3,886 | MiniLM → mpnet | 11 bugs corregidos; modelo multilingüe | Banco 10Q: 0/10 → 10/10 cubiertas |
| 2026-08-21 / 09-10 | "corpus v2" (agosto) | 6,259 | mpnet | 1 artículo = 1 chunk (antes se fusionaban) | Art. 53 Admisión: fuera del top-40 → rank 2; OE5 16/17 M02 |
| 2026-09-17 | **V2** (scripts) | 5,126 | mpnet | Pipeline en 7 scripts, 4 rutas de extracción, taxonomía unificada | 0 errores, 1,403 warnings. **En producción en OE5** |
| 2026-11-20 | V3 (código) | 5,260 | mpnet | Auditoría: 50.6% de tokens truncados en V2; `config.py`, tablas en su página, `pagina`, tests | 100% tokens embebidos, 100% cobertura |
| 2026-11-21 | V3 + M1–M6 | 5,053 | mpnet | Perfiles de modelo, evaluación R@k/MRR, limpieza de pies e índices | Con re-ranker MRR 0.709 (banco 162) |
| 2026-11-21 | **V3 + bge-m3** | **5,117** | **bge-m3** | Comparación de 5 modelos; corrección NBSP (+62 artículos) | **Semántico MRR 0.756** (mpnet 0.358) |

### V2 frente a V3 + bge-m3

| Métrica | V2 (OE5 hoy) | V3 + bge-m3 |
|---|---|---|
| Chunks | 5,126 | 5,117 |
| Modelo / dimensión | mpnet / 768 | bge-m3 / 1024 |
| Tokens que llegan al embedding | **49.4%** (mpnet trunca a 128) | **100%** (máx 382 de 384) |
| Cobertura de texto del PDF | ~94.7% (TUPA ~53%) | **100%** |
| Campo `pagina` | 0% lleno | **100%** |
| Validación | 0 errores / 1,403 warnings (mayoría falsos positivos) | 0 errores / 5 warnings (artículos cortos legítimos) |
| TUPA | texto revuelto | filas limpias (`1.3 \| Pre-Ingreso … \| S/. 120.00`) |

### Comparación de modelos (V3, banco de 154 positivas + 8 negativas, nivel artículo)

| Modelo | Sem. R@1 | Sem. R@5 | **Sem. MRR@10** (IC95%) | Re-ranker MRR | AUC pos/neg | Chunks |
|---|---|---|---|---|---|---|
| mpnet (V2/OE5) | 0.28 | 0.47 | 0.358 (0.29–0.43) | 0.709 | 0.981 | 5,053 |
| e5base | 0.60 | 0.81 | 0.689 (0.62–0.75) | 0.743 | 0.994 | 5,061 |
| e5large-instruct | 0.60 | 0.86 | 0.700 (0.64–0.76) | 0.747 | 0.982 | 5,053 |
| bge-m3 | 0.60 | 0.88 | 0.724 (0.67–0.78) | 0.746 | 0.994 | 5,072 |
| qwen3-0.6B | 0.58 | 0.82 | 0.678 (0.62–0.74) | 0.735 | 0.997 | 5,723 |
| **bge-m3 tras fix NBSP** | **0.64** | **0.92** | **0.756** | **0.776** | — | **5,117** |

- Los 4 modelos nuevos **duplican** el MRR semántico de mpnet (+0.32 a +0.37, significativo).
- Entre los 4 nuevos la diferencia **no es significativa**; se eligió **bge-m3** por tener el mejor MRR y R@5,
  no necesitar prefijos y tener licencia MIT.
- Con re-ranker (`BAAI/bge-reranker-v2-m3`) todos convergen (0.735–0.776).
- El híbrido RRF con BM25 **empeora** a bge-m3 (0.756 → 0.681): en OE5 conviene re-rankear sobre candidatos semánticos.
- Detalle: `corpus_upeu/metadatos/v3/evaluacion/resumen.csv` y `CLAUDE.md` §9.

---

## 2. Requisitos e instalación

- Docker + Docker Compose, GPU NVIDIA con NVIDIA Container Toolkit (CUDA 12.1 en la imagen).
- ~4 GB de VRAM para bge-m3 (Qwen3 necesita ~6 GB con lote 16; usar `EMBED_BATCH_SIZE=4`).
- Modelos de HuggingFace en `.hf_cache/` (se descargan la primera vez).

Versiones fijas (`requirements.txt`): torch 2.1.2+cu121, sentence-transformers 3.1.1, transformers 4.51.3,
tokenizers 0.21.1, huggingface_hub 0.36.2, chromadb 0.4.22, rank-bm25 0.2.2, pytest 8.2.2.
La imagen anterior (transformers 4.39.3, con la que se generó V2) quedó respaldada como
`oe1_arquitectura_corpus-jupyter:tf439`; se verificó que mpnet produce los mismos vectores en ambas.

```powershell
docker-compose build
docker-compose up -d          # contenedor oe1_corpus, Jupyter en http://localhost:8888
docker exec oe1_corpus python -m pytest /home/jupyteruser/work/scripts/v2/tests -q   # 39 tests
```

---

## 3. Ejecución del pipeline V3

La versión se elige con `CORPUS_VERSION` (por defecto `v3`) y el modelo con `MODEL_KEY`
(`mpnet` | `e5base` | `e5large` | `bgem3` | `qwen3`). La V2 está protegida contra escritura.

```powershell
# Etapas 01-04 (no dependen del modelo)
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/01_inventario.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/02_clasificacion.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/03_diagnostico.py
docker exec oe1_corpus python /home/jupyteruser/work/scripts/v2/04_extraccion_v2.py

# Etapas 05-09 con el modelo ganador
docker exec -e MODEL_KEY=bgem3 oe1_corpus python /home/jupyteruser/work/scripts/v2/05_chunking_v2.py
docker exec -e MODEL_KEY=bgem3 oe1_corpus python /home/jupyteruser/work/scripts/v2/06_validacion_corpus.py
docker exec -e MODEL_KEY=bgem3 -e VSTORE_DIR=vector_store_v3_bgem3 oe1_corpus python /home/jupyteruser/work/scripts/v2/07_indexacion_v2.py --force
docker exec -e MODEL_KEY=bgem3 oe1_corpus python /home/jupyteruser/work/scripts/v2/08_indice_bm25.py
docker exec -e MODEL_KEY=bgem3 oe1_corpus python /home/jupyteruser/work/scripts/v2/09_evaluacion.py
```

- `VSTORE_DIR` (opcional) indexa en una carpeta limpia; sin ella se usa `corpus_upeu/vector_store_v3/`.
- En Windows, las carpetas nuevas (`txt_bruto_vN`, `vector_store_*`) deben crearse **desde el host**.
- En Git Bash, anteponer `MSYS_NO_PATHCONV=1` a los `docker exec`.
- Documentación completa: [`scripts/v2/README_PIPELINE_V2.md`](scripts/v2/README_PIPELINE_V2.md).

---

## 4. Migración a OE5

**Pendiente** (se hará en la fase OE5). Copiar la carpeta sola no basta, porque OE5 busca la colección
`corpus_upeu_v2` y codifica las preguntas con mpnet.

| Qué | Dónde en OE5 | Cambio |
|---|---|---|
| Carpeta | `vector_store/` | Respaldar la actual y copiar `corpus_upeu/vector_store_v3_bgem3/` (81 MB) |
| Colección | `backend/rag_pipeline.py:137`, `backend/config.py:200` | `corpus_upeu_v2` → `corpus_upeu_v3_bgem3` |
| Modelo | `backend/rag_pipeline.py:138`, `backend/config.py:199` | `BAAI/bge-m3`, revisión `5617a9f61b028005a4858fdac845db406aefb181`, sin prefijos |
| Umbral | `UMBRAL_DISTANCIA_COSENO` (0.40) | Recalibrar: la escala de distancias cambia con el modelo |
| Re-ranker | pipeline | Re-rankear sobre candidatos semánticos (no RRF con BM25) |

Guía detallada: [`scripts/v2/MIGRACION_OE5_V2.md`](scripts/v2/MIGRACION_OE5_V2.md).

---

## 5. Estructura del proyecto

```
oe1_arquitectura_corpus/
├── corpus_upeu/
│   ├── pdfs/                         # 48 PDFs (no en git)
│   ├── config/
│   │   ├── clasificacion_manual.csv  # clasificación revisada (entrada de 02)
│   │   └── banco_preguntas.csv       # 154 positivas + 8 negativas (revisado = no)
│   ├── txt_bruto_v2/  txt_bruto_v3/  # texto extraído por versión
│   ├── metadatos/
│   │   ├── v2/                       # V2 canónica (OE5)
│   │   └── v3/                       # chunks/embeddings/MANIFEST por modelo (_e5base, _bgem3…)
│   │       └── evaluacion/           # resumen.csv + detalle por corrida
│   ├── vector_store_v2/              # V2 (mpnet, 5,126) — la que usa OE5
│   ├── vector_store_v3/              # V3 mpnet + bge-m3 (de trabajo, con carpetas huérfanas)
│   └── vector_store_v3_bgem3/        # ← ENTREGABLE PARA OE5 (solo corpus_upeu_v3_bgem3)
├── scripts/
│   ├── v2/                           # pipeline 01–09 + config.py, chunking.py, tests/
│   └── *.py                          # legacy agosto 2026 (rechunk_fixed, reembed_index…)
├── notebooks/                        # legacy V1 (1–9)
├── vector_store*/                    # legacy V1/agosto (6,259 chunks cada uno)
├── CLAUDE.md                         # estado detallado y bitácora técnica
├── RESUMEN_FINAL.md                  # resumen ejecutivo por etapas
├── DIAGNOSTICO_CAMBIOS.md            # diagnóstico V1 (histórico)
├── Dockerfile, docker-compose.yml, requirements.txt
```

---

## 6. Historia (resumen)

- **Junio 2026 — V1:** se corrigieron 11 bugs de los notebooks (marcadores de página indexados, whitelist que
  borraba `°`, chunks de 400 tokens, ChromaDB usando un modelo en inglés por defecto…). Se eligió mpnet
  frente a MiniLM y bge-m3 con un banco de 10 preguntas. Detalle en `DIAGNOSTICO_CAMBIOS.md`.
- **Agosto/septiembre 2026 — corpus v2 y V2:** se corrigió la fusión de artículos (6,259 chunks) y luego se
  pasó todo a scripts reproducibles (V2, 5,126 chunks), que es lo que OE5 consume (16/17 M02 end-to-end).
- **20–21 noviembre 2026 — V3:** auditoría con los archivos reales. Hallazgo principal: mpnet truncaba a 128
  tokens y la mitad del texto nunca llegaba al embedding. Se reescribió el pipeline (V3), se creó un banco
  de 162 preguntas con artículo esperado, se compararon 5 modelos y se eligió **bge-m3**. La conclusión de
  junio (mpnet > bge-m3) se invierte: con chunks bien hechos, sin truncamiento y midiendo aciertos de artículo
  (no solo distancia top-1), bge-m3 duplica el MRR de mpnet.

## Notas

- Los PDFs no se suben a git. Las carpetas `corpus_upeu/vector_store_v*/` están en `.gitignore`
  (versionarlas requiere `git add -f`).
- Distancia = 1 − coseno. Los umbrales 0.30/0.50 de V1 eran para mpnet y no aplican a bge-m3.

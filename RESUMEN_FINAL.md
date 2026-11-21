# Resumen Final — Optimización RAG UPeU

## ¿Qué se hizo?

Se optimizó el pipeline RAG (Retrieval-Augmented Generation) del corpus documental
de la Universidad Peruana Unión (49 PDFs, 2488 chunks) en 2 etapas:

### Etapa 1 — Diagnóstico y corrección de bugs
Se revisaron los 6 notebooks del pipeline ETL (extracción → limpieza → chunking →
indexación → verificación) y se corrigieron **11 issues reales**:

- **Críticos** (4): marcadores `[Página N]` en texto indexado, whitelist que rompía
  artículos, `chunk_size=400` que excedía el contexto del LLM, modelo por defecto
  en inglés que daba distancias 0.5–0.7 (ruido)
- **Medios** (5): umbral OCR bajo, tablas destruidas, chunking sin respeto a
  estructura, categorización mal-ordenada, umbral de cobertura falso positivo
- **Bajos** (2): metadata insuficiente, excepciones genéricas de ChromaDB

**Resultado:** cobertura de 0/10 → 8/10 con `paraphrase-multilingual-MiniLM-L12-v2`.

### Etapa 2 — Comparativa de modelos de embeddings
Se evaluaron 3 modelos de embeddings en GPU con el banco de 10 preguntas:

| Modelo | Dimensión | Cubiertas | Excelentes | Cat match |
|---|---|---|---|---|
| MiniLM-L12-v2 | 384 | 8/10 | 5/10 | 3/10 |
| **mpnet-base-v2** | **768** | **10/10** | **6/10** | **8/10** |
| bge-m3 | 1024 | 10/10 | 2/10 | 6/10 |

## ¿Por qué se hicieron los cambios?

1. **El corpus tenía información correcta pero no se recuperaba.** Las distancias
   de 0.5–0.7 indicaban que el modelo de embeddings estaba fallando, no el corpus.
   La causa raíz: ChromaDB usaba por defecto `all-MiniLM-L6-v2` (inglés) en lugar
   del modelo multilingüe configurado.

2. **El chunking destruía artículos completos.** Con `chunk_size=400` y overlap
   por ventana deslizante, las búsquedas a menudo devolvían chunks a mitad de
   artículo, sin contexto suficiente para que el LLM respondiera. Se reemplazó
   por chunking estructural que respeta límites de `Artículo N` / `Capítulo X`.

3. **Categorización de documentos incorrecta.** El corpus tiene 5 categorías
   (A: gobierno, B: académico, C: investigación, D: bienestar, E: políticas),
   pero el código original asignaba categorías por orden de keywords,
   fallando en "REGLAMENTO DE ESTUDIOS" (lo marcaba como D en vez de B).

4. **bge-m3 no era la mejor opción pese a ser "estado del arte".** Para corpus
   cortos y estructurados (artículos de reglamentos), mpnet (768 dim) supera
   consistentemente a bge-m3 (1024 dim) tanto en distancia promedio (0.29 vs 0.33)
   como en match por categoría (8/10 vs 6/10).

5. **Caso de uso real expuesto:** la pregunta "¿Cómo solicito una beca?" con
   MiniLM devolvía `REGLAMENTO SERVICIO APRENDIZAJE IDIOMAS` (d=0.74, incorrecto).
   Con mpnet devuelve `REGLAMENTO BECAS 2021 ACTUALIZADO` (d=0.157, excelente),
   mostrando el chunk exacto con los 5 requisitos de beca.

## ¿Qué se mantuvo?

- ✅ ChromaDB 0.4.22 (versión del proyecto)
- ✅ Distancia coseno como métrica
- ✅ `normalize_embeddings=True` en SentenceTransformer
- ✅ Estructura de 5 categorías A-E
- ✅ Pipeline ETL (extracción → limpieza → chunking → indexación)

## Archivos del proyecto

### Modificados
- `notebooks/1_extraccion_texto.ipynb` — extracción PDF con `pdfplumber`
- `notebooks/2_limpieza_corpus.ipynb` — whitelist ampliada + detección de items
- `notebooks/3_chunking_embeddings.ipynb` — chunking estructural 300/80 tokens
- `notebooks/4_indexacion_vectorial.ipynb` — `CATEGORIAS_KEYWORDS`, `extraer_articulo`
- `notebooks/5_verificacion_cobertura.ipynb` — umbrales 0.30/0.50 + `query_embeddings=`
- `Dockerfile` — PyTorch 2.1.2 con CUDA 12.1
- `docker-compose.yml` — GPU passthrough + 3 volúmenes para comparativa

### Nuevos
- `notebooks/7_evaluacion_miniLM_gpu.ipynb` — evaluación MiniLM en GPU
- `notebooks/8_evaluacion_mpnet_gpu.ipynb` — evaluación mpnet en GPU
- `notebooks/9_evaluacion_bge_m3_gpu.ipynb` — evaluación bge-m3 en GPU
- `vector_store_miniLM/`, `vector_store_mpnet/`, `vector_store_bge_m3/` — índices separados
- `DIAGNOSTICO_CAMBIOS.md` — documentación técnica detallada

## Métricas finales (producción con mpnet)

```
Cobertura:           10/10 (100%)   — vs 0/10 inicial
Excelentes (d<0.30):  6/10 (60%)    — vs 0 inicial
Categoría match:      8/10 (80%)    — vs n/a inicial
Distancia promedio:   0.2954        — vs 0.59 inicial
Distancia mínima:     0.1342
Distancia máxima:     0.4619
```

## Migración a OE5 (chatbot)

Cambios aplicados en `oe5_chatbot_upeu`:

1. `backend/rag_pipeline.py` → modelo cambiado a `paraphrase-multilingual-mpnet-base-v2`
2. `vector_store/` → regenerado con embeddings de 768 dim
3. `backend/config.py` → `UMBRAL_DISTANCIA_COSENO` ajustado de 0.35 a 0.40

**Estado:** ✅ Producción migrada, listo para pruebas con usuarios reales.

---

## Actualización 2026-08-21 — Corpus v2 (re-chunking)

**Motivo:** el chatbot oe5 devolvía M04 en preguntas con respuesta en el corpus
(caso "nota mínima admisión"). Diagnóstico: `chunk_text_estructural` fusionaba
artículos pequeños conservando solo la primera cabecera (Art. 52/53 absorbidos
bajo header "Artículo 51°"), más `Subcapítulo` sin partir. 14.5% de chunks
afectados. Detalle en `scripts/README.md`.

**Cambios:** 1 artículo = 1 chunk (sin fusionar), corte en Subcapítulo,
cabecera siempre presente. 3886 → **6259 chunks** (el original además
duplicaba contenido en cada overflow del buffer).

**Re-comparación 3 modelos (mismo banco 10Q):**

| Modelo | Dimensión | Cubiertas | Excelentes | Cat match |
|---|---|---|---|---|
| MiniLM-L12-v2 | 384 | 9/10 | 6/10 | 7/10 |
| **mpnet-base-v2** | **768** | **10/10** | **7/10** | **8/10** |
| bge-m3 | 1024 | 10/10 | 4/10 | 6/10 |

**Veredicto:** mpnet sigue ganando y mejoró (6→7 excelentes). Caso Art. 53:
fuera del top-40 → rank #2. Cobertura oe5 en umbral 0.40: 94.1% → 100%.

**Validación end-to-end en oe5** (banco 17Q contra `/consulta` en vivo,
`evaluacion/evaluar_modelo.ps1`): **16/17 M02**, tiempo prom 8.32s.
Q11 constancia y Q12 cambio de carrera resueltos. Pendiente: Q04 matrícula
(recupera 4 fuentes d=0.26 pero el LLM no redacta → M04 de generación).

---

## Actualización 2026-09-17 — V2 en scripts reproducibles

El pipeline pasó de notebooks a 7 scripts (`scripts/v2/`): 4 rutas de extracción según diagnóstico
(DIGITAL / MIXTO / TUPA_TABLAS / ESCANEADO), 18 metadatos, relaciones padre-hijo y taxonomía unificada con OE5
(becas = D). Resultado: **5,126 chunks**, 0 errores de validación, "beca" → Art. 49° (d = 0.245, categoría D).
Es la versión que OE5 tiene en producción.

---

## Actualización 2026-11-20/21 — V3 y cambio de modelo a bge-m3

### ¿Qué se hizo?

1. **Auditoría de la V2 con los archivos reales (2026-11-20).** El problema principal: mpnet solo lee
   **128 tokens** y la V2 tenía chunks de hasta 700, así que **el 50.6% del texto nunca llegaba al embedding**.
   Además, la validación daba ~1,400 falsos positivos, el campo `pagina` estaba vacío, las tablas del TUPA
   quedaban revueltas (~53% de cobertura) y no había forma de reproducir los resultados.
2. **Pipeline V3 (2026-11-20).** Configuración central, versiones protegidas, tablas en su página, chunks que
   nunca superan el límite del modelo, breadcrumb de contexto, validación de cobertura, tests automáticos.
3. **Mejoras M1–M6 (2026-11-21).** Perfiles para comparar modelos, evaluación con R@k/MRR (semántico, BM25,
   híbrido y re-ranker), limpieza del pie de página postal y descarte de tablas de contenido.
4. **Banco de preguntas (2026-11-21).** 154 preguntas con documento y artículo esperado (2–5 por cada uno de
   los 48 documentos) + 8 preguntas fuera de dominio. Provisional (sin revisión humana).
5. **Comparación de 5 modelos (2026-11-21)** con los mismos chunks y el mismo banco.
6. **Corrección NBSP (2026-11-21).** Dos reglamentos usaban un espacio no separable después de "Artículo";
   se recuperaron 62 artículos.
7. **Base de datos final (2026-11-21):** `corpus_upeu/vector_store_v3_bgem3/`.

### Resultados

| Modelo | Dimensión | MRR semántico | Artículo correcto en top-5 | Con re-ranker |
|---|---|---|---|---|
| mpnet (el de V2/OE5) | 768 | 0.358 | 47% | 0.709 |
| e5-base | 768 | 0.689 | 81% | 0.743 |
| e5-large-instruct | 1024 | 0.700 | 86% | 0.747 |
| Qwen3-Embedding-0.6B | 1024 | 0.678 | 82% | 0.735 |
| **bge-m3** | **1024** | **0.724** | **88%** | **0.746** |
| **bge-m3 + fix NBSP (final)** | **1024** | **0.756** | **92%** | **0.776** |

| Corpus | V2 | V3 + bge-m3 |
|---|---|---|
| Chunks | 5,126 | 5,117 |
| Texto que llega al embedding | 49.4% | **100%** |
| Cobertura del texto del PDF | ~94.7% | **100%** |
| Errores / warnings de validación | 0 / 1,403 | 0 / 5 |

### ¿Por qué cambió la conclusión de junio?

En junio bge-m3 "perdió" frente a mpnet, pero esa prueba usaba 10 preguntas, medía solo la distancia del
primer resultado (las distancias no son comparables entre modelos) y se hizo con chunks que fusionaban artículos.
Con chunks correctos, sin truncamiento y midiendo si se recupera **el artículo correcto**, bge-m3 duplica a
mpnet. Los cuatro modelos modernos quedan sin diferencia significativa entre sí; se eligió bge-m3 por tener las
mejores cifras, no necesitar prefijos y tener licencia MIT.

### Pendiente

- Migrar OE5 (colección, modelo, umbral, re-ranker). Hasta entonces OE5 sigue con la V2.
- Revisión humana del banco de preguntas.

---

*Documento generado: Junio 2026 · última actualización: 2026-11-21*
*Versión recomendada: `BAAI/bge-m3` (1024 dim, coseno) sobre el corpus V3 — OE5 aún usa mpnet/V2*

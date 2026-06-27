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

*Documento generado: Junio 2026*
*Versión recomendada: `paraphrase-multilingual-mpnet-base-v2` (768 dim, coseno)*

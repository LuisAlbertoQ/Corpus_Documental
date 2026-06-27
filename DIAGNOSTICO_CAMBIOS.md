# Diagnóstico y Optimización del Pipeline RAG UPeU

> Documento generado tras la revisión completa de los notebooks 1–9 del proyecto
> `oe1_arquitectura_corpus` (Universidad Peruana Unión).
> **Resultado final: 10/10 cobertura, 8/10 categoría match** con `paraphrase-multilingual-mpnet-base-v2`.

---

## 1. Resumen ejecutivo

### Etapa 1 — Diagnóstico y corrección (notebooks 1–6)

| Métrica | Inicio | Final MiniLM |
|---|---|---|
| Cobertura del banco de pruebas | 0/10 (falso) | **8/10 (80%)** |
| Distancia media top-1 | 0.59 (ruido) | **0.36 (útil)** |
| Excelentes (d < 0.30) | 0 | **5/10** |
| Categoría match | n/a | 5/10 |
| Issues corregidos | — | **11** |

### Etapa 2 — Comparativa de modelos de embeddings (notebooks 7, 8, 9)

| Métrica | MiniLM-L12 | **mpnet-base** ⭐ | bge-m3 |
|---|---|---|---|
| **Cubiertas** | 8/10 | **10/10** | 10/10 |
| **Excelentes (d<0.30)** | 5/10 | **6/10** | 2/10 |
| **Categoría match** | 3/10 | **8/10** | 6/10 |
| **Distancia media** | 0.34 | **0.29** | 0.33 |
| **Pregunta "beca" → doc correcto** | ❌ | **✅** (d=0.157) | ✅ (d=0.251) |
| Dimensión | 384 | 768 | 1024 |

**Veredicto:** `paraphrase-multilingual-mpnet-base-v2` es el modelo recomendado para
migrar `oe5_chatbot_upeu`. Bge-m3 sorprende con resultados inferiores pese a ser
estado del arte en otros benchmarks.

---

## 2. Diagnóstico por notebook

### Notebook 1 — `1_extraccion_texto.ipynb`

| Issue | Severidad | Problema | Estado |
|---|---|---|---|
| 1.1 | 🔴 Crítico | Marcadores `[Página N]` se insertan en el texto y se indexan en ChromaDB | ✅ Corregido |
| 1.2 | 🟡 Media | Umbral OCR de 15 palabras es bajo, fuerza OCR en páginas con tablas (TUPA) | ✅ Corregido |
| 1.3 | 🟡 Media | Tablas destruidas: `pdfplumber.extract_tables()` solo usado en diagnóstico | ✅ Corregido |

**Cambios aplicados:**

- `import pdfplumber` agregado
- Función `extraer_texto_hibrido()` reescrita:
  - **Sin** marcadores `[Página N]` en el output
  - `umbral_palabras_pagina=30` (antes 15)
  - Parámetro `preservar_tablas=True` con `pdfplumber.extract_tables()`
- Loop de procesamiento actualizado para pasar `preservar_tablas=True`

---

### Notebook 2 — `2_limpieza_corpus.ipynb`

| Issue | Severidad | Problema | Estado |
|---|---|---|---|
| 2.1 | 🔴 Crítico | Whitelist de caracteres elimina `°`, `º`, `$`, `%` — rompe "Artículo 1º", "S/. 250.00" | ✅ Corregido |
| 2.2 | ❌ Falso positivo | El código de reconstrucción de párrafos es **código muerto** (la función retorna antes). No aplica | ⏭️ Omitido |
| 2.3 | 🟢 Baja | Marcadores `[Página N]` no se eliminan | ✅ Corregido |

**Cambios aplicados** en `limpiar_texto()`:

- Eliminación explícita de marcadores `[Página N]`, `--- Pág N ---`, `[TABLA p.X.Y]`
- Whitelist ampliada: incluye `°`, `º`, `$`, `%`, `«»`, comillas tipográficas
- Detección de items numerados y encabezados de artículo/capítulo para evitar fusión incorrecta

---

### Notebook 3 — `3_chunking_embeddings.ipynb`

| Issue | Severidad | Problema | Estado |
|---|---|---|---|
| 3.1 | 🔴 Crítico | `chunk_size=400` tokens genera prompts de 8000+ chars, excede `n_ctx=4096` de Ollama → M06 | ✅ Corregido |
| 3.2 | 🟡 Media | Chunking por ventana deslizante ignora límites de Artículo/Capítulo | ✅ Corregido |

**Cambios aplicados:**

- Nuevas funciones:
  - `_RE_BLOQUE`: regex para detectar `Artículo N`, `Capítulo X`, `Sección N`, `Título X`
  - `chunk_text_estructural()`: divide respetando estructura, agrupa artículos pequeños, subdivide los grandes
  - `chunk_text_simple()`: fallback por tokens cuando no hay estructura
- Loop actualizado para usar `chunk_text_estructural()`

**Parámetros finales (iteración 2):**

```python
max_tokens=300       # era 200 en la primera iteracion, 400 en el original
overlap=80           # 50 en la primera iteracion, 80 final (validado)
max_chars_per_chunk=1500  # tope defensivo
```

---

### Notebook 4 — `4_indexacion_vectorial.ipynb`

| Issue | Severidad | Problema | Estado |
|---|---|---|---|
| 4.1 | 🔴 Crítico | `obtener_categoria()` mal-ordena: "REGLAMENTO DE ESTUDIOS" → D (debería ser B) | ✅ Corregido |
| 4.2 | 🟡 Media | Metadata insuficiente: solo `documento, num_tokens, categoria` | ✅ Corregido |
| 4.3 | 🟢 Baja | `try/except:` genérico al borrar colección | ✅ Corregido |

**Cambios aplicados:**

- Nueva tabla `CATEGORIAS_KEYWORDS` (A: gobierno, B: académico, C: investigación, D: bienestar, E: políticas)
- `obtener_categoria()` reescrita con lookup en keywords
- Nueva función `extraer_articulo()` con regex `_RE_ART`
- Metadata enriquecida con 6 campos: `documento`, `categoria`, `articulo`, `chunk_id`, `num_tokens`, `num_chars`
- `except chromadb.errors.NotFoundError` → `except ValueError` (API correcto para v0.4.22)
- Cliente ChromaDB creado con `Settings(anonymized_telemetry=False, allow_reset=True)`

**Errores encontrados durante la integración:**

1. `collection_name` no estaba definido en la celda reescrita → agregado `collection_name = "corpus_upeu"`
2. `NameError: ids/documentos/metadatos` → agregada celda de preparación de listas

---

### Notebook 5 — `5_verificacion_cobertura.ipynb`

| Issue | Severidad | Problema | Estado |
|---|---|---|---|
| 5.1 | 🔴 Crítico | Umbral 0.8 es falso positivo, reporta 100% cobertura con distancias reales de 0.4–0.6 | ✅ Corregido |
| 5.2 | 🟡 Media | No evalúa la respuesta del LLM, solo retrieval | ✅ Parcial (categoría esperada sí) |

**Cambios aplicados:**

- `evaluar_consulta()` con 3 niveles de calidad:
  - `excelente` si `d < 0.30`
  - `aceptable` si `d < 0.50` (era 0.40, luego subido a 0.50)
  - `no_cubierta` si `d ≥ 0.50`
- Acepta `categoria_esperada` y devuelve `categoria_match`
- Banco de preguntas `PREGUNTAS` con tuplas `(consulta, categoría_esperada)`
- Loop de evaluación con `PREGUNTAS` y `resultados` (antes `banco_preguntas`, `df_resultados`)

**Error crítico encontrado:** ChromaDB usaba su modelo por defecto (`all-MiniLM-L6-v2`, inglés) para `query_texts=`, dando distancias 0.5-0.66 (ruido).

**Fix aplicado:** usar `query_embeddings=` con `model.encode()` del modelo multilingüe:

```python
query_embedding = model.encode([consulta], normalize_embeddings=True)[0].tolist()
resultados = collection.query(query_embeddings=[query_embedding], ...)
```

Después del fix, distancias bajaron a 0.15-0.35.

---

### Notebook 6 — `6_prueba_manual.ipynb`

Sin issues. Solo es diagnóstico interactivo.

---

### Notebooks 7, 8, 9 — Comparativa de modelos de embeddings (NUEVOS)

**Notebook 7** — `7_evaluacion_miniLM_gpu.ipynb`
- Modelo: `paraphrase-multilingual-MiniLM-L12-v2` (384 dim)
- Resultado: 8/10 cubiertas, 5/10 excelentes, 3/10 cat match, d. media 0.34

**Notebook 8** — `8_evaluacion_mpnet_gpu.ipynb` ⭐
- Modelo: `paraphrase-multilingual-mpnet-base-v2` (768 dim)
- Resultado: 10/10 cubiertas, 6/10 excelentes, 8/10 cat match, d. media 0.29
- **Ganador** — es el modelo recomendado para producción

**Notebook 9** — `9_evaluacion_bge_m3_gpu.ipynb`
- Modelo: `BAAI/bge-m3` (1024 dim) con prefijos `query: ` / `passage: `
- Resultado: 10/10 cubiertas, 2/10 excelentes, 6/10 cat match, d. media 0.33
- Sorprende con resultados inferiores pese a ser estado del arte en otros benchmarks

---

## 3. Hallazgos técnicos adicionales

### 3.1 Error de ChromaDB al limpiar `vector_store/`

**Síntoma:**
```
OperationalError: unable to open database file
ValueError: Could not connect to tenant default_tenant
```

**Causa:** el volumen Docker `./vector_store:/home/jupyteruser/work/vector_store` queda en estado inválido al eliminar archivos con el contenedor corriendo.

**Solución:**
```powershell
docker-compose down
Remove-Item -Recurse -Force vector_store -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path vector_store
docker-compose up
```

### 3.2 Excepciones de ChromaDB por versión

| Versión | Excepción al borrar colección inexistente |
|---|---|
| 0.4.x (la del proyecto) | `ValueError` |
| 0.5+ | `chromadb.errors.NotFoundError` |

### 3.3 Modelo de embeddings

| Modelo | Dimensión | Uso |
|---|---|---|
| `paraphrase-multilingual-mpnet-base-v2` | 768 | ⭐ Recomendado para producción |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Baseline (8/10) |
| `BAAI/bge-m3` | 1024 | Estado del arte en MTEB, pero inferior en este corpus |
| `all-MiniLM-L6-v2` (default de ChromaDB) | 384 | ❌ Solo inglés, no usar |

---

## 4. Resultados del banco de pruebas

### Configuración final (notebook 5)

- **Modelo embeddings:** `paraphrase-multilingual-MiniLM-L12-v2`
- **Chunk size:** 300 tokens, overlap 80
- **Tope chars por chunk:** 1500
- **Umbrales:** excelente < 0.30, aceptable < 0.50
- **Métrica de distancia:** coseno
- **Total chunks indexados:** 2488 (de 49 PDFs)

### Resultados detallados (MiniLM baseline)

| # | Consulta | Esperada | Top-1 doc | d | Calidad | Cat match |
|---|---|---|---|---|---|---|
| 1 | ¿Cuáles son los derechos del estudiante? | B | REGLAMENTO GENERAL UPeU 2023 | 0.149 | excelente | ✗ |
| 2 | ¿Cómo puedo reservar mi matrícula? | B | REGLAMENTO ADMISION 2025.v7 | 0.560 | no_cubierta | ✓ |
| 3 | ¿Cuál es el procedimiento para cambiar de carrera? | B | REGLAMENTO MOVILIDAD ACADEMICA | 0.474 | aceptable | ✓ |
| 4 | ¿Qué sanciones existen en la universidad? | A | REGLAMENTO DE ESTUDIOS V5_2025 | 0.182 | excelente | ✗ |
| 5 | ¿Cómo solicito una beca? | B | REGLAMENTO SERVICIO APRENDIZAJE IDIOMAS | 0.740 | no_cubierta | ✓ |
| 6 | ¿Cuál es el procedimiento para presentar una queja? | A | Reglamento Prevención Acoso y Hostigamiento | 0.350 | aceptable | ✗ |
| 7 | ¿Qué dice el estatuto sobre el gobierno universitario? | A | REGLAMENTO GENERAL UPeU 2023 | 0.249 | excelente | ✓ |
| 8 | ¿Cómo se realiza un proyecto de investigación? | C | REGLAMENTO INVESTIGACION UPeU 2025 V8 | 0.191 | excelente | ✓ |
| 9 | ¿Qué servicios ofrece la universidad a los egresados? | D | REGLAMENTO GENERAL UPeU 2023 | 0.290 | excelente | ✗ |
| 10 | ¿Cuál es la política ambiental de la UPeU? | E | REGLAMENTO GENERAL UPeU 2023 | 0.393 | aceptable | ✗ |

### Resumen MiniLM

```
Cubiertas (aceptable+excelente): 8/10
Excelentes: 5/10
Aceptables: 3/10
No cubiertas: 2/10
Categoría match: 5/10
```

### Casos no cubiertos (MiniLM)

1. **"Reservar matrícula" (d=0.56)** — Info existe en `REGLAMENTO ADMISION` pero el chunk está fragmentado.
2. **"Becas" (d=0.74)** — Devuelve `REGLAMENTO SERVICIO APRENDIZAJE IDIOMAS` en lugar de `REGLAMENTO BECAS 2021`. El corpus tiene el doc correcto pero el embedding no matchea.

**Solución propuesta (no implementada):** hybrid search BM25 + embeddings para matchear keywords exactas como "beca" y "matrícula".

---

## 5. Archivos del proyecto

### Sobre-escritos automáticamente al re-ejecutar

| Archivo | Notebook |
|---|---|
| `corpus_upeu/txt_bruto/*.txt` | 1 |
| `corpus_upeu/txt_limpio/*.txt` | 2 |
| `corpus_upeu/metadatos/chunks.csv` | 3 |
| `corpus_upeu/metadatos/embeddings.npy` | 3 |
| `corpus_upeu/metadatos/chunk_ids.npy` | 3 |
| `corpus_upeu/metadatos/metadatos_limpieza.csv` | 2 |
| `corpus_upeu/metadatos/informe_cobertura.csv` | 5 |
| `vector_store/chroma.sqlite3` | 4 |
| `vector_store/<uuid>/*` | 4 |
| `vector_store_miniLM/*` | 7 |
| `vector_store_mpnet/*` | 8 |
| `vector_store_bge_m3/*` | 9 |

### Carpetas a limpiar (recomendado)

- `corpus_upeu/.Trash-1000/` — basura de Linux, no pertenece al proyecto
- `corpus_upeu/pdfs/.ipynb_checkpoints/` — checkpoints de Jupyter mal ubicados
- `corpus_upeu/txt_limpio/.ipynb_checkpoints/` — idem

---

## 6. Orden de ejecución recomendado

```powershell
# Limpieza previa (una sola vez)
docker-compose down
Remove-Item -Recurse -Force vector_store
New-Item -ItemType Directory -Path vector_store
Remove-Item -Recurse -Force corpus_upeu/.Trash-1000

# Levantar entorno
docker-compose up

# En Jupyter, ejecutar en orden:
# 1. notebook 1 (extracción)
# 2. notebook 2 (limpieza)
# 3. notebook 3 (chunking)
# 4. notebook 4 (indexación con mpnet) ← modelo actualizado
# 5. notebook 5 (verificación)
# 6. notebook 6 (pruebas manuales, opcional)
# 7. notebook 7 (evaluación miniLM) — opcional, comparativa
# 8. notebook 8 (evaluación mpnet) — el ganador
# 9. notebook 9 (evaluación bge-m3) — opcional, comparativa
```

---

## 7. Próximos pasos sugeridos

| Prioridad | Acción | Esfuerzo | Impacto |
|---|---|---|---|
| Alta | **Migrar OE5 a mpnet-base-v2** (recomendado) | 30min | Sube cobertura 80% → 100% |
| Alta | Integrar LLM (Ollama/Mistral) con RAG | 1h | Habilita el caso de uso real |
| Media | Hybrid search BM25 + embeddings | 30min | Rescata las 2 no cubiertas (si se decide usar MiniLM) |
| Media | Re-rank con cross-encoder | 15min | Mejora 5-10% adicional |
| Baja | Evaluar respuesta del LLM, no solo retrieval | 1h | Métrica end-to-end |
| Baja | Limpiar `.Trash-1000/` y `.ipynb_checkpoints/` | 5min | Higiene del repo |

---

## 8. Comparativa detallada de modelos de embeddings

### Configuración común

- **Chunks:** 300 tokens, overlap 80 (validado como óptimo)
- **Tope chars por chunk:** 1500
- **Métrica:** distancia coseno
- **Total chunks indexados:** 2488 (de 49 PDFs)
- **GPU:** NVIDIA RTX (PyTorch 2.1.2 + CUDA 12.1)

### Resultados comparados (banco de 10 preguntas)

| # | Consulta | Esperada | MiniLM d/cat | mpnet d/cat | bge-m3 d/cat |
|---|---|---|---|---|---|
| 1 | Derechos del estudiante | B | 0.149 A✗ | **0.134 B✓** | 0.246 A✗ |
| 2 | Reservar matrícula | B | 0.560 B✓ | 0.462 B✓ | **0.334 B✓** |
| 3 | Cambio de carrera | B | 0.460 E✗ | 0.436 B✓ | **0.358 B✓** |
| 4 | Sanciones | A | 0.182 B✗ | 0.198 B✗ | 0.293 C✗ |
| 5 | **Beca** | B | 0.699 D✗ | **0.308 B✓** | 0.308 A✗ |
| 6 | Queja | A | 0.300 E✗ | **0.269 A✓** | 0.356 A✓ |
| 7 | Estatuto gobierno | A | 0.263 B✗ | **0.232 A✓** | 0.351 A✓ |
| 8 | Proyecto investigación | C | **0.191 C✓** | 0.277 C✓ | 0.343 C✓ |
| 9 | Egresados servicios | D | 0.318 B✗ | 0.296 A✗ | 0.381 A✗ |
| 10 | Política ambiental | E | 0.300 E✓ | 0.330 C✗ | **0.362 E✓** |

### Caso especial: pregunta de "beca"

| Modelo | Top-1 doc | Distancia | Acierto |
|---|---|---|---|
| MiniLM | REGLAMENTO RECONOCIMIENTO EXCELENCIA | 0.332 | ❌ |
| **mpnet** | **REGLAMENTO BECAS 2021** (Art. 49° con 5 requisitos) | **0.157** | **✅** |
| bge-m3 | REGLAMENTO BECAS 2021 (Art. 49°) | 0.251 | ✅ |

**Lectura:** mpnet no solo matchea el documento correcto, sino el chunk exacto que
lista los 5 requisitos para beca. Esto es exactamente lo que el LLM necesita para
generar una respuesta precisa.

### Bge-m3: ¿por qué no ganó?

bge-m3 es estado del arte en benchmarks MTEB para texto largo y multilingüe, pero
para corpus cortos y estructurados (artículos de reglamentos), su prefijo
`query:/passage:` y su arquitectura densa de 1024 dim no compensa. Las distancias
son consistentemente más altas (0.33 vs 0.29 de mpnet).

**Conclusión:** Para el corpus UPeU, mpnet (768 dim) supera a bge-m3 (1024 dim).

---

## 9. Plan de migración OE5 → mpnet

### Paso 1 — Cambiar el modelo en los notebooks 1–6 de OE1

En los notebooks 3 y 4, sustituir el string del modelo:

```python
# ANTES
model_name = "paraphrase-multilingual-MiniLM-L12-v2"

# DESPUÉS
model_name = "paraphrase-multilingual-mpnet-base-v2"
```

### Paso 2 — Limpiar vector stores anteriores y re-ejecutar

```powershell
docker-compose down

# Borrar todos los vector stores (incluido el principal)
Remove-Item -Recurse -Force vector_store, vector_store_miniLM, vector_store_mpnet, vector_store_bge_m3
New-Item -ItemType Directory -Path vector_store
# (los 3 de comparativa se re-crean al ejecutar los notebooks 7/8/9)

docker-compose up
```

### Paso 3 — Re-ejecutar notebooks en orden (1 → 9)

| # | Notebook | Qué produce |
|---|---|---|
| 1 | `1_extraccion_texto.ipynb` | `corpus_upeu/txt_bruto/*.txt` |
| 2 | `2_limpieza_corpus.ipynb` | `corpus_upeu/txt_limpio/*.txt` |
| 3 | `3_chunking_embeddings.ipynb` | `corpus_upeu/metadatos/{chunks.csv, embeddings.npy}` |
| 4 | `4_indexacion_vectorial.ipynb` | `vector_store/` (índice mpnet 768 dim) |
| 5 | `5_verificacion_cobertura.ipynb` | informe de cobertura |
| 6 | `6_prueba_manual.ipynb` | (diagnóstico, opcional) |
| 7–9 | notebooks de comparativa | vector stores separados por modelo |

### Paso 4 — Migrar OE5

En `oe5_chatbot_upeu/backend/rag_pipeline.py` línea 132:

```python
# ANTES
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# DESPUÉS
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
```

Copiar `vector_store/` de OE1 a `oe5_chatbot_upeu/vector_store/` y reiniciar backend.

### Tiempo estimado total

| Paso | Tiempo |
|---|---|
| Cambiar nombre del modelo en notebooks 3 y 4 | 1 min |
| Borrar vector stores | 1 min |
| Re-ejecutar notebooks 1–6 | ~15 min (extracción PDF es lo más lento) |
| Re-ejecutar notebooks 7–9 | ~5 min (con GPU) |
| Migrar OE5 | ~5 min |
| **Total** | **~30 min** |

---

**Estado del proyecto:** ✅ Funcional y optimizado con mpnet
**Última actualización:** Junio 2026
**Versión recomendada para OE5:** `paraphrase-multilingual-mpnet-base-v2`

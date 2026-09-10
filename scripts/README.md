# Scripts de re-chunking y re-evaluación (2026-08-21)

Pipeline reproducible que corrige el bug de fusión de artículos
detectado en `notebooks/3_chunking_embeddings.ipynb`.

## Bug corregido

`chunk_text_estructural` (nb 3) fusionaba artículos pequeños para
llenar `max_tokens=300`, pero **conservaba solo la primera cabecera**:
los artículos siguientes aportaban cuerpo sin header. Además
`Subcapítulo` no era punto de corte. Consecuencias:

- Metadata `articulo` errónea (ej. contenido de Art. 52/53 bajo
  header "Artículo 51°" en REGLAMENTO ADMISION 2025.v7).
- Embeddings diluidos: el dato quedaba como cola irrelevante.
- 563/3886 chunks (14.5%) con 2+ cabeceras; el chunk correcto ni
  entraba al top-40 → M04 falsos (caso "nota mínima admisión").

## Scripts (orden de ejecución)

| # | Script | Equivale a | Salida |
|---|---|---|---|
| 1 | `rechunk_fixed.py` | nb 3 (fix) | `corpus_upeu/metadatos/chunks_fixed.csv` |
| 2 | `reembed_index.py <modelo> <coleccion> [batch]` | nb 4 + embeddings | `vector_store*/` + `embeddings_fixed_<modelo>.npy` |
| 3 | `evaluar_modelos.py` | nb 7/8/9 | veredicto en consola |

Fix aplicado en (1): un artículo = un chunk (sin fusionar),
`Subcapítulo` como punto de corte, preámbulo conservado,
cabecera siempre presente (`Artículo N°` + cuerpo).

## Resultado

- 3886 → **6259 chunks** (sin duplicados del buffer original).
- Re-comparación 10Q: **mpnet sigue ganando**
  (10/10 cobertura, 7/10 excelentes, 8/10 cat-match).
- Caso Art. 53 Admisión: fuera del top-40 → **rank #2**.
- Cobertura banco 17Q en umbral 0.40: 94.1% → **100%**.

## Nota sobre lotes ChromaDB

`chromadb==0.4.22` corrompe el índice HNSW con vectores de
dim 1024 (bge-m3) en lotes de 500 (`Index with capacity 100...`).
Usar lotes ≤100 para bge-m3 (`reembed_index.py bgem3 <col> 100`).

# Guía de Migración V2 → OE5 Chatbot

**Objetivo:** Reemplazar el índice vectorial del chatbot OE5 por la versión V2 (chunking semántico, mpnet 768, taxonomía unificada).

---

## 📦 Qué se migra

| Componente | Origen (OE1 V2) | Destino (OE5) |
|---|---|---|
| Base vectorial | `corpus_upeu/vector_store_v2/` | `vector_store/` |
| Colección ChromaDB | `corpus_upeu_v2` | `corpus_upeu` (misma) |
| Embeddings | mpnet 768-dim (ya compatible) | Sin cambios |
| Taxonomía categorías | A-E unificada (becas=D) | **Actualizar `config.py`** |

---

## ⚡ Migración Rápida (2 minutos)

### 1. Backup índice actual OE5
```bash
cd oe5_chatbot_upeu
mv vector_store vector_store_backup_$(date +%Y%m%d)
```

### 2. Copiar índice V2
```bash
# Desde raíz del proyecto
cp -r ../oe1_arquitectura_corpus/corpus_upeu/vector_store_v2 vector_store
```

### 3. Actualizar taxonomía en OE5 (`backend/config.py`)
```python
# ANTES (V1 - becas en B)
MAPEO_CATEGORIAS = {
    "D": {
        "nombre": "Bienestar estudiantil",
        "descripcion": "Reglamento del estudiante, becas, residencias...",  # ❌ becas mencionada pero en B
    },
    "B": {
        "nombre": "Académico y estudios",
        "descripcion": "...becas, admision, credito, matricula, egresado",  # ❌ becas aquí
    },
}

# DESPUÉS (V2 - becas en D - unificado)
MAPEO_CATEGORIAS = {
    "D": {
        "nombre": "Bienestar estudiantil",
        "descripcion": "Reglamento del estudiante, becas, residencias universitarias, defensoría y reconocimientos.",
    },
    "B": {
        "nombre": "Académico y estudios",
        "descripcion": "Reglamentos de estudios, admisión, grados y títulos, idiomas, movilidad académica, publicaciones y fondo, pago servicios académicos.",
    },
}
```
> **Nota:** La descripción de B ya no incluye "becas". La categoría D ya la tenía.

### 4. Verificar embedding model (sin cambios)
```python
# backend/rag_pipeline.py - YA USA mpnet
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
# ✅ No tocar, ya es el correcto
```

### 5. Reiniciar backend OE5
```bash
docker-compose restart backend
# o
docker-compose up -d --build backend
```

### 6. Test de verificación
```bash
curl -X POST http://localhost:8000/consulta \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Cómo solicito una beca en la UPeU?"}'
```

**Respuesta esperada:**
```json
{
  "respuesta": "...",
  "fuentes": [
    {
      "documento": "REGLAMENTO BECAS 2021 ACTUALIZADO",
      "articulo": "Artículo 49°",
      "categoria": "D",
      "distancia": 0.245
    }
  ]
}
```
- ✅ Distancia ~0.24 (excelente < 0.30)
- ✅ Categoría D (Bienestar) 
- ✅ Artículo 49° con 5 requisitos

---

## 🔍 Verificación Post-Migración

| Check | Comando | Esperado |
|---|---|---|
| Colección existe | `docker exec oe5_backend python -c "import chromadb; c=chromadb.PersistentClient('/data/vector_store'); print(c.get_collection('corpus_upeu').count())"` | `5126` |
| Embedding dim | `docker exec oe5_backend python -c "import chromadb; c=chromadb.PersistentClient('/data/vector_store'); col=c.get_collection('corpus_upeu'); r=col.get(include=['embeddings'], limit=1); print(len(r['embeddings'][0]))"` | `768` |
| Metadatos padre-hijo | `docker exec oe5_backend python -c "import chromadb; c=chromadb.PersistentClient('/data/vector_store'); col=c.get_collection('corpus_upeu'); r=col.get(include=['metadatas'], limit=10); print([m.get('padre_chunk_id') for m in r['metadatas']])"` | Algunos no vacíos |
| Categoría becas | Test pregunta "beca" → fuente con `categoria: "D"` | ✅ |

---

## 🔄 Rollback (si algo falla)

```bash
cd oe5_chatbot_upeu
rm -rf vector_store
mv vector_store_backup_YYYYMMDD vector_store
docker-compose restart backend
```

---

## 📝 Cambios en OE5 (resumen para commit)

```
OE5 Migration to Corpus V2:
- vector_store/ → replaced with V2 (5126 chunks, semantic chunking)
- backend/config.py → MAPEO_CATEGORIAS updated (becas=D unified)
- Embedding model unchanged (mpnet 768 already in use)
- Tested: "¿Cómo solicito una beca?" → Art. 49°, dist=0.245, cat=D
```

---

## ⚠️ Notas Importantes

1. **No regenerar embeddings** — OE5 ya usa `paraphrase-multilingual-mpnet-base-v2` (igual que V2)
2. **Colección ChromaDB** — misma nombre (`corpus_upeu`), solo reemplazar archivos
3. **Taxonomía** — ÚNICO cambio en código OE5 es `config.py` (becas D unificada)
4. **Metadatos nuevos** — OE5 verá campos extra (`padre_chunk_id`, `es_parte`, `tipo_bloque`, etc.) que puede ignorar o usar para re-ranking futuro

---

## 🆘 Troubleshooting

| Problema | Causa | Solución |
|---|---|---|
| `Collection corpus_upeu not found` | Carpeta copiada mal | Verificar `vector_store/chroma.sqlite3` existe |
| Distancia > 0.5 en test | Embedding model distinto | Verificar `rag_pipeline.py` usa mpnet |
| Categoría muestra B en becas | `config.py` no actualizado | Editar `MAPEO_CATEGORIAS` y reiniciar |
| Error `padre_chunk_id` en logs | OE5 no espera ese campo | Ignorar, es metadata extra V2 |
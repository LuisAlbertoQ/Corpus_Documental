#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_indexacion_v2.py — Indexación en ChromaDB persistente
Lee: chunks_v2.csv, embeddings_v2.npy, chunk_ids_v2.npy
Escribe: corpus_upeu/vector_store_v2/ (colección corpus_upeu_v2)
"""

import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
META_FOLDER = BASE / "corpus_upeu" / "metadatos"
CHUNKS_CSV = META_FOLDER / "chunks_v2.csv"
EMBEDDINGS_NPY = META_FOLDER / "embeddings_v2.npy"
CHUNK_IDS_NPY = META_FOLDER / "chunk_ids_v2.npy"
VECTOR_STORE = BASE / "corpus_upeu" / "vector_store_v2"
COLLECTION_NAME = "corpus_upeu_v2"

VECTOR_STORE.mkdir(parents=True, exist_ok=True)

# Cargar datos
print("Cargando datos...")
df = pd.read_csv(CHUNKS_CSV)
embeddings = np.load(EMBEDDINGS_NPY)
chunk_ids = np.load(CHUNK_IDS_NPY, allow_pickle=True)

print(f"Chunks: {len(df)} | Embeddings: {embeddings.shape} | IDs: {len(chunk_ids)}")

# Verificaciones
assert len(df) == embeddings.shape[0] == len(chunk_ids), "Inconsistencia en datos"
assert (df["chunk_id"] == chunk_ids).all(), "Chunk IDs no coinciden"

# Cliente ChromaDB persistente
client = chromadb.PersistentClient(
    path=str(VECTOR_STORE),
    settings=Settings(anonymized_telemetry=False, allow_reset=True)
)

# Eliminar colección si existe
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Colección previa '{COLLECTION_NAME}' eliminada.")
except Exception:
    pass

# Crear colección con espacio coseno
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Corpus UPeU v2 - chunking semántico mpnet 768", "hnsw:space": "cosine"}
)
print(f"Colección '{COLLECTION_NAME}' creada (cosine).")

# Preparar listas para inserción
ids = df["chunk_id"].astype(str).tolist()
documentos = df["texto"].tolist()
metadatos = []
for _, row in df.iterrows():
    meta = {
        "documento": row["documento"],
        "chunk_id": str(row["chunk_id"]),
        "tipo_documental": row["tipo_documental"],
        "categoria_tematica": row["categoria_tematica"],
        "ruta_extraccion": row["ruta_extraccion"],
        "titulo_documento": row["titulo_documento"],
        "capitulo": row["capitulo"],
        "seccion": row["seccion"],
        "articulo": row["articulo"],
        "tipo_bloque": row["tipo_bloque"],
        "padre_chunk_id": str(row["padre_chunk_id"]) if row["padre_chunk_id"] else "",
        "es_parte": bool(row["es_parte"]),
        "parte_num": int(row["parte_num"]) if not pd.isna(row["parte_num"]) else 0,
        "total_partes": int(row["total_partes"]) if not pd.isna(row["total_partes"]) else 1,
        "num_tokens": int(row["num_tokens"]),
        "num_chars": int(row["num_chars"]),
    }
    metadatos.append(meta)

embeddings_list = embeddings.tolist()

# Insertar por lotes
BATCH_SIZE = 500
total = len(ids)
print(f"Insertando {total} chunks en lotes de {BATCH_SIZE}...")

for i in tqdm(range(0, total, BATCH_SIZE), desc="Indexando"):
    end = min(i + BATCH_SIZE, total)
    collection.add(
        ids=ids[i:end],
        documents=documentos[i:end],
        metadatas=metadatos[i:end],
        embeddings=embeddings_list[i:end]
    )

# Verificación final
count = collection.count()
print(f"\n✅ Indexación completada.")
print(f"   Documentos en colección: {count}")
print(f"   Ubicación: {VECTOR_STORE}")

# Test rápido
print("\n--- Test de recuperación ---")
test_query = "¿Cómo solicito una beca en la UPeU?"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
q_emb = model.encode([test_query], normalize_embeddings=True)[0].tolist()

results = collection.query(
    query_embeddings=[q_emb],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
    print(f"\n  Resultado {i+1} (dist={dist:.4f}):")
    print(f"    Doc: {meta['documento']}")
    print(f"    Art: {meta['articulo']}")
    print(f"    Cat: {meta['categoria_tematica']}")
    print(f"    Texto: {doc[:200]}...")
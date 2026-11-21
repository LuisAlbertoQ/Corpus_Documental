#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_indexacion_v2.py — Indexa chunks + embeddings en ChromaDB.

Salida: corpus_upeu/vector_store_<version>/  colección corpus_upeu_<version>

Cambios V3 respecto a V2:
  * F7  No borra nada por defecto: si la colección ya existe, aborta.
        `--force` la reemplaza; la V2 sigue protegida por config.asegurar_version_escribible().
  * Parámetros HNSW explícitos (search_ef=100 > top_k de OE5; V2 usaba el default 10).
  * Metadatos nuevos: nombre_documento, grupo_articulo_id, pagina, char_inicio/char_fin.
    Se mantienen los de V2 (documento, articulo, categoria_tematica, titulo_documento…)
    para que OE5 siga funcionando sin cambios.
  * Corrige el "nan" que V2 guardaba como padre_chunk_id.
  * El texto guardado como `document` es el texto limpio (sin breadcrumb) que verá el LLM.
"""

import argparse
import json

import chromadb
import numpy as np
from chromadb.config import Settings
from tqdm import tqdm

import config as C
from chunking import leer_chunks

CAMPOS_TEXTO = ["documento", "nombre_documento", "chunk_id", "tipo_documental", "categoria_tematica",
                "ruta_extraccion", "titulo_documento", "titulo_estructura", "capitulo", "seccion",
                "articulo", "pagina", "pagina_fin", "tipo_bloque", "grupo_articulo_id", "padre_chunk_id"]
CAMPOS_INT = ["parte_num", "total_partes", "num_tokens", "num_tokens_embedding", "num_chars",
              "char_inicio", "char_fin"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="reemplazar la colección si ya existe")
    args = parser.parse_args()

    C.asegurar_version_escribible()
    df = leer_chunks(C.CHUNKS_CSV)
    embeddings = np.load(C.EMBEDDINGS_NPY)
    chunk_ids = np.load(C.CHUNK_IDS_NPY, allow_pickle=True)
    print(f"Chunks: {len(df)} | Embeddings: {embeddings.shape} | IDs: {len(chunk_ids)}")
    assert len(df) == embeddings.shape[0] == len(chunk_ids), "Inconsistencia en datos"
    assert (df["chunk_id"].values == chunk_ids).all(), "Chunk IDs no coinciden"
    assert embeddings.shape[1] == C.EMBEDDING_DIM, f"Dimensión {embeddings.shape[1]} != {C.EMBEDDING_DIM}"

    C.VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(C.VSTORE_DIR),
                                       settings=Settings(anonymized_telemetry=False, allow_reset=True))
    existentes = [c.name for c in client.list_collections()]
    if C.COLLECTION_NAME in existentes:
        if not args.force:
            raise SystemExit(f"❌ La colección '{C.COLLECTION_NAME}' ya existe en {C.VSTORE_DIR}. "
                             "Usa --force para reemplazarla (o cambia CORPUS_VERSION).")
        client.delete_collection(name=C.COLLECTION_NAME)
        print(f"Colección previa '{C.COLLECTION_NAME}' eliminada (--force).")

    collection = client.create_collection(
        name=C.COLLECTION_NAME,
        metadata={"description": f"Corpus UPeU {C.VERSION} - {C.CHUNK_MODE} - mpnet max_seq {C.MAX_SEQ_LENGTH}",
                  **C.HNSW_PARAMS},
    )
    print(f"Colección '{C.COLLECTION_NAME}' creada con {C.HNSW_PARAMS}")

    metadatos = []
    for _, row in df.iterrows():
        meta = {c: str(row[c]) for c in CAMPOS_TEXTO if c in df.columns}
        meta.update({c: int(row[c]) for c in CAMPOS_INT if c in df.columns})
        meta["es_parte"] = bool(row["es_parte"])
        metadatos.append(meta)

    ids, docs, embs = df["chunk_id"].tolist(), df["texto"].tolist(), embeddings.tolist()
    for i in tqdm(range(0, len(ids), C.INDEX_BATCH_SIZE), desc="Indexando"):
        j = i + C.INDEX_BATCH_SIZE
        collection.add(ids=ids[i:j], documents=docs[i:j], metadatas=metadatos[i:j], embeddings=embs[i:j])
    count = collection.count()
    print(f"\n✅ {count} chunks indexados en {C.VSTORE_DIR}")

    if C.MANIFEST_JSON.exists():
        manifest = json.loads(C.MANIFEST_JSON.read_text(encoding="utf-8"))
        manifest["indice"] = {"vector_store": str(C.VSTORE_DIR.relative_to(C.BASE)),
                              "collection": C.COLLECTION_NAME, "count": count, "hnsw": C.HNSW_PARAMS}
        C.MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Prueba rápida (el modelo debe ser el mismo y con la misma revisión)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(C.MODEL_NAME, revision=C.MODEL_REVISION)
    for q in ["¿Cómo solicito una beca en la UPeU?", "¿Cuánto cuesta el trámite de pre-ingreso?"]:
        r = collection.query(query_embeddings=[model.encode([C.PREFIJO_PREGUNTA + q], normalize_embeddings=True)[0].tolist()],
                             n_results=3, include=["metadatas", "distances"])
        print(f"\n  {q}")
        for meta, dist in zip(r["metadatas"][0], r["distances"][0]):
            print(f"    d={dist:.4f}  {meta['nombre_documento']} · {meta['articulo'] or meta['tipo_bloque']} · cat {meta['categoria_tematica']}")


if __name__ == "__main__":
    main()

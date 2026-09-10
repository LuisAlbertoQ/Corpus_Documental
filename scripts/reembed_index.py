# -*- coding: utf-8 -*-
"""Re-embeddings + reindexación con chunks corregidos (staging).

Replica notebooks/4_indexacion_vectorial.ipynb (categorías, artículo,
lotes de 500, hnsw cosine) + embeddings normalizados (como nb 8).

Uso:
    docker run --rm --gpus all \
      -v "<OE1>/corpus_upeu:/work/corpus_upeu" \
      -v "<OE1>/scripts:/work/scripts" \
      -v "<HF_CACHE>:/root/.cache/huggingface" \
      -v "<OE1>/vector_store_<modelo>:/work/out_store" \
      oe5_chatbot_upeu-backend \
      python /work/scripts/reembed_index.py <modelo> <coleccion>

Modelos: minilm | mpnet | bgem3
"""

import csv
import re
import sys
from pathlib import Path

WORK = Path("/work/corpus_upeu")
CHUNKS_CSV = WORK / "metadatos" / "chunks_fixed.csv"
OUT_STORE = Path("/work/out_store")
OUT_NPY = WORK / "metadatos" / "embeddings_fixed.npy"

MODELOS = {
    "minilm": "paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet": "paraphrase-multilingual-mpnet-base-v2",
    "bgem3": "BAAI/bge-m3",
}

CATEGORIAS_KEYWORDS = {
    "A": ["estatuto", "general upeu", "defensor", "comite electoral",
          "tupa", "reglamento interno de trabajo", "honores"],
    "B": ["estudios", "estudiante unionista", "docencia", "idiomas",
          "movilidad", "publicaciones y fondo", "pago servicios academicos",
          "becas", "admision", "credito", "matricula", "egresado"],
    "C": ["investigacion", "investigadores", "incentivos investigacion",
          "propiedad intelectual", "codigo etica investigacion", "etica"],
    "D": ["promocion", "recreacion", "deporte", "residencias",
          "multimedia", "seguimiento de egresados", "servicio psicologico"],
    "E": ["politica institucional", "politica-ambiental", "ambiental",
          "capacitacion docente", "auditoria interna", "seguridad y salud",
          "comedor", "transporte", "biblioteca"],
}


def obtener_categoria(doc_name):
    nombre = doc_name.lower()
    for cat, kws in CATEGORIAS_KEYWORDS.items():
        for kw in kws:
            if kw in nombre:
                return cat
    return "E"


_RE_ART = re.compile(
    r'(Art[íi]culo\s+\d+[ºo°]?(?:\s*[.-]\s*\d+)?|'
    r'Cap[íi]tulo\s+[IVXLCDM\d]+|'
    r'Subcap[íi]tulo\s+[IVXLCDM\d]+|'
    r'Secci[óo]n\s+\d+|'
    r'T[íi]tulo\s+[IVXLCDM\d]+)',
    re.IGNORECASE
)


def extraer_articulo(texto):
    m = _RE_ART.search((texto or "")[:200])
    return m.group(1).strip() if m else ""


def main():
    import numpy as np
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    modelo_key, coleccion = sys.argv[1], sys.argv[2]
    # BATCH configurable: chromadb 0.4.x corrompe el índice HNSW con
    # vectores grandes (bge-m3, dim 1024) en lotes de 500. Con 100 va bien.
    BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    model_name = MODELOS[modelo_key]

    print(f"Modelo: {model_name}")
    model = SentenceTransformer(model_name, device="cuda")

    rows = list(csv.DictReader(
        open(CHUNKS_CSV, encoding="utf-8")))
    print(f"Chunks: {len(rows)}")
    textos = [r["texto"] for r in rows]

    print("Generando embeddings...")
    embeddings = model.encode(
        textos, batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True)
    print("Shape:", embeddings.shape)
    np.save(str(OUT_NPY).replace("fixed", f"fixed_{modelo_key}"),
            embeddings.astype("float32"))

    metadatas = [{
        "documento": r["documento"],
        "categoria": obtener_categoria(r["documento"]),
        "articulo": extraer_articulo(r["texto"]),
        "chunk_id": r["chunk_id"],
        "num_chars": len(r["texto"]),
    } for r in rows]

    OUT_STORE.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(OUT_STORE),
        settings=Settings(anonymized_telemetry=False, allow_reset=True))
    try:
        client.delete_collection(name=coleccion)
        print(f"Colección '{coleccion}' eliminada (previa).")
    except Exception:
        pass
    collection = client.create_collection(
        name=coleccion,
        metadata={"description": f"Corpus UPeU {model_name} (chunks fijos)",
                  "hnsw:space": "cosine"})

    ids = [r["chunk_id"] for r in rows]
    print(f"Lotes de {BATCH}...")
    for i in range(0, len(ids), BATCH):
        end = min(i + BATCH, len(ids))
        collection.add(
            ids=ids[i:end], documents=textos[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end].tolist())
    print(f"Indexación completa: {collection.count()} chunks "
          f"en {OUT_STORE} / {coleccion}")


if __name__ == "__main__":
    main()

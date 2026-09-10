# -*- coding: utf-8 -*-
"""Reconstrucción LIMPIA de los 3 vector stores desde embeddings guardados.

Motivo: chromadb 0.4.x no purga la tabla `embeddings` al borrar
colecciones → los sqlite acumularon filas muertas de 3 generaciones
(4152 + 2558 + 6259) y superaron el límite GitHub de 100MB.

Estrategia: borrar cada store y recrearlo desde cero con los
embeddings ya calculados (sin re-encodear). Rápido (~1 min c/u).

Uso:
    docker run --rm \
      -v "<OE1>/corpus_upeu:/work/corpus_upeu" \
      -v "<OE1>/scripts:/work/scripts" \
      -v "<OE1>/vector_store_mpnet:/work/s_mpnet" \
      -v "<OE1>/vector_store_miniLM:/work/s_minilm" \
      -v "<OE1>/vector_store_bge_m3:/work/s_bgem3" \
      oe5_chatbot_upeu-backend \
      python /work/scripts/rebuild_clean.py
"""

import csv
import re
import shutil
import sys
from pathlib import Path

WORK = Path("/work/corpus_upeu")
CHUNKS_CSV = WORK / "metadatos" / "chunks_fixed.csv"

TARGETS = [
    # (nombre, npy, dir salida, colección, batch)
    ("mpnet", WORK / "metadatos" / "embeddings_fixed_mpnet.npy",
     Path("/work/s_mpnet"), "corpus_upeu_mpnet", 500),
    ("minilm", WORK / "metadatos" / "embeddings_fixed_minilm.npy",
     Path("/work/s_minilm"), "corpus_upeu_minilm", 500),
    ("bgem3", WORK / "metadatos" / "embeddings_fixed_bgem3.npy",
     Path("/work/s_bgem3"), "corpus_upeu_bgem3", 100),
]

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
    re.IGNORECASE)


def extraer_articulo(texto):
    m = _RE_ART.search((texto or "")[:200])
    return m.group(1).strip() if m else ""


def main():
    import numpy as np
    import chromadb
    from chromadb.config import Settings

    rows = list(csv.DictReader(open(CHUNKS_CSV, encoding="utf-8")))
    textos = [r["texto"] for r in rows]
    ids = [r["chunk_id"] for r in rows]
    metadatas = [{
        "documento": r["documento"],
        "categoria": obtener_categoria(r["documento"]),
        "articulo": extraer_articulo(r["texto"]),
        "chunk_id": r["chunk_id"],
        "num_chars": len(r["texto"]),
    } for r in rows]
    print(f"Chunks: {len(rows)}")

    for nombre, npy_path, out_dir, coleccion, batch in TARGETS:
        print(f"\n=== {nombre} -> {out_dir} ===")
        embeddings = np.load(str(npy_path))
        assert len(embeddings) == len(rows), (len(embeddings), len(rows))

        if out_dir.exists():
            # Vaciar contenidos (no el dir: puede ser un mount busy)
            for child in out_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=str(out_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True))
        collection = client.create_collection(
            name=coleccion,
            metadata={"description": f"Corpus UPeU v2 ({nombre})",
                      "hnsw:space": "cosine"})
        for i in range(0, len(ids), batch):
            end = min(i + batch, len(ids))
            collection.add(
                ids=ids[i:end], documents=textos[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end].tolist())
        print(f"OK: {collection.count()} chunks")

    print("\nReconstrucción limpia completa.")


if __name__ == "__main__":
    main()

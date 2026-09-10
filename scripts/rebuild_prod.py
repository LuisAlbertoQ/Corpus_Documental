# -*- coding: utf-8 -*-
"""Rebuild limpio del store de PRODUCCIÓN (colección corpus_upeu, mpnet).

Genera un sqlite sin filas muertas (<100MB, apto para git).
"""
import csv
import re
import sys
from pathlib import Path

WORK = Path("/work/corpus_upeu")
CHUNKS_CSV = WORK / "metadatos" / "chunks_fixed.csv"
NPY = WORK / "metadatos" / "embeddings_fixed_mpnet.npy"
OUT = Path("/work/out_prod")

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


def main():
    import numpy as np
    import chromadb
    from chromadb.config import Settings

    rows = list(csv.DictReader(open(CHUNKS_CSV, encoding="utf-8")))
    embeddings = np.load(str(NPY))
    assert len(embeddings) == len(rows)

    for child in OUT.iterdir():
        import shutil
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    client = chromadb.PersistentClient(
        path=str(OUT), settings=Settings(anonymized_telemetry=False,
                                         allow_reset=True))
    col = client.create_collection(
        name="corpus_upeu",
        metadata={"description": "Corpus UPeU v2 prod (mpnet)",
                  "hnsw:space": "cosine"})
    BATCH = 500
    for i in range(0, len(rows), BATCH):
        end = min(i + BATCH, len(rows))
        col.add(
            ids=[r["chunk_id"] for r in rows[i:end]],
            documents=[r["texto"] for r in rows[i:end]],
            metadatas=[{
                "documento": r["documento"],
                "categoria": obtener_categoria(r["documento"]),
                "articulo": (_RE_ART.search((r["texto"] or "")[:200]).group(1).strip()
                             if _RE_ART.search((r["texto"] or "")[:200]) else ""),
                "chunk_id": r["chunk_id"],
                "num_chars": len(r["texto"]),
            } for r in rows[i:end]],
            embeddings=embeddings[i:end].tolist())
    print(f"OK prod: {col.count()} chunks")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Re-comparación de los 3 modelos sobre chunks corregidos.

Replica la evaluación de notebooks 7/8/9 (mismo banco 10Q, mismos
umbrales excelente=0.30 / aceptable=0.50, n=3).

Uso:
    docker run --rm --gpus all \
      -v "<OE1>:/work/oe1" \
      -v "<HF_CACHE>:/root/.cache/huggingface" \
      oe5_chatbot_upeu-backend \
      python /work/oe1/scripts/evaluar_modelos.py
"""

import sys
from pathlib import Path

BASE = Path("/work/oe1")
STORES = {
    # (store, colección EXPLÍCITA, modelo) — no usar cols[0]: puede haber
    # colecciones stale de corridas previas.
    "minilm": (BASE / "vector_store_miniLM", "corpus_upeu_minilm",
               "paraphrase-multilingual-MiniLM-L12-v2"),
    "mpnet": (BASE / "vector_store_mpnet", "corpus_upeu_mpnet",
              "paraphrase-multilingual-mpnet-base-v2"),
    "bgem3": (BASE / "vector_store_bge_m3", "corpus_upeu_bgem3",
              "BAAI/bge-m3"),
}

PREGUNTAS = [
    ("¿Cuáles son los derechos del estudiante?", "B"),
    ("¿Cómo puedo reservar mi matrícula?", "B"),
    ("¿Cuál es el procedimiento para cambiar de carrera?", "B"),
    ("¿Qué sanciones existen en la universidad?", "A"),
    ("¿Cómo solicito una beca?", "B"),
    ("¿Cuál es el procedimiento para presentar una queja?", "A"),
    ("¿Qué dice el estatuto sobre el gobierno universitario?", "A"),
    ("¿Cómo se realiza un proyecto de investigación?", "C"),
    ("¿Qué servicios ofrece la universidad a los egresados?", "D"),
    ("¿Cuál es la política ambiental de la UPeU?", "E"),
]


def evaluar(model, collection, umbral_exc=0.30, umbral_ace=0.50, n=3):
    res = []
    for pregunta, cat_esp in PREGUNTAS:
        emb = model.encode([pregunta], normalize_embeddings=True)[0].tolist()
        r = collection.query(
            query_embeddings=[emb], n_results=n,
            include=["documents", "metadatas", "distances"])
        metas = r["metadatas"][0] or []
        dists = r["distances"][0] or [1.0]
        d1 = dists[0]
        if d1 < umbral_exc:
            calidad, cubierta = "excelente", True
        elif d1 < umbral_ace:
            calidad, cubierta = "aceptable", True
        else:
            calidad, cubierta = "no_cubierta", False
        match = (metas[0].get("categoria") == cat_esp) if metas else None
        res.append({
            "pregunta": pregunta, "cubierta": cubierta, "calidad": calidad,
            "top1_dist": round(d1, 4),
            "top1_doc": metas[0].get("documento", "?") if metas else "?",
            "top1_cat": metas[0].get("categoria", "?") if metas else "?",
            "cat_esp": cat_esp, "match": match,
        })
    return res


def main():
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    print("=" * 78)
    print("RE-COMPARACIÓN DE MODELOS SOBRE CHUNKS CORREGIDOS (6259 chunks)")
    print("=" * 78)

    resumen = []
    for key, (store, coleccion_nombre, model_name) in STORES.items():
        print(f"\n--- {key}: {model_name} ---")
        model = SentenceTransformer(model_name, device="cuda")
        client = chromadb.PersistentClient(
            path=str(store),
            settings=Settings(anonymized_telemetry=False))
        collection = client.get_collection(coleccion_nombre)
        print(f"colección: {coleccion_nombre} ({collection.count()} chunks)")

        res = evaluar(model, collection)
        for x in res:
            e = "OK" if x["cubierta"] else "XX"
            cm = "✓" if x["match"] else ("✗" if x["match"] is False else "?")
            print(f"  {e} [{x['calidad']:11}] d={x['top1_dist']:.3f} "
                  f"cat={x['top1_cat']}{cm} {x['top1_doc'][:38]}")

        n_cub = sum(1 for x in res if x["cubierta"])
        n_exc = sum(1 for x in res if x["calidad"] == "excelente")
        n_match = sum(1 for x in res if x["match"])
        resumen.append((key, model_name, n_cub, n_exc, n_match))
        try:
            del model
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("\n" + "=" * 78)
    print("VEREDICTO")
    print("=" * 78)
    for key, name, cub, exc, match in resumen:
        print(f"  {key:8s} cobertura {cub}/10 | excelentes {exc}/10 | "
              f"cat-match {match}/10   ({name})")

    mejor = max(resumen, key=lambda t: (t[2], t[3], t[4]))
    print(f"\nGANADOR: {mejor[0]} ({mejor[1]})")


if __name__ == "__main__":
    main()

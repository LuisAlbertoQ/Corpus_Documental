#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_indice_bm25.py — Corpus léxico para búsqueda híbrida (semántica + BM25).

Salida: metadatos/<version>/bm25_corpus_<version>.json
    {"version", "tokenizador", "campo", "ids": [...], "tokens": [[...], ...]}

Se guardan los tokens (no un pickle del modelo) para que OE5 construya
rank_bm25.BM25Okapi al arrancar sin depender de versiones de pickle.
Se indexa `texto_embedding` (breadcrumb + texto): así "beca" también
encuentra los artículos del REGLAMENTO BECAS aunque el artículo no repita la palabra.
"""

import json

import config as C
from chunking import leer_chunks
from texto_lexico import tokenizar, VERSION_TOKENIZADOR


def main():
    C.asegurar_version_escribible()
    df = leer_chunks(C.CHUNKS_CSV)
    tokens = [tokenizar(t) for t in df["texto_embedding"]]
    data = {
        "version": C.VERSION,
        "tokenizador": VERSION_TOKENIZADOR,
        "campo": "texto_embedding",
        "ids": df["chunk_id"].tolist(),
        "tokens": tokens,
    }
    C.BM25_JSON.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    vacios = sum(1 for t in tokens if not t)
    print(f"📄 {len(tokens)} documentos léxicos → {C.BM25_JSON} (vacíos: {vacios})")

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 no instalado: se omite la prueba (pip install rank-bm25)")
        return
    bm25 = BM25Okapi(tokens)
    for q in ["requisitos para solicitar beca", "costo pre-ingreso S/. 120.00", "artículo 53 nota mínima admisión"]:
        scores = bm25.get_scores(tokenizar(q))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        print(f"\n  {q}")
        for i in top:
            print(f"    {scores[i]:6.2f}  {df['nombre_documento'].iat[i]} · {df['articulo'].iat[i] or df['tipo_bloque'].iat[i]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_evaluacion.py — Evaluación de recuperación con un banco de preguntas con respuesta conocida (M4).

Banco: corpus_upeu/config/banco_preguntas.csv
    id, pregunta, tipo (positiva|negativa), categoria_esperada, documento_esperado
    (subcadena del nombre del PDF), articulo_esperado (número o vacío), revisado, notas

Métricas por modo (sobre las preguntas positivas):
    R@1, R@3, R@5, R@10 y MRR@10 a nivel ARTÍCULO cuando la pregunta trae artículo esperado
    (si no, a nivel documento), y R@k a nivel DOCUMENTO para todas.
    Para las negativas se reporta el puntaje coseno top-1 (para calibrar el umbral "fuera de dominio").

Modos: semantico | bm25 | hibrido (RRF k=60) | rerank (bge-reranker-v2-m3 sobre el top-30 híbrido)

La búsqueda semántica es exacta (numpy) sobre embeddings_<version>.npy: mide los embeddings,
no el índice HNSW. Solo LEE los artefactos; escribe en metadatos/<CORPUS_VERSION>/evaluacion/.

Ejemplos (Docker):
    python 09_evaluacion.py                                   # v3, mpnet, todos los modos
    python 09_evaluacion.py --version v2                      # V2 canónica (solo lectura)
    python 09_evaluacion.py --breadcrumb ninguno --modos semantico   # ablación: re-codifica sin breadcrumb
    MODEL_KEY=e5large python 09_evaluacion.py                 # otro modelo (tras correr 05 con ese modelo)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from chunking import leer_chunks, construir_breadcrumb, partes_breadcrumb
from texto_lexico import tokenizar

RERANKER = "BAAI/bge-reranker-v2-m3"
TOP_N = 50
KS = (1, 3, 5, 10)


def num_articulo(s) -> str:
    m = re.search(r"\d+", str(s or ""))
    return m.group(0) if m else ""


def cargar(version: str, modelo: str):
    meta = C.CORPUS_DIR / "metadatos" / version
    suf = "" if modelo == "mpnet" else f"_{modelo}"
    df = leer_chunks(meta / f"chunks_{version}{suf}.csv")
    emb = np.load(meta / f"embeddings_{version}{suf}.npy")
    man = meta / f"MANIFEST{suf}.json"
    cfg = json.loads(man.read_text(encoding="utf-8"))["config"] if man.exists() else {}
    cfg.setdefault("max_seq_length", 128)                 # V2: mpnet por defecto, sin breadcrumb
    cfg.setdefault("breadcrumb_modo", "ninguno")
    cfg.setdefault("max_tokens_breadcrumb", 60)
    assert len(df) == emb.shape[0], f"{version}: CSV ({len(df)}) y embeddings ({emb.shape[0]}) no coinciden"
    return df, emb, cfg


def rrf(listas, k=60):
    puntaje = {}
    for lista in listas:
        for pos, i in enumerate(lista):
            puntaje[i] = puntaje.get(i, 0.0) + 1.0 / (k + pos + 1)
    return [i for i, _ in sorted(puntaje.items(), key=lambda x: -x[1])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=C.VERSION)
    ap.add_argument("--modelo", default=C.MODEL_KEY, choices=list(C.PERFILES_MODELO))
    ap.add_argument("--modos", default="semantico,bm25,hibrido,rerank")
    ap.add_argument("--breadcrumb", choices=["completo", "documento", "ninguno"],
                    help="re-codifica los chunks con otro breadcrumb (ablación, sin re-chunkear)")
    ap.add_argument("--banco", type=Path, default=C.BANCO_PREGUNTAS_CSV)
    args = ap.parse_args()
    modos = [m.strip() for m in args.modos.split(",") if m.strip()]

    C.asegurar_version_escribible()                       # la salida va a metadatos/<CORPUS_VERSION>/
    out_dir = C.EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    banco = pd.read_csv(args.banco, dtype=str, keep_default_na=False)
    pos = banco[banco["tipo"] == "positiva"].reset_index(drop=True)
    neg = banco[banco["tipo"] == "negativa"].reset_index(drop=True)
    no_rev = int((banco["revisado"].str.lower() != "si").sum())

    df, emb, cfg = cargar(args.version, args.modelo)
    perfil = C.PERFILES_MODELO[args.modelo]
    bc_modo = args.breadcrumb or cfg["breadcrumb_modo"]
    tag = f"{args.version}_{args.modelo}_bc-{bc_modo}"
    print(f"Evaluando {tag}: {len(df)} chunks | {len(pos)} positivas, {len(neg)} negativas "
          f"({no_rev} preguntas sin revisar) | modos {modos}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(perfil["nombre"], revision=perfil["revision"])
    model.max_seq_length = int(cfg["max_seq_length"])

    if args.breadcrumb and args.breadcrumb != cfg["breadcrumb_modo"]:
        if "nombre_documento" not in df.columns:
            raise SystemExit("--breadcrumb solo aplica a versiones con metadatos V3")
        tok = model.tokenizer
        contar = lambda t: len(tok(t, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])
        textos = []
        for r in df.itertuples():
            bc = construir_breadcrumb(partes_breadcrumb(args.breadcrumb, r.nombre_documento, r.titulo_estructura,
                                                        r.capitulo, r.seccion, r.articulo),
                                      int(cfg["max_tokens_breadcrumb"]), contar)
            textos.append(f"{bc}\n{r.texto}" if bc else r.texto)
        print(f"Re-codificando {len(textos)} chunks con breadcrumb={args.breadcrumb}...")
        emb = model.encode([perfil["prefijo_documento"] + t for t in textos], batch_size=C.EMBED_BATCH_SIZE,
                           normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    campo_lex = "texto_embedding" if "texto_embedding" in df.columns else "texto"
    bm25 = None
    if {"bm25", "hibrido", "rerank"} & set(modos):
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([tokenizar(t) for t in df[campo_lex]])
    reranker = None
    if "rerank" in modos:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANKER, max_length=512)

    docs = df["documento"].str.lower().tolist()
    arts = [num_articulo(a) for a in df["articulo"]]

    def buscar(pregunta: str, modo: str):
        q = model.encode([perfil["prefijo_pregunta"] + pregunta], normalize_embeddings=True)[0]
        sem_scores = emb @ q
        sem = list(np.argsort(-sem_scores)[:TOP_N])
        if modo == "semantico":
            return sem, float(sem_scores[sem[0]])
        lex_scores = bm25.get_scores(tokenizar(pregunta))
        lex = list(np.argsort(-lex_scores)[:TOP_N])
        if modo == "bm25":
            return lex, float(sem_scores[lex[0]])
        hib = rrf([sem, lex])[:TOP_N]
        if modo == "hibrido":
            return hib, float(sem_scores[hib[0]])
        cand = hib[:30]
        s = reranker.predict([(pregunta, df["texto"].iat[i][:2000]) for i in cand], batch_size=16)
        orden = [cand[j] for j in np.argsort(-np.asarray(s))]
        return orden, float(sem_scores[orden[0]])

    filas, resumen = [], []
    for modo in modos:
        ranks, ranks_doc = [], []
        for p in pos.itertuples():
            ranking, top1 = buscar(p.pregunta, modo)
            doc_esp, art_esp = p.documento_esperado.lower(), num_articulo(p.articulo_esperado)
            rel_doc = [doc_esp in docs[i] for i in ranking[:10]]
            rel = [doc_esp in docs[i] and (not art_esp or arts[i] == art_esp) for i in ranking[:10]]
            r = rel.index(True) + 1 if True in rel else None
            rd = rel_doc.index(True) + 1 if True in rel_doc else None
            ranks.append(r)
            ranks_doc.append(rd)
            i0 = ranking[0]
            filas.append({"tag": tag, "modo": modo, "id": p.id, "pregunta": p.pregunta,
                          "nivel": "articulo" if art_esp else "documento", "rank": r, "rank_documento": rd,
                          "top1_documento": df["documento"].iat[i0], "top1_articulo": df["articulo"].iat[i0],
                          "top1_coseno": round(top1, 4)})
        cos_neg = []
        for p in neg.itertuples():
            _, top1 = buscar(p.pregunta, "semantico")
            cos_neg.append(top1)
            filas.append({"tag": tag, "modo": modo, "id": p.id, "pregunta": p.pregunta, "nivel": "negativa",
                          "top1_coseno": round(top1, 4)})
        n = len(ranks)
        fila = {"tag": tag, "version": args.version, "modelo": args.modelo, "breadcrumb": bc_modo, "modo": modo,
                "n_positivas": n, "sin_revisar": no_rev}
        for k in KS:
            fila[f"R@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 3)
        fila["MRR@10"] = round(sum(1 / r for r in ranks if r) / n, 3)
        for k in (1, 5):
            fila[f"Rdoc@{k}"] = round(sum(1 for r in ranks_doc if r and r <= k) / n, 3)
        cos_pos = [f["top1_coseno"] for f in filas if f["modo"] == modo and f["nivel"] in ("articulo", "documento")]
        fila["coseno_top1_pos"] = round(float(np.mean(cos_pos)), 4)
        fila["coseno_top1_neg"] = round(float(np.mean(cos_neg)), 4) if cos_neg else None
        resumen.append(fila)

    pd.DataFrame(filas).to_csv(out_dir / f"{tag}.csv", index=False, encoding="utf-8")
    res_path = out_dir / "resumen.csv"
    previo = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()
    nuevo = pd.DataFrame(resumen)
    if len(previo):
        previo = previo[~(previo["tag"].isin(nuevo["tag"]) & previo["modo"].isin(nuevo["modo"]))]
    pd.concat([previo, nuevo], ignore_index=True).to_csv(res_path, index=False, encoding="utf-8")

    cols = ["modo", "R@1", "R@3", "R@5", "R@10", "MRR@10", "Rdoc@1", "Rdoc@5", "coseno_top1_pos", "coseno_top1_neg"]
    print(f"\n{tag}\n" + nuevo[cols].to_string(index=False))
    print(f"\n📄 Detalle: {out_dir / (tag + '.csv')}\n📄 Resumen acumulado: {res_path}")
    if no_rev:
        print(f"⚠️  {no_rev} preguntas del banco aún no fueron revisadas por una persona: resultados provisionales.")


if __name__ == "__main__":
    main()

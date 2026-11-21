#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_chunking_v2.py — Chunking estructural + embeddings mpnet (lógica en chunking.py).

Entradas : txt_bruto_<version>/, diagnostico_<version>.csv
Salidas  : chunks_<version>.csv, embeddings_<version>.npy, chunk_ids_<version>.npy, MANIFEST.json

Cambios V3 respecto a V2:
  * P1  model.max_seq_length = config.MAX_SEQ_LENGTH y ningún chunk lo supera
        (V2 truncaba en silencio ~50% de los tokens a 128).
  * F5  se embebe `texto_embedding` = breadcrumb de contexto + texto.
  * F2/F3/F11/F12 y P3 (sección, página, duplicados, chunks basura): ver chunking.py.

Opciones:
    --txt-dir RUTA   re-chunkear otro directorio de .txt (p.ej. txt_bruto_v2 para el experimento 2a)
    --sin-embeddings genera solo chunks_<version>.csv (rápido, sin GPU)
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
from chunking import procesar_documento


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.BASE, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--txt-dir", type=Path, default=C.TXT_DIR)
    parser.add_argument("--sin-embeddings", action="store_true")
    args = parser.parse_args()

    C.asegurar_version_escribible()
    C.META_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Versión {C.VERSION} | modelo {C.MODEL_KEY} ({C.MODEL_NAME}) | modo {C.CHUNK_MODE} | "
          f"max_seq_length {C.MAX_SEQ_LENGTH} | chunk_max {C.CHUNK_MAX_TOKENS} | overlap {C.CHUNK_OVERLAP_TOKENS} | "
          f"breadcrumb {C.BREADCRUMB_MODO} | txt {args.txt_dir}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(C.MODEL_NAME, revision=C.MODEL_REVISION)
    model.max_seq_length = C.MAX_SEQ_LENGTH
    tokenizer = model.tokenizer

    def contar(texto: str) -> int:
        # truncation=False: contar TODOS los tokens (si truncara, el chunk "cabría" falsamente)
        return len(tokenizer(texto, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])

    # El prefijo de documento (p.ej. "passage: " en e5) también ocupa tokens del modelo.
    reservados = C.TOKENS_ESPECIALES + (contar(C.PREFIJO_DOCUMENTO) if C.PREFIJO_DOCUMENTO else 0)

    df = pd.read_csv(C.DIAG_CSV)
    all_chunks, sin_txt, descartados = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        nombre = row["nombre_archivo"]
        txt_path = args.txt_dir / f"{nombre}.txt"
        if not txt_path.exists() or not txt_path.read_text(encoding="utf-8").strip():
            sin_txt.append(nombre)
            continue
        all_chunks.extend(procesar_documento(
            nombre, txt_path.read_text(encoding="utf-8"), contar,
            max_seq_length=C.MAX_SEQ_LENGTH,
            chunk_max_tokens=C.CHUNK_MAX_TOKENS,
            overlap_tokens=C.CHUNK_OVERLAP_TOKENS,
            min_tokens_bloque=C.MIN_TOKENS_BLOQUE,
            max_tokens_breadcrumb=C.MAX_TOKENS_BREADCRUMB,
            tokens_especiales=reservados,
            breadcrumb_modo=C.BREADCRUMB_MODO,
            descartados=descartados,
            metadatos_doc={
                "tipo_documental": row["tipo_documental"],
                "categoria_tematica": row["categoria_tematica"],
                "ruta_extraccion": row["ruta_extraccion"],
            },
        ))
    for n in sin_txt:
        print(f"  ⚠️  Sin texto: {n}")

    df_chunks = pd.DataFrame(all_chunks)
    df_chunks.to_csv(C.CHUNKS_CSV, index=False, encoding="utf-8")
    df_desc = pd.DataFrame(descartados, columns=["documento", "motivo", "char_inicio", "char_fin", "texto"])
    df_desc.to_csv(C.DESCARTADOS_CSV, index=False, encoding="utf-8")
    print(f"📄 {len(df_desc)} spans descartados (índices / duplicados) → {C.DESCARTADOS_CSV}")
    print(f"\n📄 {len(df_chunks)} chunks → {C.CHUNKS_CSV}")

    limite = C.MAX_SEQ_LENGTH - reservados
    exceden = int((df_chunks["num_tokens_embedding"] > limite).sum())
    stats = {
        "documentos": int(df_chunks["documento"].nunique()),
        "chunks": int(len(df_chunks)),
        "tokens_texto_total": int(df_chunks["num_tokens"].sum()),
        "tokens_texto_media": round(float(df_chunks["num_tokens"].mean()), 1),
        "tokens_texto_mediana": float(df_chunks["num_tokens"].median()),
        "tokens_texto_max": int(df_chunks["num_tokens"].max()),
        "tokens_embedding_max": int(df_chunks["num_tokens_embedding"].max()),
        "chunks_exceden_max_seq": exceden,
        "tokens_reservados": reservados,
        "pct_tokens_embebidos": round(100 * float(np.minimum(df_chunks["num_tokens_embedding"], limite).sum()
                                                  / df_chunks["num_tokens_embedding"].sum()), 2),
        "chunks_partidos": int((df_chunks["total_partes"] > 1).sum()),
        "descartados": df_desc["motivo"].value_counts().to_dict(),
        "tipo_bloque": df_chunks["tipo_bloque"].value_counts().to_dict(),
        "documentos_sin_texto": sin_txt,
    }
    print(json.dumps({k: v for k, v in stats.items() if k != "tipo_bloque"}, ensure_ascii=False, indent=2))
    if exceden:
        print(f"❌ {exceden} chunks superan max_seq_length: se truncarían. Revisa chunking.py antes de seguir.")

    archivos = {"chunks": C.CHUNKS_CSV}
    if not args.sin_embeddings:
        print("Generando embeddings de texto_embedding...")
        emb = model.encode([C.PREFIJO_DOCUMENTO + t for t in df_chunks["texto_embedding"]], show_progress_bar=True,
                           batch_size=C.EMBED_BATCH_SIZE, normalize_embeddings=True)
        np.save(C.EMBEDDINGS_NPY, emb.astype(np.float32))
        np.save(C.CHUNK_IDS_NPY, df_chunks["chunk_id"].values)
        print(f"📄 Embeddings {emb.shape} → {C.EMBEDDINGS_NPY}")
        archivos.update(embeddings=C.EMBEDDINGS_NPY, chunk_ids=C.CHUNK_IDS_NPY)

    manifest = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "txt_dir": str(args.txt_dir.relative_to(C.BASE)) if args.txt_dir.is_relative_to(C.BASE) else str(args.txt_dir),
        "config": C.resumen_config(),
        "estadisticas": stats,
        "sha256": {k: sha256(p) for k, p in archivos.items()},
    }
    C.MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📄 Manifest → {C.MANIFEST_JSON}")


if __name__ == "__main__":
    main()

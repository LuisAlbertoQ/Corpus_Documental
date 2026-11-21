#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_validacion_corpus.py — Validación automática del corpus (chunks + embeddings).

Salidas: reporte_validacion_<version>.md, validacion_<version>_detalle.csv

Cambios V3 respecto a V2:
  * El límite de tokens sale de config.MAX_SEQ_LENGTH (V2 asumía 512 y no veía
    el truncamiento real a 128). Se valida num_tokens_embedding, que es lo embebido.
  * Regex de encabezado sin distinguir mayúsculas (V2: 3,931 falsos "sin header")
    y solo sobre la parte 1 de cada artículo (las partes 2+ no lo repiten por diseño).
  * Padre-hijo validado por grupo_articulo_id (V2 comparaba contra el chunk anterior).
  * NUEVO 13: cobertura de texto por documento (F6) — habría detectado el preámbulo perdido.
  * NUEVO 14: trazabilidad texto == txt_limpio[char_inicio:char_fin].
  * NUEVO 15: falsos encabezados ("TÍTULO D", "Capítulo de…").
Termina con código 1 si hay errores, para poder encadenar con 07 en scripts.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from chunking import separar_paginas, leer_chunks

alertas = []


def alerta(nivel, categoria, mensaje, chunk_id=""):
    alertas.append({"nivel": nivel, "categoria": categoria, "mensaje": mensaje, "chunk_id": chunk_id})


def detalle(nivel, categoria, filas, mensaje_fn, max_filas=20):
    for _, r in filas.head(max_filas).iterrows():
        alerta(nivel, categoria, mensaje_fn(r), r["chunk_id"])


def tokens_reservados() -> int:
    """Tokens especiales + prefijo de documento que usó 05 (p.ej. "passage: " en e5)."""
    if C.MANIFEST_JSON.exists():
        return int(json.loads(C.MANIFEST_JSON.read_text(encoding="utf-8"))
                   .get("estadisticas", {}).get("tokens_reservados", C.TOKENS_ESPECIALES))
    return C.TOKENS_ESPECIALES


def txt_dir_usado() -> Path:
    """El directorio de .txt que usó 05 (puede ser txt_bruto_v2 en el experimento 2a)."""
    if C.MANIFEST_JSON.exists():
        d = json.loads(C.MANIFEST_JSON.read_text(encoding="utf-8")).get("txt_dir")
        if d:
            p = Path(d)
            return p if p.is_absolute() else C.BASE / p
    return C.TXT_DIR


def main():
    print(f"Validando corpus {C.VERSION} (max_seq_length={C.MAX_SEQ_LENGTH})...")
    df = leer_chunks(C.CHUNKS_CSV)
    df_diag = pd.read_csv(C.DIAG_CSV)
    tiene_emb = C.EMBEDDINGS_NPY.exists() and C.CHUNK_IDS_NPY.exists()
    embeddings = np.load(C.EMBEDDINGS_NPY) if tiene_emb else None
    chunk_ids = np.load(C.CHUNK_IDS_NPY, allow_pickle=True) if tiene_emb else None
    limite = C.MAX_SEQ_LENGTH - tokens_reservados()

    # 1. Integridad
    print("1. Integridad")
    if not tiene_emb:
        alerta("ERROR", "integridad", "Faltan embeddings/chunk_ids (¿05 con --sin-embeddings?)")
    else:
        if len(df) != embeddings.shape[0]:
            alerta("ERROR", "integridad", f"Chunks CSV ({len(df)}) != embeddings ({embeddings.shape[0]})")
        if len(df) != len(chunk_ids) or not (df["chunk_id"].values == chunk_ids).all():
            alerta("ERROR", "integridad", "chunk_ids.npy no coincide con el CSV")

    # 2. Campos
    print("2. Campos")
    obligatorios = ["documento", "chunk_id", "texto", "texto_embedding", "categoria_tematica",
                    "tipo_bloque", "grupo_articulo_id", "char_inicio", "char_fin", "num_tokens_embedding"]
    informativos = ["nombre_documento", "titulo_estructura", "capitulo", "seccion", "articulo", "pagina"]
    for campo in obligatorios:
        if campo not in df.columns:
            alerta("ERROR", "metadatos", f"Campo obligatorio faltante: {campo}")
        elif (df[campo].astype(str).str.strip() == "").any():
            alerta("ERROR", "metadatos", f"Campo '{campo}' vacío en {(df[campo].astype(str).str.strip() == '').sum()} chunks")
    for campo in informativos:
        if campo in df.columns:
            vacios = int((df[campo].astype(str).str.strip() == "").sum())
            alerta("INFO", "metadatos", f"Campo '{campo}' vacío en {vacios} chunks ({100*vacios/len(df):.1f}%)")
    if "nombre_documento" in df.columns and (df["nombre_documento"] == "").any():
        alerta("ERROR", "metadatos", "nombre_documento vacío (debe derivarse siempre del archivo)")

    # 3. Contenido
    print("3. Contenido")
    vacios = df[df["texto"].str.strip() == ""]
    if len(vacios):
        alerta("ERROR", "contenido", f"{len(vacios)} chunks vacíos")
    cortos = df[df["num_chars"] < 50]
    if len(cortos):
        alerta("WARNING", "contenido", f"{len(cortos)} chunks < 50 chars")
        detalle("WARNING", "contenido", cortos, lambda r: f"Chunk corto ({r['num_chars']} chars)")
    dups = df[df.duplicated(subset=["documento", "texto"], keep=False)]
    if len(dups):
        alerta("WARNING", "contenido", f"{len(dups)} chunks con texto duplicado dentro del mismo documento")

    # 4. Tokens: lo que de verdad entra al modelo
    print("4. Tokens")
    exceden = df[df["num_tokens_embedding"] > limite]
    if len(exceden):
        alerta("ERROR", "tokens", f"{len(exceden)} chunks superan max_seq_length={C.MAX_SEQ_LENGTH} (se truncarían)")
        detalle("ERROR", "tokens", exceden, lambda r: f"{r['num_tokens_embedding']} tokens de embedding")
    pct_embebido = 100 * np.minimum(df["num_tokens_embedding"], limite).sum() / df["num_tokens_embedding"].sum()

    # 5. Basura
    print("5. Basura")
    patrones = [
        (r"Casilla\s*3564|postmast\S*\s*\S*upeu\.edu\.pe", "pie de página con dirección"),
        (r"(?m)^\s*\d+\s*$", "folio suelto"),
        (r"\[\[PAG \d+\]\]|--- Pág \d+ \(OCR\) ---|--- Contexto p\d+ ---", "marcador de página sin limpiar"),
    ]
    for patron, desc in patrones:
        m = df[df["texto"].str.contains(patron, regex=True)]
        if len(m):
            nivel = "ERROR" if "marcador" in desc else "WARNING"
            alerta(nivel, "basura", f"{len(m)} chunks con {desc}")
            detalle(nivel, "basura", m, lambda r, d=desc: f"{d} detectado", 5)

    # 6. Tablas en documentos TUPA
    print("6. Tablas")
    for doc in df_diag.loc[df_diag["ruta_extraccion"] == "TUPA_TABLAS", "nombre_archivo"]:
        d = df[df["documento"] == doc]
        if len(d) and not d["texto"].str.contains(r"\[TABLA", regex=True).any() \
                and not d["texto_embedding"].str.contains("Tabla:", regex=False).any():
            alerta("WARNING", "tablas", f"Documento TUPA_TABLAS sin tablas en sus chunks: {doc}")

    # 7. Encabezado en la parte 1 de cada artículo
    print("7. Estructura")
    primeras = df[(df["tipo_bloque"] == "articulo") & (df["parte_num"] == 1)]
    sin_header = primeras[~primeras["texto"].str.contains(r"ART[IÍ]CULO\s+\d+", flags=re.IGNORECASE, regex=True)]
    if len(sin_header):
        alerta("WARNING", "estructura", f"{len(sin_header)} artículos (parte 1) sin encabezado 'Artículo N'")
        detalle("WARNING", "estructura", sin_header, lambda r: "Artículo sin encabezado en su texto")

    # 8. Grupos padre-hijo
    print("8. Grupos")
    ids = set(df["chunk_id"])
    for grupo, g in df.groupby("grupo_articulo_id", sort=False):
        if grupo not in ids:
            alerta("ERROR", "grupos", f"grupo_articulo_id inexistente: {grupo}", g["chunk_id"].iat[0])
            continue
        nums = g["parte_num"].tolist()
        if nums != list(range(1, len(g) + 1)) or (g["total_partes"] != len(g)).any():
            alerta("ERROR", "grupos", f"Partes no secuenciales en {grupo}: {nums}", grupo)
        if g["documento"].nunique() > 1:
            alerta("ERROR", "grupos", f"Grupo {grupo} mezcla documentos", grupo)

    # 9. Categorías
    print("9. Categorías")
    invalidas = set(df["categoria_tematica"]) - {"A", "B", "C", "D", "E"}
    if invalidas:
        alerta("ERROR", "categoria", f"Categorías inválidas: {invalidas}")

    # 10. Embeddings
    print("10. Embeddings")
    norma_media = None
    if tiene_emb:
        if not np.isfinite(embeddings).all():
            alerta("ERROR", "embeddings", "Embeddings con NaN/Inf")
        if embeddings.shape[1] != C.EMBEDDING_DIM:
            alerta("ERROR", "embeddings", f"Dimensión {embeddings.shape[1]} (esperado {C.EMBEDDING_DIM})")
        normas = np.linalg.norm(embeddings, axis=1)
        norma_media = float(normas.mean())
        if not np.allclose(normas, 1.0, atol=1e-3):
            alerta("WARNING", "embeddings", f"{int((~np.isclose(normas, 1.0, atol=1e-3)).sum())} embeddings no normalizados")

    # 11. IDs duplicados
    print("11. Duplicados")
    if df["chunk_id"].duplicated().any():
        alerta("ERROR", "duplicados", f"{int(df['chunk_id'].duplicated().sum())} chunk_ids duplicados")

    # 12. Coherencia con diagnóstico
    print("12. Coherencia")
    for doc in set(df_diag["nombre_archivo"]) - set(df["documento"]):
        alerta("ERROR", "coherencia", f"Documento del diagnóstico sin chunks: {doc}")

    # 13 y 14. Cobertura y trazabilidad contra el .txt
    txt_dir = txt_dir_usado()
    print(f"13-14. Cobertura y trazabilidad ({txt_dir})")
    cobertura = {}
    desc = pd.read_csv(C.DESCARTADOS_CSV, keep_default_na=False) if C.DESCARTADOS_CSV.exists() else None
    if desc is not None and len(desc):
        alerta("INFO", "descartados", f"Spans fuera del índice: {desc['motivo'].value_counts().to_dict()}")
    for doc, g in df.groupby("documento"):
        path = txt_dir / f"{doc}.txt"
        if not path.exists():
            alerta("ERROR", "cobertura", f"No se encuentra {path} para medir cobertura")
            continue
        limpio, _ = separar_paginas(path.read_text(encoding="utf-8"))
        cubierto = np.zeros(len(limpio), dtype=bool)
        for _, r in g.iterrows():
            ini, fin = int(r["char_inicio"]), int(r["char_fin"])
            if limpio[ini:fin] != r["texto"]:
                alerta("ERROR", "trazabilidad", "texto no coincide con el .txt en char_inicio:char_fin", r["chunk_id"])
            cubierto[ini:fin] = True
        if desc is not None:                      # índices y duplicados descartados a propósito
            for _, r in desc[desc["documento"] == doc].iterrows():
                cubierto[int(r["char_inicio"]):int(r["char_fin"])] = True
        no_blanco = np.array([not ch.isspace() for ch in limpio], dtype=bool)
        cov = float(cubierto[no_blanco].mean()) if no_blanco.any() else 1.0
        cobertura[doc] = cov
        if cov < C.COBERTURA_MIN_ERROR:
            alerta("ERROR", "cobertura", f"Cobertura {cov:.1%} < {C.COBERTURA_MIN_ERROR:.0%}: {doc}")
        elif cov < C.COBERTURA_MIN_WARNING:
            alerta("WARNING", "cobertura", f"Cobertura {cov:.1%}: {doc}")
    cobertura_global = float(np.mean(list(cobertura.values()))) if cobertura else 0.0

    # 15. Falsos encabezados
    print("15. Falsos encabezados")
    # (?=(?P<n>…))(?P=n) emula un grupo atómico: el numeral se toma entero, sin backtracking
    # (si no, "TÍTULO II" se leería como "I" + letra "I").
    patron_falso = r"^(?:T[IÍ]TULO|CAP[IÍ]TULO)[ \t]+(?=(?P<n>[IVXLCDM]+))(?P=n)[A-Za-zÁÉÍÓÚÑáéíóúñ]"
    with warnings.catch_warnings():       # pandas avisa por el grupo con nombre; es intencional
        warnings.simplefilter("ignore", UserWarning)
        falsos = df[df["articulo"].str.contains(patron_falso, regex=True)
                    | df["titulo_estructura"].str.contains(patron_falso, regex=True)
                    | df["capitulo"].str.contains(patron_falso, regex=True)]
    if len(falsos):
        alerta("ERROR", "estructura", f"{len(falsos)} chunks con falso encabezado")

    # --- Reporte -------------------------------------------------------------
    df_alertas = pd.DataFrame(alertas, columns=["nivel", "categoria", "mensaje", "chunk_id"])
    df_alertas.to_csv(C.DETALLE_CSV, index=False, encoding="utf-8")
    n = df_alertas["nivel"].value_counts().to_dict()
    errores, n_warnings = n.get("ERROR", 0), n.get("WARNING", 0)

    md = [
        f"# Reporte de Validación Corpus {C.VERSION}", "",
        f"**Fecha:** {pd.Timestamp.now():%Y-%m-%d %H:%M}  ",
        f"**Modo:** {C.CHUNK_MODE} · **max_seq_length:** {C.MAX_SEQ_LENGTH} · **txt:** `{txt_dir.name}`", "",
        "## Resumen", "", "| Métrica | Valor |", "|---|---|",
        f"| Chunks | {len(df):,} |",
        f"| Documentos | {df['documento'].nunique()} |",
        f"| Tokens texto (media / mediana / máx) | {df['num_tokens'].mean():.1f} / {df['num_tokens'].median():.0f} / {df['num_tokens'].max()} |",
        f"| Tokens embedding máx (límite {limite}) | {df['num_tokens_embedding'].max()} |",
        f"| **% tokens que llegan al embedding** | **{pct_embebido:.2f}%** |",
        f"| **Cobertura media de texto** | **{cobertura_global:.2%}** (mín {min(cobertura.values(), default=0):.2%}) |",
        f"| Chunks < 50 chars | {len(cortos)} |",
        f"| Artículos partidos (total_partes > 1) | {int((df['total_partes'] > 1).sum())} |",
        f"| Norma media embeddings | {norma_media:.4f} |" if norma_media is not None else "| Norma media embeddings | n/d |",
        "", "## Alertas", "", "| Nivel | Cantidad |", "|---|---|",
        f"| 🔴 ERROR | {errores} |", f"| 🟡 WARNING | {n_warnings} |", f"| 🔵 INFO | {n.get('INFO', 0)} |",
        "", "## Por categoría", "", "| Categoría | Nivel | Cantidad |", "|---|---|---|",
    ]
    for (cat, niv), cnt in df_alertas.groupby(["categoria", "nivel"]).size().items():
        md.append(f"| {cat} | {niv} | {cnt} |")
    md += ["", "## Distribución tipo_bloque", "", "| tipo | N |", "|---|---|"]
    md += [f"| {t} | {c} |" for t, c in df["tipo_bloque"].value_counts().items()]
    md += ["", "## Primeras 60 alertas", "", "| Nivel | Categoría | Chunk | Mensaje |", "|---|---|---|---|"]
    for _, a in df_alertas[df_alertas["nivel"] != "INFO"].head(60).iterrows():
        md.append(f"| {a['nivel']} | {a['categoria']} | {a['chunk_id']} | {a['mensaje']} |")
    md += ["", f"*Detalle completo: `{C.DETALLE_CSV.name}` ({len(df_alertas)} filas)*"]
    C.REPORTE_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n{'=' * 50}\nERROR: {errores} | WARNING: {n_warnings} | tokens embebidos {pct_embebido:.2f}% | "
          f"cobertura {cobertura_global:.2%}\n📄 {C.REPORTE_MD}")
    if errores:
        for _, a in df_alertas[df_alertas["nivel"] == "ERROR"].head(20).iterrows():
            print(f"  ❌ {a['categoria']}: {a['mensaje']} {a['chunk_id']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

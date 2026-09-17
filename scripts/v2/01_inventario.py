#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_inventario.py — Genera inventario_v2.csv de los PDFs en corpus_upeu/pdfs/
Campos: nombre_archivo, hash_sha256, formato, tamanio_mb, num_paginas,
        anio, version, url_origen, fecha_oficial, vigencia, fecha_fs
"""

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

PDF_FOLDER = Path(__file__).resolve().parents[1] / "corpus_upeu" / "pdfs"
METADATA_FOLDER = Path(__file__).resolve().parents[1] / "corpus_upeu" / "metadatos"
OUTPUT_CSV = METADATA_FOLDER / "inventario_v2.csv"

os.makedirs(METADATA_FOLDER, exist_ok=True)

# --- Heurísticas sobre nombre de archivo ---
RE_ANIO = re.compile(r"\b(20\d{2})\b")
RE_VERSION = re.compile(r"[._\s-]v?(\d+(?:\.\d+)?)\b", re.IGNORECASE)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def extraer_anio_version(nombre: str):
    anio = None
    version = None
    m_anio = RE_ANIO.search(nombre)
    if m_anio:
        anio = int(m_anio.group(1))
    m_ver = RE_VERSION.search(nombre)
    if m_ver:
        version = m_ver.group(1)
    return anio, version

def main():
    pdfs = sorted(PDF_FOLDER.glob("*.pdf"))
    print(f"Encontrados {len(pdfs)} PDFs en {PDF_FOLDER}")

    rows = []
    for pdf_path in pdfs:
        stat = pdf_path.stat()
        nombre = pdf_path.name
        hash_ = sha256_file(pdf_path)

        # Metadatos FS
        tamanio_mb = round(stat.st_size / (1024 * 1024), 3)
        fecha_fs = datetime.fromtimestamp(stat.st_mtime).isoformat()

        # PyMuPDF: num páginas
        try:
            with fitz.open(pdf_path) as doc:
                num_paginas = len(doc)
        except Exception as e:
            print(f"  ⚠️  {nombre}: error leyendo páginas -> {e}")
            num_paginas = None

        # Año y versión del nombre
        anio, version = extraer_anio_version(nombre)

        rows.append({
            "nombre_archivo": nombre,
            "hash_sha256": hash_,
            "formato": "pdf",
            "tamanio_mb": tamanio_mb,
            "num_paginas": num_paginas,
            "anio": anio,
            "version": version,
            "url_origen": "",       # ← TÚ lo llenas después
            "fecha_oficial": "",    # ← TÚ
            "vigencia": "",         # ← TÚ (VIGENTE/SUPERSEDED/ABROGADO)
            "fecha_fs": fecha_fs,
        })
        print(f"  ✅ {nombre} ({tamanio_mb} MB, {num_paginas} págs, año={anio}, v={version})")

    df = pd.DataFrame(rows, columns=[
        "nombre_archivo", "hash_sha256", "formato", "tamanio_mb", "num_paginas",
        "anio", "version", "url_origen", "fecha_oficial", "vigencia", "fecha_fs"
    ])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n📄 Inventario guardado en {OUTPUT_CSV}")
    print(f"   {len(df)} filas")

    # Duplicados exactos por hash
    dup = df[df.duplicated(subset=["hash_sha256"], keep=False)]
    if not dup.empty:
        print("\n⚠️  Duplicados exactos (mismo hash):")
        for h, g in dup.groupby("hash_sha256"):
            print(f"   Hash {h[:12]}...:")
            for _, r in g.iterrows():
                print(f"     - {r['nombre_archivo']}")
    else:
        print("\n✅ Sin duplicados exactos por hash")

if __name__ == "__main__":
    main()
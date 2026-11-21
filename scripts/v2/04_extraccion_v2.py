#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_extraccion_v2.py — Extracción por ruta según el diagnóstico.
Rutas: DIGITAL, MIXTO, TUPA_TABLAS, ESCANEADO (lógica en extraccion_comun.py)
Salida: corpus_upeu/txt_bruto_<version>/{nombre_archivo}.txt + extraccion_<version>_metadatos.csv

Uso:
    python 04_extraccion_v2.py                      # todos los documentos del diagnóstico
    python 04_extraccion_v2.py --faltantes          # solo los que aún no tienen .txt
    python 04_extraccion_v2.py --solo "TUPA V6 2023 UPeU.pdf"
(--faltantes / --solo reemplazan a los antiguos check_missing.py y extraer_faltantes.py)
"""

import argparse

import pandas as pd
from tqdm import tqdm

from config import PDF_DIR, DIAG_CSV, TXT_DIR, META_DIR, EXTRACCION_CSV, asegurar_version_escribible
from extraccion_comun import extraer_pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faltantes", action="store_true", help="solo documentos sin .txt extraído")
    parser.add_argument("--solo", nargs="+", metavar="PDF", help="procesar solo estos nombres de archivo")
    args = parser.parse_args()

    asegurar_version_escribible()
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    if not DIAG_CSV.exists():
        print(f"❌ No existe {DIAG_CSV}. Ejecuta 03_diagnostico.py primero.")
        return

    df = pd.read_csv(DIAG_CSV)
    objetivo = df
    if args.solo:
        objetivo = df[df["nombre_archivo"].isin(args.solo)]
        no_encontrados = set(args.solo) - set(objetivo["nombre_archivo"])
        for n in sorted(no_encontrados):
            print(f"  ⚠️  No está en el diagnóstico: {n}")
    elif args.faltantes:
        objetivo = df[~df["nombre_archivo"].apply(lambda n: (TXT_DIR / f"{n}.txt").exists())]
    print(f"Procesando {len(objetivo)} de {len(df)} documentos del diagnóstico...")

    stats = {"DIGITAL": 0, "MIXTO": 0, "TUPA_TABLAS": 0, "ESCANEADO": 0, "ERROR": 0}
    info = {}
    for _, row in tqdm(objetivo.iterrows(), total=len(objetivo), desc="Extrayendo"):
        nombre, ruta = row["nombre_archivo"], row["ruta_extraccion"]
        try:
            res = extraer_pdf(PDF_DIR / nombre, ruta)
            (TXT_DIR / f"{nombre}.txt").write_text(res.texto, encoding="utf-8")
            stats[ruta] += 1
            info[nombre] = res
            print(f"  ✅ {nombre} → {ruta} ({len(res.texto)} chars, {res.paginas_ocr} págs OCR, {res.num_tablas} tablas)")
        except Exception as e:
            print(f"  ❌ {nombre}: {e}")
            stats["ERROR"] += 1

    print("\n--- Resumen extracción ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Metadatos de todos los documentos del diagnóstico (no solo los procesados ahora)
    meta_rows = []
    for _, row in df.iterrows():
        nombre = row["nombre_archivo"]
        path = TXT_DIR / f"{nombre}.txt"
        res = info.get(nombre)
        meta_rows.append({
            "nombre_archivo": nombre,
            "ruta_extraccion": row["ruta_extraccion"],
            "caracteres_extraidos": len(path.read_text(encoding="utf-8")) if path.exists() else 0,
            "paginas_ocr": res.paginas_ocr if res else "",
            "num_tablas": res.num_tablas if res else "",
            "errores_pagina": "; ".join(res.errores) if res else "",
            "tipo_documental": row["tipo_documental"],
            "categoria_tematica": row["categoria_tematica"],
        })
    pd.DataFrame(meta_rows).to_csv(EXTRACCION_CSV, index=False, encoding="utf-8")
    faltan = [r["nombre_archivo"] for r in meta_rows if r["caracteres_extraidos"] == 0]
    print(f"\n📄 Metadatos de extracción: {EXTRACCION_CSV}")
    if faltan:
        print(f"⚠️  {len(faltan)} documentos sin texto extraído: {faltan}")


if __name__ == "__main__":
    main()

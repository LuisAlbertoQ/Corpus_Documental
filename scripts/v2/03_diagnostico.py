#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_diagnostico.py — Diagnóstico por documento para definir ruta de extracción
Lee inventario_v2.csv + clasificacion_v2.csv (solo VIGENTE),
analiza cada PDF y escribe diagnostico_v2.csv con:
  nombre_archivo, tipo_documental, categoria_tematica,
  total_paginas, pct_paginas_digitales, pct_paginas_con_tablas,
  pct_paginas_con_imagenes, num_tablas_total, num_imagenes_total,
  tiene_columnas, ruta_extraccion, observaciones
"""

import re
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PDF_FOLDER = BASE / "corpus_upeu" / "pdfs"
INVENTARIO_CSV = BASE / "corpus_upeu" / "metadatos" / "inventario_v2.csv"
CLASIF_CSV = BASE / "corpus_upeu" / "metadatos" / "clasificacion_v2.csv"
OUTPUT_CSV = BASE / "corpus_upeu" / "metadatos" / "diagnostico_v2.csv"

# Umbral palabras/página para considerar "digital" (igual que notebook 1)
UMBRAL_PALABRAS = 30

# Heurística columnas: dispersión de coordenadas x
def detectar_columnas(page) -> bool:
    """Detecta si la página tiene layout de múltiples columnas."""
    try:
        blocks = page.get_text("dict")["blocks"]
        x_coords = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        x_coords.append(span["bbox"][0])  # x0
        if len(x_coords) < 20:
            return False
        # Bimodalidad simple: histograma con 2 picos claros
        import numpy as np
        hist, bins = np.histogram(x_coords, bins=10)
        picos = sum(1 for h in hist if h > len(x_coords) * 0.15)
        return picos >= 2
    except Exception:
        return False

def analizar_pdf(nombre_archivo: Path) -> dict:
    pdf_path = PDF_FOLDER / nombre_archivo
    resultado = {
        "total_paginas": 0,
        "paginas_digitales": 0,
        "paginas_con_tablas": 0,
        "paginas_con_imagenes": 0,
        "num_tablas_total": 0,
        "num_imagenes_total": 0,
        "tiene_columnas": False,
    }

    try:
        with fitz.open(pdf_path) as doc:
            resultado["total_paginas"] = len(doc)
            for page_num, page in enumerate(doc):
                # Texto digital
                texto = page.get_text()
                palabras = len([w for w in texto.split() if len(w) > 2])
                if palabras >= UMBRAL_PALABRAS:
                    resultado["paginas_digitales"] += 1

                # Imágenes
                imgs = page.get_images(full=True)
                if imgs:
                    resultado["paginas_con_imagenes"] += 1
                    resultado["num_imagenes_total"] += len(imgs)

                # Columnas (muestreo: primera página + mitad + última)
                if page_num in (0, len(doc)//2, len(doc)-1):
                    if detectar_columnas(page):
                        resultado["tiene_columnas"] = True

        # Tablas con pdfplumber (más confiable para conteo)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tablas = page.extract_tables()
                if tablas:
                    resultado["paginas_con_tablas"] += 1
                    resultado["num_tablas_total"] += len(tablas)

    except Exception as e:
        resultado["error"] = str(e)

    # Porcentajes
    tot = resultado["total_paginas"] or 1
    resultado["pct_paginas_digitales"] = round(100 * resultado["paginas_digitales"] / tot, 1)
    resultado["pct_paginas_con_tablas"] = round(100 * resultado["paginas_con_tablas"] / tot, 1)
    resultado["pct_paginas_con_imagenes"] = round(100 * resultado["paginas_con_imagenes"] / tot, 1)

    return resultado

def decidir_ruta(diag: dict, tipo: str) -> str:
    """Regla de decisión según diagrama."""
    pct_dig = diag.get("pct_paginas_digitales", 0)
    pct_tab = diag.get("pct_paginas_con_tablas", 0)
    tot_tab = diag.get("num_tablas_total", 0)

    # Caso especial TUPA/Tablas: tabla-primero
    if tipo == "TUPA_TARIFARIO" and tot_tab > 0:
        return "TUPA_TABLAS"

    if pct_dig >= 90:
        return "DIGITAL"
    elif pct_dig >= 20:
        return "MIXTO"
    else:
        return "ESCANEADO"

def main():
    if not INVENTARIO_CSV.exists() or not CLASIF_CSV.exists():
        print("❌ Faltan inventario_v2.csv o clasificacion_v2.csv")
        return

    df_inv = pd.read_csv(INVENTARIO_CSV)
    df_clasif = pd.read_csv(CLASIF_CSV)

    # Merge y filtrar solo VIGENTE
    df = df_inv.merge(df_clasif, on="nombre_archivo", how="inner")
    df_vig = df[df["estado"] == "VIGENTE"].copy()
    print(f"Analizando {len(df_vig)} documentos VIGENTE de {len(df)} totales...")

    rows = []
    for _, row in df_vig.iterrows():
        nombre = row["nombre_archivo"]
        tipo = row["tipo_documental"]
        print(f"  🔍 {nombre}...")

        diag = analizar_pdf(nombre)
        ruta = decidir_ruta(diag, tipo)

        obs = []
        if diag.get("error"):
            obs.append(f"ERROR: {diag['error']}")
        if diag.get("tiene_columnas"):
            obs.append("Layout multicolumna detectado")
        if diag["pct_paginas_con_tablas"] > 50:
            obs.append("Mayoría de páginas con tablas")

        rows.append({
            "nombre_archivo": nombre,
            "tipo_documental": tipo,
            "categoria_tematica": row["categoria_tematica"],
            "total_paginas": diag["total_paginas"],
            "pct_paginas_digitales": diag["pct_paginas_digitales"],
            "pct_paginas_con_tablas": diag["pct_paginas_con_tablas"],
            "pct_paginas_con_imagenes": diag["pct_paginas_con_imagenes"],
            "num_tablas_total": diag["num_tablas_total"],
            "num_imagenes_total": diag["num_imagenes_total"],
            "tiene_columnas": diag["tiene_columnas"],
            "ruta_extraccion": ruta,
            "observaciones": "; ".join(obs) if obs else "",
        })
        print(f"     → {ruta} (digital={diag['pct_paginas_digitales']}%, tablas={diag['pct_paginas_con_tablas']}%, imgs={diag['pct_paginas_con_imagenes']}%, cols={diag['tiene_columnas']})")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n📄 Diagnóstico guardado en {OUTPUT_CSV}")

    # Resumen por ruta
    print("\n--- Distribución ruta_extraccion ---")
    print(out_df["ruta_extraccion"].value_counts().to_string())

    # Detalle para revisión
    print("\n--- Detalle por documento ---")
    for _, r in out_df.iterrows():
        print(f"  {r['nombre_archivo']:<60} {r['ruta_extraccion']:<12} dig={r['pct_paginas_digitales']:>5}% tab={r['pct_paginas_con_tablas']:>5}% img={r['pct_paginas_con_imagenes']:>5}%")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extracción solo para docs faltantes"""

import re
import io
from pathlib import Path

import fitz
import pdfplumber
import pytesseract
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
PDF_FOLDER = BASE / "corpus_upeu" / "pdfs"
OUTPUT_FOLDER = BASE / "corpus_upeu" / "txt_bruto_v2"
DIAGNOSTICO_CSV = BASE / "corpus_upeu" / "metadatos" / "diagnostico_v2.csv"

UMBRAL_PALABRAS_PAGINA = 30
DPI_OCR = 250
IDIOMA_OCR = "spa"

RE_PAGINA = re.compile(r"^\s*\d+\s*$")
RE_HEADER_FOOTER = re.compile(r"^(Universidad Peruana Uni[oó]n|UPeU|www\.upeu\.edu\.pe).*$", re.IGNORECASE | re.MULTILINE)

FALTANTES = [
    "TUPA V6 2023 UPeU.pdf",
    "reglamento_deauditoria_interna.pdf",
]

def extraer_digital(pdf_path: Path) -> str:
    partes = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            lineas = []
            for linea in page.get_text().splitlines():
                linea = linea.strip()
                if not linea or RE_PAGINA.match(linea) or RE_HEADER_FOOTER.match(linea):
                    continue
                lineas.append(linea)
            if lineas:
                partes.append("\n".join(lineas))
    texto = "\n\n".join(partes)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                for t_idx, tabla in enumerate(page.extract_tables() or [], 1):
                    filas = [" | ".join(str(c) if c else "" for c in fila) for fila in tabla if any(fila)]
                    if filas:
                        texto += f"\n\n[TABLA p{i}.{t_idx}]\n" + "\n".join(filas)
    except Exception as e:
        print(f"  ⚠️  Tablas: {e}")
    return texto

def extraer_mixto(pdf_path: Path) -> str:
    partes = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, 1):
            texto_digital = page.get_text()
            palabras = len([w for w in texto_digital.split() if len(w) > 2])
            if palabras >= UMBRAL_PALABRAS_PAGINA:
                lineas = [l.strip() for l in texto_digital.splitlines() if l.strip() and not RE_PAGINA.match(l.strip()) and not RE_HEADER_FOOTER.match(l.strip())]
                if lineas:
                    partes.append("\n".join(lineas))
                print(f"    Pág {page_num}: digital ({palabras} palabras)")
            else:
                try:
                    pix = page.get_pixmap(dpi=DPI_OCR)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    texto_ocr = pytesseract.image_to_string(img, lang=IDIOMA_OCR)
                    if texto_ocr.strip():
                        partes.append(f"--- Pág {page_num} (OCR) ---\n{texto_ocr.strip()}")
                    print(f"    Pág {page_num}: OCR ({len(texto_ocr.split())} palabras)")
                except Exception as e:
                    print(f"    Pág {page_num}: error OCR -> {e}")
    texto = "\n\n".join(partes)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                for t_idx, tabla in enumerate(page.extract_tables() or [], 1):
                    filas = [" | ".join(str(c) if c else "" for c in fila) for fila in tabla if any(fila)]
                    if filas:
                        texto += f"\n\n[TABLA p{i}.{t_idx}]\n" + "\n".join(filas)
    except Exception as e:
        print(f"  ⚠️  Tablas: {e}")
    return texto

def extraer_tupa_tablas(pdf_path: Path) -> str:
    partes = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            texto_pag = page.extract_text() or ""
            if texto_pag.strip():
                lineas = [l.strip() for l in texto_pag.splitlines() if l.strip() and not RE_PAGINA.match(l.strip())]
                if lineas:
                    partes.append(f"--- Contexto p{i} ---\n" + "\n".join(lineas))
            for t_idx, tabla in enumerate(page.extract_tables() or [], 1):
                if not tabla:
                    continue
                header = " | ".join(str(c) if c else "" for c in tabla[0])
                partes.append(f"\n[TABLA p{i}.{t_idx}] {header}")
                for row in tabla[1:]:
                    if any(row):
                        partes.append(" | ".join(str(c) if c else "" for c in row))
    return "\n".join(partes)

def main():
    df = pd.read_csv(DIAGNOSTICO_CSV)
    for _, row in df.iterrows():
        nombre = row["nombre_archivo"]
        if nombre not in FALTANTES:
            continue
        ruta = row["ruta_extraccion"]
        pdf_path = PDF_FOLDER / nombre
        output_path = OUTPUT_FOLDER / f"{nombre}.txt"
        print(f"🔄 {nombre} → {ruta}...")

        try:
            if ruta == "DIGITAL":
                texto = extraer_digital(pdf_path)
            elif ruta == "MIXTO":
                texto = extraer_mixto(pdf_path)
            elif ruta == "TUPA_TABLAS":
                texto = extraer_tupa_tablas(pdf_path)
            elif ruta == "ESCANEADO":
                texto = extraer_escaneado(pdf_path)  # no implementado aquí, pero no hay ESCANEADO en faltantes
            else:
                print(f"  ❌ Ruta desconocida: {ruta}")
                continue

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(texto)
            print(f"  ✅ {len(texto)} chars")
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    import pandas as pd
    main()
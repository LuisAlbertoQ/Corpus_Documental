#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_extraccion_v2.py — Extracción por ruta según diagnostico_v2.csv
Rutas: DIGITAL, MIXTO, TUPA_TABLAS, ESCANEADO
Salida: corpus_upeu/txt_bruto_v2/{nombre_archivo}.txt
"""

import re
import io
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
PDF_FOLDER = BASE / "corpus_upeu" / "pdfs"
DIAGNOSTICO_CSV = BASE / "corpus_upeu" / "metadatos" / "diagnostico_v2.csv"
OUTPUT_FOLDER = BASE / "corpus_upeu" / "txt_bruto_v2"
METADATA_FOLDER = BASE / "corpus_upeu" / "metadatos"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
METADATA_FOLDER.mkdir(parents=True, exist_ok=True)

# Configuración
UMBRAL_PALABRAS_PAGINA = 30  # igual que notebook 1
DPI_OCR = 250
IDIOMA_OCR = "spa"

# Regex para limpieza básica
RE_PAGINA = re.compile(r"^\s*\d+\s*$")
RE_HEADER_FOOTER = re.compile(
    r"^(Universidad Peruana Uni[oó]n|UPeU|www\.upeu\.edu\.pe).*$",
    re.IGNORECASE | re.MULTILINE
)


def extraer_digital(pdf_path: Path) -> str:
    """Extracción directa PyMuPDF + tablas pdfplumber."""
    partes = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, 1):
            texto = page.get_text()
            # Filtrar líneas vacías, folios, headers/footers
            lineas_limpias = []
            for linea in texto.splitlines():
                linea = linea.strip()
                if not linea:
                    continue
                if RE_PAGINA.match(linea):
                    continue
                if RE_HEADER_FOOTER.match(linea):
                    continue
                lineas_limpias.append(linea)
            if lineas_limpias:
                partes.append("\n".join(lineas_limpias))
    texto_principal = "\n\n".join(partes)

    # Tablas con pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                tablas = page.extract_tables() or []
                for t_idx, tabla in enumerate(tablas, 1):
                    filas = [" | ".join(str(c) if c else "" for c in fila) for fila in tabla if any(fila)]
                    if filas:
                        texto_principal += f"\n\n[TABLA p{i}.{t_idx}]\n" + "\n".join(filas)
    except Exception as e:
        print(f"  ⚠️  Tablas: {e}")

    return texto_principal


def extraer_mixto(pdf_path: Path) -> str:
    """Híbrido: página con texto → digital, página sin texto → OCR."""
    partes = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, 1):
            texto_digital = page.get_text()
            palabras = len([w for w in texto_digital.split() if len(w) > 2])

            if palabras >= UMBRAL_PALABRAS_PAGINA:
                # Filtrar igual que digital
                lineas = []
                for linea in texto_digital.splitlines():
                    linea = linea.strip()
                    if not linea or RE_PAGINA.match(linea) or RE_HEADER_FOOTER.match(linea):
                        continue
                    lineas.append(linea)
                if lineas:
                    partes.append("\n".join(lineas))
                print(f"    Pág {page_num}: digital ({palabras} palabras)")
            else:
                # OCR
                try:
                    pix = page.get_pixmap(dpi=DPI_OCR)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    texto_ocr = pytesseract.image_to_string(img, lang=IDIOMA_OCR)
                    if texto_ocr.strip():
                        partes.append(f"--- Pág {page_num} (OCR) ---\n{texto_ocr.strip()}")
                        print(f"    Pág {page_num}: OCR ({len(texto_ocr.split())} palabras)")
                except Exception as e:
                    print(f"    Pág {page_num}: error OCR -> {e}")

    texto_principal = "\n\n".join(partes)

    # Tablas (siempre intentamos)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                tablas = page.extract_tables() or []
                for t_idx, tabla in enumerate(tablas, 1):
                    filas = [" | ".join(str(c) if c else "" for c in fila) for fila in tabla if any(fila)]
                    if filas:
                        texto_principal += f"\n\n[TABLA p{i}.{t_idx}]\n" + "\n".join(filas)
    except Exception as e:
        print(f"  ⚠️  Tablas: {e}")

    return texto_principal


def extraer_tupa_tablas(pdf_path: Path) -> str:
    """Modo tabla-primero: extrae tablas como filas estructuradas, texto como contexto."""
    partes = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            # Texto de la página (contexto)
            texto_pag = page.extract_text() or ""
            if texto_pag.strip():
                lineas = [l.strip() for l in texto_pag.splitlines() if l.strip() and not RE_PAGINA.match(l.strip())]
                if lineas:
                    partes.append(f"--- Contexto p{i} ---\n" + "\n".join(lineas))

            # Tablas como filas
            tablas = page.extract_tables() or []
            for t_idx, tabla in enumerate(tablas, 1):
                if not tabla:
                    continue
                # Encabezado de tabla
                header = tabla[0] if tabla else []
                header_str = " | ".join(str(c) if c else "" for c in header)
                partes.append(f"\n[TABLA p{i}.{t_idx}] {header_str}")
                # Filas de datos
                for row in tabla[1:]:
                    if any(row):
                        row_str = " | ".join(str(c) if c else "" for c in row)
                        partes.append(row_str)
    return "\n".join(partes)


def extraer_escaneado(pdf_path: Path) -> str:
    """OCR completo página a página."""
    partes = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, 1):
            try:
                pix = page.get_pixmap(dpi=DPI_OCR)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                texto_ocr = pytesseract.image_to_string(img, lang=IDIOMA_OCR)
                if texto_ocr.strip():
                    partes.append(f"--- Pág {page_num} (OCR) ---\n{texto_ocr.strip()}")
                print(f"    Pág {page_num}: OCR ({len(texto_ocr.split())} palabras)")
            except Exception as e:
                print(f"    Pág {page_num}: error OCR -> {e}")
    return "\n\n".join(partes)


def main():
    if not DIAGNOSTICO_CSV.exists():
        print(f"❌ No existe {DIAGNOSTICO_CSV}")
        return

    df = pd.read_csv(DIAGNOSTICO_CSV)
    print(f"Procesando {len(df)} documentos según diagnóstico...")

    stats = {"DIGITAL": 0, "MIXTO": 0, "TUPA_TABLAS": 0, "ESCANEADO": 0, "ERROR": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extrayendo"):
        nombre = row["nombre_archivo"]
        ruta = row["ruta_extraccion"]
        pdf_path = PDF_FOLDER / nombre
        output_path = OUTPUT_FOLDER / f"{nombre}.txt"

        try:
            if ruta == "DIGITAL":
                texto = extraer_digital(pdf_path)
            elif ruta == "MIXTO":
                texto = extraer_mixto(pdf_path)
            elif ruta == "TUPA_TABLAS":
                texto = extraer_tupa_tablas(pdf_path)
            elif ruta == "ESCANEADO":
                texto = extraer_escaneado(pdf_path)
            else:
                print(f"  ❌ {nombre}: ruta desconocida {ruta}")
                stats["ERROR"] += 1
                continue

            # Guardar
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(texto)

            stats[ruta] += 1
            print(f"  ✅ {nombre} → {ruta} ({len(texto)} chars)")

        except Exception as e:
            print(f"  ❌ {nombre}: error -> {e}")
            stats["ERROR"] += 1

    print("\n--- Resumen extracción ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Metadatos de extracción
    meta_rows = []
    for _, row in df.iterrows():
        nombre = row["nombre_archivo"]
        output_path = OUTPUT_FOLDER / f"{nombre}.txt"
        chars = len(output_path.read_text(encoding="utf-8")) if output_path.exists() else 0
        meta_rows.append({
            "nombre_archivo": nombre,
            "ruta_extraccion": row["ruta_extraccion"],
            "caracteres_extraidos": chars,
            "tipo_documental": row["tipo_documental"],
            "categoria_tematica": row["categoria_tematica"],
        })

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(METADATA_FOLDER / "extraccion_v2_metadatos.csv", index=False, encoding="utf-8")
    print(f"\n📄 Metadatos de extracción guardados en {METADATA_FOLDER / 'extraccion_v2_metadatos.csv'}")


if __name__ == "__main__":
    main()
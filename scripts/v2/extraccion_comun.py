#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extraccion_comun.py — Funciones de extracción PDF → texto compartidas (F4).

Cambios respecto a V2:
  * Cada página empieza con un marcador [[PAG N]] (config.MARCADOR_PAGINA).
    El chunking lo elimina del texto y lo usa para llenar el campo `pagina`.
  * Las tablas se insertan EN SU PÁGINA (no al final del documento) y el texto
    de la página se toma FUERA de las tablas → sin duplicar contenido.
  * Celdas con saltos de línea se aplanan para que cada fila sea una línea.
  * Ya no se escriben marcadores "--- Pág N (OCR) ---" ni "--- Contexto pN ---".
"""

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from config import UMBRAL_PALABRAS_PAGINA, DPI_OCR, IDIOMA_OCR, MARCADOR_PAGINA
from chunking import normalizar_espacios

RE_FOLIO = re.compile(r"^\s*\d+\s*$")
RE_HEADER_FOOTER = re.compile(r"^(Universidad Peruana Uni[oó]n|UPeU|www\.upeu\.edu\.pe)\b.*$", re.IGNORECASE)
# M5: pie de página con la dirección postal (sale por OCR en 18 documentos). Las URLs con
# ruta dentro del texto (p.ej. .../comite-etica/) son contenido útil y NO se tocan.
RE_PIE_DIRECCION = re.compile(r"Casilla\s*3564|postmast\S*\s*\S*upeu\.edu\.pe", re.IGNORECASE)


@dataclass
class ResultadoExtraccion:
    texto: str
    paginas: int = 0
    paginas_ocr: int = 0
    num_tablas: int = 0
    errores: list = field(default_factory=list)


def limpiar_lineas(texto: str) -> str:
    """Quita líneas vacías, folios y encabezados/pies institucionales."""
    lineas = []
    for linea in normalizar_espacios(texto).splitlines():
        linea = linea.strip()
        if not linea or RE_FOLIO.match(linea) or RE_HEADER_FOOTER.match(linea) or RE_PIE_DIRECCION.search(linea):
            continue
        lineas.append(linea)
    return "\n".join(lineas)


def contar_palabras(texto: str) -> int:
    return len([w for w in texto.split() if len(w) > 2])


def tabla_a_texto(filas: list, etiqueta: str) -> str:
    """Convierte una tabla de pdfplumber en líneas 'celda | celda'."""
    lineas = []
    for fila in filas or []:
        if not fila or not any(fila):
            continue
        celdas = [" ".join(str(c).split()) if c else "" for c in fila]
        lineas.append(" | ".join(celdas))
    if not lineas:
        return ""
    return f"[TABLA {etiqueta}]\n" + "\n".join(lineas)


def _fuera_de_tablas(bboxes):
    """Predicado para page.filter(): conserva objetos cuyo centro no cae en ninguna tabla."""
    def predicado(obj):
        if "x0" not in obj or "top" not in obj:
            return True
        cx = (obj["x0"] + obj.get("x1", obj["x0"])) / 2
        cy = (obj["top"] + obj.get("bottom", obj["top"])) / 2
        for x0, top, x1, bottom in bboxes:
            if x0 <= cx <= x1 and top <= cy <= bottom:
                return False
        return True
    return predicado


def ocr_pagina(page_fitz) -> str:
    import pytesseract
    from PIL import Image
    pix = page_fitz.get_pixmap(dpi=DPI_OCR)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=IDIOMA_OCR)


def extraer_pagina_digital(page_fitz, page_plumber, num_pagina: int, forzar_plumber: bool = False):
    """Texto de una página digital con sus tablas en el lugar correcto.

    Sin tablas → texto de PyMuPDF (mismo comportamiento que V2).
    Con tablas → texto de pdfplumber fuera de las tablas + tablas como filas.
    Devuelve (texto, num_tablas).
    """
    try:
        tablas = page_plumber.find_tables()
    except Exception:
        tablas = []

    if not tablas and not forzar_plumber:
        return limpiar_lineas(page_fitz.get_text()), 0

    bboxes = [t.bbox for t in tablas]
    try:
        texto = page_plumber.filter(_fuera_de_tablas(bboxes)).extract_text() or ""
    except Exception:
        texto = page_fitz.get_text() if not tablas else ""
    partes = [limpiar_lineas(texto)]
    n = 0
    for idx, tabla in enumerate(tablas, 1):
        bloque = tabla_a_texto(tabla.extract(), f"p{num_pagina}.{idx}")
        if bloque:
            partes.append(bloque)
            n += 1
    return "\n\n".join(p for p in partes if p), n


def extraer_pdf(pdf_path: Path, ruta: str, log=print) -> ResultadoExtraccion:
    """Extrae un PDF según su ruta de diagnóstico: DIGITAL, MIXTO, TUPA_TABLAS o ESCANEADO."""
    if ruta not in {"DIGITAL", "MIXTO", "TUPA_TABLAS", "ESCANEADO"}:
        raise ValueError(f"Ruta desconocida: {ruta}")

    res = ResultadoExtraccion(texto="")
    paginas_txt = []
    with fitz.open(pdf_path) as doc, pdfplumber.open(pdf_path) as pdf:
        res.paginas = len(doc)
        for i, page_fitz in enumerate(doc):
            num = i + 1
            page_plumber = pdf.pages[i]
            texto, usar_ocr = "", False

            if ruta == "ESCANEADO":
                usar_ocr = True
            elif ruta == "MIXTO":
                usar_ocr = contar_palabras(page_fitz.get_text()) < UMBRAL_PALABRAS_PAGINA

            try:
                if usar_ocr:
                    texto = limpiar_lineas(ocr_pagina(page_fitz))
                    res.paginas_ocr += 1
                else:
                    texto, n_tab = extraer_pagina_digital(
                        page_fitz, page_plumber, num, forzar_plumber=(ruta == "TUPA_TABLAS")
                    )
                    res.num_tablas += n_tab
            except Exception as e:  # una página fallida no tumba el documento
                res.errores.append(f"p{num}: {e}")
                log(f"    ⚠️  Pág {num}: {e}")

            if texto.strip():
                paginas_txt.append(MARCADOR_PAGINA.format(n=num) + "\n" + texto.strip())

    res.texto = "\n\n".join(paginas_txt)
    return res

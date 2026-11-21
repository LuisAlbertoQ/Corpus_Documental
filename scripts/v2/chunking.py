#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunking.py — Estructuración y chunking del corpus (lógica pura, sin modelos).

Lo usa 05_chunking_v2.py (con el tokenizer real de mpnet) y los tests
(con un contador de tokens falso). No importa config.py: todos los
parámetros llegan como argumentos.

Garantías del diseño V3 (verificadas por 06_validacion_corpus.py):
  * Cobertura: todo el texto del .txt termina en algún chunk. Se conserva el
    preámbulo (F2) y los encabezados cortos se anteponen al bloque siguiente.
  * Trazabilidad: chunk.texto == texto_limpio[char_inicio:char_fin] siempre.
  * Sin truncamiento: tokens(texto_embedding) + especiales <= max_seq_length.
  * Cortes en límites de párrafo / oración / ítem / fila de tabla, no a mitad
    de palabra (salvo una "oración" que por sí sola exceda el presupuesto).
  * Cada chunk lleva un breadcrumb de contexto solo en texto_embedding (F5):
    "Documento | TÍTULO II … | CAPÍTULO III … | Artículo 49°".
  * Todas las partes de un bloque comparten grupo_articulo_id (F12).
"""

import bisect
import re
from dataclasses import dataclass
from typing import Callable

ContarTokens = Callable[[str], int]

# ---------------------------------------------------------------------------
# Marcadores de página
# ---------------------------------------------------------------------------
# [[PAG N]] lo escribe extraccion_comun.py (V3). Los otros dos son de V2 y se
# aceptan para poder re-chunkear txt_bruto_v2 sin re-extraer.
RE_MARCADOR_PAGINA = re.compile(
    r"^[ \t]*(?:\[\[PAG (\d+)\]\]|--- Pág (\d+) \(OCR\) ---|--- Contexto p(\d+) ---)[ \t]*(?:\n|$)",
    re.MULTILINE,
)


def separar_paginas(texto: str) -> tuple[str, list[tuple[int, int]]]:
    """Quita los marcadores de página.

    Devuelve (texto_limpio, paginas) donde `paginas` es una lista ordenada de
    (offset_en_texto_limpio, numero_pagina).
    """
    partes, paginas = [], []
    pos, largo = 0, 0
    for m in RE_MARCADOR_PAGINA.finditer(texto):
        partes.append(texto[pos:m.start()])
        largo += m.start() - pos
        numero = next(g for g in m.groups() if g)
        paginas.append((largo, int(numero)))
        pos = m.end()
    partes.append(texto[pos:])
    return "".join(partes), paginas


def pagina_en(offset: int, paginas: list[tuple[int, int]]) -> str:
    """Número de página (como texto) en el que cae `offset`; '' si no se sabe."""
    if not paginas:
        return ""
    offsets = [o for o, _ in paginas]
    i = bisect.bisect_right(offsets, offset) - 1
    return str(paginas[i][1]) if i >= 0 else ""


# ---------------------------------------------------------------------------
# Encabezados estructurales (F3)
# ---------------------------------------------------------------------------
# * [ \t]+ en vez de \s+ → el encabezado no cruza saltos de línea
#   (antes "Título\nIntroducción" era un falso TÍTULO).
# * \b tras el numeral → "Título de…" / "Capítulo del…" ya no matchean
#   (antes la D de "DE" se tomaba como número romano: "TÍTULO D").
# * SUBCAPÍTULO / APARTADO cuentan como sección (antes `seccion` era 100% nulo).
_NUM = r"(?:[IVXLCDM]+|\d+)\b"
RE_ENCABEZADO = re.compile(
    r"^[ \t]*(?P<header>"
    r"(?P<titulo>T[IÍ]TULO[ \t]+" + _NUM + r")"
    r"|(?P<capitulo>CAP[IÍ]TULO[ \t]+" + _NUM + r")"
    r"|(?P<seccion>(?:SUB[ \t]?CAP[IÍ]TULO|SECCI[OÓ]N|APARTADO)[ \t]+(?:[IVXLCDM]+\b|\d+(?:\.\d+)*))"
    r"|(?P<articulo>ART[IÍ]CULO[ \t]+\d+(?:\.\d+)?(?:[ \t]*[º°])?(?:-[A-Z]\b)?)"
    r"|(?P<paso>PASO[ \t]+\d+)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
TIPOS_ESTRUCTURALES = ("titulo", "capitulo", "seccion")


@dataclass
class Bloque:
    tipo: str        # titulo|capitulo|seccion|articulo|paso|preambulo|texto_libre
    header: str      # "CAPÍTULO III", "Artículo 49°", "" para preámbulo
    nombre: str      # "DE LAS BECAS" (solo títulos/capítulos/secciones)
    inicio: int
    fin: int


def _nombre_de_encabezado(texto: str, inicio_header: int, fin_header: int) -> str:
    """Nombre que acompaña a un TÍTULO/CAPÍTULO/SECCIÓN (resto de línea o línea siguiente)."""
    fin_linea = texto.find("\n", fin_header)
    fin_linea = len(texto) if fin_linea == -1 else fin_linea
    resto = texto[fin_header:fin_linea].strip(" \t.:-–—")
    if resto:
        return resto if len(resto) <= 120 else ""
    sig_fin = texto.find("\n", fin_linea + 1)
    sig_fin = len(texto) if sig_fin == -1 else sig_fin
    siguiente = texto[fin_linea + 1:sig_fin].strip()
    if siguiente and len(siguiente) <= 120 and not RE_ENCABEZADO.match(siguiente):
        return siguiente
    return ""


# M6: línea de índice / tabla de contenidos: "CAPÍTULO III DE LAS BECAS ........ 11"
RE_LINEA_INDICE = re.compile(r"(?:\.{4,}|…{2,}|(?:\.[ \t]){4,})[ \t]*\d{1,3}[ \t]*$", re.MULTILINE)


def _es_entrada_de_indice(texto: str, pos: int) -> bool:
    """True si la línea que empieza en `pos` (o la siguiente, si el título sigue abajo) es de índice."""
    fin1 = texto.find("\n", pos)
    fin1 = len(texto) if fin1 == -1 else fin1
    if RE_LINEA_INDICE.search(texto, pos, fin1):
        return True
    fin2 = texto.find("\n", fin1 + 1)
    fin2 = len(texto) if fin2 == -1 else fin2
    return fin1 < len(texto) and bool(RE_LINEA_INDICE.search(texto, fin1 + 1, fin2))


def es_chunk_indice(cuerpo: str) -> bool:
    """Un chunk es índice si al menos 3 líneas, o el 30% de ellas, son entradas de índice."""
    lineas = [l for l in cuerpo.splitlines() if l.strip()]
    n = sum(1 for l in lineas if RE_LINEA_INDICE.search(l))
    return bool(lineas) and (n >= 3 or n / len(lineas) >= 0.3)


def detectar_bloques(texto: str) -> list[Bloque]:
    """Divide el texto limpio en bloques estructurales contiguos que lo cubren entero."""
    matches = []
    for m in RE_ENCABEZADO.finditer(texto):
        header = m.group("header")
        # Una referencia en prosa ("…según el\nartículo 12…") empieza en minúscula.
        if not header[0].isupper():
            continue
        # M6: una entrada del índice no es un encabezado real (contaminaba `capitulo`).
        if _es_entrada_de_indice(texto, m.start()):
            continue
        matches.append(m)

    bloques: list[Bloque] = []
    if not matches:
        if texto.strip():
            bloques.append(Bloque("texto_libre", "", "", 0, len(texto)))
        return bloques

    if texto[:matches[0].start()].strip():                       # F2: preámbulo (V2 lo descartaba)
        bloques.append(Bloque("preambulo", "", "", 0, matches[0].start()))

    for i, m in enumerate(matches):
        inicio = 0 if (i == 0 and not bloques) else m.start()
        fin = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        tipo = next(t for t in ("titulo", "capitulo", "seccion", "articulo", "paso") if m.group(t))
        header = " ".join(m.group("header").split())
        nombre = _nombre_de_encabezado(texto, m.start("header"), m.end("header")) if tipo in TIPOS_ESTRUCTURALES else ""
        bloques.append(Bloque(tipo, header, nombre, inicio, fin))
    return bloques


# ---------------------------------------------------------------------------
# División de bloques largos
# ---------------------------------------------------------------------------
_RE_CORTE = re.compile(
    r"\n[ \t]*\n"                                                     # párrafo
    r"|\n(?=[ \t]*(?:\d+(?:\.\d+)*[.)°º]?[ \t]|[a-zñ][.)][ \t]|[-•▪·][ \t]|\[TABLA))"  # ítems / numerales / tabla
    r"|(?<=[.;:])[ \t]*\n"                                            # línea que cierra con puntuación
    r"|(?<=[.;])[ \t]+(?=[A-ZÁÉÍÓÚÑ¿¡])"                               # oración dentro de la línea
)
_RE_TABLA = re.compile(r"^\[TABLA [^\]]*\][^\n]*\n?", re.MULTILINE)
COLA_MIN_TOKENS = 15


def _segmentos_tabla(texto: str, inicio: int, fin: int) -> list[tuple[int, int, int]]:
    """(inicio, fin, fin_cabecera) de cada tabla dentro de [inicio, fin). Termina en línea en blanco."""
    segs = []
    for m in _RE_TABLA.finditer(texto, inicio, fin):
        fin_tabla = texto.find("\n\n", m.end(), fin)
        fin_tabla = fin if fin_tabla == -1 else fin_tabla
        fin_cab = texto.find("\n", m.end(), fin_tabla)          # primera fila = cabecera de columnas
        fin_cab = fin_tabla if fin_cab == -1 else fin_cab
        segs.append((m.start(), fin_tabla, fin_cab))
    return segs


def _unidades(texto: str, inicio: int, fin: int) -> list[tuple[int, int]]:
    """Spans mínimos indivisibles (oraciones, ítems, filas de tabla) dentro de [inicio, fin)."""
    cortes = {inicio, fin}
    for m in _RE_CORTE.finditer(texto, inicio, fin):
        cortes.add(m.end())
    for t_ini, t_fin, _ in _segmentos_tabla(texto, inicio, fin):   # cada fila de tabla es una unidad
        p = texto.find("\n", t_ini, t_fin)
        while p != -1:
            cortes.add(p + 1)
            p = texto.find("\n", p + 1, t_fin)
    puntos = sorted(c for c in cortes if inicio <= c <= fin)
    unidades = []
    for a, b in zip(puntos, puntos[1:]):
        if texto[a:b].strip():
            unidades.append((a, b))
        elif unidades:                                          # espacio en blanco → al anterior
            unidades[-1] = (unidades[-1][0], b)
    return unidades


def _partir_por_palabras(texto: str, inicio: int, fin: int, max_tok: int, contar: ContarTokens):
    """Último recurso para una unidad que excede el presupuesto: corta entre palabras."""
    palabras = [(m.start(), m.end()) for m in re.finditer(r"\S+", texto[inicio:fin])]
    piezas, ini_pieza, ultimo_fin = [], None, None
    for a, b in palabras:
        a, b = a + inicio, b + inicio
        if ini_pieza is None:
            ini_pieza = a
        elif contar(texto[ini_pieza:b]) > max_tok:
            piezas.append((ini_pieza, ultimo_fin))
            ini_pieza = a
        ultimo_fin = b
    if ini_pieza is not None:
        piezas.append((ini_pieza, ultimo_fin))
    return piezas


def dividir_span(texto: str, inicio: int, fin: int, max_tok: int, overlap_tok: int,
                 contar: ContarTokens) -> list[tuple[int, int]]:
    """Divide [inicio, fin) en piezas <= max_tok uniendo unidades enteras, con solape por unidades."""
    if contar(texto[inicio:fin]) <= max_tok:
        return [(inicio, fin)]

    unidades = []
    for a, b in _unidades(texto, inicio, fin):
        if contar(texto[a:b]) > max_tok:
            unidades.extend(_partir_por_palabras(texto, a, b, max_tok, contar))
        else:
            unidades.append((a, b))

    piezas: list[tuple[int, int]] = []
    actuales: list[tuple[int, int]] = []
    for u in unidades:
        if actuales and contar(texto[actuales[0][0]:u[1]]) > max_tok:
            piezas.append((actuales[0][0], actuales[-1][1]))
            # Solape: últimas unidades de la pieza anterior hasta overlap_tok (nunca todas).
            solape, tok = [], 0
            for v in reversed(actuales[1:]):
                t = contar(texto[v[0]:v[1]])
                if tok + t > overlap_tok:
                    break
                solape.insert(0, v)
                tok += t
            actuales = solape + [u]
            if contar(texto[actuales[0][0]:u[1]]) > max_tok:
                actuales = [u]
        else:
            actuales.append(u)
    if actuales:
        piezas.append((actuales[0][0], actuales[-1][1]))
    # M5: una cola diminuta (p.ej. "8. Otras que señalen los reglamentos.") se une a la pieza
    # anterior si así no se pasa del presupuesto; si no, se queda sola.
    if len(piezas) > 1 and contar(texto[piezas[-1][0]:piezas[-1][1]]) < COLA_MIN_TOKENS \
            and contar(texto[piezas[-2][0]:piezas[-1][1]]) <= max_tok:
        piezas[-2:] = [(piezas[-2][0], piezas[-1][1])]
    return piezas


# ---------------------------------------------------------------------------
# Lectura del CSV de chunks (compartida por 06, 07 y 08)
# ---------------------------------------------------------------------------
COLUMNAS_TEXTO = [
    "documento", "nombre_documento", "chunk_id", "texto", "texto_embedding", "tipo_documental",
    "categoria_tematica", "ruta_extraccion", "titulo_estructura", "titulo_documento", "capitulo",
    "seccion", "articulo", "pagina", "pagina_fin", "tipo_bloque", "grupo_articulo_id", "padre_chunk_id",
]


def leer_chunks(path):
    """Lee chunks_<version>.csv sin que pandas convierta textos a NaN o números."""
    import pandas as pd
    cabecera = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, dtype={c: str for c in COLUMNAS_TEXTO if c in cabecera}, keep_default_na=False)
    if "es_parte" in df.columns:
        df["es_parte"] = df["es_parte"].astype(str).str.lower() == "true"
    return df


# ---------------------------------------------------------------------------
# Documento → chunks
# ---------------------------------------------------------------------------
def nombre_legible(nombre_archivo: str) -> str:
    """'REGLAMENTO-de-becas_2021.pdf' → 'REGLAMENTO de becas 2021' (F11: nunca vacío)."""
    base = re.sub(r"\.pdf$", "", nombre_archivo, flags=re.IGNORECASE)
    return " ".join(re.sub(r"[_\-]+", " ", base).split())


def _recortar(s: str, max_chars: int = 140) -> str:
    return s if len(s) <= max_chars else s[:max_chars].rsplit(" ", 1)[0] + "…"


def partes_breadcrumb(modo: str, nombre_doc: str, titulo: str, capitulo: str, seccion: str, articulo: str) -> list[str]:
    """M2: niveles de contexto según BREADCRUMB_MODO (completo | documento | ninguno)."""
    if modo == "ninguno":
        return []
    if modo == "documento":
        return [nombre_doc, articulo]
    if modo == "completo":
        return [nombre_doc, titulo, capitulo, seccion, articulo]
    raise ValueError(f"breadcrumb_modo inválido: {modo!r}")


def construir_breadcrumb(partes: list[str], max_tok: int, contar: ContarTokens) -> str:
    """Une las partes no vacías con ' | ' y, si excede max_tok, quita niveles intermedios."""
    partes = [_recortar(p) for p in partes if p]
    while len(partes) > 2 and contar(" | ".join(partes)) > max_tok:
        partes.pop(1)                     # se sacrifica primero el nivel más general tras el documento
    bc = " | ".join(partes)
    while bc and contar(bc) > max_tok:
        bc = bc[: int(len(bc) * 0.85)]
    return bc


def procesar_documento(
    nombre_archivo: str,
    texto_bruto: str,
    contar: ContarTokens,
    *,
    max_seq_length: int,
    chunk_max_tokens: int,
    overlap_tokens: int,
    min_tokens_bloque: int,
    max_tokens_breadcrumb: int,
    tokens_especiales: int = 2,
    breadcrumb_modo: str = "completo",
    metadatos_doc: dict | None = None,
    descartados: list | None = None,
) -> list[dict]:
    """Convierte el texto extraído de un documento en una lista de chunks con metadatos.

    Si se pasa `descartados` (lista), ahí se agregan los spans excluidos (duplicados e índices).
    """
    metadatos_doc = metadatos_doc or {}
    texto, paginas = separar_paginas(texto_bruto)
    bloques = detectar_bloques(texto)
    nombre_doc = nombre_legible(nombre_archivo)
    stem = re.sub(r"\.pdf$", "", nombre_archivo, flags=re.IGNORECASE)
    limite_embedding = max_seq_length - tokens_especiales

    ctx = {"titulo": "", "capitulo": "", "seccion": ""}
    chunks: list[dict] = []
    vistos: set[str] = set()
    pendiente_inicio = None

    for idx_b, b in enumerate(bloques):
        etiqueta = f"{b.header} {b.nombre}".strip()
        if b.tipo == "titulo":
            ctx = {"titulo": etiqueta, "capitulo": "", "seccion": ""}
        elif b.tipo == "capitulo":
            ctx.update(capitulo=etiqueta, seccion="")
        elif b.tipo == "seccion":
            ctx["seccion"] = etiqueta
        articulo = b.header if b.tipo in ("articulo", "paso") else ""

        inicio = pendiente_inicio if pendiente_inicio is not None else b.inicio
        es_ultimo = idx_b == len(bloques) - 1
        # Encabezados cortos ("CAPÍTULO III / DE LAS BECAS") se anteponen al bloque siguiente.
        # Se mide el bloque solo (no el prefijo acumulado): varios encabezados seguidos
        # ("TÍTULO I" + "CAPÍTULO I") terminan todos dentro del primer artículo.
        if b.tipo in TIPOS_ESTRUCTURALES and not es_ultimo and contar(texto[b.inicio:b.fin]) < min_tokens_bloque:
            pendiente_inicio = inicio
            continue
        pendiente_inicio = None

        bc_base = construir_breadcrumb(
            partes_breadcrumb(breadcrumb_modo, nombre_doc, ctx["titulo"], ctx["capitulo"], ctx["seccion"], articulo),
            max_tokens_breadcrumb, contar,
        )
        presupuesto = min(chunk_max_tokens, limite_embedding - contar(bc_base) - 1)
        tablas = _segmentos_tabla(texto, inicio, b.fin)

        for _ in range(6):   # reintenta con menos presupuesto si algún embedding se pasara del límite
            piezas = dividir_span(texto, inicio, b.fin, presupuesto, overlap_tokens, contar)
            nuevos = []
            for p_ini, p_fin in piezas:
                bruto = texto[p_ini:p_fin]
                p_ini += len(bruto) - len(bruto.lstrip())
                p_fin -= len(bruto) - len(bruto.rstrip())
                cuerpo = texto[p_ini:p_fin]
                if not cuerpo:
                    continue
                bc = bc_base
                # Pieza que empieza dentro de una tabla: añadir la cabecera de columnas al breadcrumb.
                for t_ini, t_fin, t_cab in tablas if breadcrumb_modo != "ninguno" else []:
                    if t_cab < p_ini < t_fin:
                        cabecera = f"Tabla: {_recortar(' '.join(texto[t_ini:t_cab].split()), 160)}"
                        candidato = f"{bc_base} | {cabecera}" if bc_base else cabecera
                        if contar(f"{candidato}\n{cuerpo}") <= limite_embedding:
                            bc = candidato
                        break
                emb = f"{bc}\n{cuerpo}" if bc else cuerpo
                nuevos.append((p_ini, p_fin, cuerpo, emb))
            if all(contar(e) <= limite_embedding for *_, e in nuevos):
                break
            presupuesto = int(presupuesto * 0.9)

        # Fuera del índice vectorial (quedan registrados en `descartados` para la cobertura):
        #  * duplicados exactos dentro del documento
        #  * M6: piezas que son el índice / tabla de contenidos del reglamento
        conservados = []
        for n in nuevos:
            motivo = "duplicado" if n[2] in vistos else ("indice" if es_chunk_indice(n[2]) else "")
            if motivo:
                if descartados is not None:
                    descartados.append({"documento": nombre_archivo, "motivo": motivo,
                                        "char_inicio": n[0], "char_fin": n[1], "texto": n[2]})
            else:
                conservados.append(n)
            vistos.add(n[2])
        nuevos = conservados
        total = len(nuevos)
        grupo_id = ""
        for j, (p_ini, p_fin, cuerpo, emb) in enumerate(nuevos, 1):
            chunk_id = f"{stem}_{len(chunks):04d}"
            grupo_id = chunk_id if j == 1 else grupo_id
            chunks.append({
                "documento": nombre_archivo,
                "nombre_documento": nombre_doc,
                "chunk_id": chunk_id,
                "texto": cuerpo,
                "texto_embedding": emb,
                "num_tokens": contar(cuerpo),
                "num_tokens_embedding": contar(emb),
                "num_chars": len(cuerpo),
                "tipo_documental": metadatos_doc.get("tipo_documental", ""),
                "categoria_tematica": metadatos_doc.get("categoria_tematica", ""),
                "ruta_extraccion": metadatos_doc.get("ruta_extraccion", ""),
                "titulo_estructura": ctx["titulo"],
                "titulo_documento": ctx["titulo"],     # alias V2 (OE5 compat); ver nombre_documento
                "capitulo": ctx["capitulo"],
                "seccion": ctx["seccion"],
                "articulo": articulo,
                "pagina": pagina_en(p_ini, paginas),
                "pagina_fin": pagina_en(max(p_fin - 1, p_ini), paginas),
                "tipo_bloque": b.tipo,
                "grupo_articulo_id": grupo_id,
                "padre_chunk_id": grupo_id if j > 1 else "",
                "es_parte": j > 1,
                "parte_num": j,
                "total_partes": total,
                "char_inicio": p_ini,
                "char_fin": p_fin,
            })
    return chunks

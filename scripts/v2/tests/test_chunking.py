"""Tests de chunking.py con un contador de tokens falso (1 palabra = 1 token).

Ejecutar:  docker exec oe1_corpus python -m pytest /home/jupyteruser/work/scripts/v2/tests -q
"""
import pytest

import chunking as C


def contar(texto: str) -> int:
    return len(texto.split())


PARAMS = dict(max_seq_length=60, chunk_max_tokens=40, overlap_tokens=8,
              min_tokens_bloque=6, max_tokens_breadcrumb=12, tokens_especiales=2)


def chunks_de(texto, nombre="REGLAMENTO PRUEBA 2025.pdf", **kw):
    return C.procesar_documento(nombre, texto, contar, **{**PARAMS, **kw})


# --- F3: encabezados -------------------------------------------------------
@pytest.mark.parametrize("linea, tipo", [
    ("TÍTULO III", "titulo"),
    ("Título II DE LOS ESTUDIANTES", "titulo"),
    ("CAPÍTULO IV", "capitulo"),
    ("Capítulo 2", "capitulo"),
    ("Artículo 49°", "articulo"),
    ("ARTÍCULO 12º.- El estudiante", "articulo"),
    ("Artículo 5-A", "articulo"),
    ("SUBCAPÍTULO I", "seccion"),
    ("Sección 2.1", "seccion"),
    ("Paso 3", "paso"),
])
def test_encabezados_validos(linea, tipo):
    m = C.RE_ENCABEZADO.match(linea)
    assert m and m.group(tipo), linea


@pytest.mark.parametrize("linea", [
    "Título de segunda especialidad",      # antes: "TÍTULO D"
    "TÍTULO DE MAESTRÍA",
    "Capítulo del reglamento",
    "Capítulo IVY",
    "Título\nIntroducción",                 # no cruza líneas
])
def test_falsos_encabezados(linea):
    assert C.RE_ENCABEZADO.match(linea) is None, linea


def test_referencia_en_minuscula_no_corta():
    texto = "Artículo 1° Objeto del reglamento según el\nartículo 12 del estatuto vigente."
    bloques = C.detectar_bloques(texto)
    assert [b.header for b in bloques] == ["Artículo 1°"]


# --- Páginas ---------------------------------------------------------------
def test_separar_paginas_y_pagina_en():
    raw = "[[PAG 1]]\nhola mundo\n\n[[PAG 2]]\nsegunda página\n--- Pág 3 (OCR) ---\ntercera"
    limpio, paginas = C.separar_paginas(raw)
    assert "PAG" not in limpio and "OCR" not in limpio
    assert C.pagina_en(limpio.index("hola"), paginas) == "1"
    assert C.pagina_en(limpio.index("segunda"), paginas) == "2"
    assert C.pagina_en(limpio.index("tercera"), paginas) == "3"
    assert C.pagina_en(0, []) == ""


# --- F2: preámbulo y cobertura ---------------------------------------------
DOC = """[[PAG 1]]
PRESENTACIÓN
Este reglamento regula las becas de la universidad y su alcance general.
TÍTULO I
DISPOSICIONES GENERALES
CAPÍTULO I
DE LAS BECAS
Artículo 1° La beca cubre la pensión de enseñanza del estudiante.
Artículo 2° Derogado.
[[PAG 2]]
Artículo 3° Requisitos para postular a la beca:
a) Tener promedio ponderado mayor a 15.
b) No tener deudas pendientes con la universidad.
c) Presentar la solicitud en la fecha indicada por bienestar universitario.
d) Adjuntar la ficha socioeconómica firmada por el estudiante y el apoderado.
e) Haber aprobado todos los cursos del ciclo anterior sin excepción alguna.
"""


def cobertura(texto_bruto, chunks):
    limpio, _ = C.separar_paginas(texto_bruto)
    cubierto = [False] * len(limpio)
    for ch in chunks:
        for i in range(ch["char_inicio"], ch["char_fin"]):
            cubierto[i] = True
    no_blanco = [i for i, c in enumerate(limpio) if not c.isspace()]
    return sum(cubierto[i] for i in no_blanco) / len(no_blanco)


def test_preambulo_se_conserva():
    chunks = chunks_de(DOC)
    assert chunks[0]["tipo_bloque"] == "preambulo"
    assert "PRESENTACIÓN" in chunks[0]["texto"]


def test_cobertura_total_y_trazabilidad():
    chunks = chunks_de(DOC)
    limpio, _ = C.separar_paginas(DOC)
    assert cobertura(DOC, chunks) == 1.0
    for ch in chunks:
        assert limpio[ch["char_inicio"]:ch["char_fin"]] == ch["texto"]


def test_encabezados_cortos_se_anteponen_al_articulo():
    chunks = chunks_de(DOC)
    art1 = next(c for c in chunks if c["articulo"] == "Artículo 1°")
    assert art1["texto"].startswith("TÍTULO I")          # no queda un chunk "TÍTULO I" suelto
    assert not any(c["texto"].strip() in {"TÍTULO I", "CAPÍTULO I"} for c in chunks)
    assert art1["capitulo"] == "CAPÍTULO I DE LAS BECAS"
    assert art1["titulo_estructura"] == "TÍTULO I DISPOSICIONES GENERALES"


def test_articulo_corto_no_se_fusiona():
    chunks = chunks_de(DOC)
    assert any(c["articulo"] == "Artículo 2°" and c["texto"] == "Artículo 2° Derogado." for c in chunks)


# --- P1/F5: límites de tokens y breadcrumb ---------------------------------
def test_ningun_embedding_supera_max_seq():
    chunks = chunks_de(DOC * 3, chunk_max_tokens=20, overlap_tokens=5)
    limite = PARAMS["max_seq_length"] - PARAMS["tokens_especiales"]
    assert chunks and all(c["num_tokens_embedding"] <= limite for c in chunks)
    assert all(c["num_tokens"] <= 20 for c in chunks)


def test_breadcrumb_en_texto_embedding_no_en_texto():
    chunks = chunks_de(DOC)
    art3 = next(c for c in chunks if c["articulo"] == "Artículo 3°")
    assert art3["texto_embedding"].startswith("REGLAMENTO PRUEBA 2025")
    assert "Artículo 3°" in art3["texto_embedding"].split("\n")[0]
    assert not art3["texto"].startswith("REGLAMENTO PRUEBA")


# --- F12: partes y grupos --------------------------------------------------
def test_partes_comparten_grupo_y_cortan_en_items():
    chunks = chunks_de(DOC, chunk_max_tokens=25, overlap_tokens=0)
    partes = [c for c in chunks if c["articulo"] == "Artículo 3°"]
    assert len(partes) > 1
    grupo = partes[0]["chunk_id"]
    assert all(p["grupo_articulo_id"] == grupo for p in partes)
    assert [p["parte_num"] for p in partes] == list(range(1, len(partes) + 1))
    assert all(p["total_partes"] == len(partes) for p in partes)
    assert partes[0]["padre_chunk_id"] == "" and all(p["padre_chunk_id"] == grupo for p in partes[1:])
    # Cortes en límites de ítem: cada parte (salvo la 1) empieza con "x)"
    assert all(p["texto"][:2] in {"a)", "b)", "c)", "d)", "e)"} for p in partes[1:])
    assert partes[0]["pagina"] == "2"


def test_overlap_repite_unidades():
    chunks = chunks_de(DOC, chunk_max_tokens=25, overlap_tokens=12)
    partes = [c for c in chunks if c["articulo"] == "Artículo 3°"]
    assert len(partes) > 1
    assert partes[1]["char_inicio"] < partes[0]["char_fin"]   # hay solape


def test_duplicados_dentro_del_documento_se_eliminan():
    texto = "Artículo 1° Texto repetido idéntico.\n" * 1 + "Artículo 1° Texto repetido idéntico.\n"
    chunks = chunks_de(texto)
    assert len([c for c in chunks if c["texto"].startswith("Artículo 1°")]) == 1


# --- Tablas ----------------------------------------------------------------
def test_tabla_se_parte_por_filas_con_cabecera_en_breadcrumb():
    filas = "\n".join(f"{i}.1 | Trámite número {i} de la tabla | S/ {i}0.00" for i in range(1, 30))
    texto = f"[[PAG 1]]\n[TABLA p1.1]\nCódigo | Procedimiento | Costo\n{filas}\n"
    chunks = chunks_de(texto, nombre="TUPA PRUEBA.pdf", chunk_max_tokens=30, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(c["texto"].split("\n")[0].count("|") == 2 for c in chunks[1:])       # empiezan en una fila entera
    assert any("Tabla: [TABLA p1.1] Código | Procedimiento | Costo" in c["texto_embedding"] for c in chunks[1:])
    assert cobertura(texto, chunks) == 1.0


def test_nombre_legible():
    assert C.nombre_legible("Reglamento-de-docentes_unionista.pdf") == "Reglamento de docentes unionista"


# --- M6: índices / tablas de contenido ---------------------------------------
INDICE = """[[PAG 1]]
ÍNDICE
CAPÍTULO I DISPOSICIONES GENERALES ............ 3
CAPÍTULO II DE LAS BECAS ...................... 5
CAPÍTULO III DE LOS REQUISITOS ................ 7
[[PAG 3]]
CAPÍTULO I
DISPOSICIONES GENERALES
Artículo 1° El presente reglamento regula las becas que otorga la universidad a sus estudiantes.
"""


def test_entradas_de_indice_no_son_encabezados():
    limpio, _ = C.separar_paginas(INDICE)
    headers = [b.header for b in C.detectar_bloques(limpio)]
    assert headers.count("CAPÍTULO I") == 1 and "CAPÍTULO II" not in headers


def test_chunk_indice_se_descarta_pero_cuenta_para_cobertura():
    descartados = []
    chunks = chunks_de(INDICE, descartados=descartados)
    assert any(d["motivo"] == "indice" for d in descartados)
    assert not any(C.es_chunk_indice(c["texto"]) for c in chunks)
    limpio, _ = C.separar_paginas(INDICE)
    cubierto = [False] * len(limpio)
    for s in chunks + descartados:
        for i in range(s["char_inicio"], s["char_fin"]):
            cubierto[i] = True
    assert all(cubierto[i] for i, ch in enumerate(limpio) if not ch.isspace())


def test_duplicado_queda_registrado():
    descartados = []
    chunks_de("Artículo 1° Texto repetido idéntico.\nArtículo 1° Texto repetido idéntico.\n", descartados=descartados)
    assert [d["motivo"] for d in descartados] == ["duplicado"]


# --- M2: modos de breadcrumb -------------------------------------------------
@pytest.mark.parametrize("modo, esperado", [
    ("completo", "REGLAMENTO PRUEBA 2025 | TÍTULO I DISPOSICIONES GENERALES | CAPÍTULO I DE LAS BECAS | Artículo 1°"),
    ("documento", "REGLAMENTO PRUEBA 2025 | Artículo 1°"),
    ("ninguno", "TÍTULO I"),
])
def test_breadcrumb_modos(modo, esperado):
    chunks = chunks_de(DOC, breadcrumb_modo=modo, max_tokens_breadcrumb=40)
    art1 = next(c for c in chunks if c["articulo"] == "Artículo 1°")
    assert art1["texto_embedding"].split("\n")[0] == esperado
    if modo == "ninguno":
        assert art1["texto_embedding"] == art1["texto"]


# --- M5: cola diminuta ------------------------------------------------------
def test_cola_diminuta_se_une_a_la_pieza_anterior():
    texto = "uno dos tres cuatro cinco seis siete ocho nueve diez. " * 3 + "fin."
    piezas = C.dividir_span(texto, 0, len(texto), max_tok=31, overlap_tok=0, contar=contar)
    assert all(contar(texto[a:b]) >= C.COLA_MIN_TOKENS or len(piezas) == 1 for a, b in piezas)


def test_nbsp_tras_articulo_se_detecta():
    # PDFs como Jefes de prácticas traen "Artículo\xa01": antes no se detectaba el encabezado.
    doc = ("[[PAG 1]]\nCAPÍTULO I\nDISPOSICIONES GENERALES\n"
           "Artículo\xa01.\xa0El presente reglamento regula la función de los jefes de práctica en la universidad.\n"
           "Artículo\xa02. Los jefes de práctica apoyan al docente en las sesiones de laboratorio y taller.\n")
    chunks = chunks_de(doc)
    arts = [c["articulo"] for c in chunks if c["articulo"]]
    assert len(arts) == 2 and "1" in arts[0] and "2" in arts[1]
    limpio, _ = C.separar_paginas(doc)
    assert "\xa0" not in limpio and len(limpio) == len(doc) - len("[[PAG 1]]\n")
    for ch in chunks:
        assert limpio[ch["char_inicio"]:ch["char_fin"]] == ch["texto"]

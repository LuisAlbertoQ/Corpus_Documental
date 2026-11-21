#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py — Fuente única de rutas y parámetros del pipeline del corpus UPeU.

Todos los scripts 01..08 hacen `from config import ...`. Funciona igual en el
host y dentro del contenedor Docker (/home/jupyteruser/work), porque BASE se
calcula desde la ubicación de este archivo: scripts/v2/config.py -> raíz.

La versión del corpus se elige con la variable de entorno CORPUS_VERSION
(por defecto "v3"). La V2 está protegida: ningún script escribe sobre ella
salvo que se defina ALLOW_OVERWRITE_V2=1 explícitamente.

Ejemplo (Docker):
    docker exec -e CORPUS_VERSION=v3 oe1_corpus python /home/jupyteruser/work/scripts/v2/05_chunking_v2.py
"""

import os
from pathlib import Path

# --- Rutas -------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]          # raíz del repo
CORPUS_DIR = BASE / "corpus_upeu"
PDF_DIR = CORPUS_DIR / "pdfs"
CONFIG_DIR = CORPUS_DIR / "config"                  # entradas manuales versionadas en git

VERSION = os.environ.get("CORPUS_VERSION", "v3").strip()
VERSIONES_PROTEGIDAS = {"v2"}                       # canónica, la consume OE5

META_DIR = CORPUS_DIR / "metadatos" / VERSION
TXT_DIR = CORPUS_DIR / f"txt_bruto_{VERSION}"
VSTORE_DIR = CORPUS_DIR / f"vector_store_{VERSION}"

# --- Modelo de embeddings (M1: configurable por perfil) -----------------------
# MODEL_KEY elige un perfil. "mpnet" es el de V2/OE5 y conserva los nombres de
# archivo y colección originales; cualquier otro añade el sufijo _<modelo> para
# poder comparar modelos dentro de la misma versión sin pisarse.
# Cambiar el modelo final exige re-indexar Y ajustar OE5 (prefijo de la pregunta).
PERFILES_MODELO = {
    "mpnet": dict(
        nombre="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        revision="4328cf26390c98c5e3c738b4460a05b95f4911f5",
        dim=768, max_seq_entrenado=128, prefijo_pregunta="", prefijo_documento=""),
    "e5base": dict(
        nombre="intfloat/multilingual-e5-base", revision=None,
        dim=768, max_seq_entrenado=512, prefijo_pregunta="query: ", prefijo_documento="passage: "),
    "e5large": dict(
        nombre="intfloat/multilingual-e5-large-instruct", revision=None,
        dim=1024, max_seq_entrenado=512,
        prefijo_pregunta=("Instruct: Dada una pregunta de un estudiante, recupera los artículos de los "
                          "reglamentos universitarios que la responden\nQuery: "),
        prefijo_documento=""),
    "bgem3": dict(
        nombre="BAAI/bge-m3", revision=None,
        dim=1024, max_seq_entrenado=8192, prefijo_pregunta="", prefijo_documento=""),
    # Requiere transformers>=4.51 (la imagen actual tiene 4.39.3): ver CLAUDE.md §8.
    "qwen3": dict(
        nombre="Qwen/Qwen3-Embedding-0.6B", revision=None,
        dim=1024, max_seq_entrenado=32768,
        prefijo_pregunta=("Instruct: Dada una pregunta de un estudiante, recupera los artículos de los "
                          "reglamentos universitarios que la responden\nQuery:"),
        prefijo_documento=""),
}
MODEL_KEY = os.environ.get("MODEL_KEY", "mpnet").strip()
if MODEL_KEY not in PERFILES_MODELO:
    raise ValueError(f"MODEL_KEY inválido: {MODEL_KEY!r} (opciones: {', '.join(PERFILES_MODELO)})")
PERFIL = PERFILES_MODELO[MODEL_KEY]
MODEL_NAME = PERFIL["nombre"]
MODEL_REVISION = os.environ.get("MODEL_REVISION") or PERFIL["revision"]
EMBEDDING_DIM = PERFIL["dim"]
PREFIJO_PREGUNTA = PERFIL["prefijo_pregunta"]
PREFIJO_DOCUMENTO = PERFIL["prefijo_documento"]

SUFIJO = "" if MODEL_KEY == "mpnet" else f"_{MODEL_KEY}"
COLLECTION_NAME = f"corpus_upeu_{VERSION}{SUFIJO}"

# Archivos por etapa (01-04 no dependen del modelo; 05-08 sí)
INVENTARIO_CSV = META_DIR / f"inventario_{VERSION}.csv"
CLASIF_CSV = META_DIR / f"clasificacion_{VERSION}.csv"
CLASIF_MANUAL_CSV = CONFIG_DIR / "clasificacion_manual.csv"
BANCO_PREGUNTAS_CSV = CONFIG_DIR / "banco_preguntas.csv"
DIAG_CSV = META_DIR / f"diagnostico_{VERSION}.csv"
EXTRACCION_CSV = META_DIR / f"extraccion_{VERSION}_metadatos.csv"
CHUNKS_CSV = META_DIR / f"chunks_{VERSION}{SUFIJO}.csv"
EMBEDDINGS_NPY = META_DIR / f"embeddings_{VERSION}{SUFIJO}.npy"
CHUNK_IDS_NPY = META_DIR / f"chunk_ids_{VERSION}{SUFIJO}.npy"
DESCARTADOS_CSV = META_DIR / f"chunks_descartados_{VERSION}{SUFIJO}.csv"
REPORTE_MD = META_DIR / f"reporte_validacion_{VERSION}{SUFIJO}.md"
DETALLE_CSV = META_DIR / f"validacion_{VERSION}{SUFIJO}_detalle.csv"
BM25_JSON = META_DIR / f"bm25_corpus_{VERSION}{SUFIJO}.json"
MANIFEST_JSON = META_DIR / f"MANIFEST{SUFIJO}.json"
EVAL_DIR = META_DIR / "evaluacion"

# P1: mpnet trae max_seq_length=128 y todo lo que excede se trunca en silencio.
#   Experimento 2a -> CHUNK_MODE=articulo, MAX_SEQ_LENGTH=384 (chunks grandes, todo embebido)
#   Plan 2b        -> CHUNK_MODE=pequeno,  MAX_SEQ_LENGTH=128 (chunks <=120 tok + grupo_articulo_id)
CHUNK_MODE = os.environ.get("CHUNK_MODE", "articulo").strip()
if CHUNK_MODE not in {"articulo", "pequeno"}:
    raise ValueError(f"CHUNK_MODE inválido: {CHUNK_MODE!r} (usa 'articulo' o 'pequeno')")

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "384" if CHUNK_MODE == "articulo" else "128"))
TOKENS_ESPECIALES = 2                               # <s> y </s> que añade el tokenizer al codificar
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))

# M2: qué contexto se antepone al texto en texto_embedding.
#   completo  -> "Documento | TÍTULO … | CAPÍTULO … | Sección … | Artículo N"
#   documento -> "Documento | Artículo N"
#   ninguno   -> solo el texto (como V2)
BREADCRUMB_MODO = os.environ.get("BREADCRUMB_MODO", "completo").strip()
if BREADCRUMB_MODO not in {"completo", "documento", "ninguno"}:
    raise ValueError(f"BREADCRUMB_MODO inválido: {BREADCRUMB_MODO!r}")

# --- Chunking ----------------------------------------------------------------
# Tope del cuerpo (texto) de un chunk. El texto_embedding (breadcrumb + texto)
# nunca supera MAX_SEQ_LENGTH - TOKENS_ESPECIALES: así 0 tokens se truncan.
CHUNK_MAX_TOKENS = int(os.environ.get("CHUNK_MAX_TOKENS", "350" if CHUNK_MODE == "articulo" else "120"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "40" if CHUNK_MODE == "articulo" else "20"))
# Bloques más cortos que esto (p.ej. "CAPÍTULO III / DE LAS BECAS") no se emiten
# solos: se anteponen al bloque siguiente (elimina chunks basura y duplicados).
MIN_TOKENS_BLOQUE = int(os.environ.get("MIN_TOKENS_BLOQUE", "25"))
# Máximo de tokens que puede ocupar el breadcrumb de contexto
# ("Documento | Título | Capítulo | Artículo"). En modo pequeno se reduce
# para no comerse el presupuesto de 128 tokens.
MAX_TOKENS_BREADCRUMB = int(os.environ.get("MAX_TOKENS_BREADCRUMB", "60" if CHUNK_MODE == "articulo" else "30"))

# --- Extracción --------------------------------------------------------------
UMBRAL_PALABRAS_PAGINA = 30     # palabras (>2 letras) para considerar una página "digital"
DPI_OCR = 300                   # recomendado por Tesseract (V2 usaba 250)
IDIOMA_OCR = "spa"
MARCADOR_PAGINA = "[[PAG {n}]]"  # se inserta al extraer; el chunking lo quita y lo usa para `pagina`

# --- Validación --------------------------------------------------------------
COBERTURA_MIN_ERROR = 0.97      # texto del .txt que debe terminar en chunks
COBERTURA_MIN_WARNING = 0.99

# --- ChromaDB ----------------------------------------------------------------
HNSW_PARAMS = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 100,
    "hnsw:M": 16,
}
INDEX_BATCH_SIZE = 500


def asegurar_version_escribible() -> None:
    """Aborta si se intenta escribir sobre una versión protegida (V2)."""
    if VERSION in VERSIONES_PROTEGIDAS and os.environ.get("ALLOW_OVERWRITE_V2") != "1":
        raise SystemExit(
            f"❌ CORPUS_VERSION={VERSION} está protegida (la usa OE5). "
            "Usa otra versión (p.ej. v3) o define ALLOW_OVERWRITE_V2=1 si de verdad quieres sobrescribirla."
        )


def resumen_config() -> dict:
    """Parámetros que se guardan en MANIFEST.json."""
    return {
        "version": VERSION,
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "embedding_dim": EMBEDDING_DIM,
        "prefijo_pregunta": PREFIJO_PREGUNTA,
        "prefijo_documento": PREFIJO_DOCUMENTO,
        "breadcrumb_modo": BREADCRUMB_MODO,
        "chunk_mode": CHUNK_MODE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "min_tokens_bloque": MIN_TOKENS_BLOQUE,
        "max_tokens_breadcrumb": MAX_TOKENS_BREADCRUMB,
        "dpi_ocr": DPI_OCR,
        "umbral_palabras_pagina": UMBRAL_PALABRAS_PAGINA,
        "hnsw": HNSW_PARAMS,
        "collection": COLLECTION_NAME,
    }

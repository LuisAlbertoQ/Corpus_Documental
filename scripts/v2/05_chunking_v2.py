#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_chunking_v2.py — Chunking semántico por estructura documental
Lee: corpus_upeu/txt_bruto_v2/ + diagnostico_v2.csv + clasificacion_v2.csv
Escribe: corpus_upeu/metadatos/chunks_v2.csv, embeddings_v2.npy, chunk_ids_v2.npy
"""

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
TXT_FOLDER = BASE / "corpus_upeu" / "txt_bruto_v2"
DIAG_CSV = BASE / "corpus_upeu" / "metadatos" / "diagnostico_v2.csv"
CLASIF_CSV = BASE / "corpus_upeu" / "metadatos" / "clasificacion_v2.csv"
META_FOLDER = BASE / "corpus_upeu" / "metadatos"
CHUNKS_CSV = META_FOLDER / "chunks_v2.csv"
EMBEDDINGS_NPY = META_FOLDER / "embeddings_v2.npy"
CHUNK_IDS_NPY = META_FOLDER / "chunk_ids_v2.npy"

META_FOLDER.mkdir(parents=True, exist_ok=True)

# Modelo embeddings (mpnet 768 dim)
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
print(f"Cargando modelo {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# --- Regex de estructura documental ---
RE_TITULO_DOC = re.compile(r"^(?:T[IÍ]TULO\s+[IVXLCDM\d]+|TITLE\s+\d+)\b", re.IGNORECASE)
RE_CAPITULO = re.compile(r"^(?:CAP[IÍ]TULO\s+[IVXLCDM\d]+|CHAPTER\s+\d+)\b", re.IGNORECASE)
RE_SECCION = re.compile(r"^(?:SECCI[OÓ]N\s+\d+(?:\.\d+)*|SECTION\s+\d+(?:\.\d+)*)\b", re.IGNORECASE)
RE_ARTICULO = re.compile(r"^(?:ART[IÍ]CULO\s+\d+[º°]?(?:\s*[.-]\s*\d+)?|ARTICLE\s+\d+)\b", re.IGNORECASE)
RE_SU_ARTICULO = re.compile(r"^\d+[º°]?\.\s", re.IGNORECASE)  # 1º., 2º., etc.
RE_NUMERAL = re.compile(r"^(?:\d+[.)]|[a-z][.)])\s", re.IGNORECASE)  # 1. 2. a) b)
RE_LISTA_BULLET = re.compile(r"^[-•▪·]\s")
RE_PASO = re.compile(r"^(?:PASO\s+\d+|STEP\s+\d+)\b", re.IGNORECASE)
RE_TABLA_MARKER = re.compile(r"^\[TABLA\s+p\d+\.\d+\]", re.IGNORECASE)
RE_IMAGEN = re.compile(r"^\[IMAGEN|\[FIGURA|\[FIGURE", re.IGNORECASE)

# Combinado para detectar cualquier inicio de bloque estructural
RE_BLOQUE = re.compile(
    r"(?:^|\n)\s*("
    r"T[IÍ]TULO\s+[IVXLCDM\d]+|"
    r"CAP[IÍ]TULO\s+[IVXLCDM\d]+|"
    r"SECCI[OÓ]N\s+\d+(?:\.\d+)*|"
    r"ART[IÍ]CULO\s+\d+[º°]?(?:\s*[.-]\s*\d+)?|"
    r"PASO\s+\d+"
    r")",
    re.IGNORECASE
)

# Configuración chunking
MAX_TOKENS_ARTICULO_CORTO = 400
MAX_TOKENS_ARTICULO_LARGO = 700
MAX_TOKENS_OTROS = 500
OVERLAP_PCT = 0.12  # 12% para artículos partidos
OVERLAP_TOKENS_FIJO = 60  # fallback

def contar_tokens(texto: str) -> int:
    return len(model.tokenizer.encode(texto, add_special_tokens=False))

def dividir_texto_largo(texto: str, max_tokens: int, overlap: int) -> list[str]:
    """Divide texto largo en chunks con overlap por tokens. Garantiza <= max_tokens."""
    tokens = model.tokenizer.encode(texto, add_special_tokens=False)
    chunks = []
    start = 0
    # Usar max_tokens - 1 como margen de seguridad para evitar 701 tokens
    safe_max = max_tokens - 1
    while start < len(tokens):
        end = min(start + safe_max, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk = model.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk)
        if end >= len(tokens):
            break
        start += (safe_max - overlap)
    return chunks

def detectar_bloques(texto: str) -> list[dict]:
    """
    Detecta bloques estructurales en el texto.
    Retorna lista de dicts: {tipo, titulo, contenido, inicio, fin, nivel}
    nivel: 1=título doc, 2=capítulo, 3=sección, 4=artículo, 5=sub-artículo/lista/paso
    """
    matches = list(RE_BLOQUE.finditer(texto))
    bloques = []
    
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        inicio = m.end()
        fin = matches[i+1].start() if i+1 < len(matches) else len(texto)
        contenido = texto[inicio:fin].strip()
        
        # Determinar tipo y nivel
        if RE_TITULO_DOC.match(header):
            tipo, nivel = "titulo", 1
        elif RE_CAPITULO.match(header):
            tipo, nivel = "capitulo", 2
        elif RE_SECCION.match(header):
            tipo, nivel = "seccion", 3
        elif RE_ARTICULO.match(header):
            tipo, nivel = "articulo", 4
        elif RE_PASO.match(header):
            tipo, nivel = "paso", 5
        else:
            tipo, nivel = "otro", 4
        
        bloques.append({
            "tipo": tipo,
            "header": header,
            "contenido": contenido,
            "inicio": inicio,
            "fin": fin,
            "nivel": nivel,
            "texto_completo": f"{header}\n{contenido}" if contenido else header
        })
    
    # Si no hay bloques detectados, tratar todo como un bloque
    if not bloques:
        bloques.append({
            "tipo": "texto_libre",
            "header": "",
            "contenido": texto,
            "inicio": 0,
            "fin": len(texto),
            "nivel": 0,
            "texto_completo": texto
        })
    
    return bloques

def extraer_sub_bloques(contenido: str) -> list[str]:
    """Extrae sub-bloques: listas, numerales, pasos dentro de un artículo."""
    lineas = contenido.split('\n')
    sub_bloques = []
    actual = []
    
    for linea in lineas:
        linea_strip = linea.strip()
        if not linea_strip:
            if actual:
                sub_bloques.append('\n'.join(actual))
                actual = []
            continue
        
        # Detectar inicio de item de lista/numeral/paso
        es_inicio = (RE_NUMERAL.match(linea_strip) or 
                     RE_LISTA_BULLET.match(linea_strip) or
                     RE_SU_ARTICULO.match(linea_strip) or
                     RE_PASO.match(linea_strip))
        
        if es_inicio and actual:
            sub_bloques.append('\n'.join(actual))
            actual = [linea]
        else:
            actual.append(linea)
    
    if actual:
        sub_bloques.append('\n'.join(actual))
    
    return sub_bloques if sub_bloques else [contenido]

def procesar_documento(nombre_archivo: str, texto: str, tipo_doc: str, cat_tematica: str, ruta_ext: str) -> list[dict]:
    """
    Procesa un documento completo y retorna lista de chunks con metadatos.
    """
    chunks = []
    bloques = detectar_bloques(texto)
    
    # Contexto jerárquico actual
    contexto = {"titulo_doc": "", "capitulo": "", "seccion": "", "articulo_actual": ""}
    
    for bloque in bloques:
        # Actualizar contexto
        if bloque["nivel"] == 1:
            contexto["titulo_doc"] = bloque["header"]
        elif bloque["nivel"] == 2:
            contexto["capitulo"] = bloque["header"]
        elif bloque["nivel"] == 3:
            contexto["seccion"] = bloque["header"]
        elif bloque["nivel"] == 4 and bloque["tipo"] == "articulo":
            contexto["articulo_actual"] = bloque["header"]
        
        texto_bloque = bloque["texto_completo"]
        tokens = contar_tokens(texto_bloque)
        
        # Determinar límite de tokens según tipo
        if bloque["tipo"] == "articulo":
            max_tokens = MAX_TOKENS_ARTICULO_LARGO if tokens > MAX_TOKENS_ARTICULO_CORTO else MAX_TOKENS_ARTICULO_CORTO
        elif bloque["tipo"] in ("tabla", "lista", "paso"):
            max_tokens = MAX_TOKENS_OTROS
        else:
            max_tokens = MAX_TOKENS_OTROS
        
        if tokens <= max_tokens:
            # Chunk único
            chunk_id = f"{nombre_archivo.replace('.pdf', '')}_{len(chunks):04d}"
            chunks.append({
                "documento": nombre_archivo,
                "chunk_id": chunk_id,
                "texto": texto_bloque,
                "num_tokens": tokens,
                "num_chars": len(texto_bloque),
                "tipo_documental": tipo_doc,
                "categoria_tematica": cat_tematica,
                "ruta_extraccion": ruta_ext,
                "titulo_documento": contexto["titulo_doc"],
                "capitulo": contexto["capitulo"],
                "seccion": contexto["seccion"],
                "articulo": contexto["articulo_actual"] if bloque["tipo"] == "articulo" else bloque["header"],
                "pagina": "",  # TODO: mapear desde PDF original si se necesita
                "tipo_bloque": bloque["tipo"],
                "padre_chunk_id": "",
                "es_parte": False,
                "parte_num": 0,
                "total_partes": 1
            })
        else:
            # Artículo largo → partir en partes con overlap
            overlap = int(max_tokens * OVERLAP_PCT)
            partes = dividir_texto_largo(texto_bloque, max_tokens, overlap)
            
            for j, parte in enumerate(partes):
                chunk_id = f"{nombre_archivo.replace('.pdf', '')}_{len(chunks):04d}"
                es_parte = j > 0  # Solo las partes 2+ son "partes"
                chunks.append({
                    "documento": nombre_archivo,
                    "chunk_id": chunk_id,
                    "texto": parte,
                    "num_tokens": contar_tokens(parte),
                    "num_chars": len(parte),
                    "tipo_documental": tipo_doc,
                    "categoria_tematica": cat_tematica,
                    "ruta_extraccion": ruta_ext,
                    "titulo_documento": contexto["titulo_doc"],
                    "capitulo": contexto["capitulo"],
                    "seccion": contexto["seccion"],
                    "articulo": contexto["articulo_actual"],
                    "pagina": "",
                    "tipo_bloque": bloque["tipo"],
                    "padre_chunk_id": chunks[-1]["chunk_id"] if j > 0 else "",
                    "es_parte": es_parte,
                    "parte_num": j + 1,
                    "total_partes": len(partes)
                })
    
    return chunks

def main():
    # Cargar datos de diagnóstico y clasificación
    df_diag = pd.read_csv(DIAG_CSV)
    df_clasif = pd.read_csv(CLASIF_CSV)
    
    # Merge para tener toda la info (diagnostico ya tiene tipo_documental y categoria_tematica)
    df = df_diag.merge(df_clasif[["nombre_archivo", "estado"]], on="nombre_archivo", how="left")
    
    all_chunks = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        nombre = row["nombre_archivo"]
        txt_path = TXT_FOLDER / f"{nombre}.txt"
        
        if not txt_path.exists():
            print(f"  ⚠️  No existe: {txt_path}")
            continue
        
        texto = txt_path.read_text(encoding="utf-8")
        if not texto.strip():
            print(f"  ⚠️  Vacío: {nombre}")
            continue
        
        chunks_doc = procesar_documento(
            nombre_archivo=nombre,
            texto=texto,
            tipo_doc=row["tipo_documental"],
            cat_tematica=row["categoria_tematica"],
            ruta_ext=row["ruta_extraccion"]
        )
        all_chunks.extend(chunks_doc)
    
    print(f"\nTotal chunks generados: {len(all_chunks)}")
    
    # DataFrame de chunks
    df_chunks = pd.DataFrame(all_chunks)
    
    # Guardar chunks CSV (sin embeddings)
    df_chunks.to_csv(CHUNKS_CSV, index=False, encoding="utf-8")
    print(f"📄 Chunks guardados en {CHUNKS_CSV}")
    
    # Generar embeddings
    print("Generando embeddings...")
    textos = df_chunks["texto"].tolist()
    embeddings = model.encode(textos, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    
    # Guardar embeddings
    np.save(EMBEDDINGS_NPY, embeddings.astype(np.float32))
    print(f"📄 Embeddings guardados en {EMBEDDINGS_NPY} (shape: {embeddings.shape})")
    
    # Guardar chunk_ids
    np.save(CHUNK_IDS_NPY, df_chunks["chunk_id"].values)
    print(f"📄 Chunk IDs guardados en {CHUNK_IDS_NPY}")
    
    # Estadísticas
    print("\n--- Estadísticas ---")
    print(f"Documentos procesados: {df['nombre_archivo'].nunique()}")
    print(f"Chunks totales: {len(df_chunks)}")
    print(f"Tokens promedio: {df_chunks['num_tokens'].mean():.0f}")
    print(f"Tokens máx: {df_chunks['num_tokens'].max()}")
    print(f"Chunks por tipo_bloque:")
    print(df_chunks["tipo_bloque"].value_counts().to_string())
    print(f"Chunks partidos (es_parte=True): {df_chunks['es_parte'].sum()}")
    print(f"Relaciones padre-hijo: {(df_chunks['padre_chunk_id'] != '').sum()}")

if __name__ == "__main__":
    main()
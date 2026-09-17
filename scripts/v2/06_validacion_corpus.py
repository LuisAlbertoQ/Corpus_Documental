#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_validacion_corpus.py — Validación de calidad del corpus chunked
Lee: chunks_v2.csv, embeddings_v2.npy, chunk_ids_v2.npy, diagnostico_v2.csv
Escribe: reporte_validacion_v2.md, validacion_v2_detalle.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parents[1]
META_FOLDER = BASE / "corpus_upeu" / "metadatos"
CHUNKS_CSV = META_FOLDER / "chunks_v2.csv"
EMBEDDINGS_NPY = META_FOLDER / "embeddings_v2.npy"
CHUNK_IDS_NPY = META_FOLDER / "chunk_ids_v2.npy"
DIAG_CSV = META_FOLDER / "diagnostico_v2.csv"
REPORT_MD = META_FOLDER / "reporte_validacion_v2.md"
DETALLE_CSV = META_FOLDER / "validacion_v2_detalle.csv"

# Cargar datos
print("Cargando datos...")
df = pd.read_csv(CHUNKS_CSV)
embeddings = np.load(EMBEDDINGS_NPY)
chunk_ids = np.load(CHUNK_IDS_NPY, allow_pickle=True)
df_diag = pd.read_csv(DIAG_CSV)

print(f"Chunks: {len(df)} | Embeddings: {embeddings.shape} | IDs: {len(chunk_ids)}")

# --- Verificaciones ---

resultados = []
alertas = []

def agregar_alerta(nivel, categoria, mensaje, chunk_id=""):
    alertas.append({
        "nivel": nivel,      # ERROR, WARNING, INFO
        "categoria": categoria,
        "mensaje": mensaje,
        "chunk_id": chunk_id
    })

# 1. Integridad básica
print("\n1. Verificando integridad básica...")
if len(df) != embeddings.shape[0]:
    agregar_alerta("ERROR", "integridad", f"Chunks CSV ({len(df)}) != Embeddings ({embeddings.shape[0]})")
if len(df) != len(chunk_ids):
    agregar_alerta("ERROR", "integridad", f"Chunks CSV ({len(df)}) != Chunk IDs ({len(chunk_ids)})")
if not (df["chunk_id"] == chunk_ids).all():
    agregar_alerta("ERROR", "integridad", "Chunk IDs no coinciden con CSV")

# 2. Campos obligatorios
print("2. Verificando campos obligatorios...")
campos_obligatorios = [
    "documento", "chunk_id", "texto", "num_tokens", "num_chars",
    "tipo_documental", "categoria_tematica", "ruta_extraccion",
    "titulo_documento", "capitulo", "seccion", "articulo",
    "tipo_bloque", "padre_chunk_id", "es_parte", "parte_num", "total_partes"
]
for campo in campos_obligatorios:
    if campo not in df.columns:
        agregar_alerta("ERROR", "metadatos", f"Campo obligatorio faltante: {campo}")
    else:
        nulos = df[campo].isna().sum()
        if nulos > 0:
            agregar_alerta("WARNING", "metadatos", f"Campo '{campo}' tiene {nulos} nulos")

# 3. Chunks vacíos o muy cortos
print("3. Verificando chunks vacíos/cortos...")
vacíos = df[df["texto"].str.strip() == ""]
if len(vacíos) > 0:
    agregar_alerta("ERROR", "contenido", f"{len(vacíos)} chunks con texto vacío")
    for _, r in vacíos.iterrows():
        agregar_alerta("ERROR", "contenido", "Chunk vacío", r["chunk_id"])

cortos = df[df["num_chars"] < 50]
if len(cortos) > 0:
    agregar_alerta("WARNING", "contenido", f"{len(cortos)} chunks < 50 chars")
    for _, r in cortos.iterrows():
        agregar_alerta("WARNING", "contenido", f"Chunk muy corto ({r['num_chars']} chars)", r["chunk_id"])

# 4. Tokens fuera de rango
print("4. Verificando tokens...")
# mpnet max_seq_length = 512 tokens reales, pero tokenizer puede dar más
# Verificar > 512 (truncamiento real del modelo)
exceden = df[df["num_tokens"] > 512]
if len(exceden) > 0:
    agregar_alerta("WARNING", "tokens", f"{len(exceden)} chunks > 512 tokens (se truncarán en embedding)")
    for _, r in exceden.iterrows():
        agregar_alerta("WARNING", "tokens", f"{r['num_tokens']} tokens", r["chunk_id"])

# Verificar > 700 (nuestro límite diseñado)
muy_largos = df[df["num_tokens"] > 700]
if len(muy_largos) > 0:
    agregar_alerta("ERROR", "tokens", f"{len(muy_largos)} chunks > 700 tokens (excede diseño)")
    for _, r in muy_largos.iterrows():
        agregar_alerta("ERROR", "tokens", f"{r['num_tokens']} tokens", r["chunk_id"])

# 5. Texto basura (headers/footers residuales, OCR noise)
print("5. Verificando texto basura...")
patrones_basura = [
    (r"Universidad Peruana Uni[oó]n", "header UPeU"),
    (r"www\.upeu\.edu\.pe", "URL footer"),
    (r"^\s*\d+\s*$", "folio solo"),
    (r"P[aá]gina\s+\d+", "marcador página"),
    (r"--- Pág \d+ \(OCR\) ---", "marcador OCR no limpio"),
]
for patron, desc in patrones_basura:
    matches = df[df["texto"].str.contains(patron, regex=True, na=False)]
    if len(matches) > 0:
        agregar_alerta("WARNING", "basura", f"{len(matches)} chunks con {desc}")
        for _, r in matches.head(5).iterrows():
            agregar_alerta("WARNING", "basura", f"{desc} detectado", r["chunk_id"])

# 6. Tablas preservadas
print("6. Verificando tablas...")
con_tabla = df[df["texto"].str.contains(r"\[TABLA p\d+\.\d+\]", regex=True, na=False)]
print(f"   Chunks con marcador [TABLA]: {len(con_tabla)}")
# Verificar docs diagnosticados como TUPA_TABLAS
docs_tupa = df_diag[df_diag["ruta_extraccion"] == "TUPA_TABLAS"]["nombre_archivo"].tolist()
for doc in docs_tupa:
    chunks_doc = df[df["documento"] == doc]
    tablas_doc = chunks_doc[chunks_doc["texto"].str.contains(r"\[TABLA", regex=True, na=False)]
    if len(tablas_doc) == 0:
        agregar_alerta("WARNING", "tablas", f"Doc TUPA_TABLAS '{doc}' sin marcadores [TABLA] en chunks")

# 7. Estructura jerárquica (artículos con header)
print("7. Verificando estructura artículos...")
chunks_art = df[df["tipo_bloque"] == "articulo"]
sin_articulo_header = chunks_art[~chunks_art["texto"].str.contains(r"ART[IÍ]CULO\s+\d+", regex=True, na=False)]
if len(sin_articulo_header) > 0:
    agregar_alerta("WARNING", "estructura", f"{len(sin_articulo_header)} chunks tipo 'articulo' sin header 'ARTICULO N'")

# 8. Relaciones padre-hijo
print("8. Verificando relaciones padre-hijo...")
partes = df[df["es_parte"] == True]
for _, r in partes.iterrows():
    padre_id = r["padre_chunk_id"]
    # Manejar NaN/empty string
    if pd.isna(padre_id) or padre_id == "" or str(padre_id).lower() == "nan":
        agregar_alerta("ERROR", "padre_hijo", f"Parte sin padre_chunk_id válido", r["chunk_id"])
        continue
    if padre_id not in df["chunk_id"].values:
        agregar_alerta("ERROR", "padre_hijo", f"Padre '{padre_id}' no existe", r["chunk_id"])
    else:
        padre = df[df["chunk_id"] == padre_id].iloc[0]
        if padre["documento"] != r["documento"]:
            agregar_alerta("ERROR", "padre_hijo", f"Padre en documento distinto", r["chunk_id"])
        if padre["articulo"] != r["articulo"]:
            agregar_alerta("WARNING", "padre_hijo", f"Padre con artículo distinto", r["chunk_id"])

# Verificar secuencia de partes
for padre_id, grupo in partes.groupby("padre_chunk_id"):
    nums = sorted(grupo["parte_num"].tolist())
    if nums != list(range(1, len(nums)+1)):
        agregar_alerta("WARNING", "padre_hijo", f"Partes no secuenciales para padre {padre_id}: {nums}")

# 9. Categorías temáticas válidas
print("9. Verificando categorías temáticas...")
cats_validas = {"A", "B", "C", "D", "E"}
cats_invalidas = set(df["categoria_tematica"].unique()) - cats_validas
if cats_invalidas:
    agregar_alerta("ERROR", "categoria", f"Categorías inválidas: {cats_invalidas}")

# 10. Embeddings: NaN, Inf, dimensiones
print("10. Verificando embeddings...")
if np.isnan(embeddings).any():
    agregar_alerta("ERROR", "embeddings", "Embeddings contienen NaN")
if np.isinf(embeddings).any():
    agregar_alerta("ERROR", "embeddings", "Embeddings contienen Inf")
if embeddings.shape[1] != 768:
    agregar_alerta("ERROR", "embeddings", f"Dimensión inesperada: {embeddings.shape[1]} (esperado 768)")

# Verificar normalización (norma ~1.0)
norms = np.linalg.norm(embeddings, axis=1)
if not np.allclose(norms, 1.0, atol=1e-3):
    no_norm = np.sum(~np.isclose(norms, 1.0, atol=1e-3))
    agregar_alerta("WARNING", "embeddings", f"{no_norm} embeddings no normalizados (norma != 1.0)")

# 11. Duplicados de chunk_id
print("11. Verificando duplicados...")
dup_ids = df[df.duplicated(subset=["chunk_id"], keep=False)]
if len(dup_ids) > 0:
    agregar_alerta("ERROR", "duplicados", f"{len(dup_ids)} chunk_ids duplicados")

# 12. Coherencia documento-diagnóstico
print("12. Verificando coherencia con diagnóstico...")
for _, row in df_diag.iterrows():
    doc = row["nombre_archivo"]
    chunks_doc = df[df["documento"] == doc]
    if len(chunks_doc) == 0:
        agregar_alerta("WARNING", "coherencia", f"Doc '{doc}' en diagnóstico sin chunks generados")

# --- Generar reporte ---
print("\n--- Generando reporte ---")

df_alertas = pd.DataFrame(alertas)
df_alertas.to_csv(DETALLE_CSV, index=False, encoding="utf-8")

# Resumen por nivel
resumen_nivel = df_alertas["nivel"].value_counts().to_dict() if len(df_alertas) > 0 else {}
resumen_cat = df_alertas["categoria"].value_counts().to_dict() if len(df_alertas) > 0 else {}

# Estadísticas generales
stats = {
    "total_chunks": len(df),
    "total_documentos": df["documento"].nunique(),
    "tokens_promedio": round(df["num_tokens"].mean(), 1),
    "tokens_mediana": df["num_tokens"].median(),
    "tokens_max": int(df["num_tokens"].max()),
    "chars_promedio": round(df["num_chars"].mean(), 1),
    "chunks_articulo": int((df["tipo_bloque"] == "articulo").sum()),
    "chunks_capitulo": int((df["tipo_bloque"] == "capitulo").sum()),
    "chunks_partidos": int(df["es_parte"].sum()),
    "relaciones_padre_hijo": int((df["padre_chunk_id"] != "").sum()),
    "embedding_dim": embeddings.shape[1],
    "embedding_norma_media": round(norms.mean(), 4),
    "alertas_total": len(df_alertas),
    "alertas_error": resumen_nivel.get("ERROR", 0),
    "alertas_warning": resumen_nivel.get("WARNING", 0),
    "alertas_info": resumen_nivel.get("INFO", 0),
}

# Markdown report
md = f"""# Reporte de Validación Corpus v2

**Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Versión corpus:** v2 (chunking semántico)

## Resumen Ejecutivo

| Métrica | Valor |
|---|---|
| Total chunks | {stats['total_chunks']:,} |
| Total documentos | {stats['total_documentos']} |
| Tokens promedio | {stats['tokens_promedio']} |
| Tokens mediana | {stats['tokens_mediana']} |
| Tokens máximo | {stats['tokens_max']} |
| Chars promedio | {stats['chars_promedio']} |
| Chunks tipo artículo | {stats['chunks_articulo']} ({stats['chunks_articulo']/stats['total_chunks']*100:.1f}%) |
| Chunks tipo capítulo | {stats['chunks_capitulo']} |
| Chunks partidos (>500 tok) | {stats['chunks_partidos']} |
| Relaciones padre-hijo | {stats['relaciones_padre_hijo']} |
| Embedding dimensión | {stats['embedding_dim']} |
| Embedding norma media | {stats['embedding_norma_media']} |

## Alertas

| Nivel | Cantidad |
|---|---|
| 🔴 ERROR | {stats['alertas_error']} |
| 🟡 WARNING | {stats['alertas_warning']} |
| 🔵 INFO | {stats['alertas_info']} |
| **Total** | **{stats['alertas_total']}** |

## Alertas por Categoría

"""
for cat, cnt in resumen_cat.items():
    md += f"| {cat} | {cnt} |\n"

md += "\n## Detalle de Alertas (primeras 50)\n\n"
if len(df_alertas) > 0:
    md += "| Nivel | Categoría | Chunk ID | Mensaje |\n"
    md += "|---|---|---|---|\n"
    for _, a in df_alertas.head(50).iterrows():
        emoji = "🔴" if a["nivel"] == "ERROR" else "🟡" if a["nivel"] == "WARNING" else "🔵"
        md += f"| {emoji} {a['nivel']} | {a['categoria']} | {a['chunk_id']} | {a['mensaje']} |\n"
else:
    md += "✅ **Sin alertas**\n"

md += f"\n---\n*Reporte completo en `{DETALLE_CSV}` ({len(df_alertas)} filas)*\n"

REPORT_MD.write_text(md, encoding="utf-8")
print(f"📄 Reporte: {REPORT_MD}")
print(f"📄 Detalle: {DETALLE_CSV}")

# Resumen en consola
print(f"\n{'='*50}")
print(f"VALIDACIÓN COMPLETADA")
print(f"{'='*50}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Alertas: {stats['alertas_total']} (ERROR: {stats['alertas_error']}, WARNING: {stats['alertas_warning']})")
print(f"Tokens: avg={stats['tokens_promedio']}, max={stats['tokens_max']}")
print(f"Artículos: {stats['chunks_articulo']}, Partidos: {stats['chunks_partidos']}, Padre-hijo: {stats['relaciones_padre_hijo']}")

if stats['alertas_error'] > 0:
    print(f"\n❌ {stats['alertas_error']} ERRORES - REVISAR ANTES DE INDEXAR")
    for _, a in df_alertas[df_alertas["nivel"] == "ERROR"].iterrows():
        print(f"  - {a['categoria']}: {a['mensaje']} ({a['chunk_id']})")
elif stats['alertas_warning'] > 0:
    print(f"\n⚠️  {stats['alertas_warning']} WARNINGS - revisar")
else:
    print(f"\n✅ SIN ERRORES - corpus listo para indexar")
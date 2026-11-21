#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_clasificacion.py — Propone tipo_documental y categoria_tematica por nombre
Lee inventario_v2.csv, escribe clasificacion_v2.csv con columnas:
  nombre_archivo, tipo_documental, categoria_tematica (A-E unificada OE5), estado, observaciones
"""

import re
from pathlib import Path

import pandas as pd

from config import INVENTARIO_CSV, CLASIF_CSV as OUTPUT_CSV, CLASIF_MANUAL_CSV
from config import asegurar_version_escribible

# Columnas que la revisión humana puede fijar en corpus_upeu/config/clasificacion_manual.csv
CAMPOS_MANUALES = ["tipo_documental", "categoria_tematica", "estado", "observaciones"]

# --- Reglas por palabras clave en nombre ---
TIPO_KEYWORDS = [
    (r"^ESTATUTO\b", "ESTATUTO"),
    (r"^REGLAMENTO\b", "REGLAMENTO"),
    (r"^RESOLUCION\b|^RESOLUCI[OÓ]N\b", "RESOLUCION"),
    (r"^DIRECTIVA\b", "DIRECTIVA"),
    (r"^POLITICA\b|^POL[IÍ]TICA\b", "POLITICA"),
    (r"^TUPA\b|^TABLAS\b", "TUPA_TARIFARIO"),
    (r"^GUIA\b|^GU[IÍ]A\b|^MANUAL\b", "MANUAL_GUIA"),
    (r"^MODELO\b", "MODELO_FORMATO"),
    (r"^REQUISITOS\b", "OTRO"),            # REQUISITOS PARA TRASLADO...
    (r"^MODIFICACI[OÓ]N\b", "OTRO"),        # MODIFICACIÓN DIRECTIVA...
    (r"^REGLAMENTO\s+CAPACITACION\b", "REGLAMENTO"),  # edge case
]

# --- Categoría temática unificada OE5 (becas → D) ---
# Cada tupla: (regex, categoria)
CAT_KEYWORDS = [
    (r"ESTATUTO|GENERAL\s+UPEU|DEFENSORIA|COMITE\s+ELECTORAL|TUPA|INTERNO\s+DE\s+TRABAJO", "A"),
    (r"ESTUDIOS|ESTUDIANTE\s+UNIONISTA|DOCENCIA|IDIOMAS|MOVILIDAD|PUBLICACIONES\s+Y\s+FONDO|PAGO\s+SERVICIOS|ADMISION|CREDITO|MATRICULA|EGRESADO|SEGUIMIENTO\s+EGRESADOS|GRADOS\s+Y\s+TITULOS|LEGAJO|BACHILLER|TRASLADO", "B"),
    (r"INVESTIGACION|INVESTIGADORES|INCENTIVOS\s+INVESTIGACION|PROPIEDAD\s+INTELECTUAL|CODIGO\s+ETICA\s+INVESTIGACION|TABLAS\s+INCENTIVOS", "C"),
    (r"BECAS|RESIDENCIAS|PROMOCION|RECREACION|DEPORTE|MULTIMEDIA|EXCELENCIA\s+ACADEMICA|RECONOCIMIENTO\s+EXCELENCIA", "D"),
    (r"POLITICA\s+INSTITUCIONAL|POLITICA[-_]AMBIENTAL|AMBIENTAL|CAPACITACION\s+DOCENTE|AUDITORIA\s+INTERNA|SEGURIDAD|SALUD|COMEDOR|TRANSPORTE|BIBLIOTECA|IDENTIDAD\s+VISUAL|REDES\s+SOCIALES|PRODAC|ACOSO\s+HOSTIGAMIENTO|TRABAJO\s+DIGNO|INCLUSION|DIVERSIDAD|DUPLICADO\s+DIPLOMAS|JEFES\s+PRACTICAS", "E"),
]

def clasificar_tipo(nombre: str) -> str:
    nombre_upper = nombre.upper()
    for pattern, tipo in TIPO_KEYWORDS:
        if re.search(pattern, nombre_upper):
            return tipo
    return "OTRO"

def clasificar_categoria(nombre: str) -> str:
    nombre_upper = nombre.upper()
    for pattern, cat in CAT_KEYWORDS:
        if re.search(pattern, nombre_upper):
            return cat
    return "E"  # fallback

def aplicar_clasificacion_manual(df: pd.DataFrame) -> pd.DataFrame:
    """F8: la revisión humana queda versionada en git, no editada a mano en metadatos/.

    Cada valor no vacío de clasificacion_manual.csv sobrescribe la propuesta
    automática. Los PDFs que no figuran en el archivo quedan en REVISAR y,
    por lo tanto, 03_diagnostico.py los excluye (solo procesa VIGENTE).
    """
    if not CLASIF_MANUAL_CSV.exists():
        print(f"⚠️  No existe {CLASIF_MANUAL_CSV}: se usa solo la clasificación automática.")
        return df

    manual = pd.read_csv(CLASIF_MANUAL_CSV, dtype=str).fillna("")
    manual = manual.set_index("nombre_archivo")
    df = df.copy()
    aplicados, sin_revision = 0, []
    for i, row in df.iterrows():
        nombre = row["nombre_archivo"]
        if nombre not in manual.index:
            sin_revision.append(nombre)
            continue
        for campo in CAMPOS_MANUALES:
            valor = manual.at[nombre, campo] if campo in manual.columns else ""
            if valor:
                df.at[i, campo] = valor
        aplicados += 1

    huerfanos = sorted(set(manual.index) - set(df["nombre_archivo"]))
    print(f"✅ Clasificación manual aplicada a {aplicados}/{len(df)} documentos ({CLASIF_MANUAL_CSV.name})")
    for nombre in sin_revision:
        print(f"   ⚠️  Sin revisión manual (queda {df.loc[df['nombre_archivo'] == nombre, 'estado'].iat[0]}): {nombre}")
    for nombre in huerfanos:
        print(f"   ⚠️  En clasificacion_manual.csv pero no en el inventario: {nombre}")
    return df


def main():
    asegurar_version_escribible()
    if not INVENTARIO_CSV.exists():
        print(f"❌ No existe {INVENTARIO_CSV}. Ejecuta 01_inventario.py primero.")
        return

    df = pd.read_csv(INVENTARIO_CSV)
    print(f"Cargados {len(df)} registros de {INVENTARIO_CSV}")

    rows = []
    for _, r in df.iterrows():
        nombre = r["nombre_archivo"]
        tipo = clasificar_tipo(nombre)
        cat = clasificar_categoria(nombre)

        # Estado por defecto: REVISAR (tú confirmas)
        estado = "REVISAR"
        obs = ""

        # Heurísticas de estado
        if "COPIA" in nombre.upper() or "(1)" in nombre:
            estado = "DUPLICADO"
            obs = "Posible duplicado por nombre"
        elif "v." in nombre.lower() or "v" + str(r.get("version", "")) in nombre.lower():
            pass  # versiones explícitas, mantener

        rows.append({
            "nombre_archivo": nombre,
            "tipo_documental": tipo,
            "categoria_tematica": cat,
            "estado": estado,
            "observaciones": obs,
        })

    out_df = pd.DataFrame(rows, columns=[
        "nombre_archivo", "tipo_documental", "categoria_tematica", "estado", "observaciones"
    ])
    out_df = aplicar_clasificacion_manual(out_df)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"📄 Clasificación propuesta en {OUTPUT_CSV}")

    # Resumen
    print("\n--- Distribución tipo_documental ---")
    print(out_df["tipo_documental"].value_counts().to_string())
    print("\n--- Distribución categoria_tematica (A-E) ---")
    print(out_df["categoria_tematica"].value_counts().sort_index().to_string())
    print("\n--- Estado ---")
    print(out_df["estado"].value_counts().to_string())

    # Ambigüos (tipo OTRO o estado REVISAR)
    ambiguos = out_df[(out_df["tipo_documental"] == "OTRO") | (out_df["estado"] == "REVISAR")]
    print(f"\n⚠️  {len(ambiguos)} filas para revisar manual (tipo=OTRO o estado=REVISAR):")
    for _, r in ambiguos.iterrows():
        print(f"   - {r['nombre_archivo']:<60} tipo={r['tipo_documental']:<15} cat={r['categoria_tematica']}  ({r['observaciones']})")

if __name__ == "__main__":
    main()
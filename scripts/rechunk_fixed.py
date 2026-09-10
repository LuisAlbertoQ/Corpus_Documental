# -*- coding: utf-8 -*-
"""Re-chunking corregido del corpus UPeU (fix Tier 3 / hallazgo OE1).

Correcciones respecto a notebooks/3_chunking_embeddings.ipynb:
  FIX-1: al fusionar artículos pequeños en un mismo chunk se conserva
         la cabecera de CADA artículo (antes solo se guardaba la
         primera y el resto aportaba cuerpo sin header → metadata
         `articulo` errónea + embedding diluido).
  FIX-2: `Subcapítulo` agregado al regex de corte (antes no partía y
         quedaba pegado al artículo previo).

Parámetros idénticos a producción: max_tokens=300, overlap=50,
max_chars_per_chunk=1500, tokenizer de paraphrase-multilingual-mpnet.

Uso (contenedor backend con GPU + mounts OE1):
    docker run --rm --gpus all \
      -v "<OE1>/corpus_upeu:/work/corpus_upeu" \
      -v "<HF_CACHE>:/root/.cache/huggingface" \
      oe5_chatbot_upeu-backend \
      python /work/scripts/rechunk_fixed.py

Salida: corpus_upeu/metadatos/chunks_fixed.csv (staging, NO pisa
chunks.csv hasta verificación explícita).
"""

import csv
import re
import sys
from pathlib import Path

WORK = Path("/work/corpus_upeu")
INPUT_FOLDER = WORK / "txt_limpio"
OUT_CSV = WORK / "metadatos" / "chunks_fixed.csv"

# --- FIX-2: Subcapítulo agregado -------------------------------------------
_RE_BLOQUE = re.compile(
    r'(?:^|\n)\s*('
    r'Art[íi]culo\s+\d+[ºo°]?(?:\s*[.-]\s*\d+)?'
    r'|Cap[íi]tulo\s+[IVXLCDM\d]+'
    r'|Subcap[íi]tulo\s+[IVXLCDM\d]+'
    r'|Secci[óo]n\s+\d+'
    r'|T[íi]tulo\s+[IVXLCDM\d]+'
    r')',
    re.IGNORECASE
)


def chunk_text_estructural(text, tokenizer, max_tokens=300, overlap=50,
                           max_chars_per_chunk=1500):
    """Divide texto en UN chunk por bloque estructural (sin fusionar).

    FIX-1 (v2, estricto): cada artículo/capítulo/sección/título es su
    propio chunk. La fusión de artículos pequeños diluía el embedding
    (caso Art 51/52/53 de Admisión: el dato de "nota mínima" quedaba
    como cola irrelevante y nunca rankeaba). Solo se subdividen los
    bloques que exceden max_tokens, conservando siempre su cabecera.
    El preámbulo previo a la primera cabecera se conserva como chunk.
    """
    matches = list(_RE_BLOQUE.finditer(text))

    if not matches:
        return chunk_text_simple(text, tokenizer, max_tokens, overlap,
                                 max_chars_per_chunk)

    chunks = []

    # Preámbulo anterior a la primera cabecera (portadas, presentación)
    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            chunks.extend(
                chunk_text_simple(pre, tokenizer, max_tokens, overlap,
                                  max_chars_per_chunk))

    for i, m in enumerate(matches):
        cabecera = m.group(1).strip()
        inicio = m.end()
        fin = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        contenido = text[inicio:fin].strip()

        bloque = f"{cabecera}\n{contenido}" if contenido else cabecera
        tokens = tokenizer.encode(bloque, add_special_tokens=False)

        if len(tokens) <= max_tokens:
            chunks.append(bloque)
        else:
            for j in range(0, len(tokens), max_tokens - overlap):
                sub = tokenizer.decode(tokens[j:j + max_tokens],
                                       skip_special_tokens=True)
                sub = sub[:max_chars_per_chunk]
                chunks.append(f"{cabecera}\n{sub}")

    return [c for c in chunks if c.strip()]


def chunk_text_simple(text, tokenizer, max_tokens=300, overlap=80,
                      max_chars_per_chunk=1500):
    """Fallback: chunking por tokens sin estructura."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        chunk = tokenizer.decode(tokens[start:start + max_tokens],
                                 skip_special_tokens=True)
        if len(chunk) > max_chars_per_chunk:
            chunk = chunk[:max_chars_per_chunk]
        chunks.append(chunk)
        start += (max_tokens - overlap)
    return chunks


def main():
    from sentence_transformers import SentenceTransformer

    print("Cargando tokenizer mpnet...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    tokenizer = model.tokenizer
    del model

    txt_files = sorted(INPUT_FOLDER.glob("*.txt"))
    print(f"Documentos: {len(txt_files)}")

    total = 0
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["documento", "chunk_id", "texto", "num_tokens"])
        w.writeheader()
        for txt_path in txt_files:
            doc = txt_path.stem
            text = txt_path.read_text(encoding="utf-8")
            blocos = chunk_text_estructural(
                text, tokenizer, max_tokens=300, overlap=50,
                max_chars_per_chunk=1500)
            for i, ch in enumerate(blocos):
                w.writerow({
                    "documento": doc,
                    "chunk_id": f"{doc}_{i:04d}",
                    "texto": ch,
                    "num_tokens": len(tokenizer.encode(ch)),
                })
            total += len(blocos)
            print(f"  {doc[:48]:48s} -> {len(blocos):4d}")

    print(f"\nTOTAL chunks fijos: {total} -> {OUT_CSV}")

    # Verificación del caso Art 51/52/53 (Admisión)
    print("\n--- Verificación Art 51/52/53 ---")
    with open(OUT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if ("ADMISION" in row["documento"]
                    and "Nota m" in (row["texto"] or "")
                    and "nima aprobatoria" in (row["texto"] or "")):
                print("chunk_id:", row["chunk_id"])
                print("primeras 3 lineas:",
                      " / ".join(row["texto"].split("\n")[:3]))
                break


if __name__ == "__main__":
    sys.exit(main())

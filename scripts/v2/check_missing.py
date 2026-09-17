import pandas as pd
import os

df = pd.read_csv("/home/jupyteruser/work/corpus_upeu/metadatos/diagnostico_v2.csv")
existing = set(f.replace(".txt", "") for f in os.listdir("/home/jupyteruser/work/corpus_upeu/txt_bruto_v2") if f.endswith(".txt"))
all_docs = set(df["nombre_archivo"].tolist())
missing = all_docs - existing
print(f"Total en diagnóstico: {len(all_docs)}")
print(f"Extraídos: {len(existing)}")
print(f"Faltantes ({len(missing)}):")
for m in sorted(missing):
    print(f"  - {m}")
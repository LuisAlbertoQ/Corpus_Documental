import sys
from pathlib import Path

# Permite `import chunking`, `import texto_lexico` desde scripts/v2/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

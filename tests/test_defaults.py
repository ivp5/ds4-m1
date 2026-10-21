"""Fail if the generated header diverges from its canonical profile data."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gguf-tools'))
from generate_defaults import generate
if (ROOT/'ds4_defaults.h').read_text()!=generate():
    raise SystemExit('ds4_defaults.h is stale; run python3 gguf-tools/generate_defaults.py')
print('Generated defaults: PASS')

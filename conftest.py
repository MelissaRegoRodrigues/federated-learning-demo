"""Puts src/ on sys.path so `import flapp...` works in every test file without packaging."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

"""pytest configuration: add project root to sys.path so src/ and data/ are importable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

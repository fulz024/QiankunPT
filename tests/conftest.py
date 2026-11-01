import sys
from pathlib import Path

# Run source-tree tests without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

project_root = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
os.chdir(project_root)
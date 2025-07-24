# notebooks/_setup_env.py
# This script sets up the environment for the notebook by configuring the system path
# to include the parent directory of the current working directory.
# This allows for importing modules from the parent directory.

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
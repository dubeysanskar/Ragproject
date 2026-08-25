"""Put apps/api on sys.path so `goarag` imports from the repo root.

The eval suite adds the *project root* to sys.path, but the package lives one
level down in apps/api/. This keeps the real code where it belongs instead of
duplicating it up here for the harness's benefit.
"""
from __future__ import annotations

import sys
from pathlib import Path

_API = Path(__file__).resolve().parents[1] / "apps" / "api"
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

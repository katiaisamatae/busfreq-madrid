"""Configuración de pytest: añade codigo/back a sys.path para importar los
paquetes del proyecto (demand, optimizer, ingest) sin instalación previa."""
import sys
from pathlib import Path

BACK = Path(__file__).resolve().parents[1]  # codigo/back
if str(BACK) not in sys.path:
    sys.path.insert(0, str(BACK))

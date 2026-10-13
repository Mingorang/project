"""Export reproducible metrics, with no implicit plotting dependencies."""
import json
from pathlib import Path
def save_metrics(metrics,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")

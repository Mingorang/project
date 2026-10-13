"""Check minimum artifact contract before publishing results."""
from pathlib import Path
def audit(folder):
    p=Path(folder)
    required=["predictions.csv","fold_metrics.csv","model_card.md","experiment_manifest.json"]
    return {name:(p/name).exists() for name in required}

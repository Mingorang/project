"""Paired loss comparison on identical forecast origins."""
import numpy as np
def paired_improvement(baseline_losses,model_losses):
    d=np.asarray(baseline_losses)-np.asarray(model_losses)
    return {"mean_improvement":float(d.mean()),"std_difference":float(d.std(ddof=1))}

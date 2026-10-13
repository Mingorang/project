"""Compare return distribution without treating fidelity as prediction skill."""
import numpy as np
def diagnostics(real,synthetic):
    a=np.asarray(real,float); b=np.asarray(synthetic,float)
    return {k:float(v) for k,v in {"mean_gap":abs(a.mean()-b.mean()),"vol_gap":abs(a.std()-b.std()),"tail95_gap":abs(np.quantile(abs(a),.95)-np.quantile(abs(b),.95))}.items()}

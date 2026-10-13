"""Explicit bounded similarity, not an estimated probability of success."""
import numpy as np
def similarity(actual,pred,scale):
    if scale<=0: raise ValueError("Validation-derived scale must be positive")
    a=np.asarray(actual,float); p=np.asarray(pred,float)
    if a.shape!=p.shape or a.size==0: raise ValueError("Paths must align")
    error=float(np.mean(np.abs(a-p)))
    return {"mae":error,"similarity":float(np.exp(-error/scale))}

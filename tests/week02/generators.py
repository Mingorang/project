"""Seeded log-return null and block-bootstrap generators."""
import numpy as np
def iid_paths(training_returns, n_paths, horizon, seed=42):
    rng=np.random.default_rng(seed)
    x=np.asarray(training_returns,float)
    return rng.choice(x,size=(n_paths,horizon),replace=True)
def block_paths(training_returns,n_paths,horizon,block=5,seed=42):
    rng=np.random.default_rng(seed); x=np.asarray(training_returns,float)
    if not 1<=block<=len(x): raise ValueError("Invalid block")
    out=[]
    for _ in range(n_paths):
        pieces=[]
        while len(pieces)<horizon:
            start=rng.integers(0,len(x)-block+1)
            pieces.extend(x[start:start+block])
        out.append(pieces[:horizon])
    return np.array(out)

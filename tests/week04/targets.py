"""Forward labels: never join to features until after chronological split."""
import numpy as np
def targets(c,horizon=5):
    future=np.log(c.Close.shift(-horizon)/c.Close)
    return future.rename("forward_return"),(future>0).astype("Int64").where(future.notna()).rename("up")

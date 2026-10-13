"""Feature provenance assertion."""
import pandas as pd
def check(feature_times,cutoff_times,label_end_times):
    f=pd.to_datetime(feature_times,utc=True); c=pd.to_datetime(cutoff_times,utc=True); y=pd.to_datetime(label_end_times,utc=True)
    if not ((f<=c)&(y>c)).all(): raise AssertionError("Temporal leakage or invalid target")

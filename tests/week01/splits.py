"""Chronological partition; embargo must be >= forecast horizon."""
def split_indices(n, horizon=5, train_fraction=.6, validation_fraction=.2):
    a=int(n*train_fraction); b=int(n*(train_fraction+validation_fraction))
    if not 0<a<a+horizon<b<b+horizon<n: raise ValueError("Insufficient observations")
    return slice(0,a),slice(a+horizon,b),slice(b+horizon,n)

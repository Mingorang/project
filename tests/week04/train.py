"""Baseline Random Forest. Caller supplies strictly chronological arrays."""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
def build(seed=42):
    return Pipeline([("impute",SimpleImputer(strategy="median")),("rf",RandomForestClassifier(n_estimators=300,min_samples_leaf=10,random_state=seed,n_jobs=-1))])
def fit(train_x,train_y,seed=42):
    model=build(seed); model.fit(train_x,train_y); return model

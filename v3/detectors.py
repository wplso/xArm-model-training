"""Exact Detector and threshold definitions from the original training dependency."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

class Detector:
    def __init__(self,kind,seed=20261031):
        self.kind=kind;self.seed=seed
    def fit(self,X):
        X=np.asarray(X,float)
        if not np.isfinite(X).any(axis=0).all():raise ValueError("All-missing training feature")
        self.imputer=SimpleImputer(strategy="median",keep_empty_features=True)
        self.scaler=StandardScaler();Z=self.scaler.fit_transform(self.imputer.fit_transform(X))
        if self.kind=="IsolationForest":self.model=IsolationForest(n_estimators=200,random_state=self.seed,n_jobs=1).fit(Z)
        elif self.kind=="PCA":self.model=PCA(n_components=.95,svd_solver="full").fit(Z)
        elif self.kind!="Statistical":raise ValueError(self.kind)
        return self
    def score(self,X):
        Z=self.scaler.transform(self.imputer.transform(np.asarray(X,float)))
        if self.kind=="Statistical":return np.mean(Z**2,axis=1)
        if self.kind=="IsolationForest":return -self.model.score_samples(Z)
        return np.mean((Z-self.model.inverse_transform(self.model.transform(Z)))**2,axis=1)

def threshold(scores,target=.05):return float(np.quantile(scores,1-target,method="higher"))

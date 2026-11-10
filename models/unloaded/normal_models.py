"""Inference class referenced by the archived No.3 and No.3/4/5 joblibs."""

import numpy as np


class Detector:
    """Read a fitted imputer, scaler and estimator from a trusted archive."""

    def score(self, values):
        standardized = self.scaler.transform(self.imputer.transform(np.asarray(values, float)))
        if self.kind == "Statistical":
            return np.mean(standardized**2, axis=1)
        if self.kind == "IsolationForest":
            return -self.model.score_samples(standardized)
        if self.kind == "PCA":
            reconstructed = self.model.inverse_transform(self.model.transform(standardized))
            return np.mean((standardized - reconstructed) ** 2, axis=1)
        raise ValueError(f"Unknown detector {self.kind!r}")

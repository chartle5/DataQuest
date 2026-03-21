from typing import Dict, List, Tuple

import numpy as np

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

from sklearn.ensemble import GradientBoostingRegressor


class TrialRanker:
    def __init__(self) -> None:
        self.model = None
        self.uses_lightgbm = False
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, group: List[int],
            feature_names: List[str] = None) -> None:
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        if lgb is not None:
            self.model = lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=5,
                verbose=-1,
            )
            self.model.fit(X, y, group=group, feature_name=self.feature_names)
            self.uses_lightgbm = True
        else:
            self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05)
            self.model.fit(X, y)
            self.uses_lightgbm = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        if self.uses_lightgbm and self.feature_names:
            import pandas as pd
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict(X)

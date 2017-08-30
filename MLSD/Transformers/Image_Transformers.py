import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.image import extract_patches_2d


class BasicImage(TransformerMixin):
    def __init__(self, Dreduction=None, *args, **kwargs):

        self.Dreduction = Dreduction

    def fit(self, X, y=None):
        features = []
        for i in X.values:
            patch = extract_patches_2d(x, *args, *kwargs)
            dims = np.cumprod(np.asarray(patch.shape))
            features.append(np.asarray(patch).reshape(1, dims))
        self.features = pd.DataFrame(features, index=X.index)

    def transform(self, X, y=None):

        features = []
        for i in X.values:
            patch = extract_patches_2d(x, *args, *kwargs)
            dims = np.cumprod(np.asarray(patch.shape))
            features.append(np.asarray(patch).reshape(1, dims))
        self.features = pd.DataFrame(features, index=X.index)

        return self.features

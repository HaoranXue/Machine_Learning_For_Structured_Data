import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.base import TransformerMixin
from .pyFDA import bspline
from .pyFDA.register import localRegression


class BasicSeries(TransformerMixin):

    def __init__(self, Dreduction= None):

        self.Dreduction = Dreduction

    def fit(self, X,*args,**kwargs):

        def first_order_d(X):
            X= np.asarray(X)
            return X[1:]-X[:-1]
        def second_order_d(X):
            X= np.asarray(X)
            first_order = first_order_d(X)
            return first_order[1:]-first_order[:-1]
        def fo_mean(X):
            return np.mean(first_order_d(X))
        def fo_std(X):
            return np.std(first_order_d(X))
        def fo_min(X):
            return np.min(first_order_d(X))
        def fo_max(X):
            return np.max(first_order_d(X))
        def fo_median(X):
            return np.median(first_order_d(X))
        def fo_skew(X):
            return skew(first_order_d(X))
        def fo_kurt(X):
            return kurtosis(first_order_d(X))
        def so_mean(X):
            return np.mean(second_order_d(X))
        def so_std(X):
            return np.std(second_order_d(X))
        def so_min(X):
            return np.min(second_order_d(X))
        def so_max(X):
            return np.max(second_order_d(X))
        def so_median(X):
            return np.median(second_order_d(X))
        def so_skew(X):
            return skew(second_order_d(X))
        def so_kurt(X):
            return kurtosis(second_order_d(X))

        self.features = pd.DataFrame(np.asarray([X.min(),
                                    X.max(),
                                    X.mean(),
                                    X.std(),
                                    X.apply(skew),
                                    X.apply(kurtosis),
                                    X.apply(np.median),
                                    X.apply(fo_mean),
                                    X.apply(fo_std),
                                    X.apply(fo_min),
                                    X.apply(fo_median),
                                    X.apply(fo_max),
                                    X.apply(fo_skew),
                                    X.apply(fo_kurt),
                                    X.apply(so_mean),
                                    X.apply(so_std),
                                    X.apply(so_min),
                                    X.apply(so_median),
                                    X.apply(so_max),
                                    X.apply(so_skew),
                                    X.apply(so_kurt)]).T).dropna(1)


    def transform(self,X,*args,**kwargs):

        def first_order_d(X):
            X= np.asarray(X)
            return X[1:]-X[:-1]
        def second_order_d(X):
            X = np.asarray(X)
            first_order = first_order_d(X)
            return first_order[1:]-first_order[:-1]
        def fo_mean(X):
            return np.mean(first_order_d(X))
        def fo_std(X):
            return np.std(first_order_d(X))
        def fo_min(X):
            return np.min(first_order_d(X))
        def fo_max(X):
            return np.max(first_order_d(X))
        def fo_median(X):
            return np.median(first_order_d(X))
        def fo_skew(X):
            return skew(first_order_d(X))
        def fo_kurt(X):
            return kurtosis(first_order_d(X))
        def so_mean(X):
            return np.mean(second_order_d(X))
        def so_std(X):
            return np.std(second_order_d(X))
        def so_min(X):
            return np.min(second_order_d(X))
        def so_max(X):
            return np.max(second_order_d(X))
        def so_median(X):
            return np.median(second_order_d(X))
        def so_skew(X):
            return skew(second_order_d(X))
        def so_kurt(X):
            return kurtosis(second_order_d(X))

        features = pd.DataFrame(np.asarray([X.min(),
                                    X.max(),
                                    X.mean(),
                                    X.std(),
                                    X.apply(skew),
                                    X.apply(kurtosis),
                                    X.apply(np.median),
                                    X.apply(fo_mean),
                                    X.apply(fo_std),
                                    X.apply(fo_min),
                                    X.apply(fo_median),
                                    X.apply(fo_max),
                                    X.apply(fo_skew),
                                    X.apply(fo_kurt),
                                    X.apply(so_mean),
                                    X.apply(so_std),
                                    X.apply(so_min),
                                    X.apply(so_median),
                                    X.apply(so_max),
                                    X.apply(so_skew),
                                    X.apply(so_kurt)]).T).dropna(1)
        return features

#
# class BsplineSeries(TransformerMixin):
#
#
# class localRSeries(TransformerMixin):
#

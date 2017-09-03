import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from tsfresh import extract_features, extract_relevant_features
import patsy as ps
from .pyFDA import bspline
from .pyFDA.register import localRegression


class BasicSeries(TransformerMixin):
    def __init__(self, Dreduction=None):

        self.Dreduction = Dreduction

    def fit(self, X, y=None, *args, **kwargs):
        def first_order_d(X):
            X = np.asarray(X)
            return X[1:] - X[:-1]

        def second_order_d(X):
            X = np.asarray(X)
            first_order = first_order_d(X)
            return first_order[1:] - first_order[:-1]

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

        self.features = pd.DataFrame(
            np.asarray([
                X.min(), X.max(), X.mean(), X.std(), X.apply(skew), X.apply(
                    kurtosis), X.apply(np.median), X.apply(fo_mean), X.apply(
                        fo_std), X.apply(fo_min), X.apply(fo_median),
                X.apply(fo_max), X.apply(fo_skew), X.apply(fo_kurt), X.apply(
                    so_mean), X.apply(so_std), X.apply(so_min), X.apply(
                        so_median), X.apply(so_max), X.apply(so_skew), X.apply(
                            so_kurt)
            ]).T).dropna(1)

    def transform(self, X, y=None, *args, **kwargs):
        def first_order_d(X):
            X = np.asarray(X)
            return X[1:] - X[:-1]

        def second_order_d(X):
            X = np.asarray(X)
            first_order = first_order_d(X)
            return first_order[1:] - first_order[:-1]

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

        features = pd.DataFrame(
            np.asarray([
                X.min(), X.max(), X.mean(), X.std(), X.apply(skew), X.apply(
                    kurtosis), X.apply(np.median), X.apply(fo_mean), X.apply(
                        fo_std), X.apply(fo_min), X.apply(fo_median),
                X.apply(fo_max), X.apply(fo_skew), X.apply(fo_kurt), X.apply(
                    so_mean), X.apply(so_std), X.apply(so_min), X.apply(
                        so_median), X.apply(so_max), X.apply(so_skew), X.apply(
                            so_kurt)
            ]).T).dropna(1)
        return features

    def fit_transform(self, X, y=None, *args, **kwargs):
        def first_order_d(X):
            X = np.asarray(X)
            return X[1:] - X[:-1]

        def second_order_d(X):
            X = np.asarray(X)
            first_order = first_order_d(X)
            return first_order[1:] - first_order[:-1]

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

        self.features = pd.DataFrame(
            np.asarray([
                X.min(), X.max(), X.mean(), X.std(), X.apply(skew), X.apply(
                    kurtosis), X.apply(np.median), X.apply(fo_mean), X.apply(
                        fo_std), X.apply(fo_min), X.apply(fo_median),
                X.apply(fo_max), X.apply(fo_skew), X.apply(fo_kurt), X.apply(
                    so_mean), X.apply(so_std), X.apply(so_min), X.apply(
                        so_median), X.apply(so_max), X.apply(so_skew), X.apply(
                            so_kurt)
            ]).T).dropna(1)
        return self.features


class tsfreshSeries(TransformerMixin):
    def __init__(self, *args, **kwargs):

        print('Using tsfresh as backend')

    def fit(self, X, y=None, *args, **kwargs):

        if y == None:
            self.model = extract_features
        else:
            self.model = extract_relevant_features

    def transform(self, X, y=None, *args, **kwargs):

        data = X
        X = data[0]
        time = data[0].index.values
        column_id = np.repeat(0, len(X.values))

        for i in range(1, len(data)):
            column_id = np.concatenate(
                [column_id, np.repeat(i, len(data[i].values))])
            time = np.concatenate([time, data[i].index.values])
            X = pd.concat([X, data[i]])

        dataset = pd.DataFrame(np.asarray([column_id, time, X.values]).T)
        dataset.columns = ['id', 'time', 'X']

        features = self.model(dataset, column_id='id', column_sort='time')

        return features.dropna(1)

    def fit_transform(self, X, y=None, *args, **kwargs):

        if y == None:
            self.model = extract_features
        else:
            self.model = extract_relevant_features

        data = X
        X = data[0]
        time = data[0].index.values
        column_id = np.repeat(0, len(X.values))

        for i in range(1, len(data)):
            column_id = np.concatenate(
                [column_id, np.repeat(i, len(data[i].values))])
            time = np.concatenate([time, data[i].index.values])
            X = pd.concat([X, data[i]])

        dataset = pd.DataFrame(np.asarray([column_id, time, X.values]).T)
        dataset.columns = ['id', 'time', 'X']

        features = features = self.model(
            dataset, column_id='id', column_sort='time')

        return features.dropna(1)


class BsplineSeries(TransformerMixin):
    def __init__(self, degrees, knots):
        self.degrees = degrees
        self.knots = knots

    def fit(self, X, y=None):

        self.model_list = []
        for i in X:
            smoothed_array = ps.builtins.bs(
                i.values, degrees=self.degrees, df=self.knots)
            model = LinearRegression()
            model.fit(smoothed_array, i.values)
            self.model_list.append(model)

    def transform(self, X):

        param_matrix = []
        for i in self.model_list:
            params = i.coef_
            param_matrix.append(params)

        return pa.DataFrame(np.asarray(param_matrix).T)

    def fit_transform(self, X):

        self.model_list = []
        for i in X:
            smoothed_array = ps.builtins.bs(
                i.values, degrees=self.degrees, df=self.knots)
            model = LinearRegression()
            model.fit(smoothed_array, i.values)
            self.model_list.append(model)

        param_matrix = []

        for i in self.model_list:
            params = i.coef_
            param_matrix.append(params)

        return pa.DataFrame(np.asarray(param_matrix).T)


class localRSeries(TransformerMixin):
    def __init__(self, fractions):
        self.fractions = franctions

    def fit(self, X, y=None):

        self.model_list = []
        for i in X:
            smoothed_array = localRegression.RegisterLocalRegression(
                i.values)
            model = LinearRegression()
            model.fit(smoothed_array, i.values)
            self.model_list.append(model)

    def transform(self, X):

        param_matrix = []
        for i in self.model_list:
            params = i.coef_
            param_matrix.append(params)

        return pa.DataFrame(np.asarray(param_matrix).T)

    def fit_transform(self, X):

        self.model_list = []
        for i in X:
            smoothed_array = localRegression.RegisterLocalRegression(
                i.values)
            model = LinearRegression()
            model.fit(smoothed_array, i.values)
            self.model_list.append(model)

        param_matrix = []

        for i in self.model_list:
            params = i.coef_
            param_matrix.append(params)

        return pa.DataFrame(np.asarray(param_matrix).T)

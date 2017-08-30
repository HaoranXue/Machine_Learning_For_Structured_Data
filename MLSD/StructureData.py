import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .Transformers import BasicSeries, BasicBag, BasicText, BasicImage


class SData(object):
    '''
    SData is a feature container for Structed feature.
    SData has four different dtype:

    Series: Series data is a sequence plus time information,
            sequence data can be classified in this type too.

    Bag:    A set of features without particular order, we also
            classify mesh data in this case, there will be a specific
            transformers for the features extractions of mesh data.

    Image:  A sex of Image with default transformers from sklearn.

    Text:   A set of Text data with default transormers from sklearn.
    '''

    def __init__(self,
                 x,
                 index=None,
                 column=None,
                 dtype='Series',
                 transformer=None):

        # Store the Structed DATA and Check if the input x is right.
        if type(x) == list:
            self.values = np.asarray(x)
        elif type(x) == np.ndarray:
            self.values = x
        else:
            print('Error: Input for x shold be a list or numpy array')

        # Store the dtype and Check if the dtype is right.
        if dtype == 'Series':
            for i in range(len(self.values)):
                if type(self.values[i]) != pd.Series:
                    self.values[i] = pd.Series(self.values[i])
                else:
                    pass
            self.dtype = dtype
            if transformer == None:
                self.transformer = BasicSeries(Dreduction=None)

        elif dtype == 'Bag':
            for i in range(len(self.values)):
                if type(self.values[i]) != pd.Series:
                    self.values[i] = pd.Series(self.values[i])
                else:
                    pass
            self.dtype = dtype
            if transformer == None:
                self.transformer = BasicBag()

        elif dtype == 'Image':
            self.dtype = dtype
            if transformer == None:
                self.transformer = BasicImage()

        elif dtype == 'Text':
            self.dtype = dtype
            if transformer == None:
                self.transformer = BasicText()

        else:
            print('Error: dtype should be Series, Bag, Image or Text')

        #Setting index.
        if index == None:

            self.index = range(len(self.values))

        # Store the transformer object.
        if transformer != None:

            self.transformer = transformer

        self.column = column

    def __len__(self):

        return len(self.values)

    def __getitem__(self, n):

        return self.values[n]

    def __str__(self):

        return 'SData'

    def __call__(self):

        pass

    def plot(self):
        for i in self:
            plt.plot(i)

        plt.show()

    @property
    def p_dtype(self):

        return type(self.values)

    @property
    def size(self):
        return [len(i) for i in self.values]

    def append(self, new_row):

        self.values = np.row_stack((self.values, new_row))

    def reindex(self):

        self.index = range(len(self.values))

    def min(self):
        if self.dtype in ['Image', 'Text']:
            print('Error: Image or Text data cannot caculate the min')
        else:
            return np.asarray([np.min(i) for i in self.values])

    def max(self):
        if self.dtype in ['Image', 'Text']:
            print('Error: Image or Text data cannot caculate the max')
        else:
            return np.asarray([np.max(i) for i in self.values])

    def std(self):
        if self.dtype in ['Image', 'Text']:
            print('Error: Image or Text data cannot caculate the std')
        else:
            return np.asarray([np.std(i) for i in self.values])

    def mean(self):
        if self.dtype in ['Image', 'Text']:
            print('Error: Image or Text data cannot caculate the mean')
        else:
            return np.asarray([np.mean(i) for i in self.values])

    def shift(self, n):
        self.values = self.values[:-1]
        self.index = self.index[1:]

    def inner_shift(self, n):
        if self.dtype in ['Image', 'Text', 'Bag']:
            print(
                'Error: Image,Text and bag data don not have inner_shift method'
            )
        else:
            return np.asarray([i.shift(n) for i in self.values])

    def fillna(self, x):
        if self.dtype in ['Image', 'Text']:
            print('Error: Image,Text data don not have fillna method')
        else:
            for i in range(len(self.values)):
                self.values[i] = self.values[i].fillna(x)

    def ffill(self, x):
        if self.dtype in ['Image', 'Text']:
            print('Error: Image,Text data don not have ffill method')
        else:
            for i in range(len(self.values)):
                self.values[i] = self.values[i].ffill

    def bfill(self, x):
        if self.dtype in ['Image', 'Text']:
            print('Error: Image,Text data don not have bfill method')
        else:
            for i in range(len(self.values)):
                self.values[i] = self.values[i].bfill

    def split_train_test(self,y,test_ratio=0.3):

        length = len(self.values)
        test_len = int(test_ratio * length)
        train_len = length - test_len
        index = np.asarray(range(length))
        np.random.shuffle(index)
        train_index = index[:train_len]
        test_index = index[train_len:]

        y_test = y[test_index]
        y_train = y[train_index]

        X_test = SData(
            self.values[test_index],
            dtype=self.dtype,
            transformer=self.transformer)

        X_train = SData(
            self.values[train_index],
            dtype=self.dtype,
            transformer=self.transformer)

        return X_train,y_train, X_test,y_test

    @property
    def extracted_features(self):
        transformer = self.transformer
        return transformer.fit_transform(self)

    def resample(cls, freq, func):

        if self.dtype == 'Series':
            self.values = np.asarray(
                [i.resample(freq).apply(func) for i in self.values])
        else:
            print('Only Series type can be resampled')

    @classmethod
    def C_resample(cls, freq, func):

        if self.dtype == 'Series':
            return cls([i.resample(freq).apply(func) for i in self.values],
                       dtype='Series',
                       transformer=self.transformer)
        else:
            print('Only Series type can be resampled')

    def apply(self, func):

        return np.asarray([func(i) for i in self.values])

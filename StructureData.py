import numpy as np
import pandas as pd


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

    def __init__(self, x, index = None, dtype='Series', transformer=None):

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

        elif dtype == 'Bag':
            for i in range(len(self.values)):
                if type(self.values[i]) != pd.Series:
                    self.values[i] = pd.Series(self.values[i])
                else:
                    pass
            self.dtype = dtype

        elif dtype == 'Image':
            self.dtype = dtype

        elif dtype == 'Text':
            self.dtype = dtype

        else:
            print('Error: dtype should be Series, Bag, Image or Text')

        #Setting index.
        if index == None:

            self.index = range(len(self.values))

        # Store the transformer object.
        self.transformer = transformer

    def __len__(self):

        return len(self.values)

    def __getitem__(self, n):

        return self.values[n]

    def __str__(self):

        return 'SData'

    def __call__(self):

        pass

    def append(self, new_row):

        self.values = np.row_stack((self.values, new_row))

    def reindex(self):

        self.index = range(len(self.values))

    def std(self):
        if self.dtype in ['Image', 'Text']:
            print('Error: Image or Text data cannot caculate the std')
        else:
            return [np.std(i) for i in self.values]

    def mean(self):
        if self.dtype in ['Image', 'Text']:
            print('Error: Image or Text data cannot caculate the std')
        else:
            return [np.mean(i) for i in self.values]

    @property
    def extracted_features(self):

        return self.transformer.fit(self.values).transform()

    def resample(cls, freq, func):

        if self.dtype == 'Series':
            self.values = [i.resample(freq).apply(func) for i in self.values]
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

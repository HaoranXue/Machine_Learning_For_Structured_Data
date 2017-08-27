import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .StructureData import *

class SDataFrame(object):
    '''
    StructureDataFrame is a data container to store sereis or structed data as features
    for machine learning models.
    '''

    def __init__(self, data, columns=None, index=None, transformers= None):

        #Check data's format.
        len_list = [len(i) for i in data]
        if len(np.unique(len_list)) == 1:
            pass
        else:
            print('Error: Different features have different length.')

        self.len = len_list[0]

        #Check columns format.
        if columns == None:
            self.columns = range(len(self.data))
        elif len(columns) == len(self.data):
            self.columns = columns
        else:
            print('Error: columns have a different len from the data.')

        #Check index format.
        if index == None:
            index = range(self.len)
        elif len(index) == self.len:
            self.index = index
        else:
            print('Error: index have a different len.')

        self.data = []
        for i in data:
            if type(i) == list or type(i) == np.ndarray:
                self.data.append(SData(x=i, index =index))
            elif print(i) == 'SData':
                self.data.append(i)
            else:
                print('Error: Elements of data have a wrong type.')

        if transformers == None:
            pass
        else:
            self.transformers = transformers
        # dtype of Structed data

        self.dtype = np.asarray([i.dtype for i in self.data])
        self.size = np.asarray([i.size for i in self.data])

    def min(self):
        if 'Image' in self.dtype or 'Text' in self.dtype :
            print('Error: Image or Text data cannot caculate the min')
        else:
            return pd.DataFrame(np.asarray([i.min() for i in self.data]), index= self.index,columns=self.columns)

    def max(self):
        if 'Image' in self.dtype or 'Text' in self.dtype :
            print('Error: Image or Text data cannot caculate the max')
        else:
            return pd.DataFrame(np.asarray([i.max() for i in self.data]), index= self.index,columns=self.columns)

    def mean(self):
        if 'Image' in self.dtype or 'Text' in self.dtype :
            print('Error: Image or Text data cannot caculate the mean')
        else:
            return pd.DataFrame(np.asarray([i.mean() for i in self.data]), index= self.index,columns=self.columns)

    def std(self):
        if 'Image' in self.dtype or 'Text' in self.dtype :
            print('Error: Image or Text data cannot caculate the std')
        else:
            return pd.DataFrame(np.asarray([i.std() for i in self.data]), index= self.index,columns=self.columns)

    def shift(self,n):
        self.data = [i.shift(n) for i in self.data]
        self.index = self.index[1:]

    def inner_shift(self,n):
        self.data = [i.inner_shift(n) for i in self.data]

    def iloc(self,x,y):
        if type(x) == np.int and type(y) == np.int:
            return self.data[y][x]
        else:
            print('Error: x and y should be int')

    def __len__(self):

        return self.len

    def __str__(self):

        pass

    def __call__(self):

        pass

    def __getitem__(self, n):

        item = [i == n for i in self.index]
        return [i[item] for i in self.data]

    def __getattr__(self, attr):

        for i in range(len(self.columns)):
            if attr == self.columns[i]:
                return self.data[i]
            else:
                pass

    def resample(self,freq,func):

        for i in range(len(self.data)):
            if type(self.data[i]) == np.ndarray:
                pass

            elif self.data[i].dtype == 'Series':
                self.data[i].resample(freq).apply(func)

            else:
                pass

    def fillna(self,x):
        for i in self.data:
            i.fillna(x)

    def ffill(self,x):
        for i in self.data:
            i.ffill()

    def bfill(self,x):
        for i in self.data:
            i.bfill()

    @property
    def extracted_features(self):
        features = self.data[0].extracted_features
        for i in range(len(self.data)):
            features.join(
            self.data[i].extracted_features)

        return features



    @classmethod
    def join(clf,new_sdata):

        if print(new_sdata) == 'SData':

            sdata=[]
            for i in self.index:
                sdata.append(new_sdata.values[new_sdata.index == i] )

            join_sdata = SData(sdata, index =self.index, column = new_sdata.column, dtype=new_sdata.dtype, transformer=new_sdata.transformer)


            return clf(data = self.data.append(join_sdata), columns = self.columns.append(join_sdata.column), index = self.index, transformer= self.transformer)

    @classmethod
    def C_resample(cls,freq,func):

        new_data = []
        for i in range(len(self.data)):
            if type(self.data[i]) == np.ndarray:
                new_data.append(self.data[i])
            elif self.data[i].dtype == 'Series':
                new_data.append(self.data[i].C_resample(freq).apply(func))
            else:
                new_data.append(self.data[i])

        return cls(data= new_data, columns = self.columns, index=self.index)

    @classmethod
    def apply(cls,func):
        new_data = []
        for i in range(len(self.values)):
            new_data.append(self.data.apply(func))

        return cls(new_data, index = self.index, columns = self.columns,       transformer = self.transformer)

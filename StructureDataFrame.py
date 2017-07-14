import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SDataFrame(object):
    '''
    StructureDataFrame is a data container to store sereis or structed data as features
    for machine learning models.
    '''

    def __init__(self, data, columns=None, index=None):

        #Check data's format.
        len_list = [len(i) for i in data]
        if len(np.unique(len_list)) == 1:
            pass
        else:
            print('Error: Different features have different length.')

        self.len = len_list[0]

        self.data = []
        for i in data:
            if type(i) == list:
                self.data.append(np.asarray(i))
            elif print(i) == 'SData':
                self.data.append(i)
            else:
                print('Error: Elements of data have a wrong type.')

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

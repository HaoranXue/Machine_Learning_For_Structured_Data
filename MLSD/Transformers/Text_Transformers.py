import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class BasicText(TransformerMixin):

    def __init__(self, Dreduction= None, *args,**kwargs):

        self.Dreduction = Dreduction

    def fit(self, X, y =None):
        self.trans = CountVectorizer(*args,**kwargs)
        self.TFid = TfidfTransformer()
        self.trans.fit(X.values)
        self.TFid.fit(self.trans.fit_transform(X.values).toarray())


    def transform(self,X):

        self.features = pd.DataFrame(self.TFid.transform(X.values), index=X.index)
        
        return self.features

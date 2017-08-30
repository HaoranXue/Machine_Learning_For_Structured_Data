from sklearn.base import TransformerMixin


class activeTrans(TransformerMixin):
    def __init__(self,
                 ifSData=False,
                 Trans_in=True,
                 Multi_col=False,
                 New_Trans=None,
                 Reset_default=False):

        self.Trans_in = Trans_in
        self.Multi_col = Multi_col
        self.New_Trans = New_Trans
        self.Reset_default = Reset_default
        self.ifSData = ifSData

    def fit(self, X, y=None):

        if self.New_Trans != None:
            X.transformer = self.New_Trans

        if self.ifSData == True:
            trans = X.transformer
            trans.fit(X)
            self.trans = trans

        elif self.ifSData == False:
            self.trans = []
            for i in X.data:
                self.trans.append(i.transformer.fit(i))

    def transform(self, X, y=None):

        if self.New_Trans != None:
            X.transformer = self.New_Trans

        if self.ifSData == True:

            return self.trans.transform(X)

        elif self.ifSData == False:

            features = self.trans[0].transform(X.data[0])
            for i in range(2, len(X.data)):
                features.join(self.trans[i].transform(X.data[i]))
            return features

    def fit_transform(self, X, y=None):
        
        if self.New_Trans != None:
            X.transformer = self.New_Trans

        if self.ifSData == True:
            self.trans = X.transformer
            return self.trans.fit_transform(X)

        elif self.ifSData == False:
            self.trans = []
            for i in X.data:
                self.trans.append(i.transformer)
            features = self.trans[0].fit_transform(X.data[0])
            for i in range(2, len(X.data)):
                features.join(self.trans[i].fit_transform(X.data[i]))
            return features

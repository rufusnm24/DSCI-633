import pandas as pd
import numpy as np
import sys


sys.path.insert(0,'../..')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
class my_model():
    def fit(self, X, y):
        # do not exceed 29 mins
        data=pd.concat([X,y], axis=1)
        # preprocessing text data
        col = ["title", "location", "description", "requirements"]
        for i in col:
            data[i] = data[i].str.lower()
            data[i] = data[i].str.replace('[^a-zA-Z]', ' ')

        data['text'] = data['description'] + data['requirements']

        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75, norm='l2')
        train = self.vectorizer.fit_transform(data['text'].astype('U'))

        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1, 0.1, 0.01, 0.001],
                      'kernel': ['rbf']}
        self.grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
        self.grid.fit(train, y)

        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        col = ["title", "location", "description", "requirements"]
        for i in col:
            X[i] = X[i].str.lower()
            X[i] = X[i].str.replace('[^a-z]', ' ')
        X['text'] = X['description'] + X['requirements']
        test = self.vectorizer.transform(X['text'].astype('U'))
        predictions = self.grid.predict(test)
        return predictions





import pandas as pd
import numpy as np
from copy import deepcopy

class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50):
        # Multi-class Adaboost algorithm (SAMME)
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator roundsc
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]

    def error(self,X, y):
        w = np.full(len(y), (1 / len(y)))
        self.estimators.fit(X, y, sample_weight=w)
        pred = self.estimators.predict(X)
        error = sum(w * (pred != y))
        return error;


    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        n = len(y)
        #initalizing weights
        w = np.full(n, (1 / n))
        self.alpha = np.zeros(self.n_estimators)
        for i in range(self.n_estimators):
            self.estimators[i].fit(X, y, sample_weight=w)
            pred = self.estimators[i].predict(X)
            error = sum(w * (pred != y))
            while(error >= (1-(1/k))):
                 error=error(X,y)
            self.alpha[i] = np.log((1 - error) / error) + np.log(k - 1)
            w = w * np.exp(self.alpha[i] * (pred != y))
            w = w / sum(w)
        self.alpha = self.alpha / sum(self.alpha)
        return self

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        probs = self.predict_proba(X)
        key = probs.idxmax(axis=1, skipna=True)
        predictions = key.tolist()
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = {}
        for y in self.classes_:
            probs[y] = np.zeros(len(X))
            for i in range(len(self.estimators)):
                probs[y] += self.alpha[i] * ((self.estimators[i].predict(X)) == y)
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs






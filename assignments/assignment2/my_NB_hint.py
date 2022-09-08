import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        # list of classes for this model
        self.classes_ = list(set(list(y)))
        # for calculation of P(y)
        self.P_y = Counter(y)
        for i in self.P_y:
            self.P_y[i] = self.P_y[i] /sum(self.P_y.values())
        #print(self.P_y)
        # self.P[yj][Xi][xi] = P(xi|yj) where Xi is the feature name and xi is the feature value, yj is a specific class label
        # make sure to use self.alpha in the __init__() function as the smoothing factor when calculating P(xi|yj)
        self.P={}
        self.features = list(X.columns)

        for outcome in np.unique(y):
            self.P[outcome] = {}
            out_count=sum(y==outcome)
            for feature in self.features:
                self.P[outcome][feature] = {}
                #likelihood = X[feature][y[y==outcome].index.values.tolist()].value_counts().to_dict()
                ni= len(pd.unique(X[feature]))
                for val in np.unique(X[feature]):
                    count=X[feature][y==outcome][X[feature]==val].count()
                    self.P[outcome][feature][val] = (count + self.alpha) / (out_count + (ni * self.alpha))
        #print(self.P)



    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # P(yj|x) = P(x|yj)P(yj)/P(x)
        # P(x|yj) = P(x1|yj)P(x2|yj)...P(xk|yj) = self.P[yj][X1][x1]*self.P[yj][X2][x2]*...*self.P[yj][Xk][xk]
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for key in X:
                p *= X[key].apply(lambda value: self.P[label][key][value] if value in self.P[label][key] else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        #print("sum",sums)
        probs = probs.apply(lambda v: v / sums)
        #print(probs)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # Hint: predicted class is the class with highest prediction probability (from self.predict_proba)
        probs = self.predict_proba(X)
        #print(probs)
        #print((probs.iloc[2]))
        key = probs.idxmax(axis=1)
        #print(key)
        predictions = key.tolist()
        #print(predictions)
        return predictions






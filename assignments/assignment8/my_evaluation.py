import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion_matrix = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below
        correct = self.predictions == self.actuals
        self.acc = float(Counter(correct)[True]) / len(correct)
        self.confusion_matrix = {}
        for label in self.classes_:
            tp = Counter(self.predictions[self.actuals == label] == label)[True]
            tn = Counter(self.predictions[self.actuals != label] != label)[True]
            fp = Counter(self.predictions[self.actuals != label] == label)[True]
            fn = Counter(self.predictions[self.actuals == label] != label)[True]
            self.confusion_matrix[label] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
        return self

    def accuracy(self):
        if self.confusion_matrix==None:
            self.confusion()
        return self.acc

    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0

        if self.confusion_matrix==None:
            self.confusion()
        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if tp+fp == 0:
                prec = 0
            else:
                prec = float(tp) / (tp + fp)
        else:
            if average == "micro":
                prec = self.accuracy()
            else:
                prec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        prec_label = 0
                    else:
                        prec_label = float(tp) / (tp + fp)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    prec += prec_label * ratio
        return prec

    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0

        if self.confusion_matrix == None:
            self.confusion()
        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]
            if tp + fn == 0:
                recall = 0
            else:
                recall = float(tp) / (tp + fn)
        else:
            if average == "micro":
                recall = self.accuracy()
            else:
                recall = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fn = self.confusion_matrix[label]["FN"]
                    if tp + fn == 0:
                        recall_label = 0
                    else:
                        recall_label = float(tp) / (tp + fn)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    recall += recall_label * ratio
        return recall

    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        if target in self.classes_:
            prec = self.precision(target)
            recall = self.recall(target)
            if prec + recall == 0:
                f1 = 0
            else:
                f1 =float( 2 * prec * recall) / float(prec + recall)
        else:
            if average == "micro":
                f1 = self.accuracy()
            else:
                f1 = 0
                n = len(self.actuals)
                for label in self.classes_:
                    prec = self.precision(label)
                    recall = self.recall(label)
                    if prec + recall == 0:
                        f1_label = 0
                    else:
                        f1_label =float( 2 * prec * recall) / float(prec + recall)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    f1 += f1_label * ratio
        return f1

    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        if type(self.pred_proba)==type(None):
            return None
        else:
            if target in self.classes_:
                order = np.argsort(self.pred_proba[target])[::-1]
                tp = 0
                fp = 0
                fn = Counter(self.actuals)[target]
                tn = len(self.actuals) - fn
                tpr = 0
                fpr = 0
                auc_target = 0
                for i in order:
                    if self.actuals[i] == target:
                        tp = tp + 1
                        fn = fn - 1
                        tpr = float(tp) / float(tp + fn)
                    else:
                        fp = fp + 1
                        tn = tn - 1
                        p_fpr = fpr
                        fpr = float(fp) /float (fp + tn)
                        auc_target = float(auc_target) + float(fpr - p_fpr) * float(tpr)
            else:
                raise Exception("Unknown target class.")

            return auc_target



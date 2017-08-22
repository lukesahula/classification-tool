from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

class EvaluationTool():

    def __init__(self, file_path, delimiter):

        self.true, self.pred = self.read_data(file_path, delimiter)
        self.stats = self.compute_stats()


    def read_data(self, file_path, delimiter):
        labels = ['true', 'pred']
        df = pd.read_csv(file_path, delimiter, header=None, names=labels)
        return (df['true'], df['pred'])

    def compute_stats(self):
        labels = set(self.true)
        labels.union(self.pred)

        labels = list(labels)
        labels.sort()

        matrix = confusion_matrix(self.true, self.pred, labels=labels)
        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)

        stats = {}

        indexes = range(0, len(matrix[0]))

        for label, index in zip(labels, indexes):
            stats[label] = {}
            stats[label]['FP'] = FP[index]
            stats[label]['FN'] = FN[index]
            stats[label]['TP'] = TP[index]
            stats[label]['TN'] = TN[index]

        return stats

    def compute_precision(self, class_label):
        '''
        Computes precision for the given class label.
        :param class_label: Class label of the row.
        :return: Computed precision of the classifier for the given class.
        '''
        true_positive = self.stats[class_label]['TP']
        false_positive = self.stats[class_label]['FP']

        try:
            precision = true_positive / (true_positive + false_positive)
            return precision
        except ZeroDivisionError:
            print("Division by zero.")

    def compute_recall(self, class_label):
        '''
        Computes recall for the given class label.
        :param class_label: Class label of the row.
        :return: Computed recall of the classifier for the given class.
        '''
        true_positive = self.stats[class_label]['TP']
        false_negative = self.stats[class_label]['FN']

        try:
            recall = true_positive / (true_positive + false_negative)
            return recall
        except ZeroDivisionError:
            pass
        print("Exception by zero.")

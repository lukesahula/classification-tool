from collections import defaultdict

from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

class EvaluationTool():

    def __init__(self, file_path, delimiter):
        self.true = None
        self.pred = None
        self.stats = None
        self.labels = None

        self.__read_data(file_path, delimiter)
        self.__compute_stats()


    def __read_data(self, file_path, delimiter):
        """
        Reads true/pred data from a file and saves it to dataframes.
        :param file_path: Path to the file
        :param delimiter: Symbol or as tring by which the data is delimited.
        """
        columns = ['true', 'pred']
        df = pd.read_csv(file_path, delimiter, header=None, names=columns)
        self.true = df['true']
        self.pred = df['pred']

    def __compute_stats(self):
        """
        Computes TPs, FPs, TNs and FNs of the given data.
        """
        self.labels = sorted(set(self.true) | set(self.pred))

        matrix = confusion_matrix(self.true, self.pred, labels=self.labels)
        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)


        stats = defaultdict(dict)

        for i, label in zip(range(len(self.labels)), self.labels):
            stats[label]['FP'] = FP[i]
            stats[label]['FN'] = FN[i]
            stats[label]['TP'] = TP[i]
            stats[label]['TN'] = TN[i]

        self.stats = stats

    def compute_precision(self, class_label):
        """
        Computes precision for the given class label.
        :param class_label: Class label of the row.
        :return: Computed precision of the classifier for the given class.
        """
        TP = self.stats[class_label]['TP']
        FP = self.stats[class_label]['FP']

        return TP / (TP + FP)

    def compute_recall(self, class_label):
        """
        Computes recall for the given class label.
        :param class_label: Class label of the row.
        :return: Computed recall of the classifier for the given class.
        """
        TP = self.stats[class_label]['TP']
        FN = self.stats[class_label]['FN']

        return TP / (TP + FN)

    def get_avg_precision(self):
        """
        Returns average precision score from
        all class labels.
        """
        cumsum = 0

        for label in self.labels:
            cumsum += self.compute_precision(label)
        avg = cumsum / len(self.labels)

        return avg

    def get_avg_recall(self):
        """
        Return average recall score from
        all class labels.
        """
        cumsum = 0

        for label in self.labels:
            cumsum += self.compute_recall(label)
        avg = cumsum / len(self.labels)

        return avg

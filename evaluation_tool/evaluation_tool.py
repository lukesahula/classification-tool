from collections import defaultdict

from sklearn.metrics import confusion_matrix
from loading_tool.loading_tool import load_classifications

import numpy as np

class EvaluationTool():

    def __init__(self, file_path, delimiter, legit=None, agg=False):
        self.metadata = None

        self.legit = legit
        self.trues, self.preds, self.metadata = load_classifications(
            file_path,
            delimiter,
            agg
        )
        self.labels = sorted(set(self.trues) | set(self.preds))

    def compute_stats(self):
        """
        Computes TPs, FPs, TNs and FNs of the given data.
        """
        matrix = confusion_matrix(self.trues, self.preds, labels=self.labels)
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

        return stats

    def compute_aggregated_stats(self, agg_column):
        """
        Computes TPs, FPs, and FNs of the given data aggregated by
        the specified agg_collumn.
        """
        stats = defaultdict(dict)
        for label in self.labels:
            stats[label] = {}
            stats[label]['TP'] = set()
            stats[label]['FP'] = set()
            stats[label]['FN'] = set()

        keys = list(self.metadata[agg_column])

        for i in range(len(keys)):
            if self.trues[i] == self.preds[i]:
                stats[self.trues[i]]['TP'].add(keys[i])
            else:
                stats[self.trues[i]]['FN'].add(keys[i])
                stats[self.preds[i]]['FP'].add(keys[i])

        for label in self.labels:
            stats[label]['FP'] = stats[label]['FP'] - stats[label]['TP']
            stats[label]['FN'] = stats[label]['FN'] - stats[label]['TP']

            stats[label]['FP'] = len(stats[label]['FP'])
            stats[label]['FN'] = len(stats[label]['FN'])
            stats[label]['TP'] = len(stats[label]['TP'])

        return stats

    def compute_precision(self, class_label, stats):
        """
        Computes precision for the given class label.
        :param class_label: Class label of the row.
        :return: Computed precision of the classifier for the given class.
        """
        TP = stats[class_label]['TP']
        FP = stats[class_label]['FP']

        if TP + FP == 0:
            return np.nan
        return TP / (TP + FP)

    def compute_recall(self, class_label, stats):
        """
        Computes recall for the given class label.
        :param class_label: Class label of the row.
        :return: Computed recall of the classifier for the given class.
        """
        TP = stats[class_label]['TP']
        FN = stats[class_label]['FN']

        if TP + FN == 0:
            return np.nan
        return TP / (TP + FN)

    def get_avg_precision(self, stats, legit=True, nan=True):
        """
        Counts the average precision.
        :param legit: If false, legit label is skipped.
        :param nan: If false, nan precisions are skipped.
        :return: Average precision.
        """
        labels = self.labels

        if not legit:
            labels.remove(self.legit)

        precs = np.fromiter(
            [self.compute_precision(label, stats) for label in labels],
            dtype=float,
            count=len(labels)
        )

        if not nan:
            return np.nanmean(precs)
        else:
            return np.nansum(precs) / len(labels)

    def get_avg_recall(self, stats, legit=True, nan=True):
        """
        Counts the average recall.
        :param legit: If false, legit label is skipped.
        :param nan: If false, nan precisions are skipped.
        :return: Average recall.
        """
        labels = self.labels

        if not legit:
            labels.remove(self.legit)

        recs = np.fromiter(
            [self.compute_recall(label, stats) for label in labels],
            dtype=float,
            count=len(labels)
        )

        if not nan:
            return np.nanmean(recs)
        else:
            return np.nansum(recs) / len(labels)

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

    def compute_stats(self, trues, preds):
        """
        Computes TPs, FPs, TNs and FNs of the given data.
        """
        labels = sorted(set(trues) | set(preds))

        matrix = confusion_matrix(trues, preds, labels=labels)
        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)

        stats = defaultdict(dict)

        for i, label in zip(range(len(labels)), labels):
            stats[label]['FP'] = FP[i]
            stats[label]['FN'] = FN[i]
            stats[label]['TP'] = TP[i]
            stats[label]['TN'] = TN[i]

        return stats

    def filter_data(self, agg_column, agg_key):
        """
        Filters data by the aggregation key in the aggregation column.
        """
        indexes_by_key = self.metadata.index[
            self.metadata[agg_column] == agg_key
        ].tolist()

        trues_by_key = self.trues[indexes_by_key].tolist()
        preds_by_key = self.preds[indexes_by_key].tolist()
        return (trues_by_key, preds_by_key)

    def aggregate_stats(self, agg_column, agg_key):
        """
        Aggregates statistics by the aggregation key in the aggregation column.
        """
        trues, preds = self.filter_data(agg_column, agg_key)
        stats = self.compute_stats(trues, preds)
        labels = list(stats.keys())

        for label in labels:
            if stats[label]['TP'] != 0:
                stats[label]['TP'] = 1
                stats[label]['FP'] = 0
                stats[label]['FN'] = 0

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

    def get_avg_precision(self, labels, stats, legit=True, nan=True):
        """
        Counts the average precision.
        :param legit: If false, legit label is skipped.
        :param nan: If false, nan precisions are skipped.
        :return: Average precision.
        """
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

    def get_avg_recall(self, labels, stats, legit=True, nan=True):
        """
        Counts the average recall.
        :param legit: If false, legit label is skipped.
        :param nan: If false, nan precisions are skipped.
        :return: Average recall.
        """
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

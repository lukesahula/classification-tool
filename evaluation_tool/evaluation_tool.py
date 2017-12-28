from collections import defaultdict

from sklearn.metrics import confusion_matrix

import numpy as np

class EvaluationTool():

    def __init__(self, legit=None, agg=False):
        self.legit = legit
        self.labels = []


    def compute_stats(self, data_chunk):
        """
        Computes TPs, FPs and FNs of the given data using sklearn.
        :param data_chunk: A chunk of data containing trues and preds.
        :return: A dictionary of stats containing TPs, FPs, FNs for all
        labels.
        """
        trues = data_chunk[0]
        preds = data_chunk[1]
        labels = sorted(set(trues) | set(preds))

        #TODO Smelly
        self.labels = sorted(set(labels) | set(self.labels))

        matrix = confusion_matrix(trues, preds, labels=labels)
        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)

        stats = defaultdict(dict)

        for i, label in zip(range(len(labels)), labels):
            stats[label]['FP'] = FP[i]
            stats[label]['FN'] = FN[i]
            stats[label]['TP'] = TP[i]

        return dict(stats)

    def compute_aggregated_stats(self, agg_column, data_chunk):
        """
        Computes TPs, FPs and FNs of the given data aggregated by
        the specified agg_collumn.
        :param agg_column: The name of the column specifying the aggregation.
        :param data_chunk: A chunk of data containing trues and preds.
        :return: A dictionary of stats containing TPs, FPs and FNs for all
        labels.
        """

        stats = defaultdict(lambda: defaultdict(set))

        trues = data_chunk[0]
        preds = data_chunk[1]
        metadata = data_chunk[2]

        labels = sorted(set(trues) | set(preds))
        keys = list(metadata[agg_column])

        #TODO Smelly
        self.labels = sorted(set(labels) | set(self.labels))

        for i in range(len(trues)):
            if trues[i] == preds[i]:
                stats[trues[i]]['TP'].add(keys[i])
            else:
                stats[trues[i]]['FN'].add(keys[i])
                stats[preds[i]]['FP'].add(keys[i])

        for label in labels:
            stats[label]['FP'] -= stats[label]['TP']
            stats[label]['FN'] -= stats[label]['TP']

            stats[label]['FP'] = len(stats[label]['FP'])
            stats[label]['FN'] = len(stats[label]['FN'])
            stats[label]['TP'] = len(stats[label]['TP'])

        return dict(stats)

    def compute_precision(self, class_label, stats):
        """
        Computes precision for the given class label.
        :param class_label: Class label of the row.
        :param stats: Computed statistics (TPs, FPs, FNs for given label)
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
        :param stats: Computed statistics (TPs, FPs, FNs for given label)
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
        :param stats: Computed statistics (TPs, FPs, FNs for all labels)
        :param legit: If false, legit label is skipped.
        :param nan: If false, nan precisions are skipped.
        :return: Average precision.
        """
        labels = list(self.labels)

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
        :param stats: Computed statistics (TPs, FPs, FNs for all labels)
        :param legit: If false, legit label is skipped.
        :param nan: If false, nan precisions are skipped.
        :return: Average recall.
        """
        labels = list(self.labels)

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

    def get_stats_counts(self, labels, stats):
        """
        Counts TPS, FPS and FNs for given labels
        :param labels: A list of labels or a single label.
        :param stats: Computed statistics (TPs, FPs, FNs for all labels)
        :return: Dictionary of counts.
        """
        counts = {}
        counts['TP'] = 0
        counts['FP'] = 0
        counts['FN'] = 0
        if not isinstance(labels, list):
            labels = [labels]
        for label in labels:
            counts['TP'] += stats[label]['TP']
            counts['FP'] += stats[label]['FP']
            counts['FN'] += stats[label]['FN']

        return counts



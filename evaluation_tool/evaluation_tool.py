from collections import defaultdict

from sklearn.metrics import confusion_matrix

import numpy as np
import scipy as sp


class EvaluationTool():
    def __init__(self, legit=None):
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
        self.labels = sorted(set(self.labels) | set(trues) | set(preds))

        matrix = confusion_matrix(trues, preds, labels=self.labels)
        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)

        stats = defaultdict(dict)

        for i, label in zip(range(len(self.labels)), self.labels):
            stats[label]['FP'] = FP[i]
            stats[label]['FN'] = FN[i]
            stats[label]['TP'] = TP[i]

        return dict(stats)

    def aggregate_stats(self, stats):
        """
        Aggregates the prepared stats so that there are no duplicities and
        once there is a TP for the specified user (or other aggregation value),
        there can be no FPs or FNs.
        :param stats: A dictionary of label dictionaries containing all TPs,
        FPs and FNs with their associated value from the aggregation column.
        :return: A dictionary of aggregated TPs, FPs and FNs for all labels.
        """
        for label in self.labels:
            stats[label]['FP'] -= stats[label]['TP']
            stats[label]['FN'] -= stats[label]['TP']

            stats[label]['FP'] = len(stats[label]['FP'])
            stats[label]['FN'] = len(stats[label]['FN'])
            stats[label]['TP'] = len(stats[label]['TP'])

        return dict(stats)

    def compute_stats_for_agg(self, agg_column, data_chunk, relaxed=False):
        """
        Computes TPs, FPs and FNs of the given data prepared for aggregation by
        the specified agg_collumn.
        :param agg_column: The name of the column specifying the aggregation.
        :param data_chunk: A chunk of data containing trues and preds.
        :param relaxed: Do not distinguish between FP and TP if the false label
        is still positive but of a different class.
        :return: A dictionary of label dictionaries containing all TPs, FPs and
        FNs with their associated value from the agg_column.
        labels.
        """

        stats = defaultdict(lambda: defaultdict(set))

        trues = data_chunk[0]
        preds = data_chunk[1]
        metadata = data_chunk[2]

        labels = sorted(set(trues) | set(preds))
        keys = list(metadata[agg_column])

        self.labels = sorted(set(self.labels) | set(trues) | set(preds))

        for true, pred, key in zip(trues, preds, keys):
            if true == pred:
                stats[true]['TP'].add(key)
            else:
                stats[true]['FN'].add(key)
                if relaxed and true != self.legit and pred != self.legit:
                    stats[pred]['TP'].add(key)
                else:
                    stats[pred]['FP'].add(key)

        return stats

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

    def get_avg_precision(self, stats, legit=True, nan=True, par_labels=None):
        """
        Counts the average precision.
        :param stats: Computed statistics (TPs, FPs, FNs for all labels)
        :param legit: If false, legit label is skipped.
        :param nan: If false, nan precisions are skipped.
        :parama par_labels: Compute avg precision from these labels.
        :return: Average precision.
        """
        labels = list(par_labels) if par_labels else list(self.labels)

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

    def get_avg_recall(self, stats, legit=True, nan=True, par_labels=None):
        """
        Counts the average recall.
        :param stats: Computed statistics (TPs, FPs, FNs for all labels)
        :param legit: If false, legit label is skipped.
        :param nan: If false, nan precisions are skipped.
        :param par_labels: Compute avg recall from these labels.
        :return: Average recall.
        """
        labels = list(par_labels) if par_labels else list(self.labels)

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
        labels = [labels] if not isinstance(labels, list) else labels
        for label in labels:
            counts['TP'] += stats[label]['TP']
            counts['FP'] += stats[label]['FP']
            counts['FN'] += stats[label]['FN']

        return counts

    def get_labels_with_prec_above_thres(self, thres, labels, stats):
        """
        Returns those labels with precision above specified threshold.
        :param thres: Number between 0 and 1
        :param labels: A list of labels
        :param stats: Computed statistics (TPs, FPs, FNs for all labels)
        :return: A list of labels.
        """
        return [l for l in labels if self.compute_precision(l, stats) >= thres]

    def compute_corellation_matrix(self, data):
        corr_matrix = data.corr()
        return corr_matrix

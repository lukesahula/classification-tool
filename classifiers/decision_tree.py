from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.stats import entropy, mode
from math import sqrt, ceil


class LeafNode():
    def __init__(self, label):
        self.label = label

    def evaluate(self, observation):
        return self.label


class DecisionNode():
    def __init__(self, left_child, right_child, attr, value):
        self.left_child = left_child
        self.right_child = right_child
        self.attr = attr
        self.value = value

    def evaluate(self, observation):
        if observation[self.attr] <= self.value:
            return self.left_child.evaluate(observation)
        else:
            return self.right_child.evaluate(observation)


class DecisionTree():
    def __init__(self, max_features, min_samples_split, random_state):
        self.root = None
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = np.random.RandomState(random_state)

    def fit(self, feature_matrix, labels):
        self.root = self.__build_tree(feature_matrix, labels)

    def predict(self, observations):
        return [self.root.evaluate(row) for row in observations.itertuples(False)]

    def __build_tree(self, feature_matrix, labels):
        if self.__should_create_leaf_node(feature_matrix, labels):
            return self.__create_leaf_node(labels)
        sampled_columns = self.__get_feature_sample(feature_matrix)
        if sampled_columns is None:
            return self.__create_leaf_node(labels)
        attr, value = self.__find_best_split_parameters(
            sampled_columns, labels
        )
        threshold = self.__get_split_threshold(
            feature_matrix, labels, attr, value
        )
        labels_left = labels.loc[threshold]
        labels_right = labels.loc[~threshold]
        if len(labels_left) == 0 or len(labels_right) == 0:
            return self.__create_leaf_node(
                pd.concat((labels_left, labels_right))
            )
        left_child = self.__build_tree(
            feature_matrix.loc[threshold, :],
            labels_left
        )
        right_child = self.__build_tree(
            feature_matrix.loc[~threshold, :],
            labels_right
        )
        return DecisionNode(left_child, right_child, attr, value)

    def __should_create_leaf_node(self, feature_matrix, labels):
        if len(set(labels)) == 1:
            return True
        if len(feature_matrix.drop_duplicates()) == 1:
            return True
        if len(labels) < self.min_samples_split:
            return True
        return False

    def __create_leaf_node(self, labels):
        return LeafNode(mode(labels)[0][0])

    def __get_feature_sample(self, feature_matrix):
        # Drop constant columns
        feature_matrix = feature_matrix.loc[
            :, (feature_matrix != feature_matrix.iloc[0]).any()
        ]
        if len(feature_matrix.columns) <= 1:
            return None
        if self.max_features == 'sqrt':
            cols = self.random_state.choice(
                a=feature_matrix.columns,
                size=ceil(sqrt(len(feature_matrix.columns))),
                replace=False
            )
            return feature_matrix.loc[:, cols]
        else:
            return feature_matrix

    def __find_best_split_parameters(self, feature_matrix, labels):
        min_entropy = float('inf')
        split = None
        for column in feature_matrix.columns:
            counts_l, counts_r = self.__compute_class_counts(labels)
            coordinates = self.__compute_coordinates(
                feature_matrix[column], labels
            )
            for value, counts in sorted(coordinates.items()):
                for label, count in counts.items():
                    index = np.where(counts_l['class'] == label)
                    counts_l[index] = (
                        counts_l[index]['class'],
                        counts_l[index]['count'] + count
                    )
                    counts_r[index] = (
                        counts_r[index]['class'],
                        counts_r[index]['count'] - count
                    )
                current_entropy = self.__compute_entropy(counts_l, counts_r)
                if current_entropy < min_entropy:
                    min_entropy = current_entropy
                    split = column, value
        return split

    def __get_split_threshold(self, feature_matrix, labels, attr, value):
        threshold = feature_matrix.loc[:, attr].values <= value
        return threshold

    def __compute_entropy(self, counts_l, counts_r):
        size_l = sum(counts_l['count'])
        size_r = sum(counts_r['count'])
        entropy_l = size_l * entropy(counts_l['count']) if size_l != 0 else 0
        entropy_r = size_r * entropy(counts_r['count']) if size_r != 0 else 0
        sum_sizes = size_l + size_r

        return ((entropy_l + entropy_r) / sum_sizes)

    def __compute_class_counts(self, labels):
        names = ['class', 'count']
        formats = ['i8', 'i8']
        dtype = dict(names=names, formats=formats)
        counts_r = np.fromiter(
            Counter(labels).items(), dtype, count=len(set(labels))
        )
        counts_l = counts_r.copy()
        counts_l['count'] = 0

        return counts_l, counts_r

    def __compute_coordinates(self, column, labels):
        classes = set(labels)
        coordinates = defaultdict(dict)
        for v in column:
            for c in classes:
                coordinates[v][c] = 0

        for v, l in zip(column, labels):
            coordinates[v][l] += 1

        return coordinates

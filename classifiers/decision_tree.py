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
        predictions = []
        for index, row in observations.iterrows():
            predictions.append(self.root.evaluate(row))
        return predictions

    def __build_tree(self, feature_matrix, labels):
        if self.__should_create_leaf_node(feature_matrix, labels):
            return self.__create_leaf_node(labels)
        sampled_columns = self.__get_feature_sample(feature_matrix)
        attr, value = self.__find_best_split_parameters(
            sampled_columns, labels
        )
        dataset_left, dataset_right = self.__split_dataset(
            feature_matrix, labels, attr, value
        )
        if len(dataset_left[1]) == 0 or len(dataset_right[1]) == 0:
            return self.__create_leaf_node(
                pd.concat((dataset_left[1], dataset_right[1]))
            )
        left_child = self.__build_tree(dataset_left[0], dataset_left[1])
        right_child = self.__build_tree(dataset_right[0], dataset_right[1])
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

    def __get_feature_sample(self, fm):
        feature_matrix = fm.loc[:, (fm != fm.iloc[0]).any()]  # Drop const cols
        if self.max_features == 'sqrt':
            cols = feature_matrix.sample(
                n=ceil(sqrt(len(feature_matrix.columns))), axis=1,
                random_state=self.random_state)
            return cols
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

    def __split_dataset(self, feature_matrix, labels, attr, value):
        threshold = feature_matrix[attr].values <= value
        return ((feature_matrix[threshold], labels[threshold]),
                (feature_matrix[~threshold], labels[~threshold]))

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

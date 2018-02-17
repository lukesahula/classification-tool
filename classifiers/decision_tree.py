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
        if observation[self.attr] < self.value:
            return self.left_child.evaluate(observation)
        else:
            return self.right_child.evaluate(observation)


class DecisionTree():
    def __init__(self, max_features, min_samples_split, random_state):
        self.root = None
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.feature_sample_set = False

    def fit(self, feature_vectors, labels):
        self.root = self.__build_tree(feature_vectors, labels)

    def predict(self, observations):
        predictions = []
        for index, row in observations.iterrows():
            predictions.append(self.root.evaluate(row))
        return predictions

    def __build_tree(self, feature_vectors, labels):
        if self.__should_create_leaf_node(feature_vectors, labels):
            return self.__create_leaf_node(labels)
        if not self.feature_sample_set:
            feature_vectors = self.__get_feature_sample(feature_vectors)
        attr, value = self.__find_best_split_parameters(
            feature_vectors, labels
        )
        dataset_left, dataset_right = self.__split_dataset(
            feature_vectors, labels, attr, value
        )
        if len(dataset_left[1]) == 0 or len(dataset_right[1]) == 0:
            return self.__create_leaf_node(
                pd.concat((dataset_left[1], dataset_right[1]))
            )
        left_child = self.__build_tree(dataset_left[0], dataset_left[1])
        right_child = self.__build_tree(dataset_right[0], dataset_right[1])
        return DecisionNode(left_child, right_child, attr, value)

    def __should_create_leaf_node(self, feature_vectors, labels):
        if len(set(labels)) == 1:
            return True
        if len(feature_vectors.drop_duplicates()) == 1:
            return True
        if len(labels) < self.min_samples_split:
            return True
        return False

    def __create_leaf_node(self, labels):
        return LeafNode(mode(labels)[0][0])

    def __get_feature_sample(self, feature_vectors):
        self.feature_sample_set = True
        if self.max_features == 'sqrt':
            return feature_vectors.sample(
                n=ceil(sqrt(len(feature_vectors.columns))),
                random_state=self.random_state, axis=1
            )
        else:
            return feature_vectors

    def __find_best_split_parameters(self, feature_vectors, labels):
        max_gain = float('-inf')
        split = None
        for column in feature_vectors:
            counts_l, counts_r = self.__compute_class_counts(labels)
            coordinates = self.__compute_coordinates(
                feature_vectors[column], labels
            )
            for value, counts in coordinates.items():
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
                current_gain = self.__compute_gain(counts_l, counts_r)
                if current_gain >= max_gain:
                    max_gain = current_gain
                    split = column, value
        print('Split: ' + str(split[0]) + ' ' + str(split[1]))
        return split

    def __split_dataset(self, feature_vectors, labels, attr, value):
        threshold = feature_vectors[attr] < value
        return ((feature_vectors[threshold], labels[threshold]),
                (feature_vectors[~threshold], labels[~threshold]))

    def __compute_gain(self, counts_l, counts_r):
        size_l = sum(counts_l['count'])
        size_r = sum(counts_r['count'])
        if size_r == 0 or size_l == 0:
            return float('-inf')
        entropy_l = size_l * entropy(counts_l['count'])
        entropy_r = size_r * entropy(counts_r['count'])
        sum_sizes = size_l + size_r

        gain = - ((entropy_l + entropy_r) / sum_sizes)
        return gain

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

    def __compute_coordinates(self, feature_vector, labels):
        classes = set(labels)
        coordinates = defaultdict(dict)
        for v in feature_vector:
            for c in classes:
                coordinates[v][c] = 0

        for v, l in zip(feature_vector, labels):
            coordinates[v][l] += 1

        return coordinates

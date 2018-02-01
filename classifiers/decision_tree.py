from statistics import mode
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import entropy

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
    def __init__(
            self, max_features, min_samples_split, criterion, random_state
    ):
        self.root = None
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, feature_vectors, labels):
        self.root = self.__build_tree__(feature_vectors, labels)

    def predict(self, feature_vector):
        return self.root.evaluate(feature_vector)

    def __build_tree__(self, feature_vectors, labels):
        if self.__should_create_leaf_node__(feature_vectors, labels):
            return self.__create_leaf_node__(labels)
        attr, value = self.__find_best_split_parameters__(
            feature_vectors, labels
        )
        dataset_left, dataset_right = self.__split_dataset__(
            feature_vectors, labels, attr, value
        )
        left_child = self.__build_tree__(dataset_left[0], dataset_left[1])
        right_child = self.__build_tree__(dataset_right[0], dataset_right[1])
        return DecisionNode(left_child, right_child, attr, value)


    def __should_create_leaf_node__(self, feature_vectors, labels):
        # TODO prozkoumat?
        if len(labels) == 0:
            return False
        if len(set(labels)) == 1:
            return True
        if len(feature_vectors.drop_duplicates()) == 1:
            return True
        if len(labels) < self.min_samples_split:
            return True
        return False

    def __create_leaf_node__(self, labels):
        return LeafNode(mode(labels))

    def __find_best_split_parameters__(self, feature_vectors, labels):
        max_gain = float('-inf')
        split = None
        for column in feature_vectors:
            counts_l, counts_r = self.__compute_class_counts__(labels)
            coordinates = self.__compute_coordinates__(
                feature_vectors[column], labels
            )
            for point in coordinates.items():
                for c_count in point[1].items():
                    index = np.where(counts_l['class'] == c_count[0])
                    counts_l[index] = (
                        counts_l[index][0][0],
                        counts_l[index][0][1] + c_count[1]
                    )
                    counts_r[index] = (
                        counts_r[index][0][0],
                        counts_r[index][0][1] - c_count[1]
                    )
                current_gain = self.__compute_gain__(counts_l, counts_r)
                if current_gain > max_gain:
                    max_gain = current_gain
                    split = column, point[0]
        return split

    def __split_dataset__(self, feature_vectors, labels, attr, value):
        threshold = feature_vectors[attr] < value
        dataset_left = feature_vectors[threshold]
        dataset_right = feature_vectors[~threshold]
        indices_left = dataset_left.index
        indices_right = dataset_right.index
        labels_left = labels[indices_left]
        labels_right = labels[indices_right]
        dataset_left = dataset_left, labels_left
        dataset_right = dataset_right, labels_right
        return dataset_left, dataset_right

    def __compute_gain__(self, counts_l, counts_r):
        size_l = sum(counts_l['count'])
        size_r = sum(counts_r['count'])
        gain = - ((size_l * entropy(counts_l['count']) +
                  size_r * entropy(counts_r['count'])) / (size_l + size_r))
        return gain

    def __compute_class_counts__(self, labels):
        names = ['class', 'count']
        formats = ['i8', 'i8']
        dtype = dict(names=names, formats=formats)
        counts_r = np.fromiter(
            Counter(labels).items(), dtype, count=len(set(labels))
        )
        counts_l = counts_r.copy()
        counts_l['count'] = 0

        return counts_l, counts_r

    def __compute_coordinates__(self, feature_vector, labels):
        classes = set(labels)
        coordinates = defaultdict(dict)
        for v in feature_vector:
            for c in classes:
                coordinates[v][c] = 0

        for v, l in zip(feature_vector, labels):
            coordinates[v][l] += 1

        return coordinates
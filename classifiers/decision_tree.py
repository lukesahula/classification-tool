from statistics import mode
from collections import Counter, defaultdict

class DecisionNode():
    def __init__(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child


class DecisionTree():
    def __init__(
            self, max_features, min_samples_split, criterion, random_state
    ):
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, feature_vectors, labels):
        pass

    def predict(self, feature_vector):
        pass

    def __build_tree__(self, feature_vectors, labels):
        if self.__should_create_leaf_node__(feature_vectors, labels):
            return self.__create_leaf_node__(labels)
        split_param = self.__find_best_split_parameters__(
            feature_vectors, labels
        )
        dataset_left, dataset_right = self.__split_dataset__(
            feature_vectors, labels
        )
        left_child = self.__build_tree__(dataset_left[0], dataset_left[1])
        right_child = self.__build_tree__(dataset_right[0], dataset_right[1])
        return DecisionNode(left_child, right_child)


    def __should_create_leaf_node__(self, feature_vectors, labels):
        if len(set(labels)) == 1:
            return True
        if len(feature_vectors.drop_duplicates()) == 1:
            return True
        if len(labels) < self.min_samples_split:
            return True
        return False

    def __create_leaf_node__(self, labels):
        return mode(labels)

    def __find_best_split_parameters__(self, feature_vectors, labels):
        max_gain = float('-inf')
        number_of_classes = len(set(labels))
        for column in feature_vectors:
            counts_r = dict(Counter(labels))
            counts_l = dict.fromkeys(counts_r, 0)
            coordinates = self.__compute_coordinates__(
                feature_vectors[column], labels
            )

    def __split_dataset__(self, feature_vectors, labels, split_params):
        pass

    def __compute_coordinates__(self, feature_vector, labels):
        pass

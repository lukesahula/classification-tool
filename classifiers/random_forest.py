import numpy as np
from joblib import Parallel, delayed
from classifiers.decision_tree import DecisionTree
from scipy.stats import mode
import multiprocessing
import pandas as pd


class RandomForest():
    def __init__(
        self, max_features, min_samples_split, random_state, n_jobs,
        n_estimators
    ):
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = np.random.RandomState(random_state)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.trees = None
        self.feature_matrix = None
        self.labels = None

    def init_tree(self, seed):
        indices = self.sample_dataset(seed)
        tree = DecisionTree(self.max_features, self.min_samples_split, seed)
        tree.fit(self.feature_matrix.loc[indices], self.labels[indices])
        return tree

    def fit_otfi(self, feature_matrix, labels):
        self.feature_matrix = feature_matrix
        self.labels = labels
        random_seeds = self.random_state.randint(0, 10000, self.n_estimators)
        self.trees = Parallel(n_jobs=self.n_jobs)(
             delayed(self.init_tree)(seed)
             for seed in random_seeds
        )

    def fit(self, feature_matrix, labels):
        self.feature_matrix = feature_matrix
        self.labels = labels
        random_seeds = self.random_state.randint(0, 10000, self.n_estimators)
        self.trees = Parallel(n_jobs=self.n_jobs)(
             delayed(self.init_tree)(seed)
             for seed in random_seeds
        )

    def mode(self, predictions):
        values, counts = np.unique(predictions, return_counts=True)
        m = counts.argmax()
        return values[m]

    def predict(self, observations, parallel):
        ind_predictions = parallel(
            delayed(tree.predict)(
                observations,
            )
            for tree in self.trees
        )
        return [self.mode(ind_pred) for ind_pred in zip(*ind_predictions)]

    def predict_otfi(self, observations, parallel):
        for column in observations.columns:
            observations[column] = observations[column].apply(
                lambda x: np.where(pd.isnull(x), self.feature_matrix[column].sample(replace=True), x)
            )
        self.predict(observations, parallel)

    def sample_dataset(self, seed):
        random_state = np.random.RandomState(seed)
        indices = random_state.choice(
            self.feature_matrix.index.values, len(self.feature_matrix)
        )
        return indices

from sklearn import datasets
import os
import glob
import pandas as pd
import numpy as np
import csv

class ClassificationTool():

    def __init__(self, classifier):

        self.classifier = classifier

    def load_dataset(self, path, samples=None):

        pos = os.path.join(path, 'pos', 'pos_test')
        neg = os.path.join(path, 'neg_seed_01_samples_' + samples, 'sampled')

        data = pd.DataFrame()
        data = pd.concat((data, pd.read_csv(pos, sep='\t', header=None)))
        data = pd.concat((data, pd.read_csv(neg, sep='\t', header=None)))

        data.replace(
            to_replace=np.nan,
            value=-1000000,
            inplace=True
        )

        return (data[data.columns[4:]], data[data.columns[3]])

    def train_classifier(self, tr_path, samples):
        self.tr_data = self.load_dataset(tr_path, samples)
        self.classifier.fit(self.tr_data[0], self.tr_data[1])
        self.tr_data = None

    def save_predictions(self, t_path, samples, output_file):

        def chunks(dataframe, n):
            """
            Yield successive n-sized chunks from dataframe.
            """
            for i in range(0, len(dataframe), n):
                yield dataframe[i:i + n]


        self.t_data = self.load_dataset(t_path, samples)

        with open(output_file, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=';')

            preds = []
            for chunk in chunks(self.t_data[0].index, 1000):
                to_predict = self.t_data[0].ix[chunk]
                preds = preds + list(self.classifier.predict(to_predict))

            writer.writerows(zip(
                    self.t_data[1],
                    preds
                )
            )

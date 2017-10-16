from sklearn import datasets
import os
import glob
import pandas as pd
import numpy as np
import csv

class ClassificationTool():

    def __init__(self, classifier, tr_path, t_path, zipped=False):

        self.classifier = classifier
        self.tr_path = tr_path
        self.t_path = t_path
        self.zipped = zipped


    def load_dataset(self, path, zipped=False):

        if not zipped:
            data = datasets.load_svmlight_file(path)
            return (pd.DataFrame(data[0].toarray()), data[1])
        else:
            files = glob.glob(os.path.join(path, 'neg', '*.gz'))
            files += glob.glob(os.path.join(path, 'pos', '*.gz'))

            data = pd.concat(
                pd.read_csv(f, sep='\t', header=None) for f in files
            )

            data.replace(
                to_replace=np.nan,
                value=-1000000,
                inplace=True
            )

            return (data[data.columns[4:]], data[data.columns[3]])

    def train_classifier(self):
        self.tr_data = self.load_dataset(self.tr_path, self.zipped)
        self.classifier.fit(self.tr_data[0], self.tr_data[1])
        self.tr_data = None

    def save_predictions(self, output_file):

        def chunks(dataframe, n):
            """
            Yield successive n-sized chunks from dataframe.
            """
            for i in range(0, len(dataframe), n):
                yield dataframe[i:i + n]


        self.t_data = self.load_dataset(self.t_path, zipped=self.zipped)

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

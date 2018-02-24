import csv


class ClassificationTool():
    def __init__(self, classifier):
        self.classifier = classifier

    def train_classifier(self, tr_data):
        """
        Trains the classifier.
        :param tr_data: Training data in a tuple (features, labels)
        Features are a pd.DataFrame, labels a list
        """
        self.classifier.fit(tr_data[0], tr_data[1])

    def save_predictions(
        self, t_data, output_file, parallel=None, metadata=True, legit=0
    ):
        """
        Predicts labels on testing data and writes it to csv file.
        :param t_data: Testing data in a tuple (features, labels, metadata)
        Features are a pd.DataFrame, labels a list, metadata a pd.DataFrame
        :param output_file: Path to output file.
        """
        preds = (self.classifier.predict(t_data[0], parallel) if parallel
                 else list(self.classifier.predict(t_data[0])))

        # Get indexes of True Negatives.
        if legit is not None:
            t_n_indexes = [i for i in t_data[1].index
                           if t_data[1][i] == preds[i]
                           and preds[i] == legit]

            # Remove True Negatives from preds and trues.
            t_n_indexes.reverse()
            [preds.pop(i) for i in t_n_indexes]
            t_data[1].drop(t_data[1].index[t_n_indexes], inplace=True)
            if metadata:
                t_data[2].drop(t_data[2].index[t_n_indexes], inplace=True)

        with open(output_file, 'a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=';')

            if metadata:
                writer.writerows(
                    zip(
                        t_data[1],
                        preds,
                        t_data[2][0],
                        t_data[2][1],
                        t_data[2][2]
                    )
                )
            else:
                writer.writerows(
                    zip(
                        t_data[1],
                        preds,
                    )
                )

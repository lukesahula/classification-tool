class EvaluationTool():

    def __init__(self):
        self.stats = {}

    def compute_stats(self, predicted, real):
        '''
        Takes predicted and real labels and increments
        TPs, FPs, FNs correspondingly.
        :param predicted: Predicted label
        :param real: Real label
        '''
        if predicted not in self.stats:
            self.stats[predicted] = {}
            self.stats[predicted]['TP'] = 0
            self.stats[predicted]['FP'] = 0
            self.stats[predicted]['FN'] = 0

        if real not in self.stats:
            self.stats[real] = {}
            self.stats[real]['TP'] = 0
            self.stats[real]['FP'] = 0
            self.stats[real]['FN'] = 0

        if predicted == real:
            self.stats[predicted]['TP'] += 1
        else:
            self.stats[predicted]['FP'] += 1
            self.stats[real]['FN'] += 1

    def compute_precision(self, class_label):
        '''
        Computes precision for the given class label.
        :param class_label: Class label of the row.
        :return: Computed precision of the classifier for the given class.
        '''
        true_positive = self.stats[class_label]['TP']
        false_positive = self.stats[class_label]['FP']

        if true_positive == false_positive == 0:
            return 'Undefined'

        precision = true_positive / (true_positive + false_positive)
        return precision

    def compute_recall(self, class_label):
        '''
        Computes recall for the given class label.
        :param class_label: Class label of the row.
        :return: Computed recall of the classifier for the given class.
        '''
        true_positive = self.stats[class_label]['TP']
        false_negative = self.stats[class_label]['FN']

        if true_positive == false_negative == 0:
            return 'Undefined'

        recall = true_positive / (true_positive + false_negative)
        return recall

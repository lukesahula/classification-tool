import os

from classification_tool.classification_tool import ClassificationTool
from evaluation_tool.evaluation_tool import EvaluationTool
from sklearn.ensemble import RandomForestClassifier as RFC

class LetterRunner():

    def execute_run(self):
        letter_training = 'classification_tool/datasets/letter.scale.tr'
        letter_testing = 'classification_tool/datasets/letter.scale.t'

        rfc = RFC(n_estimators=100, criterion="entropy", n_jobs=-1)
        clas_tool = ClassificationTool()
        clas_tool.training_data = clas_tool.load_dataset(letter_training)
        clas_tool.train_classifier(rfc)
        clas_tool.testing_data = clas_tool.load_dataset(letter_testing)

        predictions_output = 'classification_tool/outputs/rfc.letter'

        clas_tool.save_predictions(predictions_output)

        eval_tool = EvaluationTool(predictions_output, ';')
        print("Average precision: %f" %(eval_tool.get_avg_precision()))
        print("Average recall: %f" %(eval_tool.get_avg_recall()))

        print("\nPrecisions per label:")
        for label in eval_tool.labels:
            print("Label: %.1f, precision: %f"
                  %(label, eval_tool.compute_precision(label)))

        print("\nRecalls per label:")
        for label in eval_tool.labels:
            print("Label: %.1f, recall: %f"
                  %(label, eval_tool.compute_recall(label)))

runner = LetterRunner()
runner.execute_run()

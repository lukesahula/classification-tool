from utils.utils import tee
from loading_tool.loading_tool import LoadingTool
from classification_tool.classification_tool import ClassificationTool
from evaluation_tool.evaluation_tool import EvaluationTool
from sklearn.ensemble import RandomForestClassifier as RFC


class CiscoRunner():

    def execute_run(self):
        cisco_training = (
            'classification_tool/datasets/cisco_datasets/data/test_tr'
        )
        cisco_testing = (
            'classification_tool/datasets/cisco_datasets/data/test_t'
        )

        eval_output = 'evaluation_tool/outputs/rfc.cisco'

        n_estimators = 100
        max_features = 'sqrt'
        min_samples_split = 2
        criterion = 'entropy'
        n_jobs = -1
        random_state = 0

        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 10000,
            'bin_samples': 5000,
            'seed': random_state
        }

        with open(eval_output, 'w', encoding='utf-8') as f:
            tee('Running cisco runner with the following configuration:\n', f)
            tee('Classifier: {}'.format(str(RFC)), f)
            tee('Number of trees in the forest: {}'
                .format(str(n_estimators)), f)
            tee('Max number of features for a split: {}'
                .format(str(max_features)), f)
            tee('Min number of samples for a split: {}'
                .format(str(min_samples_split)), f)
            tee('Criterion: {}'.format(criterion), f)
            tee('Number of jobs: {}'.format(str(n_jobs)), f)
            tee('Number of negative samples: {}'
                .format(str(sampling_settings['neg_samples'])), f)
            tee('Number of samples for quantization: {}'
                .format(str(sampling_settings['bin_samples'])), f)
            tee('Bin count: {}'
                .format(str(sampling_settings['bin_count'])), f)
            tee('Seed: {}\n'.format(random_state), f)


        rfc = RFC(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            criterion=criterion,
            n_jobs=n_jobs,
            random_state=random_state
        )
        loading_tool = LoadingTool(sampling_settings)
        clas_tool = ClassificationTool(rfc, loading_tool)
        clas_tool.train_classifier(cisco_training)
        predictions_output = 'classification_tool/outputs/rfc.cisco'
        clas_tool.save_predictions(
            cisco_testing,
            predictions_output,
        )

        eval_tool = EvaluationTool(predictions_output, ';')
        stats = eval_tool.compute_stats()
        with open(eval_output, 'a', encoding='utf-8') as f:
            tee('Average precision: {}'
                .format(eval_tool.get_avg_precision(stats)), f)
            tee('Average recall: {}'
                .format(eval_tool.get_avg_recall(stats)), f)
            tee('\nPrecisions per label:', f)

            for label in eval_tool.labels:
                tee('Label: %.1f, precision: %f'
                    %(label, eval_tool.compute_precision(label, stats)), f)

            tee('\nRecalls per label:', f)
            for label in eval_tool.labels:
                tee('Label: %.1f, recall: %f'
                    %(label, eval_tool.compute_recall(label, stats)), f)

runner = CiscoRunner()
runner.execute_run()

import os

import datetime

from utils.utils import tee
from loading_tool.loading_tool import LoadingTool
from classification_tool.classification_tool import ClassificationTool
from evaluation_tool.evaluation_tool import EvaluationTool
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.externals import joblib
from collections import defaultdict


class SerializableClassifier():
    def __init__(self, classifier, bins=None):
        self.classifier = classifier
        self.bins = bins


class CiscoRunner():

    def execute_run(self, clas_path=None, agg_by=None, relaxed=False):
        tr_path = (
            'classification_tool/datasets/cisco_datasets/data/test_tr'
        )
        t_path = (
            'classification_tool/datasets/cisco_datasets/data/test_t'
        )

        output_dir = datetime.datetime.now().isoformat()
        output_dir = os.path.join('runner_outputs', output_dir)
        os.makedirs(output_dir)
        eval_output = os.path.join(output_dir, 'eval')
        predictions_output = os.path.join(output_dir, 'clas')
        clsfr_output = os.path.join(output_dir, 'clsfr')

        # Configuration for the RFC
        n_estimators = 100
        max_features = 'sqrt'
        min_samples_split = 2
        criterion = 'entropy'
        n_jobs = -1
        random_state = 0

        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 100000,
            'bin_samples': 50000,
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
            tee('Seed: {}'.format(random_state), f)
            tee('Aggregated by: {}'.format(agg_by), f)
            tee('Relaxed: {}\n'.format(relaxed), f)


        if not clas_path:
            rfc = RFC(
                n_estimators=n_estimators,
                max_features=max_features,
                min_samples_split=min_samples_split,
                criterion=criterion,
                n_jobs=n_jobs,
                random_state=random_state
            )
            loading_tool = LoadingTool(sampling_settings)
            clas_tool = ClassificationTool(rfc)
            tr_data = loading_tool.load_training_data(tr_path)
            tr_data = loading_tool.quantize_data(tr_data)
            clas_tool.train_classifier(tr_data)
            tr_data = None
        else:
            ser_classifier = joblib.load(clas_path)
            loading_tool = LoadingTool(sampling_settings, ser_classifier.bins)
            clas_tool = ClassificationTool(ser_classifier.classifier)

        if os.path.isfile(predictions_output):
            os.remove(predictions_output)

        for t_data in loading_tool.load_testing_data(t_path):
            t_data = loading_tool.quantize_data(t_data)
            clas_tool.save_predictions(
                t_data,
                predictions_output,
            )
        t_data = None

        eval_tool = EvaluationTool(legit=0, agg=True)

        if agg_by:
            stats = defaultdict(lambda: defaultdict(set))
        else:
            stats = defaultdict(lambda: defaultdict(int))

        for chunk in loading_tool.load_classifications(
                predictions_output, ';', True):
            if agg_by:
                chunk_stats = eval_tool.compute_stats_for_agg(
                    agg_by, chunk, relaxed)
                for k, v in chunk_stats.items():
                    stats[k]['FP'] = stats[k]['FP'] | v['FP']
                    stats[k]['FN'] = stats[k]['FN'] | v['FN']
                    stats[k]['TP'] = stats[k]['TP'] | v['TP']
            else:
                chunk_stats = eval_tool.compute_stats(chunk)
                for label in chunk_stats:
                    stats[label]['FP'] += chunk_stats[label]['FP']
                    stats[label]['FN'] += chunk_stats[label]['FN']
                    stats[label]['TP'] += chunk_stats[label]['TP']
        if agg_by:
            stats = eval_tool.aggregate_stats(stats)

        counts = eval_tool.get_stats_counts(eval_tool.labels, stats)
        with open(eval_output, 'a', encoding='utf-8') as f:
            tee('Avg Stats:\n', f)
            tee('Average precision w/o legit & NaN: {}'
                .format(eval_tool.get_avg_precision(stats, False, False)), f)
            tee('Average precision w/o legit: {}'
                .format(eval_tool.get_avg_precision(stats, False, True)), f)
            tee('Average recall w/o legit: {}'
                .format(eval_tool.get_avg_recall(stats, False, True)), f)
            tee('\nAverage precision w/o NaN: {}'
                .format(eval_tool.get_avg_precision(stats, True, False)), f)
            tee('Average overall precision: {}'
                .format(eval_tool.get_avg_precision(stats)), f)
            tee('Average overall recall: {}'
                .format(eval_tool.get_avg_recall(stats)), f)
            tee('\nOverall stats:\n', f)
            tee('TPS: {}'.format(counts['TP']), f)
            tee('FPS: {}'.format(counts['FP']), f)
            tee('FNS: {}'.format(counts['FN']), f)

            tee('Individual stats:\n', f)
            tee('label\tprecis\trecall\ttps\tfps\tfns', f)
            for label in eval_tool.labels:
                counts = eval_tool.get_stats_counts(label, stats)
                tee('%3.0f\t%4.3f\t%4.3f %6.0f\t%6.0f\t%6.0f'
                    %(label, eval_tool.compute_precision(label, stats),
                      eval_tool.compute_recall(label, stats),
                      counts['TP'], counts['FP'], counts['FN']), f)

        ser_classifier = SerializableClassifier(
            clas_tool.classifier,
            loading_tool.bins
        )

        joblib.dump(ser_classifier, clsfr_output, compress=9)

runner = CiscoRunner()
#runner.execute_run(clas_path='classification_tool/outputs/rfc.cisco.clsfr')
runner.execute_run(agg_by='user', relaxed=False)

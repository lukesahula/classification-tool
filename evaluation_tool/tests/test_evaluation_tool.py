from unittest import TestCase

import pandas as pd

from evaluation_tool import EvaluationTool


class TestEvaluationTool(TestCase):

    def test_compute_stats_minimal(self):
        dataset = [
            [0, 1],
            [1, 1],
            [2, 2],
            [3, 0]
        ]

        evaluation_tool = EvaluationTool()

        for row in dataset:
            evaluation_tool.compute_stats(row[0], row[1])

        self.assertEqual(evaluation_tool.stats[0]['FP'], 1)
        self.assertEqual(evaluation_tool.stats[0]['FN'], 1)
        self.assertEqual(evaluation_tool.stats[0]['TP'], 0)

        self.assertEqual(evaluation_tool.stats[1]['FP'], 0)
        self.assertEqual(evaluation_tool.stats[1]['FN'], 1)
        self.assertEqual(evaluation_tool.stats[1]['TP'], 1)

        self.assertEqual(evaluation_tool.stats[2]['FP'], 0)
        self.assertEqual(evaluation_tool.stats[2]['FN'], 0)
        self.assertEqual(evaluation_tool.stats[2]['TP'], 1)

        self.assertEqual(evaluation_tool.stats[3]['FP'], 1)
        self.assertEqual(evaluation_tool.stats[3]['FN'], 0)
        self.assertEqual(evaluation_tool.stats[3]['TP'], 0)

    def test_compute_precision_minimal(self):
        dataset = [
            [0, 1],
            [1, 1],
            [2, 2],
            [3, 0]
        ]

        evaluation_tool = EvaluationTool()

        for row in dataset:
            evaluation_tool.compute_stats(row[0], row[1])

        self.assertEqual(evaluation_tool.compute_precision(0), 0)
        self.assertEqual(evaluation_tool.compute_precision(1), 1)
        self.assertEqual(evaluation_tool.compute_precision(2), 1)
        self.assertEqual(evaluation_tool.compute_precision(3), 0)

    def test_compute_recall_minimal(self):
        dataset = [
            [0, 1],
            [1, 1],
            [2, 2],
            [3, 0]
        ]

        evaluation_tool = EvaluationTool()

        for row in dataset:
            evaluation_tool.compute_stats(row[0], row[1])

        self.assertEqual(evaluation_tool.compute_recall(0), 0)
        self.assertEqual(evaluation_tool.compute_recall(1), 0.5)
        self.assertEqual(evaluation_tool.compute_recall(2), 1)
        self.assertRaises(ZeroDivisionError, evaluation_tool.compute_recall(3))

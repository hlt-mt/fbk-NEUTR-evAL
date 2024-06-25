# Copyright 2023 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import unittest

from sklearn.metrics import classification_report
from fbk_neutreval.metrics import ClassBasedF1Metric, AccuracyMetric


class TestMetrics(unittest.TestCase):
    def test_f1_metric_two_classes(self):
        self._compare_ours_f1_vs_sklearn([0, 0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1, 0], [0, 1])

    def test_zero_f1(self):
        self._compare_ours_f1_vs_sklearn([0, 0, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0], [0, 1])

    def test_three_labels(self):
        self._compare_ours_f1_vs_sklearn([0, 1, 2, 0, 1, 2, 2], [1, 1, 0, 0, 0, 0, 0], [0, 1, 2])

    def test_empty(self):
        f1_metric = ClassBasedF1Metric()
        acc_metric = AccuracyMetric()
        self.assertDictEqual(f1_metric.overall_result, {})
        self.assertDictEqual(acc_metric.overall_result, {"accuracy": 0.0})

    def test_accuracy(self):
        acc_metric = AccuracyMetric()
        for label, pred in zip([0, 0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1, 0]):
            acc_metric.add_element(pred, label)
        self.assertAlmostEqual(acc_metric.overall_result["accuracy"], 4 / 7)

    def _compare_ours_f1_vs_sklearn(self, labels, preds, classes):
        true_report = classification_report(
            y_true=labels,
            y_pred=preds,
            labels=classes,
            output_dict=True)
        f1_metric = ClassBasedF1Metric()
        for i in range(len(labels)):
            f1_metric.add_element(preds[i], labels[i])
        cand_report = f1_metric.overall_result
        true_report = {c: true_report[str(c)] for c in classes}
        for c in true_report:
            del true_report[c]['support']
            for m in true_report[c]:
                self.assertAlmostEqual(true_report[c][m], cand_report[c][m], places=15)


if __name__ == '__main__':
    unittest.main()

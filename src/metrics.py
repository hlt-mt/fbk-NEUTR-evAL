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
from typing import Dict

REGISTERED_METRICS = {}


def register_metric(name):
    def do_register_metric(cls):
        REGISTERED_METRICS[name] = cls
        return cls

    return do_register_metric


class Metric:
    """
    Base class to extend to implement metrics that can be reported in the generate.
    """
    def add_element(self, prediction: int, label: int) -> None:
        ...

    @property
    def overall_result(self) -> Dict:
        return {}

    def pretty_print(self):
        ...


@register_metric("accuracy")
class AccuracyMetric(Metric):
    """
    Computes basic accuracy across all categories.
    """
    def __init__(self):
        self.correct_count = 0
        self.wrong_count = 0

    def add_element(self, prediction: int, label: int) -> None:
        if prediction == label:
            self.correct_count += 1
        else:
            self.wrong_count += 1

    @property
    def overall_result(self) -> Dict:
        total = self.correct_count + self.wrong_count
        if total > 0:
            accuracy = self.correct_count / (self.correct_count + self.wrong_count)
        else:
            accuracy = 0.0
        return {"accuracy": accuracy}

    def pretty_print(self):
        print(f"Accuracy : {(self.overall_result['accuracy'] * 100):.2f}")


@register_metric("class_f1")
class ClassBasedF1Metric(Metric):
    """
    Computes recall, precision, and F1 for each class.
    """
    def __init__(self):
        self.classes = {}

    def _ensure_class_present(self, cls):
        if cls not in self.classes:
            self.classes[cls] = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}

    def add_element(self, prediction: int, label: int) -> None:
        self._ensure_class_present(prediction)
        if prediction == label:
            self.classes[prediction]["true_positives"] += 1
        else:
            self.classes[prediction]["false_positives"] += 1
            self._ensure_class_present(label)
            self.classes[label]["false_negatives"] += 1

    def _get_class_scores(self, cls):
        if cls not in self.classes:
            return {}
        num_predicted = self.classes[cls]["true_positives"] + self.classes[cls]["false_positives"]
        if num_predicted == 0:
            precision = 0.
        else:
            precision = self.classes[cls]["true_positives"] / num_predicted
        num_to_predict = self.classes[cls]["true_positives"] + self.classes[cls]["false_negatives"]
        if num_to_predict == 0:
            recall = 0.
        else:
            recall = self.classes[cls]["true_positives"] / (
                    self.classes[cls]["true_positives"] + self.classes[cls]["false_negatives"])
        prec_plus_recall = precision + recall
        if prec_plus_recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / prec_plus_recall
        return {"precision": precision, "recall": recall, "f1-score": f1}

    @property
    def overall_result(self) -> Dict:
        return {cls: self._get_class_scores(cls) for cls in self.classes}

    def pretty_print(self):
        result = self.overall_result
        for cls in sorted(result):
            metrics_string = [
                f"{metric}: {result[cls][metric]:.2f}"
                for metric in ["f1-score", "precision", "recall"]]
            print(f"Class {cls}: {' ; '.join(metrics_string)}")

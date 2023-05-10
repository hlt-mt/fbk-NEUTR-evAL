#!/usr/bin/env python3
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
import argparse
import csv
from pathlib import Path

from sklearn.metrics import classification_report
from torch import tensor

from src.data_preprocessing import load_data, prepare_data
from src.generator import BertGenerator


def save_output(data_file: Path, save_file: Path, outputs: tensor) -> None:
    """Saves outputs together with input sentences (and labels if available) in a tsv file"""
    sentences, labels = load_data(data_file)
    assert len(sentences) == outputs.size()[0], "Number of sentences and of predicted classes does not match"
    if labels:
        header = ["SENTENCE", "LABEL", "OUTPUT"]
        rows = [
            {"SENTENCE": sentences[i], "LABEL": labels[i], "OUTPUT": outputs[i].item()} for i in range(len(sentences))]
    else:
        header = ["SENTENCE", "OUTPUT"]
        rows = [{"SENTENCE": sentences[i], "OUTPUT": outputs[i].item()} for i in range(len(sentences))]
    with open(save_file, "w") as t_f:
        writer = csv.DictWriter(t_f, fieldnames=header, delimiter="\t")
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        required=True,
        type=Path,
        help="Path of tsv file containing the data.")
    parser.add_argument(
        "--save-file",
        required=True,
        type=Path,
        help="Path of the tsv file where outputs will be saved")
    parser.add_argument("--model", required=True, type=str, help="Name of the Bert-based model.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path of the fine-tuned model.")
    parser.add_argument("--num-classes", required=True, type=str, help="Number of classes for the classification task.")
    parser.add_argument("--batch-size", required=False, type=int, default=16)
    parser.add_argument("--evaluation", required=False, type=bool, default=False)

    args = parser.parse_args()

    dataloader = prepare_data(args.data_file, args.model, shuffle=False)
    generator = BertGenerator(args.checkpoint, args.num_classes, dataloader)
    outputs = generator.generate()
    save_output(args.data_file, args.save_file, outputs)

    if args.evaluation:
        _, labels = load_data(args.data_file)
        assert len(labels) > 0, "Evaluation is set, but labels are not available"

        accuracy = sum([pred.item() != label for pred, label in zip(outputs, labels)]) / len(labels)
        print("OVERALL ACCURACY: ", accuracy)

        report = classification_report(
            y_true=labels,
            y_pred=outputs,
            labels=[i for i in range(len(args.num_classes))],
            output_dict=True)
        for label in report:
            print(
                f"Class {label}\t"
                f"PRECISION: {report[label]['precision']}\t"
                f"RECALL: {report[label]['recall']}\t"
                f"F1: {report[label]['f1']}")



if __name__ == "__main__":
    main()

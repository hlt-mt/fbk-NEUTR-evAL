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
from pathlib import Path
from typing import List

from torch.utils.data import DataLoader

from src.data_preprocessing import BertPreprocessor
from src.generator import BertGenerator
from src.metrics import REGISTERED_METRICS, Metric
from src.writers import REGISTERED_WRITERS, FileWriter


def generate(
        generator: BertGenerator,
        dataloader: DataLoader,
        preprocessor: BertPreprocessor,
        writer: FileWriter = None,
        metrics: List[Metric] = []) -> None:
    for batch in dataloader:
        output = generator.generate(batch)
        preds = output[0]
        probs = output[1]
        labels = output[2] if len(output) > 2 else None
        if writer is not None:
            sentences = preprocessor.decode(batch[0])
            if labels:
                for sent, pred, prob, label in zip(sentences, preds, probs, labels):
                    writer.write_line({
                        "SENTENCE": sent, "LABEL": label, "PREDICTION": pred, "PROBABILITY": prob})
            else:
                for sent, pred, prob in zip(sentences, preds, probs):
                    writer.write_line({
                        "SENTENCE": sent, "PREDICTION": pred, "PROBABILITY": prob})

        for metric in metrics:
            assert labels is not None, "Labels are required to compute metrics"
            for pred, prob, label in zip(preds, probs, labels):
                metric.add_element(pred, prob, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        required=True,
        type=Path,
        help="Path of tsv file containing the data.")
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Name of the Bert-based model.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path of the fine-tuned model.")
    parser.add_argument(
        "--num-classes",
        required=True,
        type=int,
        help="Number of classes for the classification task.")
    parser.add_argument("--batch-size", required=False, type=int, default=16)
    parser.add_argument("--max-seq-len", required=False, type=int, default=128)
    parser.add_argument("--lower-case", required=False, type=bool, default=False)
    parser.add_argument(
        "--save-file",
        required=False,
        type=Path,
        default=None,
        help="Path of the tsv file where outputs will be saved")
    parser.add_argument("--metrics", choices=REGISTERED_METRICS.keys(), nargs="+")
    parser.add_argument("--writer", choices=REGISTERED_WRITERS.keys(), default="tsv")

    args = parser.parse_args()
    preprocessor = BertPreprocessor(args.model, args.max_seq_len, args.lower_case)
    dataloader = preprocessor.prepare_data(
        tsv_file=args.data_file,
        shuffle=False,
        batch_size=args.batch_size)
    generator = BertGenerator(args.checkpoint, args.num_classes)
    metrics = [REGISTERED_METRICS[m]() for m in getattr(args, "metrics", [])]

    if args.save_file is None:
        generate(generator, dataloader, preprocessor, metrics=metrics)
    else:
        with REGISTERED_WRITERS[args.writer](args.save_file) as w:
            generate(generator, dataloader, preprocessor, writer=w, metrics=metrics)


if __name__ == "__main__":
    main()

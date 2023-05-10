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
import os
from pathlib import Path

from src.data_preprocessing import prepare_data
from src.model import BertForSequenceClassificationModel
from src.trainer import BertTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Name or path of the model to fine-tuned.")
    parser.add_argument("--num-classes", required=True, type=str, help="Number of classes in the classification task.")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--train", required=True, type=str)
    parser.add_argument("--validation", required=True, type=str)
    parser.add_argument("--save-dir", required=True, type=str)
    parser.add_argument("--num-epochs", required=True, type=int)
    parser.add_argument("--batch-size", required=False, type=int, default=16)
    parser.add_argument("--learning-rate", required=False, type=float, default=5e-5)
    parser.add_argument("--epsilon", required=False, type=float, default=1e-08)
    args = parser.parse_args()

    train_dataloader = prepare_data(
        Path(os.path.join(args.data_root, args.train)),
        args.model,
        args.batch_size)
    validation_dataloader = prepare_data(
        Path(os.path.join(args.data_root, args.validation)),
        args.model,
        args.batch_size)
    model = BertForSequenceClassificationModel(args.model, args.num_classes)
    trainer = BertTrainer(
        model,
        args.save_dir,
        train_dataloader,
        validation_dataloader,
        args.num_epochs,
        args.learning_rate,
        args.epsilon)
    trainer.train()


if __name__ == "__main__":
    main()
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
import os
import shutil
import unittest

import numpy as np
import torch
from torch import LongTensor
from torch.utils.data import TensorDataset, DataLoader

from src.model import SequenceClassificationModel
from src.trainer import BertTrainer


class TestBertTrainer(unittest.TestCase):
    def setUp(self) -> None:
        # Training data
        self.train_token_ids = LongTensor(np.array(
            [[101, 8795, 2050, 1041, 2474, 21111, 25312, 3366, 1010, 14255, 2226, 11192, 2050,
              4487, 18834, 18515, 6692, 4779, 3217, 19204, 2015, 1010, 21864, 102],
             [101, 8795, 2050, 1041, 2474, 2117, 2050, 25312, 3366, 1010, 15544, 4765, 2527,
              11265, 2072, 5787, 2072, 102, 0, 0, 0, 0, 0, 0],
             [101, 8795, 2050, 1041, 2474, 28774, 4143, 25312, 3366, 1010, 20704, 2527, 11687,
              4667, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [101, 8795, 2050, 1041, 2474, 24209, 8445, 2050, 25312, 3366, 1012, 102, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        self.train_attention_masks = LongTensor(np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        self.train_labels = torch.LongTensor(np.array([0, 0, 1, 1]))
        training_set = TensorDataset(
            self.train_token_ids,
            self.train_attention_masks,
            self.train_labels)
        self.training_dataloader = DataLoader(
            training_set,
            shuffle=True,
            batch_size=2)
        # Validation Data
        self.val_token_ids = LongTensor(np.array(
            [[101, 8795, 2050, 1041, 2474, 28774, 4143, 25312, 3366, 1010, 20704, 2527, 11687,
              4667, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [101, 8795, 2050, 1041, 2474, 24209, 8445, 2050, 25312, 3366, 1012, 102, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        self.val_attention_masks = LongTensor(np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        self.val_labels = torch.LongTensor(np.array([1, 1]))
        validation_set = TensorDataset(
            self.val_token_ids,
            self.val_attention_masks,
            self.val_labels)
        self.validation_dataloader = DataLoader(
            validation_set,
            shuffle=True,
            batch_size=2)

        # Loading models
        self.model_bert = SequenceClassificationModel(
            "bert-base-uncased",
            num_labels=2)  # Bert Model
        self.model_roberta = SequenceClassificationModel(
            "osiria/roberta-base-italian",
            num_labels=2)  # Roberta Model

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_path = os.path.join(root_dir, 'tests', "save_checkpoints")
        os.mkdir(self.save_path)

        # Trainer Initialization
        self.trainer_bert = BertTrainer(
            self.model_bert,
            self.save_path,
            self.training_dataloader,
            self.validation_dataloader,
            num_epochs=2)
        self.trainer_roberta = BertTrainer(
            self.model_roberta,
            self.save_path,
            self.training_dataloader,
            self.validation_dataloader,
            num_epochs=2)

    def tearDown(self) -> None:
        shutil.rmtree(self.save_path)

    # Verifying that during training weights are updated
    def test_bert_updating(self):
        self.trainer_bert.train()

        original_model = SequenceClassificationModel("bert-base-uncased", num_labels=2)
        state_dict_model = original_model.state_dict()
        ft_model1 = SequenceClassificationModel(
            os.path.join(self.save_path, "checkpoint_1"),
            num_labels=2)
        state_dict_model1 = ft_model1.state_dict()
        ft_model2 = SequenceClassificationModel(
            os.path.join(self.save_path, "checkpoint_2"),
            num_labels=2)
        state_dict_model2 = ft_model2.state_dict()

        self.assertFalse(torch.equal(
            state_dict_model["model.bert.encoder.layer.0.output.dense.bias"],
            state_dict_model1["model.bert.encoder.layer.0.output.dense.bias"]))
        self.assertFalse(torch.equal(
            state_dict_model["model.bert.encoder.layer.0.output.dense.bias"],
            state_dict_model2["model.bert.encoder.layer.0.output.dense.bias"]))
        self.assertFalse(torch.equal(
            state_dict_model1["model.bert.encoder.layer.0.output.dense.bias"],
            state_dict_model2["model.bert.encoder.layer.0.output.dense.bias"]))

    def test_roberta_updating(self):
        self.trainer_roberta.train()

        original_model = SequenceClassificationModel("osiria/roberta-base-italian", num_labels=2)
        state_dict_model = original_model.state_dict()
        ft_model1 = SequenceClassificationModel(
            os.path.join(self.save_path, "checkpoint_1"),
            num_labels=2)
        state_dict_model1 = ft_model1.state_dict()
        ft_model2 = SequenceClassificationModel(
            os.path.join(self.save_path, "checkpoint_2"),
            num_labels=2)
        state_dict_model2 = ft_model2.state_dict()

        self.assertFalse(torch.equal(
            state_dict_model["model.roberta.encoder.layer.0.output.dense.bias"],
            state_dict_model1["model.roberta.encoder.layer.0.output.dense.bias"]))
        self.assertFalse(torch.equal(
            state_dict_model["model.roberta.encoder.layer.0.output.dense.bias"],
            state_dict_model2["model.roberta.encoder.layer.0.output.dense.bias"]))
        self.assertFalse(torch.equal(
            state_dict_model1["model.roberta.encoder.layer.0.output.dense.bias"],
            state_dict_model2["model.roberta.encoder.layer.0.output.dense.bias"]))


if __name__ == '__main__':
    unittest.main()

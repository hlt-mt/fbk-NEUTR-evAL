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
import unittest

import numpy as np
import torch

from src.model import BertForSequenceClassificationModel


class TestBertPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        # Data
        self.data = [
            [101, 8795, 2050, 1041, 2474, 21111, 25312, 3366, 1010, 14255, 2226, 11192, 2050,
             4487, 18834, 18515, 6692, 4779, 3217, 19204, 2015, 1010, 21864, 102],
            [101, 8795, 2050, 1041, 2474, 2117, 2050, 25312, 3366, 1010, 15544, 4765, 2527,
             11265, 2072, 5787, 2072, 102, 0, 0, 0, 0, 0, 0],
            [101, 8795, 2050, 1041, 2474, 28774, 4143, 25312, 3366, 1010, 20704, 2527, 11687,
             4667, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [101, 8795, 2050, 1041, 2474, 24209, 8445, 2050, 25312, 3366, 1012, 102, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.token_ids = torch.tensor(np.array(self.data), dtype=torch.long)
        self.attention = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.attention_masks = torch.tensor(np.array(self.attention), dtype=torch.long)
        self.labels = [0, 0, 1, 1]
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
        # Model
        self.model = BertForSequenceClassificationModel("bert-base-uncased", num_labels=3)

    # Verifying whether forward works correctly (checking dimension of outputs)
    def test_forward(self):
        preds = self.model.forward(self.token_ids, self.attention_masks, self.labels)
        self.assertEqual(torch.Size([]), preds.loss.size())
        self.assertEqual(torch.Size([4,3]), preds.logits.size())

    # Verifying whether forward gives the same outputs for same sentences in different batches
    def test_consistency_forward(self):
        preds = self.model.forward(self.token_ids, self.attention_masks, self.labels)

        # New data
        data = [self.data[2], self.data[3], self.data[0], self.data[1]]
        new_token_ids = torch.tensor(np.array(data), dtype=torch.long)
        attention = [self.attention[2], self.attention[3], self.attention[0], self.attention[1]]
        new_attention_masks = torch.tensor(np.array(attention), dtype=torch.long)
        labels = [self.labels[2], self.labels[3], self.labels[0], self.labels[1]]
        new_labels = torch.tensor(np.array(labels), dtype=torch.long)

        new_preds = self.model.forward(new_token_ids, new_attention_masks, new_labels)

        self.assertTrue(torch.equal(new_preds.logits[0], preds.logits[2]))
        self.assertTrue(torch.equal(new_preds.logits[1], preds.logits[3]))
        self.assertTrue(torch.equal(new_preds.logits[2], preds.logits[0]))
        self.assertTrue(torch.equal(new_preds.logits[3], preds.logits[1]))
        self.assertEqual(new_preds.loss, preds.loss)


if __name__ == '__main__':
    unittest.main()

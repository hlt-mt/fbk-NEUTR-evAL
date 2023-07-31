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
from torch import LongTensor

from src.model import SequenceClassificationModel


class TestBertPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        # Data
        self.token_ids = LongTensor(np.array(
            [[101, 8795, 2050, 1041, 2474, 21111, 25312, 3366, 1010, 14255, 2226, 11192, 2050,
              4487, 18834, 18515, 6692, 4779, 3217, 19204, 2015, 1010, 21864, 102],
             [101, 8795, 2050, 1041, 2474, 2117, 2050, 25312, 3366, 1010, 15544, 4765, 2527,
              11265, 2072, 5787, 2072, 102, 0, 0, 0, 0, 0, 0],
             [101, 8795, 2050, 1041, 2474, 28774, 4143, 25312, 3366, 1010, 20704, 2527, 11687,
              4667, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [101, 8795, 2050, 1041, 2474, 24209, 8445, 2050, 25312, 3366, 1012, 102, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        self.attention_masks = LongTensor(np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        self.labels = torch.LongTensor(np.array([0, 0, 1, 1]))
        # Model
        self.model = SequenceClassificationModel("bert-base-uncased", num_labels=3)

    # Verifying whether forward works correctly (checking dimension of outputs)
    def test_forward(self):
        loss, probabilities = self.model.forward(
            self.token_ids,
            self.attention_masks,
            self.labels)
        self.assertEqual(torch.Size([]), loss.size())
        self.assertEqual(torch.Size([4, 3]), probabilities.size())

    # Verifying whether forward gives the same outputs for same sentences in different batches
    def test_forward_consistency(self):
        loss, probabilities = self.model.forward(
            self.token_ids,
            self.attention_masks,
            self.labels)

        # New data
        new_order = [2, 3, 0, 1]
        new_token_ids = self.token_ids[new_order]
        new_attention_masks = self.attention_masks[new_order]
        new_labels = self.labels[new_order]

        new_loss, new_probabilities = self.model.forward(
            new_token_ids,
            new_attention_masks,
            new_labels)

        self.assertTrue(torch.equal(new_probabilities[0], probabilities[2]))
        self.assertTrue(torch.equal(new_probabilities[1], probabilities[3]))
        self.assertTrue(torch.equal(new_probabilities[2], probabilities[0]))
        self.assertTrue(torch.equal(new_probabilities[3], probabilities[1]))
        self.assertAlmostEqual(new_loss.item(), loss.item(), delta=0.00001)

    # Verifying that the sum of the probabilities always returns 1.
    def test_probabilities(self):
        _, probabilities = self.model.forward(
            self.token_ids,
            self.attention_masks,
            self.labels)
        print(torch.sum(probabilities, dim=1).detach())
        self.assertTrue(np.allclose(
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.sum(probabilities.detach().numpy(), axis=1),
            atol=0.00001))


if __name__ == '__main__':
    unittest.main()

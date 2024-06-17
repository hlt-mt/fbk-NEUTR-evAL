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
from torch.utils.data import TensorDataset, DataLoader

from fbk_neutreval.generator import Generator


class TestBertGenerator(unittest.TestCase):
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
        data_set = TensorDataset(self.token_ids, self.attention_masks)
        self.dataloader = DataLoader(data_set, batch_size=2)
        # Training on gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create an instance of BertGenerator with a model hosted in HF,
        # not with a checkpoint's path as expected
        self.generator = Generator(
            "bert-base-uncased",
            num_labels=2)

    # Verifying that in multiple inference on the same data, the results are consistent
    def test_model_updating(self):
        for batch in self.dataloader:
            previous_output = None
            for i in range(5):
                output, _ = self.generator.generate(batch)
                if i > 0:
                    self.assertListEqual(previous_output, output)
                previous_output = output


if __name__ == '__main__':
    unittest.main()

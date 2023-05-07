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

import torch
from torch import Tensor

from src.data_preprocessing import BertPreprocessor


class TestBertPreprocessor(unittest.TestCase):
    def setUp(self):
        self.text = [
            "Questa è la prima frase, più lunga di ventiquattro tokens, quindi dovrebbe essere troncata",
            "Questa è la seconda frase, rientra nei limiti",
            "Questa è la terza frase, avrà padding",
            "Questa è la quarta frase."]
        self.labels = [0, 0, 1, 1]
        self.bert_preprocessor = BertPreprocessor("bert-base-uncased", max_seq_len=24)

    def test_preprocessing(self):
        token_ids, attention_masks, labels = self.bert_preprocessor.preprocessing(self.text, self.labels)

        self.assertIsInstance(token_ids, Tensor)
        self.assertIsInstance(attention_masks, Tensor)
        self.assertIsInstance(labels, Tensor)

        self.assertEqual(token_ids.size(), torch.Size([4, 24]))
        self.assertEqual(attention_masks.size(), torch.Size([4, 24]))
        self.assertEqual(labels.size(), torch.Size([4]))

        self.assertTrue(torch.equal(
            torch.all(attention_masks, dim=1),
            torch.tensor([True, False, False, False])))


if __name__ == '__main__':
    unittest.main()

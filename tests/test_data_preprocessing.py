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

from fbk_neutreval.data_preprocessing import Preprocessor


class TestBertPreprocessor(unittest.TestCase):
    def setUp(self):
        self.text = [
            "Questa è la prima frase, più lunga di ventiquattro tokens, "
            "quindi ci si aspetta che sia troncata",
            "Questa è la seconda frase, rientra nei limiti",
            "Questa è la terza frase, avrà padding",
            "Questa è la quarta frase."]
        self.bert_preprocessor = Preprocessor("bert-base-uncased", max_seq_len=24)
        self.roberta_preprocessor = Preprocessor("osiria/roberta-base-italian", max_seq_len=24)

# Verifying that in case that labels are not available, the labels tensor is not returned
    def test_no_labels(self):
        preprocessing_tuple = self.bert_preprocessor._preprocessing(self.text)
        self.assertEqual(2, len(preprocessing_tuple))

    # Verifying that _preprocessing() function returns the expected tensors for Bert tokenizer
    def test_bert_preprocessing(self):
        labels = [0, 0, 1, 1]

        token_ids, attention_masks, labels = self.bert_preprocessor._preprocessing(
            self.text,
            labels)

        self.assertIsInstance(token_ids, Tensor)
        self.assertIsInstance(attention_masks, Tensor)
        self.assertIsInstance(labels, Tensor)

        self.assertEqual(token_ids.size(), torch.Size([4, 24]))
        self.assertEqual(attention_masks.size(), torch.Size([4, 24]))
        self.assertEqual(labels.size(), torch.Size([4]))

        self.assertTrue(torch.equal(
            torch.all(attention_masks, dim=1),
            torch.tensor([True, False, False, False])))

    # Verifying that _preprocessing() function returns the expected tensors for Roberta tokenizer
    def test_roberta_preprocessing(self):
        labels = [0, 0, 1, 1]

        token_ids, attention_masks, labels = self.roberta_preprocessor._preprocessing(
            self.text,
            labels)

        self.assertIsInstance(token_ids, Tensor)
        self.assertIsInstance(attention_masks, Tensor)
        self.assertIsInstance(labels, Tensor)

        self.assertEqual(token_ids.size(), torch.Size([4, 24]))
        self.assertEqual(attention_masks.size(), torch.Size([4, 24]))
        self.assertEqual(labels.size(), torch.Size([4]))

        self.assertTrue(torch.equal(
            torch.all(attention_masks, dim=1),
            torch.tensor([True, False, False, False])))

    def test_decode(self):
        batch = [
            torch.tensor(
                [[101, 8795, 2050, 1041, 2474, 21111, 25312, 3366, 1010, 14255,
                  2226, 11192, 2050, 4487, 18834, 18515, 6692, 4779, 3217, 19204,
                  2015, 1010, 21864, 102],
                 [101, 8795, 2050, 1041, 2474, 2117, 2050, 25312, 3366, 1010,
                  15544, 4765, 2527, 11265, 2072, 5787, 2072, 102, 0, 0,
                  0, 0, 0, 0]]),
            torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])]
        testo = Preprocessor("bert-large-uncased").decode(batch[0])
        self.assertListEqual(
            ['questa e la prima frase, piu lunga di ventiquattro tokens, qui',
             'questa e la seconda frase, rientra nei limiti'],
            testo)


if __name__ == '__main__':
    unittest.main()

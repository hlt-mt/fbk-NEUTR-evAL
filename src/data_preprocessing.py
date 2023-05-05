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
import csv
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from transformers import BertTokenizer


class BertPreprocessor:
    def __init__(self, model: str, max_seq_len: int = 32):
        self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=True)
        self.max_seq_len = max_seq_len

    def preprocessing(self, texts, labels):
        """
        <class transformers.tokenization_utils_base.BatchEncoding> is used.
        Returns:
          - token_id (Tensor)
          - attention_mask: Tensor of indices (0,1) specifying which tokens should considered by the model
          (return_attention_mask = True)
          - labels (Tensor)
        """
        token_ids = []
        attention_masks = []
        for sent in texts:
            encoding_dict = self.tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt')
            token_ids.append(encoding_dict['input_ids'])
            attention_masks.append(encoding_dict['attention_mask'])
        token_ids = torch.cat(token_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        return token_ids, attention_masks, labels


def load_data(tsv_file: str) -> Tuple[List[str], List[int]]:
    sentences = []
    labels = []
    with open(tsv_file, "r") as t_f:
        tsv_reader = csv.DictReader(t_f, delimiter='\t')
        for line in tsv_reader:
            sentences.append(line['SENTENCE'])
            labels.append(int(line['LABEL']))
    return sentences, labels


def data_preparation(tsv_file: str, model: str, batch_size=16) -> DataLoader:
    sentences, labels = load_data(tsv_file)
    preprocessor = BertPreprocessor(model)
    test_token_ids, test_attention_masks, test_labels = preprocessor.preprocessing(sentences, labels)
    test_set = TensorDataset(test_token_ids, test_attention_masks, test_labels)
    return DataLoader(test_set, sampler=RandomSampler(test_set), batch_size=batch_size)

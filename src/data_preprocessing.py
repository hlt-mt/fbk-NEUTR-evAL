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
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import BertTokenizer


def load_data(tsv_file: Path) -> Tuple[List[str], List[int]]:
    """
    Loads the tsv file containing the sentences in the "SENTENCE" column
    (and the labels if available, in the "LABEL" column).
    It returns two lists, one for sentences, one for labels.
    If labels are not avaiblae, the second list is empty
    """
    sentences = []
    labels = []
    with open(tsv_file, "r") as t_f:
        tsv_reader = csv.DictReader(t_f, delimiter='\t')
        has_labels = 'LABEL' in tsv_reader.fieldnames
        for line in tsv_reader:
            sentences.append(line['SENTENCE'])
            if has_labels:
                labels.append(int(line['LABEL']))
    return sentences, labels


class BertPreprocessor:
    """
    The BertPreprocessor class is responsible for loading and preprocessing text data,
    tailored specifically to be used with Bert models.
    It builds on the BertTokenizer class provided by HuggingFace,
    and leverages the pretrained model's tokenizer stored in the HuggingFace archive.
    """
    def __init__(self, model: str, max_seq_len: int = 128, lower_case=False):
        self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=lower_case)
        self.max_seq_len = max_seq_len

    def _preprocessing(self, texts: List, labels: List = None):
        """
        <class transformers.tokenization_utils_base.BatchEncoding> is used
        to encode batches of text.
        It returns:
          - token_ids (Tensor): the tokens representing the text
          - attention_mask: Tensor of indices (0,1) specifying which tokens
          should considered by the model
          (return_attention_mask = True)
          - labels (Tensor)
        """
        encoding_dict = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        token_ids = encoding_dict['input_ids']
        attention_masks = encoding_dict['attention_mask']
        if labels:
            labels = torch.tensor(labels)
            return token_ids, attention_masks, labels
        else:
            return token_ids, attention_masks

    def prepare_data(self, tsv_file: Path, shuffle=True, batch_size=16) -> DataLoader:
        """Prepares data in batches and returns a dataloader object."""
        sentences, labels = load_data(tsv_file)
        data_set = TensorDataset(*self._preprocessing(sentences, labels))
        return DataLoader(data_set, shuffle=shuffle, batch_size=batch_size)

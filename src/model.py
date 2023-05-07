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
import torch

from transformers import BertForSequenceClassification


class BertForSequenceClassificationModel(torch.nn.Module):
    """
    BertForSequenceClassificationModel consists of a classifier model composed of a pretrained Bert-based
    encoder and a feed-forward layer.
    The pretrained model can be either a model from the HuggingFace archive or a checkpoint of Bert-based model built
    with HuggingFace.
    """
    def __init__(self, model, num_labels, output_attentions=False, output_hidden_states=False):
        super(BertForSequenceClassificationModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            model,  # either path or HF name (e.g, "bert-base-uncased")
            num_labels=num_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_mask,
            labels=labels)

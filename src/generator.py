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
from pathlib import Path

import torch
from torch import tensor
from torch.utils.data import DataLoader

from src.model import BertForSequenceClassificationModel


class BertGenerator:
    def __init__(self, model: Path, num_labels: int, dataloader: DataLoader):
        self.model = BertForSequenceClassificationModel(model, num_labels=num_labels)
        self.dataloader = dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Training on gpu

    def generate(self):
        outputs = tensor([])
        self.model.eval()
        for batch in self.dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                probabilities = self.model.forward(inputs["input_ids"], inputs["attention_mask"])
                predicted_label = torch.argmax(probabilities, dim=1)
                predicted_labels = predicted_label.cpu().detach()
                outputs = torch.cat((outputs, predicted_labels))
        return outputs

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

from fbk_neutreval.model import SequenceClassificationModel


class Generator:
    def __init__(self, model: Path, num_labels: int):
        self.model = SequenceClassificationModel(model, num_labels=num_labels)
        # Training on gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def generate(self, batch):
        self.model.eval()
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(self.device),
                "attention_mask": batch[1].to(self.device)}
            probs = self.model.forward(
                inputs["input_ids"],
                inputs["attention_mask"])
            predicted_labels = torch.argmax(probs, dim=1).cpu().detach().tolist()
            probabilities = []
            for row, index in zip(probs, predicted_labels):
                probabilities.append(row[index].item())
        if len(batch) > 2:
            return predicted_labels, probabilities, batch[2].cpu().tolist()
        else:
            return predicted_labels, probabilities

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
import logging
import os

import torch
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO")
LOGGER.setLevel(getattr(logging, log_level))


class BertTrainer:
    """Class to perform training of a text classifier built upon a pre-trained Bert model"""
    def __init__(
            self,
            model,
            save_path,
            train_dataloader,
            val_dataloader,
            num_epochs,
            lr=5e-5,
            eps=1e-08):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, eps=eps)
        self.num_epochs = num_epochs
        # Training on gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.save_path = save_path

    def train(self):
        total_updates = 0
        for epoch in range(self.num_epochs):
            LOGGER.info(f"Starting Epoch {epoch}")
            self.model.train()
            train_loss = 0
            num_samples = 0
            for batch in tqdm(self.train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
                self.optimizer.zero_grad()
                loss, probabilities = self.model(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"])
                loss.backward()
                self.optimizer.step()
                total_updates += 1
                train_loss += loss.item()
                num_samples += inputs["input_ids"].shape[0]
            train_loss /= num_samples

            self.model.eval()
            val_loss = 0
            num_samples = 0
            for batch in self.val_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[2]}
                    loss, probabilities = self.model(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"])
                    val_loss += loss.item()
                    num_samples += inputs["input_ids"].shape[0]
            val_loss /= num_samples

            LOGGER.info(
                f"Epoch {epoch + 1}, "
                f"NUmber updates: {total_updates}"
                f"Training loss: {train_loss:.3f}, "
                f"Validation loss: {val_loss:.3f}")

            save_path = os.path.join(self.save_path, f"checkpoint_{epoch + 1}")
            LOGGER.info(f"Saving checkpoint_{epoch + 1}")
            self.model.save_pretrained(save_path)

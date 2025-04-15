# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import pandas as pd
from pandas import Series
from typing import Dict, List, Optional, Union


class Prompt:
    """
    A class for few-shot prompts. It stores message templates for `system`, `user` and `assistant`
    role message, as well as the data to be used to populate the templates for few-shot prompting.
    Each user-assistant message pair represents an exemplar interaction with the model.
    It includes several methods to create prompts and to post-process models' outputs.
    """

    def __init__(self,
                 user_template: str = None,
                 assistant_template: str = None,
                 system_message: Optional[str] = None):
        self.system_message = system_message
        self.user_template = user_template
        self.assistant_template = assistant_template
        self._shots_data = []
        self._selected_shots = []

    @property
    def shots_data(self) -> List[Series]:
        if self._shots_data is None:
            raise ValueError('Shots data not set.')
        elif isinstance(self._shots_data, pd.DataFrame):
            return [shot for h, shot in self._shots_data.iterrows()]
        elif isinstance(self._shots_data, list):
            return self._shots_data

    @shots_data.setter
    def shots_data(self, data: Union[pd.DataFrame, List[Dict[str, str]]]) -> None:
        """Set the shots data to be used for prompt composition."""
        self._shots_data = data
        # Reset selected shots when new data is set
        self._selected_shots = None

    def load_tsv_shots_data(self, filename: str):
        """
        Load information to populate `user` and `assistant` role messages with.
        The column headers should match the placeholders to be replaced in the prompt templates.

        Parameters:
        * filename: path to the TSV file containing shots data.
        """
        with open(filename, 'r') as f:
            self._shots_data = pd.read_csv(f, sep='\t')

    def select_shots(self,
                     num_shots: Optional[int] = 0,
                     indices: Optional[List[int]] = None) -> None:
        """
        Select shots to be used in prompt composition.

        Parameters:
        * num_shots: Number of shots to select, starting from the beginning of `shots_data`.
        Required if indices not provided.
        * indices: Specific indices to select. If provided, num_shots is ignored.
        """

        if indices is not None:
            # Select specific shots
            self._selected_shots = [self.shots_data[i] for i in indices]
        elif num_shots is not None:
            # Select random shots
            available_shots = self.shots_data
            if num_shots > len(available_shots):
                raise ValueError(
                    f"Requested {num_shots} shots but only {len(available_shots)} available.")
            self._selected_shots = self.shots_data[:num_shots]
        else:
            raise ValueError("Either num_shots or indices must be provided.")

    def compose(self,
                prompt_input: Union[Dict[str, str], pd.DataFrame, pd.Series],
                system_message: bool = True) -> List[Dict[str, str]]:
        """
        Create the list of messages that make up the prompt.
        For few-shot prompting, use select_shots first.

        Parameters:
        * prompt_input: The input for the final user message.
        * system_message: Specify to include the system message.
        """

        prompt = []
        if system_message and self.system_message is not None:
            prompt.append({'role': 'system', 'content': self.system_message})

        # Add selected shots
        for shot in self._selected_shots:
            prompt.append({'role': 'user', 'content': self.user_template.format_map(shot)})
            prompt.append(
                {'role': 'assistant', 'content': self.assistant_template.format_map(shot)})

        # Add final input
        if isinstance(prompt_input, pd.DataFrame):
            prompt_input = prompt_input.to_dict('records')[0]

        prompt.append({'role': 'user', 'content': self.user_template.format_map(prompt_input)})
        return prompt

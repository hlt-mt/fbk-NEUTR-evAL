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
"""
This script is used to evaluate gender-neutrality with large language models.
"""
import argparse
import csv
import logging
import json
from pathlib import Path
from fbk_neutreval.cli.llm_as_a_judge.prompt import Prompt
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


def load_prompt_message(filename: str) -> str:
    """
    Loads messages or message templates for the prompts from plain text files and returns them as
    strings.
    """
    with open(filename, "r") as f:
        message = f.read()
        return message


def load_input(filename: str) -> List[Dict[str, str]]:
    """
    Loads an input file in TSV format and returns a list of dictionaries.
    """
    with open(filename) as f:
        return list(csv.DictReader(f, delimiter='\t'))


class Model:
    def __init__(self,
                 model_id: str,
                 schema_file: str):
        self.model_id = model_id
        self.schema_file = schema_file

    def generate(self, input_messages: List[Dict[str, str]]) -> Dict[str, Union[str, List]]:
        pass


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds common cli arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="TSV file containing input data.")
    parser.add_argument('-l', "--lang", type=str, required=True,
                        choices=['de', 'el', 'es', 'it'],
                        help="Target language to be evaluated (either 'de', 'es', 'el', or 'it')")
    parser.add_argument("-o", "--output", type=str, help="Output file", required=True)
    parser.add_argument("-p", "--prompt", default='cross-p_l', type=str,
                        choices=['mono-l', 'mono-p_l', 'cross-l', 'cross-p_l'],
                        help="Specify which prompt to use")
    parser.add_argument("-r", "--start_index", type=int, default=0,
                        help="Optional starting index")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Enable verbose mode.")
    return parser


def evaluate(model: Model,
             input_data_file: str,
             start_index: int,
             output_file: str,
             prompt_id: str,
             lang: str,
             verbose: bool):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s')

    resources_dir = Path(__file__).resolve().parent.parent.parent / "resources"

    system_message_file = f"{resources_dir}/prompts/{lang}/{prompt_id}"
    user_file = 'user_cross' if 'cross' in prompt_id else 'user_mono'
    user_message_template_file = f"{resources_dir}/prompts/{user_file}"
    system_message = load_prompt_message(system_message_file)
    user_message_template = load_prompt_message(user_message_template_file)
    prompt = Prompt(user_template=user_message_template,
                    assistant_template="{out}",
                    system_message=system_message)
    prompt.load_tsv_shots_data(f"{resources_dir}/shots/{lang}/{prompt_id}.tsv")

    input_data = load_input(input_data_file)

    if start_index > len(input_data):
        logger.error(f"Invalid start index {start_index}: input length {len(input_data)}")
        exit(1)

    for i, entry in enumerate(input_data[start_index:], start=start_index + 1):
        prompt.select_shots(num_shots=8)
        this_prompt = prompt.compose(prompt_input=entry)
        response = model.generate(this_prompt)
        response['id'] = str(i)
        logger.info(f"{i} {response}")

        with open(output_file, "a") as f:
            json.dump(response, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Evaluation saved to {output_file}")

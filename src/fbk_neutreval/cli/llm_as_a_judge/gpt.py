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
import argparse
import json
from fbk_neutreval.cli.llm_as_a_judge.generic import add_args, Model, evaluate
from openai import OpenAI
from pathlib import Path
from typing import List, Dict, Union


class GPTModel(Model):
    def __init__(self,
                 model_id: str,
                 schema_file: str,
                 key_file: str,
                 org_file: str):
        super().__init__(model_id, schema_file)

        with open(key_file, "r") as f:
            self.openai_key = f.read()

        self.openai_org = None

        if org_file is not None:
            with open(org_file, 'r') as f:
                self.org = f.read()

        self.client = OpenAI(api_key=self.openai_key, organization=self.openai_org)

        with open(self.schema_file, "r") as f:
            self.schema = json.load(f)

    def generate(self, input_messages: List[Dict[str, str]]) -> Dict[str, Union[str, List]]:
        output = self.client.chat.completions.create(
            model=self.model_id,
            temperature=0,
            messages=input_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "gnt_eval",
                    "strict": True,
                    "schema": self.schema}}
        )

        return json.loads(output.choices[0].message.content)


def main(args):
    resources_dir = Path(__file__).resolve().parent.parent / "resources"
    schema_file = f"{resources_dir}/schemas/openai/{args.prompt}.json"

    model = GPTModel(model_id=args.model,
                     schema_file=schema_file,
                     key_file=args.key_file,
                     org_file=args.org_file)

    evaluate(model=model,
             input_data_file=args.input,
             start_index=args.start_index,
             output_file=args.output,
             prompt_id=args.prompt,
             lang=args.lang,
             verbose=args.verbose)


def cli():
    """
    This script requires three arguments:

    * -i or --input: the path to a TSV file containing input data. For source-target evaluation,
    the TSV file should contain two columns with headers 'src' and 'tgt', for the source sentences
    and the candidate translations respectively. For target-only evaluation, only the 'tgt' column
    is necessary.
    * -l or --lang: the language to be evaluated (either 'de', 'es', or 'it').
    * -o or --output: the path and name of the output file. Models' generation will be saved in
    JSON Lines format.

    Optional arguments:
    * -p or --prompt: the specific prompt to be used for the evaluation (either 'mono-l',
    'mono-p_l', 'cross-l', 'cross-p_l'). Default: 'cross-p_l'.
    * -s or --include_source: Specify to include the source in the evaluation. If specified, the
    input data should be a TSV file with separate columns for the source and the target, with
    headers 'src' and 'tgt' respectively. Otherwise, the file should only include the column 'tgt'.
    * -m or --model: the OpenAi identifier for the model to use. Make sure to use models that
    support structured generation. Default: gpt-4o-2024-08-06.
    * -r or --start_index: the index of the entry in the TSV file from which the evaluation should
    (re)start. Default: 0.
    * -v or --verbose: specify to show all model outputs in the console.
    """

    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser.add_argument("-m", "--model",
                        help="Model identifier, defaults to gpt-4o-2024-08-06",
                        default='gpt-4o-2024-08-06')
    parser.add_argument('-k', "--key_file", type=str, required=True,
                        help="Path to a plain text file containing the OpenAI API key.")
    parser.add_argument("--org_file", type=str, default=None,
                        help="Path to a plain text file containing the OpenAI organization ID.")
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli()

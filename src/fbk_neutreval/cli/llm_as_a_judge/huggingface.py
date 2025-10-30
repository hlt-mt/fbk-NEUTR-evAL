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
import outlines
from fbk_neutreval.cli.llm_as_a_judge.generic import add_args, Model, evaluate
from pathlib import Path
from outlines.samplers import GreedySampler
from transformers import AutoTokenizer
from typing import List, Dict, Union


class HFModel(Model):
    def __init__(self,
                 model_id: str,
                 schema_file: str):
        super().__init__(model_id, schema_file)

        self.model = outlines.models.transformers(
            self.model_id, model_kwargs={'device_map': 'auto', 'torch_dtype': 'auto'})

        with open(self.schema_file, "r") as f:
            self.schema = f.read()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.sampler = GreedySampler()
        self.generator = outlines.generate.json(self.model, self.schema, sampler=self.sampler)

    def generate(self, input_messages: List[Dict[str, str]]) -> Dict[str, Union[str, List]]:
        prompt = self.tokenizer.apply_chat_template(input_messages, tokenize=False)
        return self.generator(prompt)


def main(args):
    resources_dir = Path(__file__).resolve().parent.parent.parent / "resources"
    schema_file = f"{resources_dir}/schemas/{args.prompt}.json"

    model = HFModel(model_id=args.model,
                    schema_file=schema_file)

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
    * -l or --lang: the language to be evaluated (either 'de', 'el', 'es', or 'it').
    * -o or --output: the path and name of the output file. Models' generation will be saved in
    JSON Lines format.

    Optional arguments:
    * -p or --prompt: the specific prompt to be used for the evaluation (either 'mono-l',
    'mono-p_l', 'cross-l', 'cross-p_l'). Default: 'cross-p_l'. For Greek, only `cross-p_l` is
    available.
    * -s or --include_source: Specify to include the source in the evaluation. If specified, the
    input data should be a TSV file with separate columns for the source and the target, with
    headers 'src' and 'tgt' respectively. Otherwise, the file should only include the column 'tgt'.
    * -m or --model: the language model to be used for the evaluation. Default:
    Qwen/Qwen2.5-72B-Instruct.
    * -r or --start_index: the index of the entry in the TSV file from which the evaluation should
    (re)start. Default: 0.
    * -v or --verbose: specify to show all model outputs in the console.
    """
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser.add_argument("-m", "--model",
                        help="Model identifier, defaults to Qwen/Qwen2.5-72B-Instruct",
                        default='Qwen/Qwen2.5-72B-Instruct')

    # Catch invalid options for Greek evaluation
    args = parser.parse_args()
    if args.lang == 'el' and args.prompt != 'cross-p_l':
        parser.error("When '--lang el' is selected, '--prompt' must be 'cross-p_l' (default).")

    main(args)


if __name__ == '__main__':
    cli()

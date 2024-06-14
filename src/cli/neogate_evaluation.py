# Copyright 2024 FBK

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
This script is used to evaluate the use of gender-inclusive neomorphemes with the Neo-GATE
benchmark.
"""

import csv
import string
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple


VERSION = '1.0'


class AnnotationError(Exception):
    """
    A general `Exception` class for annotation errors.

    Annotation errors can occur when the tagset mapping includes whitespaces or punctuation.
    """
    def __init__(self, line_index: int):
        self.message = f"Annotation error in line {line_index}. This is most likely due to the" \
            "presence of whitespaces or punctuation in the tagset mapping."

    def __str__(self):
        return self.message


class Annotation:
    """
    A class for Neo-GATE annotations.

    Annotations are designed to evaluate whether individual words were generated as desired, i.e.
    they include properly used neomorphemes.
    All annotations include the three forms of the word required for the evaluation: the
    masculine form, the feminine form, and the form with the desired neomorpheme.
    These forms are accessible under the attribute `forms` as a `dict`.

    Some annotations target function words specifically. The `is_function_word()` method allows to
    check whether the annotation targets a function word.
    Function word annotations include two extra attributes:
    - `anchor: str` which maps the function word to an anchor, in respect to which the word should
    be correctly collocated;
    - `distance: int` which indicates at what distance the anchor word should be found.
    """
    def __init__(self, annotation_string: str, hypothesis_index: int):
        self.annotation_parts = annotation_string.split(" ")
        self.function_word = False
        if len(self.annotation_parts) == 4:
            if '=' in self.annotation_parts[3]:
                self.function_word = True
                self.function_word_annotation = self.annotation_parts[3].split('=')
                if len(self.function_word_annotation) != 2:
                    raise AnnotationError(hypothesis_index)
                self.anchor = self.function_word_annotation[0]
                self.distance = int(self.function_word_annotation[1])
            else:
                raise AnnotationError(hypothesis_index)
        elif len(self.annotation_parts) != 3:
            raise AnnotationError(hypothesis_index)

    @property
    def forms(self) -> dict:
        return {
          "masculine": self.annotation_parts[0],
          "feminine": self.annotation_parts[1],
          "correct": self.annotation_parts[2]}

    def is_function_word(self) -> bool:
        """
        Returns `True` if the annotation targets a function word.
        """
        return self.function_word


def get_annotations(annotation_string: str, hypothesis_index: int) -> List[Annotation]:
    """
    Splits the annotation string into as many annotations as the words to be evaluated in the
    entry, and returns them as `Annotation` objects.
    Annotations are separated by a semicolon `;` in the `annotation_string`.

    Each annotation contains the three forms of each term to be searched for in the outputs,
    -- masculine, feminine, and the form with neomorphemes -- separated by whitespaces.
    """
    annotations = annotation_string.strip().split(';')
    return [
        Annotation(a.strip(), hypothesis_index=hypothesis_index) for a in annotations if a != '']


def to_words(hypothesis_line: str, punct: str) -> List[str]:
    """
    Pre-processes the output line to be evaluated, removing punctuation,
    and splitting it into a list of words.
    """
    hypothesis = hypothesis_line.lower()
    hypothesis = hypothesis.replace("'", ' ').translate(str.maketrans('', '', punct))
    return [word for word in hypothesis.split(' ') if word != '']


def evaluate_word(
        word: str,
        annotations: List[Annotation],
        hypothesis: list,
        word_index: int) -> Optional[Tuple[Dict, Annotation]]:
    """
    Checks whether the word matches any of the annotated forms.

    If the word matches an annotation for a function word, this function also checks its
    positioning with respect to the `anchor`.
    Anchors were created so as to consist in the longest common subword among the masculine,
    feminine, and correct form, in order to match it regardless of the form.

    If any form from the annotation is found, it returns both a `dict` which contributes to the
    overall line evaluation and the `Annotation` that was matched.
    """
    word_evaluation = {}
    for annotation in annotations:
        for form in annotation.forms:
            if word == annotation.forms[form]:
                # Checking if the annotation is for a function word
                if annotation.is_function_word():
                    anchor = annotation.anchor
                    anchor_index = word_index + annotation.distance
                    if anchor_index < len(hypothesis):
                        # The function word is correctly positioned with respect to the anchor
                        if anchor in hypothesis[anchor_index]:
                            word_evaluation['matched'] = word
                            word_evaluation[form] = word
                            return word_evaluation, annotation
                else:
                    # Content word case
                    word_evaluation['matched'] = word
                    word_evaluation[form] = word
                    return word_evaluation, annotation
    return None


def check_misgenerations(hypothesis: List[str], neomorphemes: List[str]) -> List[str]:
    """
    Checks whether the hypothesis contains mis-generations.

    This function scans the hypothesis for further words containing neomorphemes
    (i.e. the characters included in `neomorphemes`).

    As all words that contain neomorphemes and matched the annotations have been previously
    removed, this function only returns words with neomorphemes that do not match the annotations.
    """
    misgenerations = []
    for word in hypothesis:
        for n in neomorphemes:
            if n in word:
                misgenerations.append(word)
    return misgenerations


def evaluate_line(
        hypothesis: List[str],
        annotations: List[Annotation],
        neomorphemes: List[str]) -> Dict[str, Any]:
    """
    Scans all words in the hypothesis and check whether they match the annotations for the
    corresponding entry in Neo-GATE.
    """
    line_evaluation = {
        'n-annotations': len(annotations),
        'matched': [],
        'correct': [],
        'masculine': [],
        'feminine': [],
        'misgenerations': []}

    # Creating a copy of the line, to check for misgenerations
    line_double = hypothesis.copy()

    for word_index, word in enumerate(hypothesis):
        evaluation_result = evaluate_word(
            word, annotations, hypothesis, word_index)
        if evaluation_result is not None:
            word_evaluation, found_annotation = evaluation_result
            for k in word_evaluation:
                line_evaluation[k].append(word_evaluation[k])
            if ('correct' in word_evaluation) \
                or ('masculine' in word_evaluation) \
                    or ('feminine' in word_evaluation):
                # Masking the found word on the duplicate to avoid interfering with the
                # function word anchoring mechanism
                line_double[word_index] = '[X]'
                annotations.remove(found_annotation)

    line_evaluation['misgenerations'] = check_misgenerations(line_double, neomorphemes)
    return line_evaluation


def compute_scores(evaluation: list) -> dict:
    """
    Aggregates all the statistics for the input evaluated annotations and computes the metrics
    described in `Piergentili et al., "Enhancing Gender-Inclusive Machine Translation with
    Neomorphemes and Large Language Models" <https://arxiv.org/pdf/2405.08477>`__
    """
    scores = {
        'n-annotations': 0,
        'correct': 0,
        'matched': 0,
        'masculine': 0,
        'feminine': 0,
        'misgenerations': 0,  # Mis-generations count
        'coverage': 0.0,
        'accuracy': 0.0,
        'cwa': 0.0,  # Coverage-weighted accuracy
        'misgen-ratio': 0.0  # Mis-generation ratio
    }

    for line in evaluation:
        scores['n-annotations'] += line['n-annotations']
        scores['matched'] += len(line['matched'])
        scores['correct'] += len(line['correct'])
        scores['masculine'] += len(line['masculine'])
        scores['feminine'] += len(line['feminine'])
        scores['misgenerations'] += len(line['misgenerations'])

    scores['coverage'] = scores['matched'] / scores['n-annotations'] * 100
    scores['misgen-ratio'] = scores['misgenerations'] / scores['n-annotations'] * 100
    if scores['matched'] == 0:
        logging.error("No neomorpheme matched. Accuracy and coverage-weighted accuracy set to 0.")
    else:
        scores['accuracy'] = scores['correct'] / scores['matched'] * 100
        scores['cwa'] = scores['coverage'] * scores['accuracy'] / 100

    return scores


def print_overall_scores(scores: dict):
    """
    Outputs to standard out the scores for all metrics in Neo-GATE.
    """
    print(
        f"Total annotations: {scores['n-annotations']}\n"
        f"Matched: {scores['matched']}\n"
        f"Correct: {scores['correct']}\n"
        f"Masculine: {scores['masculine']}\n"
        f"Feminine: {scores['feminine']}\n"
        f"Mis-generations (count): {scores['misgenerations']}\n\n"
        f"COVERAGE: {scores['coverage']:.2f}\n"
        f"ACCURACY: {scores['accuracy']:.2f}\n"
        f"COVERAGE-WEIGHTED ACCURACY: {scores['cwa']:.2f}\n"
        f"MIS-GENERATION (ratio): {scores['misgen-ratio']:.2f}\n")


def export_detailed_evaluation(evaluation: list, filename: str):
    """
    Export a TSV file containing all the words used to compute the scores.

    For each output, the TSV file includes an entry reporting with the following columns:
    - `#`: The corresponding ID in Neo-GATE.
    - `ANNOTATIONS`: The number of annotations associated with that entry.
    - `MATCHED`: The annotations matched in the output (regardless of the form).
    - `CORRECT`: The annotated words in the neomorpheme form matched in the output.
    - `MASCULINE`: The annotated words in the masculine form matched in the output.
    - `FEMININE`: The annotated words in the feminine form matched in the output.
    - `MIS-GENERATIONS`: The mis-generations matched in the output.
    """
    header = ['#', 'ANNOTATIONS', 'MATCHED', 'CORRECT', 'MASCULINE', 'FEMININE', 'MIS-GENERATIONS']
    with open(filename + '_eval.tsv', 'w') as eval_file:
        output = csv.writer(eval_file, delimiter='\t')
        output.writerow(header)
        for index, line in enumerate(evaluation):
            output.writerow([
                index,
                line['n-annotations'],
                ' '.join(line['matched']),
                ' '.join(line['correct']),
                ' '.join(line['masculine']),
                ' '.join(line['feminine']),
                ' '.join(line['misgenerations'])])


def main(args):
    with open(args.testset, 'r') as test_file:
        test_data = test_file.readlines()
    with open(args.outputs, 'r') as system_output_file:
        system_output = system_output_file.readlines()

    assert len(test_data) == len(system_output), \
        "The number of entries in the system output does not match that of Neo-GATE."
    with open(args.neomorphemes, 'r') as neomorphemes_file:
        # List of symbols to check for over-generation evaluation
        neomorphemes = neomorphemes_file.readlines()
    neomorphemes = [symbol.strip() for symbol in neomorphemes if symbol != '']

    punct = string.punctuation  # List (string) of punctuation to remove
    # Making sure the slash symbol is not removed, so as to avoid matching double gender words,
    # e.g., `solo/a`
    punct.replace('/', '')
    # Making sure that the neomorphemes are not removed
    for nm in neomorphemes:
        punct = punct.replace(nm, '')

    evaluation = []
    for i, (test_row, out_row) in enumerate(zip(test_data, system_output)):
        # Preprocessing the hypothesis and splitting it into a list of words
        hypothesis = to_words(out_row.strip('\n'), punct)
        # List of terms to check for coverage and evaluation
        annotations = get_annotations(test_row.strip('\n'), hypothesis_index=i+1)
        evaluation.append(evaluate_line(hypothesis, annotations, neomorphemes))

    scores = compute_scores(evaluation)
    print_overall_scores(scores)
    if args.export_tsv:
        export_detailed_evaluation(evaluation, args.outputs)


def cli_script():
    """
    This script requires three arguments:
    - `--testset`: the path to the file containing Neo-GATE's annotation adjusted to the desired
    neomorpheme paradigm.
    - `--output`: the path to the file containing system outputs to evaluate.
    - `--neomorphemes`: the path to a file containing the characters used in the neomorpheme
    paradigm, one per line.

    The resulting metrics are printed to the standard output, and a detailed output that allows
    for a manual inspection of the results is saved as a TSV file.
    """
    print(f"Neo-GATE evaluation script {VERSION}.\n")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="The script computes neomorpheme coverage, accuracy, coverage-weighted "
                    "accuracy, and mis-generation rate.",
        prog="neogate_eval")
    parser.add_argument(
        "--testset",
        type=str,
        required=True,
        help="File containing the test set annotation in plain text.")
    parser.add_argument(
        "--outputs",
        type=str,
        required=True,
        help="File containing the outputs to evaluate in plain text.")
    parser.add_argument(
        "--neomorphemes",
        type=str,
        required=True,
        help="File containing the expected neomorphemes, one per line.")
    parser.add_argument(
        "--export_tsv",
        action='store_true',
        help="Set to True to export a tsv file with a detailed evaluation.")
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_script()

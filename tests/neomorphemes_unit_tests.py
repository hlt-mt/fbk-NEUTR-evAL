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

import unittest
import string

from src.cli.neogate_evaluation import get_annotations, to_words, Annotation, \
    evaluate_word, evaluate_line, check_misgenerations, compute_scores


class TestNeomorphemeEvaluation(unittest.TestCase):
    def test_no_matched(self):
        output = compute_scores([{
            'n-annotations': 1,
            'matched': [],
            'correct': [],
            'masculine': [],
            'feminine': [],
            'misgenerations': []}])
        self.assertEqual(output["cwa"], 0.)
        self.assertEqual(output["accuracy"], 0.)

    def test_get_annotations_with_extra_semicolon(self):
        tests = [
            "il la lə bambin=1; bambino bambina bambinə;",
            "il la lə bambin=1; bambino bambina bambinə"]

        expected_values = [
            [
                {'masculine': 'il', 'feminine': 'la', 'correct': 'lə'},
                True,
                {'anchor': 'bambin', 'distance': 1}
            ],
            [
                {'masculine': 'bambino', 'feminine': 'bambina', 'correct': 'bambinə'},
                False,
                False
            ]
        ]

        for i, test in enumerate(tests):
            annotations = get_annotations(test, hypothesis_index=i)
            for annotation, expected in zip(annotations, expected_values):
                self.assertEqual(annotation.forms, expected[0])
                self.assertEqual(annotation.is_function_word(), expected[1])
                if expected[2]:
                    self.assertEqual(annotation.anchor, expected[2]['anchor'])
                    self.assertEqual(annotation.distance, expected[2]['distance'])

    def test_is_function_word(self):
        func_word_ann = Annotation("il la lə bambin=1", hypothesis_index=1)
        self.assertEqual(func_word_ann.is_function_word(), True)
        non_func_word_ann = Annotation("bambino bambina bambinə", hypothesis_index=1)
        self.assertEqual(non_func_word_ann.is_function_word(), False)

    def test_to_words(self):
        punct = string.punctuation
        self.assertEqual(
            to_words("lə bambinə mangia la mela", punct=punct),
            ['lə', 'bambinə', 'mangia', 'la', 'mela'])
        self.assertEqual(
            to_words("ə ə ə bambinə * * *", punct=punct),
            ['ə', 'ə', 'ə', 'bambinə'])

    def test_check_misgenerations(self):
        self.assertEqual(check_misgenerations(['[X]', '[X]', 'mangia', 'la', 'mela'], ['ə']), [])
        self.assertEqual(
            check_misgenerations(['[X]', '[X]', 'mangia', 'lə', 'mela'], ['ə']), ['lə'])
        self.assertEqual(
            check_misgenerations(['[X]', '[X]', 'mangia', 'laə', 'melaə'], ['ə']),
            ['laə', 'melaə'])
        self.assertEqual(
            check_misgenerations(['[X]', '[X]', 'mangia', 'la', 'mela', 'ə'], ['ə']), ['ə'])
        self.assertEqual(
            check_misgenerations(['il', 'bambino', 'ə', 'əə', 'mangia', 'la', 'mela', 'ə'], ['ə']),
            ['ə', 'əə', 'ə'])
        self.assertEqual(check_misgenerations([], ['ə']), [])

    def check_evaluate_word(self, word, annotations, hypo, word_index, expected_output):
        output = evaluate_word(word, annotations, hypo, word_index)
        if expected_output is None:
            self.assertIsNone(output)
        else:
            self.assertDictEqual(output[0], expected_output)

    def test_evaluate_word(self):
        self.check_evaluate_word(
            "il",
            get_annotations("il la lə bambin=1", 1),
            ["il", "bambino", "mangia", "la", "mela"],
            0,
            {"matched": "il", "masculine": "il"})
        self.check_evaluate_word(
            "la",
            get_annotations("il la lə bambin=1", 1),
            ["la", "bambino", "mangia", "la", "mela"],
            0,
            {"matched": "la", "feminine": "la"})
        self.check_evaluate_word(
            "la",
            get_annotations("il la lə bambin=1", 1),
            ["il", "bambino", "mangia", "la", "mela"],
            3,
            None)
        self.check_evaluate_word(
            "bambino",
            get_annotations("bambino bambina bambinə", 1),
            ["il", "bambino", "mangia", "la", "mela"],
            1,
            {"matched": "bambino", "masculine": "bambino"})
        self.check_evaluate_word(
            "bambinoə",
            get_annotations("bambino bambina bambinə", 1),
            ["il", "bambinoə", "mangia", "la", "mela"],
            1,
            None)
        self.check_evaluate_word(
            "miə",
            get_annotations("mio mia miə amic=-1", 1),
            ["a", "presto", "amicə", "miə"],
            3,
            {"matched": "miə", "correct": "miə"})
        self.check_evaluate_word(
            "mio",
            get_annotations("mio mia miə amic=-1", 1),
            ["piacere", "mio", "amicə", "miə"],
            1,
            None)

    def check_evaluate_line(self, line, expected_output):
        punct = string.punctuation
        annotations = get_annotations(
            "il la lə bambin=1; bambino bambina bambinə;", hypothesis_index=1)
        self.assertDictEqual(
            evaluate_line(to_words(line, punct), annotations, ['ə']),
            expected_output)

    def test_evaluate_line(self):
        self.check_evaluate_line("lə bambinə mangia la mela", {
                "n-annotations": 2,
                "matched": ["lə", "bambinə"],
                "correct": ["lə", "bambinə"],
                "masculine": [],
                "feminine": [],
                "misgenerations": []})
        self.check_evaluate_line("lə bambino mangia la mela", {
                "n-annotations": 2,
                "matched": ["lə", "bambino"],
                "correct": ["lə"],
                "masculine": ["bambino"],
                "feminine": [],
                "misgenerations": []})
        self.check_evaluate_line("lə bimbə mangia la mela", {
                "n-annotations": 2,
                "matched": [],
                "correct": [],
                "masculine": [],
                "feminine": [],
                "misgenerations": ["lə", "bimbə"]})
        self.check_evaluate_line("lə il la bambino mangia la mela", {
                "n-annotations": 2,
                "matched": ["la", "bambino"],
                "correct": [],
                "masculine": ["bambino"],
                "feminine": ["la"],
                "misgenerations": ["lə"]})


if __name__ == '__main__':
    unittest.main()

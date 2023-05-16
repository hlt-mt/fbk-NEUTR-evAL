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
import csv
from typing import Dict


REGISTERED_WRITERS = {}


def register_writer(name):
    def do_register_writer(cls):
        REGISTERED_WRITERS[name] = cls
        return cls

    return do_register_writer


class FileWriter:
    """
    Writes results to a File.
    """
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.fp = None

    def __enter__(self):
        assert self.fp is None
        self.fp = open(self.file_name, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fp is not None:
            self.fp.close()

    def write_line(self, elements: Dict[str, str]) -> None:
        ...


@register_writer("tsv")
class TSVWriter(FileWriter):
    """
    Writes results as a TSV file.
    """
    def __init__(self, file_name: str):
        super().__init__(file_name)
        self.writer = None

    def write_line(self, elements: Dict[str, str]) -> None:
        if self.writer is None:
            self.writer = csv.DictWriter(self.fp, fieldnames=elements.keys(), delimiter="\t")
        self.writer.writerow(elements)

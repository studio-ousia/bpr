import csv
import json
import dataclasses
from typing import Iterator

import lmdb


@dataclasses.dataclass
class Passage:
    id: int
    title: str
    text: str


class PassageDB:
    def __init__(self, input_file: str):
        self._input_file = input_file
        self._db = lmdb.open(input_file, subdir=False, readonly=True)

    def __reduce__(self):
        return (self.__class__, (self._input_file,))

    def __len__(self):
        return self._db.stat()["entries"]

    def __getitem__(self, id_: int) -> Passage:
        with self._db.begin() as txn:
            json_text = txn.get(str(id_).encode("utf-8"))
            if json_text is None:
                raise KeyError("Invalid passage_id: " + str(id_))
            title, text = json.loads(json_text)
            return Passage(id_, title, text)

    def __iter__(self) -> Iterator[Passage]:
        with self._db.begin() as txn:
            cursor = txn.cursor()
            for id_str, json_str in iter(cursor):
                title, text = json.loads(json_str.decode("utf-8"))
                yield Passage(int(id_str.decode("utf-8")), title, text)


class InMemoryPassageDB(PassageDB):
    def __init__(self, input_file: str):
        self.data = {}
        with open(input_file) as i_f:
            tsv_reader = csv.reader(i_f, delimiter="\t")
            for row in tsv_reader:
                if row[0] == "id":
                    continue  # ignoring header
                _id, text, title = int(row[0]), row[1], row[2]
                self.data[_id] = (text, title)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id_: int) -> Passage:
        text, title = self.data[id_]
        return Passage(id_, title, text)

    def __iter__(self) -> Iterator[Passage]:
        for id_, (text, title) in self.data.items():
            yield Passage(id_, title, text)

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

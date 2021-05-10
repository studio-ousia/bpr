from __future__ import annotations

import dataclasses
import glob
import json
import logging
import multiprocessing
import os
import unicodedata
from contextlib import closing
from multiprocessing.pool import Pool
from typing import Dict, Generator, List, Optional, Tuple

import joblib
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetPassage:
    id: str
    index: int
    title: str
    title_token_ids: List[int]
    passage_token_ids: List[int]
    has_answer: bool
    answer_spans: List[Tuple[int, int]]


@dataclasses.dataclass
class DatasetExample:
    index: int
    answers: List[str]
    positive_passages: List[DatasetPassage]
    other_passages: List[DatasetPassage]
    is_gold_positive: bool
    query_token_ids: List[int]

    def to_tuple(self) -> tuple:
        return (
            self.index,
            self.answers,
            [dataclasses.astuple(p) for p in self.positive_passages],
            [dataclasses.astuple(p) for p in self.other_passages],
            self.is_gold_positive,
            self.query_token_ids,
        )

    @staticmethod
    def from_tuple(input_tuple: tuple) -> DatasetExample:
        return DatasetExample(
            input_tuple[0],
            input_tuple[1],
            [DatasetPassage(*t) for t in input_tuple[2]],
            [DatasetPassage(*t) for t in input_tuple[3]],
            input_tuple[4],
            input_tuple[5],
        )


class ReaderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        example_data: List[tuple],
        query_token_ids: np.ndarray,
        query_indptr: np.ndarray,
        title_token_ids: np.ndarray,
        title_indptr: np.ndarray,
        passage_token_ids: np.ndarray,
        passage_indptr: np.ndarray,
    ):
        self.example_data = example_data
        self.query_token_ids = query_token_ids
        self.query_indptr = query_indptr
        self.title_token_ids = title_token_ids
        self.title_indptr = title_indptr
        self.passage_token_ids = passage_token_ids
        self.passage_indptr = passage_indptr

    def __len__(self):
        return len(self.example_data)

    def __getitem__(self, index: int):
        example = DatasetExample.from_tuple(self.example_data[index])

        example.query_token_ids = self.query_token_ids[
            self.query_indptr[example.index] : self.query_indptr[example.index + 1]
        ].tolist()
        for passage in example.positive_passages + example.other_passages:
            passage.title_token_ids = self.title_token_ids[
                self.title_indptr[passage.index] : self.title_indptr[passage.index + 1]
            ].tolist()
            passage.passage_token_ids = self.passage_token_ids[
                self.passage_indptr[passage.index] : self.passage_indptr[passage.index + 1]
            ].tolist()

        return example

    @classmethod
    def load_dataset(
        cls,
        retriever_file: str,
        base_pretrained_model: str,
        fold: str,
        nq_gold_file: Optional[str] = None,
        mmap_mode: str = "r",
        pool_size: int = multiprocessing.cpu_count(),
        chunk_size: int = 30,
    ) -> ReaderDataset:
        cache_key = "_".join(
            [
                os.path.basename(retriever_file),
                base_pretrained_model.replace("/", "-"),
                os.path.basename(nq_gold_file) if nq_gold_file else "None",
                fold,
            ]
        )
        cache_file = os.path.join(os.path.dirname(retriever_file), "cache_" + cache_key + ".pkl")
        if os.path.exists(cache_file):
            data = joblib.load(cache_file, mmap_mode=mmap_mode)
            return ReaderDataset(**data)

        retriever_data = []
        for path in glob.glob(retriever_file):
            with open(path) as f:
                retriever_data += json.load(f)

        example_data = []

        example_index = 0
        passage_index = 0
        query_token_ids = []
        query_indptr = [0]
        title_token_ids = []
        title_indptr = [0]
        passage_token_ids = []
        passage_indptr = [0]

        with closing(
            Pool(
                pool_size,
                initializer=cls._init_dataset_worker,
                initargs=(base_pretrained_model, fold, nq_gold_file),
                context=multiprocessing.get_context("spawn"),
            )
        ) as pool:
            with tqdm(total=len(retriever_data)) as pbar:
                for example in pool.imap(cls._process_dataset_item, retriever_data, chunksize=chunk_size):
                    if example is not None:
                        for passage in example.positive_passages + example.other_passages:
                            title_token_ids += passage.title_token_ids
                            title_indptr.append(len(title_token_ids))

                            passage_token_ids += passage.passage_token_ids
                            passage_indptr.append(len(passage_token_ids))

                            # remove the following fields for computational efficiency
                            passage.title_token_ids = []
                            passage.passage_token_ids = []

                            passage.index = passage_index
                            passage_index += 1

                        query_token_ids += example.query_token_ids
                        query_indptr.append(len(query_token_ids))
                        example.query_token_ids = []

                        example.index = example_index
                        example_index += 1

                        example_data.append(example.to_tuple())

                    pbar.update()

        data = dict(
            example_data=example_data,
            query_token_ids=np.array(query_token_ids, dtype=np.uint16),
            query_indptr=np.array(query_indptr, dtype=np.uint64),
            title_token_ids=np.array(title_token_ids, dtype=np.uint16),
            title_indptr=np.array(title_indptr, dtype=np.uint64),
            passage_token_ids=np.array(passage_token_ids, dtype=np.uint16),
            passage_indptr=np.array(passage_indptr, dtype=np.uint64),
        )
        joblib.dump(data, cache_file, protocol=-1)
        return ReaderDataset(**data)

    @classmethod
    def _init_dataset_worker(cls, base_pretrained_model: str, fold: str, nq_gold_file: str) -> None:
        cls._base_pretrained_model = base_pretrained_model
        cls._fold = fold

        cls._tokenizer = BertTokenizerFast.from_pretrained(base_pretrained_model)

        if nq_gold_file:
            cls._gold_title_data, cls._tokenized_qs2original_qs = cls._load_nq_gold_dataset(nq_gold_file)
        else:
            cls._gold_title_data, cls._tokenized_qs2original_qs = {}, {}

    @staticmethod
    def _load_nq_gold_dataset(dataset_file: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        with open(dataset_file) as f:
            data = json.load(f)["data"]

        gold_title_data = {}
        tokenized_qs2original_qs = {}

        for item in data:
            query = item["question"]
            tokenized_query = item.get("question_tokens", query)
            tokenized_qs2original_qs[tokenized_query] = query

            gold_title_data[query] = item["title"]

        return gold_title_data, tokenized_qs2original_qs

    @classmethod
    def _process_dataset_item(cls, dataset_item: dict) -> Optional[DatasetExample]:
        query = dataset_item["question"]
        query = cls._tokenized_qs2original_qs.get(query, query)

        query_token_ids = cls._tokenizer.encode(query, add_special_tokens=False)

        passages = [
            DatasetPassage(
                id=ctx["id"],
                index=-1,
                title=ctx["title"],
                title_token_ids=cls._tokenizer.encode(ctx["title"], add_special_tokens=False),
                passage_token_ids=cls._tokenizer.encode(ctx["text"], add_special_tokens=False),
                answer_spans=[],
                has_answer=ctx["has_answer"],
            )
            for ctx in dataset_item["ctxs"]
        ]

        if cls._fold != "train":
            return DatasetExample(
                index=-1,
                answers=dataset_item["answers"],
                positive_passages=[],
                other_passages=passages,
                is_gold_positive=False,
                query_token_ids=query_token_ids,
            )

        answer_token_ids_list = [cls._tokenizer.encode(a, add_special_tokens=False) for a in dataset_item["answers"]]
        answer_token_ids_list += [
            cls._tokenizer.encode("".join(a.split(" ")), add_special_tokens=False) for a in dataset_item["answers"]
        ]
        positive_passages = []

        def find_answer_spans(passage_token_ids: List[int]) -> Generator[Tuple[int, int], None, None]:
            for answer_token_ids in answer_token_ids_list:
                answer_length = len(answer_token_ids)
                for i in range(len(passage_token_ids) - answer_length + 1):
                    if passage_token_ids[i : i + answer_length] == answer_token_ids:
                        yield (i, i + answer_length)

        def normalize_title(text: str) -> str:
            return unicodedata.normalize("NFD", text.lower())

        gold_title = cls._gold_title_data.get(query)
        is_gold_positive = True
        if gold_title:
            for passage in passages:
                if passage.has_answer and normalize_title(passage.title) == normalize_title(gold_title):
                    answer_spans = sorted(frozenset(find_answer_spans(passage.passage_token_ids)))
                    if answer_spans:
                        passage.answer_spans = answer_spans
                        positive_passages.append(passage)

        if not positive_passages:
            is_gold_positive = False
            for passage in passages:
                if passage.has_answer:
                    answer_spans = sorted(frozenset(find_answer_spans(passage.passage_token_ids)))
                    if answer_spans:
                        passage.answer_spans = answer_spans
                        positive_passages.append(passage)

        if not positive_passages:
            return None

        negative_passages = [p for p in passages if not p.has_answer]

        return DatasetExample(
            index=-1,
            answers=dataset_item["answers"],
            positive_passages=positive_passages,
            other_passages=negative_passages,
            is_gold_positive=is_gold_positive,
            query_token_ids=query_token_ids,
        )

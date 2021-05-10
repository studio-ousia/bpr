import dataclasses
from typing import List

import numpy as np
import torch
from tqdm import trange
from transformers import AutoTokenizer

from .biencoder import BiEncoder
from .index import FaissIndex
from .passage_db import Passage, PassageDB


@dataclasses.dataclass
class Candidate:
    id: int
    score: float
    passage: Passage


class Retriever(object):
    def __init__(self, index: FaissIndex, biencoder: BiEncoder, passage_db: PassageDB):
        self.index = index
        self._biencoder = biencoder
        self._passage_db = passage_db

        self._tokenizer = AutoTokenizer.from_pretrained(biencoder.hparams.base_pretrained_model, use_fast=True)

    def encode_queries(self, queries: List[str], batch_size: int = 256) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for start in trange(0, len(queries), batch_size):
                model_inputs = self._tokenizer.batch_encode_plus(
                    queries[start : start + batch_size],
                    return_tensors="pt",
                    max_length=self._biencoder.hparams.max_query_length,
                    pad_to_max_length=True,
                )

                model_inputs = {k: v.to(self._biencoder.device) for k, v in model_inputs.items()}
                emb = self._biencoder.query_encoder(**model_inputs).cpu().numpy()
                embeddings.append(emb)

        return np.vstack(embeddings)

    def search(self, query_embeddings: np.ndarray, k: int, **faiss_index_options) -> List[List[Candidate]]:
        scores_list, ids_list = self.index.search(query_embeddings, k, **faiss_index_options)
        return [
            [Candidate(int(id_), float(score), self._passage_db[id_]) for score, id_ in zip(scores, ids)]
            for scores, ids in zip(scores_list, ids_list)
        ]

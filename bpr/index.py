import logging
import time
from typing import List, Optional, Tuple

import faiss
import numpy as np
from tqdm import trange

logger = logging.getLogger(__name__)


class FaissIndex:
    def __init__(self, index: faiss.Index, passage_ids: List[int]):
        self.index = index
        self._passage_ids = np.array(passage_ids, dtype=np.int64)

    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        scores_arr, ids_arr = self.index.search(query_embeddings, k)
        ids_arr = self._passage_ids[ids_arr.reshape(-1)].reshape(query_embeddings.shape[0], -1)
        logger.info("Total search time: %.3f", time.time() - start_time)
        return scores_arr, ids_arr

    @classmethod
    def build(
        cls,
        passage_ids: List[int],
        passage_embeddings: np.ndarray,
        index: Optional[faiss.Index] = None,
        buffer_size: int = 50000,
    ):
        if index is None:
            index = faiss.IndexFlatIP(passage_embeddings.shape[1])
        for start in trange(0, len(passage_ids), buffer_size):
            index.add(passage_embeddings[start : start + buffer_size])

        return cls(index, passage_ids)

    def to_gpu(self):
        if faiss.get_num_gpus() == 1:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            cloner_options = faiss.GpuMultipleClonerOptions()
            cloner_options.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=cloner_options)

        return self.index


class FaissHNSWIndex(FaissIndex):
    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_embeddings = np.hstack((query_embeddings, np.zeros((query_embeddings.shape[0], 1), dtype=np.float32)))
        return super().search(query_embeddings, k)

    @classmethod
    def build(
        cls,
        passage_ids: List[int],
        passage_embeddings: np.ndarray,
        index: Optional[faiss.Index] = None,
        buffer_size: int = 50000,
    ):
        sq_norms = (passage_embeddings ** 2).sum(1)
        max_sq_norm = float(sq_norms.max())
        aux_dims = np.sqrt(max_sq_norm - sq_norms)
        passage_embeddings = np.hstack((passage_embeddings, aux_dims.reshape(-1, 1)))
        return super().build(passage_ids, passage_embeddings, index, buffer_size)


class FaissBinaryIndex(FaissIndex):
    def __init__(self, index: faiss.Index, passage_ids: List[int], passage_embeddings: np.ndarray):
        self.index = index
        self._passage_ids = np.array(passage_ids, dtype=np.int64)
        self._passage_embeddings = passage_embeddings

    def search(self, query_embeddings: np.ndarray, k: int, binary_k=None, rerank=True) -> Tuple[np.ndarray, np.ndarray]:
        if binary_k is None:
            binary_k = k * 100

        start_time = time.time()
        num_queries = query_embeddings.shape[0]
        bin_query_embeddings = np.packbits(np.where(query_embeddings > 0, 1, 0)).reshape(num_queries, -1)

        if not rerank:
            scores_arr, ids_arr = self.index.search(bin_query_embeddings, k)
            ids_arr = self._passage_ids[ids_arr.reshape(-1)].reshape(num_queries, -1)
            return scores_arr, ids_arr

        scores_arr, ids_arr = self.index.search(bin_query_embeddings, binary_k)
        logger.info("Initial search time: %.3f", time.time() - start_time)

        passage_embeddings = np.unpackbits(self._passage_embeddings[ids_arr.reshape(-1)])
        passage_embeddings = passage_embeddings.reshape(num_queries, binary_k, -1).astype(np.float32)
        passage_embeddings = passage_embeddings * 2 - 1
        scores_arr = np.einsum("ijk,ik->ij", passage_embeddings, query_embeddings)
        sorted_indices = np.argsort(-scores_arr, axis=1)

        ids_arr = ids_arr[np.arange(num_queries)[:, None], sorted_indices]
        ids_arr = self._passage_ids[ids_arr.reshape(-1)].reshape(num_queries, -1)

        scores_arr = scores_arr[np.arange(num_queries)[:, None], sorted_indices]
        logger.info("Total search time: %.3f", time.time() - start_time)

        return scores_arr[:, :k], ids_arr[:, :k]

    @classmethod
    def build(
        cls,
        passage_ids: List[int],
        passage_embeddings: np.ndarray,
        index: Optional[faiss.Index] = None,
        buffer_size: int = 50000,
    ):
        if index is None:
            index = faiss.IndexBinaryFlat(passage_embeddings.shape[1] * 8)
        for start in trange(0, len(passage_ids), buffer_size):
            index.add(passage_embeddings[start : start + buffer_size])

        return cls(index, passage_ids, passage_embeddings)

import argparse
import csv
import json
import logging
import multiprocessing
import os
import tempfile
import time
import unicodedata
from collections import defaultdict
from contextlib import closing
from typing import List

import faiss
import joblib
import numpy as np
import regex
from torch.nn.parallel import DataParallel
from tqdm import tqdm

import bpr.index
from bpr.biencoder import BiEncoder
from bpr.index import FaissBinaryIndex, FaissIndex, FaissHNSWIndex
from bpr.passage_db import PassageDB
from bpr.retriever import Retriever

logger = logging.getLogger(__name__)

# https://github.com/facebookresearch/DPR/blob/f403c3b3e179e53c0fe68a0718d5dc25371fe5df/dpr/utils/tokenizers.py#L154
ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
NON_WS = r"[^\p{Z}\p{C}]"
# https://github.com/facebookresearch/DPR/blob/f403c3b3e179e53c0fe68a0718d5dc25371fe5df/dpr/utils/tokenizers.py#L163
REGEXP = regex.compile("(%s)|(%s)" % (ALPHA_NUM, NON_WS), flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE)


def _has_answer(answer: str, passage: str) -> bool:
    def tokenize(text: str) -> List[str]:
        return [m.group() for m in REGEXP.finditer(text)]

    answer_tokens = tokenize(unicodedata.normalize("NFD", answer.lower()))
    passage_tokens = tokenize(unicodedata.normalize("NFD", passage.lower()))

    for i in range(0, len(passage_tokens) - len(answer_tokens) + 1):
        if answer_tokens == passage_tokens[i : i + len(answer_tokens)]:  # noqa: E203
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--biencoder_file", type=str, required=True)
    parser.add_argument("--embedding_file", type=str, required=True)
    parser.add_argument("--passage_db_file", type=str, required=True)
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--top_k", type=int, default=[1, 5, 20, 50, 100], action="append")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--binary_k", type=int, default=2048)
    parser.add_argument("--binary_to_float", action="store_true")
    parser.add_argument("--binary_no_rerank", action="store_true")
    parser.add_argument("--use_binary_hash", action="store_true")
    parser.add_argument("--use_hnsw", action="store_true")
    parser.add_argument("--hash_num_bits", type=int, default=768)
    parser.add_argument("--hnsw_store_n", type=int, default=512)
    parser.add_argument("--hnsw_ef_construction", type=int, default=200)
    parser.add_argument("--hnsw_ef_search", type=int, default=128)
    parser.add_argument("--biencoder_device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--index_device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--pool_size", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--chunk_size", type=int, default=32)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING, format="[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    )
    logger.setLevel(logging.INFO)
    bpr.index.logger.setLevel(logging.INFO)

    passage_db = PassageDB(args.passage_db_file)
    embedding_data = joblib.load(args.embedding_file, mmap_mode="r")
    ids, embeddings = embedding_data["ids"], embedding_data["embeddings"]
    dim_size = embeddings.shape[1]

    logger.info("Building index...")
    if embeddings.dtype == np.uint8:
        if args.binary_to_float:
            embeddings = np.unpackbits(embeddings).reshape(-1, dim_size * 8).astype(np.float32)
            embeddings = embeddings * 2 - 1
            base_index = faiss.IndexFlatIP(dim_size * 8)
            index = FaissIndex.build(ids, embeddings, base_index)

        elif args.use_binary_hash:
            base_index = faiss.IndexBinaryHash(dim_size * 8, args.hash_num_bits)
            index = FaissBinaryIndex.build(ids, embeddings, base_index)

        else:
            base_index = faiss.IndexBinaryFlat(dim_size * 8)
            index = FaissBinaryIndex.build(ids, embeddings, base_index)

    elif args.use_hnsw:
        base_index = faiss.IndexHNSWFlat(dim_size + 1, args.hnsw_store_n)
        base_index.hnsw.efSearch = args.hnsw_ef_search
        base_index.hnsw.efConstruction = args.hnsw_ef_construction
        index = FaissHNSWIndex.build(ids, embeddings, base_index)

    else:
        base_index = faiss.IndexFlatIP(dim_size)
        index = FaissIndex.build(ids, embeddings, base_index)
        if args.index_device == "cuda":
            index = index.to_gpu()

    del ids
    del embeddings

    with tempfile.NamedTemporaryFile() as f:
        if isinstance(index, FaissBinaryIndex):
            faiss.write_index_binary(index.index, f.name)
        else:
            faiss.write_index(index.index, f.name)

        logger.info("Index size: %d bytes", os.path.getsize(f.name))

    logger.info("Loading BiEncoder...")
    biencoder = BiEncoder.load_from_checkpoint(args.biencoder_file, map_location="cpu")
    biencoder = biencoder.to(args.biencoder_device)
    biencoder.eval()
    biencoder.freeze()

    if args.parallel:
        biencoder.query_encoder = DataParallel(biencoder.query_encoder)

    retriever = Retriever(index, biencoder, passage_db)

    logger.info("Loading QA pairs from %s", args.qa_file)
    with open(args.qa_file) as f:
        qa_pairs = [(row[0], eval(row[1].strip())) for row in csv.reader(f, delimiter="\t")]
    total_count = len(qa_pairs)

    logger.info("Computing query embeddings...")
    queries = [pair[0] for pair in qa_pairs]
    query_embeddings = retriever.encode_queries(queries)

    logger.info("Getting top-k results...")
    start_time = time.time()
    if isinstance(index, FaissBinaryIndex):
        topk_results = retriever.search(
            query_embeddings, max(args.top_k), binary_k=args.binary_k, rerank=not args.binary_no_rerank
        )
    else:
        topk_results = retriever.search(query_embeddings, max(args.top_k))
    query_time = time.time() - start_time
    logger.info("Elapsed time: %.2fsec", query_time)
    logger.info("Queries per sec: %.2f", total_count / query_time)

    del biencoder
    del retriever

    def process_candidates(args):
        (query, answers), candidates = args
        passage_dicts = []
        for index, candidate in enumerate(candidates):
            passage_dict = dict(id=int(candidate.passage.id), score=float(candidate.score))
            passage_dict["title"] = candidate.passage.title
            passage_dict["text"] = candidate.passage.text

            if any(_has_answer(answer, candidate.passage.text) for answer in answers):
                passage_dict["has_answer"] = True
            else:
                passage_dict["has_answer"] = False

            passage_dicts.append(passage_dict)

        return dict(question=query, answers=answers, ctxs=passage_dicts)

    logger.info("Computing evaluation metrics...")

    correct_counts = defaultdict(int)
    output_examples = []
    with tqdm(total=len(qa_pairs)) as pbar:
        with closing(multiprocessing.Pool(args.pool_size)) as pool:
            for example in pool.imap(process_candidates, zip(qa_pairs, topk_results), chunksize=args.chunk_size):
                output_examples.append(example)
                for index, passage_dict in enumerate(example["ctxs"]):
                    if passage_dict["has_answer"]:
                        for top_k in args.top_k:
                            if index < top_k:
                                correct_counts[top_k] += 1
                        break

                pbar.update()

    if not args.no_eval:
        logger.info("#total examples: %d", total_count)
        for top_k in args.top_k:
            precision = correct_counts[top_k] / total_count
            logger.info("precision@%d:%f correct_samples:%d", top_k, precision, correct_counts[top_k])

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(output_examples, f, ensure_ascii=False, indent=2)

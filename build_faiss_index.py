import argparse

import faiss
import joblib
import numpy as np
from tqdm import trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hash_num_bits", type=int, default=768)

    args = parser.parse_args()

    embedding_data = joblib.load(args.embedding_file, mmap_mode="r")

    ids = np.array(embedding_data["ids"], dtype=np.int)
    embeddings = embedding_data["embeddings"]
    dim_size = embeddings.shape[1] * 8

    index = faiss.IndexBinaryIDMap(faiss.IndexBinaryFlat(dim_size))
    for start in trange(0, ids.size, args.batch_size):
        index.add_with_ids(embeddings[start : start + args.batch_size], ids[start : start + args.batch_size])

    faiss.write_index_binary(index, args.output_file)

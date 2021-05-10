import argparse
import itertools

import joblib
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from transformers import AutoTokenizer

from bpr.biencoder import BiEncoder
from bpr.passage_db import PassageDB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--biencoder_file", type=str, required=True)
    parser.add_argument("--passage_db_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--biencoder_device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    passage_db = PassageDB(args.passage_db_file)

    torch.set_grad_enabled(False)
    biencoder = BiEncoder.load_from_checkpoint(args.biencoder_file, map_location="cpu")
    biencoder = biencoder.to(args.biencoder_device)
    biencoder.eval()
    biencoder.freeze()

    tokenizer = AutoTokenizer.from_pretrained(biencoder.hparams.base_pretrained_model, use_fast=True)

    passage_encoder = biencoder.passage_encoder
    if args.parallel:
        passage_encoder = DataParallel(passage_encoder)

    ids = []
    embeddings = []
    db_iterator = iter(passage_db)

    with tqdm(total=len(passage_db)) as pbar:
        while True:
            passages = list(itertools.islice(db_iterator, args.batch_size))
            if not passages:
                break
            passage_inputs = tokenizer.batch_encode_plus(
                [(passage.title, passage.text) for passage in passages],
                return_tensors="pt",
                max_length=biencoder.hparams.max_passage_length,
                pad_to_max_length=True,
            )
            passage_inputs = {k: v.to(args.biencoder_device) for k, v in passage_inputs.items()}
            emb = passage_encoder(**passage_inputs)
            if biencoder.hparams.binary:
                emb = biencoder.convert_to_binary_code(emb).cpu().numpy()
                emb = np.where(emb == -1, 0, emb).astype(np.bool)
                emb = np.packbits(emb).reshape(emb.shape[0], -1)
            else:
                emb = emb.cpu().numpy().astype(np.float32)

            ids += [passage.id for passage in passages]
            embeddings.append(emb)
            pbar.update(args.batch_size)

    embeddings = np.vstack(embeddings)
    joblib.dump(dict(ids=ids, embeddings=embeddings), args.output_file)

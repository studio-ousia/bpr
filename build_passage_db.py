import argparse
import csv
import json

import lmdb
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passage_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--map_size", type=int, default=21000000000)
    parser.add_argument("--chunk_size", type=int, default=1024)
    args = parser.parse_args()

    db = lmdb.open(args.output_file, map_size=args.map_size, subdir=False)
    with db.begin(write=True) as txn:
        with open(args.passage_file) as f:
            tsv_reader = csv.reader(f, delimiter="\t")
            buf = []
            for n, row in tqdm(enumerate(tsv_reader)):
                if n == 0:
                    continue
                _id, text, title = int(row[0]), row[1], row[2]
                json_str = json.dumps((title, text))
                buf.append((str(_id).encode("utf-8"), json_str.encode("utf-8")))
                if len(buf) == args.chunk_size:
                    txn.cursor().putmulti(buf)
            if buf:
                txn.cursor().putmulti(buf)

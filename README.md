# BPR

Binary Passage Retriever (BPR) is an efficient neural retrieval model for
open-domain question answering. BPR integrates a _learning-to-hash_ technique
into [Dense Passage Retriever (DPR)](https://github.com/facebookresearch/DPR) to
**represent the passage embeddings using compact binary codes** rather than
continuous vectors. It substantially **reduces the memory size without a loss of accuracy** tested on Natural Questions and TriviaQA datasets.

BPR was originally developed to improve the computational efficiency of the [Sōseki question
answering system](https://demo.soseki.ai/) submitted to the [Systems under 6GB track](https://ai.google.com/research/NaturalQuestions/efficientqa) in the
[NeurIPS 2020 EfficientQA competition](https://efficientqa.github.io/).
Please refer to [our ACL 2021 paper](#) for further technical details.

## Installation

BPR can be installed using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

The virtual environment automatically created by Poetry can be activated by
`poetry shell`.

Alternatively, you can install required libraries using pip:

```bash
pip install -r requirements.txt
```

## Trained Models

**BPR fine-tuned on the Natural Questions dataset:**

* [Checkpoint file](https://drive.google.com/file/d/1BibJ0GQn6rvKfEBksPMeyx-vl3s57vT7/view?usp=sharing) (836MB)
* [Index file](https://drive.google.com/file/d/1hTnTi1r_6lGfUmJ9RWbx3ciX8r6GDrOT/view?usp=sharing) (2.1GB)

**BPR fine-tuned on the TriviaQA dataset:**

* [Checkpoint file](https://drive.google.com/file/d/1ehbpUo0EmAW61Jc72xi1S02548p0Dw6I/view?usp=sharing) (836MB)
* [Index file](https://drive.google.com/file/d/1EqGAkxIrg6TkVG72kCYMdH7jUIsQFvte/view?usp=sharing) (2.1GB)

## Example Usage


```python
>>> from bpr import BiEncoder, FaissBinaryIndex, InMemoryPassageDB, Retriever
>>> import faiss
# Load the model from the checkpoint file
>>> biencoder = BiEncoder.load_from_checkpoint("<CHECKPOINT_FILE>")
>>> biencoder.eval()
>>> biencoder.freeze()
# Load Wikipedia passages into memory
>>> passage_db = InMemoryPassageDB("<DPR_DATASET_DIR>/wikipedia_split/psgs_w100.tsv")
# Load the index
>>> base_index = faiss.read_index_binary("<INDEX_FILE>")
>>> index = FaissBinaryIndex(base_index)
# Instantiate the Retriever
>>> retriever = Retriever(index, biencoder, passage_db)
# Encode queries into embeddings
>>> query_embeddings = retriever.encode_queries(["who is under the mask of darth vader"])
# Get top-k results
>>> retriever.search(query_embeddings, k=100)[0][0]
Candidate(id=650642, score=95.28644561767578, passage=Passage(id=650642, title='Darth Vader', text="that McQuarrie's earliest conception of Vader was so successful that very little needed to be changed for production. Working from McQuarrie's designs, the costume designer John Mollo devised a costume that could be worn by an actor on-screen using a combination of clerical robes, a motorcycle suit, a German military helmet and a gas mask. The prop sculptor Brian Muir created the helmet and armour used in the film. The sound of the respirator function of Vader's mask was created by Ben Burtt using modified recordings of scuba breathing apparatus used by divers. The sound effect is trademarked in the"))
```

The Wikipedia passage data (`psgs_w100.tsv`) is available on the [DPR website](https://github.com/facebookresearch/DPR). At the time of writing, the file can be downloaded by cloning the DPR repository and running the following command:

```bash
python data/download_data.py --resource data.wikipedia_split.psgs_w100
```

## Reproducing Experiments

Before you start, you need to download the datasets available on the
[DPR website](https://github.com/facebookresearch/DPR) into `<DPR_DATASET_DIR>`.

The experimental results on the Natural Questions dataset can be reproduced by
running the commands provided in this section. We used a server with 8 NVIDIA
Tesla V100 GPUs with 16GB memory in the experiments. The results on the TriviaQA
dataset can be reproduced by changing the file names of the input dataset to the
corresponding ones (e.g., `nq-train.json` -> `trivia-train.json`).

**1. Building passage database**

```bash
python build_passage_db.py \
    --passage_file=<DPR_DATASET_DIR>/wikipedia_split/psgs_w100.tsv \
    --output_file=<PASSAGE_DB_FILE>
```

**2. Training BPR**

```bash
python train_biencoder.py \
   --gpus=8 \
   --distributed_backend=ddp \
   --train_file=<DPR_DATASET_DIR>/retriever/nq-train.json \
   --eval_file=<DPR_DATASET_DIR>/retriever/nq-dev.json \
   --gradient_clip_val=2.0 \
   --max_epochs=40 \
   --binary
```

**3. Building passage embeddings**

```bash
python generate_embeddings.py \
   --biencoder_file=<BPR_CHECKPOINT_FILE> \
   --output_file=<EMBEDDING_FILE> \
   --passage_db_file=<PASSAGE_DB_FILE> \
   --batch_size=4096 \
   --parallel
```

**4. Evaluating BPR**

```bash
python evaluate_retriever.py \
    --binary_k=1000 \
    --biencoder_file=<BPR_CHECKPOINT_FILE> \
    --embedding_file=<EMBEDDING_FILE> \
    --passage_db_file=<PASSAGE_DB_FILE> \
    --qa_file=<DPR_DATASET_DIR>/retriever/qas/nq-test.csv \
    --parallel
```

**5. Creating dataset for reader**

```bash
python evaluate_retriever.py \
    --binary_k=1000 \
    --biencoder_file=<BPR_CHECKPOINT_FILE> \
    --embedding_file=<EMBEDDING_FILE> \
    --passage_db_file=<PASSAGE_DB_FILE> \
    --qa_file=<DPR_DATASET_DIR>/retriever/qas/nq-train.csv \
    --output_file=<READER_TRAIN_FILE> \
    --top_k=200 \
    --parallel

python evaluate_retriever.py \
    --binary_k=1000 \
    --biencoder_file=<BPR_CHECKPOINT_FILE> \
    --embedding_file=<EMBEDDING_FILE> \
    --passage_db_file=<PASSAGE_DB_FILE> \
    --qa_file=<DPR_DATASET_DIR>/retriever/qas/nq-dev.csv \
    --output_file=<READER_DEV_FILE> \
    --top_k=200 \
    --parallel

python evaluate_retriever.py \
    --binary_k=1000 \
    --biencoder_file=<BPR_CHECKPOINT_FILE> \
    --embedding_file=<EMBEDDING_FILE> \
    --passage_db_file=<PASSAGE_DB_FILE> \
    --qa_file==<DPR_DATASET_DIR>/retriever/qas/nq-test.csv \
    --output_file=<READER_TEST_FILE> \
    --top_k=200 \
    --parallel
```

**6. Training reader**

```bash
python train_reader.py \
   --gpus=8 \
   --distributed_backend=ddp \
   --train_file=<READER_TRAIN_FILE> \
   --validation_file=<READER_DEV_FILE> \
   --test_file=<READER_TEST_FILE> \
   --learning_rate=2e-5 \
   --max_epochs=20 \
   --accumulate_grad_batches=4 \
   --nq_gold_train_file=<DPR_DATASET_DIR>/gold_passages_info/nq_train.json \
   --nq_gold_validation_file=<DPR_DATASET_DIR>/gold_passages_info/nq_dev.json \
   --nq_gold_test_file=<DPR_DATASET_DIR>/gold_passages_info/nq_test.json \
   --train_batch_size=1 \
   --eval_batch_size=2 \
   --gradient_clip_val=2.0
```

**7. Evaluating reader**

```bash
python evaluate_reader.py \
    --gpus=8 \
    --distributed_backend=ddp \
    --checkpoint_file=<READER_CHECKPOINT_FILE> \
    --eval_batch_size=1
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
## Citation

If you find this work useful, please cite the following paper:

```
@inproceedings{yamada2021bpr,
  title={Efficient Passage Retrieval with Hashing for Open-domain Question Answering},
  author={Ikuya Yamada and Akari Asai and Hannaneh Hajishirzi},
  booktitle={ACL},
  year={2021}
}
```

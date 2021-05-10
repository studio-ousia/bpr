import functools
import operator
import random
import re
import string
import unicodedata
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_info
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
    PreTrainedModel,
)

from .reader_dataset import DatasetExample, ReaderDataset


class Reader(LightningModule):
    def __init__(self, hparams: Union[Namespace, dict]):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        self.hparams = hparams
        self.model = self._create_model()
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.base_pretrained_model, use_fast=False)

    def _create_model(self) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(self.hparams.base_pretrained_model)
        base_class = AutoModel.from_config(config).__class__

        class ReaderModel(base_class):
            def __init__(self, config: PretrainedConfig):
                super().__init__(config)
                self.qa_outputs = nn.Linear(config.hidden_size, 2)
                self.qa_classifier = nn.Linear(config.hidden_size, 1)

                self.init_weights()

            def forward(
                self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
            ) -> Tuple[torch.Tensor, ...]:
                outputs = super().forward(
                    input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
                )
                sequence_output = outputs[0]

                start_logits, end_logits = self.qa_outputs(sequence_output).split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)

                classifier_feature = sequence_output[:, 0, :]
                classifier_logits = self.qa_classifier(classifier_feature).squeeze(-1)

                return classifier_logits, start_logits, end_logits

        return ReaderModel.from_pretrained(self.hparams.base_pretrained_model, config=config)

    def prepare_data(self) -> None:
        ReaderDataset.load_dataset(
            self.hparams.train_file, self.hparams.base_pretrained_model, "train", self.hparams.nq_gold_train_file
        )
        ReaderDataset.load_dataset(
            self.hparams.validation_file,
            self.hparams.base_pretrained_model,
            "val",
            getattr(self.hparams, "nq_gold_validation_file", ""),
        )
        ReaderDataset.load_dataset(
            self.hparams.test_file,
            self.hparams.base_pretrained_model,
            "test",
            getattr(self.hparams, "nq_gold_test_file", ""),
        )

    def setup(self, step: str) -> None:
        if step == "test":
            self._train_dataset = []
            self._test_dataset = ReaderDataset.load_dataset(
                self.hparams.test_file,
                self.hparams.base_pretrained_model,
                "test",
                getattr(self.hparams, "nq_gold_test_file", ""),
            )
            rank_zero_info("The number of test examples: %d", len(self._test_dataset))

        else:
            self._train_dataset = ReaderDataset.load_dataset(
                self.hparams.train_file, self.hparams.base_pretrained_model, "train", self.hparams.nq_gold_train_file,
            )
            rank_zero_info("The number of training examples: %d", len(self._train_dataset))

            self._val_dataset = ReaderDataset.load_dataset(
                self.hparams.validation_file,
                self.hparams.base_pretrained_model,
                "val",
                getattr(self.hparams, "nq_gold_validation_file", ""),
            )
            rank_zero_info("The number of validation examples: %d", len(self._val_dataset))

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train", self._train_dataset, self.hparams.train_batch_size)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("val", self._val_dataset, self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test", self._test_dataset, self.hparams.eval_batch_size)

    def _get_dataloader(self, fold: str, dataset: ReaderDataset, batch_size: int) -> DataLoader:
        collate_fn = functools.partial(
            Reader._collate_fn, hparams=self.hparams, fold=fold, max_seq_length=self.hparams.max_seq_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=fold == "train",
            num_workers=self.hparams.num_dataloader_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            worker_init_fn=functools.partial(Reader._init_worker, hparams=self.hparams),
        )

    @classmethod
    def _init_worker(cls, worker_id, hparams: Namespace) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(hparams.base_pretrained_model, use_fast=True)

    @classmethod
    def _collate_fn(
        cls, batch: List[DatasetExample], hparams: Namespace, fold: str, max_seq_length: int
    ) -> Dict[str, torch.Tensor]:
        items = []
        example_indices = []
        passage_labels = []
        passage_mask = []
        start_positions = []
        end_positions = []

        if fold == "train":
            num_passages_per_query = hparams.num_train_passages
        else:
            num_passages_per_query = hparams.num_eval_passages

        for example in batch:
            example_indices.append(example.index)
            query_token_ids = example.query_token_ids

            if fold == "train":
                positive_passages = example.positive_passages
                # https://github.com/facebookresearch/DPR/blob/76db6386a07bd75f567751739b75698727b46115/dpr/data/reader_data.py#L312
                if not example.is_gold_positive and hparams.max_non_gold_positives is not None:
                    positive_passages = positive_passages[: hparams.max_non_gold_positives]
                passages = [random.choice(positive_passages)]

                # https://github.com/facebookresearch/DPR/blob/76db6386a07bd75f567751739b75698727b46115/dpr/data/reader_data.py#L325
                negative_passages = example.other_passages[: hparams.max_negatives]
                passages += list(np.random.permutation(negative_passages)[: num_passages_per_query - 1])
                passage_labels += [1] + [0] * (num_passages_per_query - 1)

            else:
                passages = example.other_passages[:num_passages_per_query]
                passage_labels += [
                    1 if (n < len(passages) and passages[n].has_answer) else 0 for n in range(num_passages_per_query)
                ]

            for n, passage in enumerate(passages):
                passage_token_ids = passage.passage_token_ids
                title_token_ids = passage.title_token_ids

                max_passage_seq_length = (
                    max_seq_length - len(query_token_ids) - len(title_token_ids) - 4
                )  # [CLS] + [SEP] * 3
                passage_token_ids = passage_token_ids[:max_passage_seq_length]
                title_passage_token_ids = title_token_ids + [cls.tokenizer.sep_token_id] + passage_token_ids

                input_ids = cls.tokenizer.build_inputs_with_special_tokens(query_token_ids, title_passage_token_ids)
                token_type_ids = cls.tokenizer.create_token_type_ids_from_sequences(
                    query_token_ids, title_passage_token_ids
                )
                attention_mask = [1] * len(input_ids)
                answer_mask = [0] * (len(input_ids) - len(passage_token_ids) - 1) + [1] * len(passage_token_ids) + [0]
                passage_offset = len(input_ids) - len(passage_token_ids) - 1  # [SEP]

                if fold == "train" and n == 0:
                    answer_spans = [span for span in passage.answer_spans if span[1] <= len(passage_token_ids)]
                    if answer_spans:
                        start_positions.append(answer_spans[0][0] + passage_offset)
                        end_positions.append(answer_spans[0][1] + passage_offset - 1)
                    else:
                        start_positions.append(-1)
                        end_positions.append(-1)

                else:
                    start_positions.append(-1)
                    end_positions.append(-1)

                items.append(
                    dict(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        answer_mask=answer_mask,
                    )
                )
                passage_mask.append(1)

            for _ in range(num_passages_per_query - len(passages)):
                items.append(
                    dict(
                        input_ids=[cls.tokenizer.pad_token_id], token_type_ids=[0], attention_mask=[0], answer_mask=[0],
                    )
                )
                passage_mask.append(0)
                start_positions.append(-1)
                end_positions.append(-1)

        def create_padded_sequence(key: str, items: list) -> torch.Tensor:
            padding_value = 0
            if key == "input_ids":
                padding_value = cls.tokenizer.pad_token_id
            tensors = [torch.tensor(o[key], dtype=torch.long) for o in items]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = {
            k: create_padded_sequence(k, items).view(len(batch), num_passages_per_query, -1) for k in items[0].keys()
        }
        ret["example_indices"] = torch.tensor(example_indices, dtype=torch.long)
        ret["passage_mask"] = torch.tensor(passage_mask, dtype=torch.long).view(len(batch), -1)
        ret["passage_labels"] = torch.tensor(passage_labels, dtype=torch.long).view(len(batch), -1)
        ret["start_positions"] = torch.tensor(start_positions, dtype=torch.long).view(len(batch), -1)
        ret["end_positions"] = torch.tensor(end_positions, dtype=torch.long).view(len(batch), -1)

        return ret

    def forward(self, batch: Dict[str, torch.LongTensor]) -> Tuple[torch.Tensor, ...]:
        return self.model(**batch)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        metrics = self._compute_metrics(batch, batch_idx, self._train_dataset, "train")
        loss = (metrics["classifier_loss"] + metrics["span_loss"]) / 3
        return dict(
            loss=loss,
            log=dict(
                train_loss=loss, train_classifier_loss=metrics["classifier_loss"], train_span_loss=metrics["span_loss"],
            ),
        )

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._compute_metrics(batch, batch_idx, self._val_dataset, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._compute_metrics(batch, batch_idx, self._test_dataset, "test")

    def _compute_metrics(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, examples: List[DatasetExample], fold: str
    ) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        batch_size, num_passages, seq_length = input_ids.size()

        example_indices = batch.pop("example_indices")
        answer_mask = batch.pop("answer_mask")
        passage_mask = batch.pop("passage_mask")
        passage_labels = batch.pop("passage_labels")
        start_positions = batch.pop("start_positions")
        end_positions = batch.pop("end_positions")

        model_outputs = self({k: t.view(-1, seq_length) for k, t in batch.items()})
        ret = {}

        classifier_logits = model_outputs[0].view(batch_size, num_passages)
        classifier_logits = classifier_logits + ((passage_mask - 1) * 10000).type_as(classifier_logits)

        if fold == "train":
            classifier_labels = classifier_logits.new_zeros(batch_size, dtype=torch.long)
            ret["classifier_loss"] = nn.CrossEntropyLoss(reduction="mean")(classifier_logits, classifier_labels)
        else:
            ret["classifier_num_correct"] = torch.gather(
                passage_labels, 1, classifier_logits.argmax(dim=1, keepdim=True)
            ).sum()

        start_logits, end_logits = model_outputs[1:]
        start_logits = start_logits + ((answer_mask.view(-1, seq_length) - 1) * 10000).type_as(start_logits)
        end_logits = end_logits + ((answer_mask.view(-1, seq_length) - 1) * 10000).type_as(end_logits)

        if fold == "train":
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions.view(-1))
            end_loss = loss_fct(end_logits, end_positions.view(-1))
            ret["span_loss"] = start_loss + end_loss

        else:
            selected_passage_indices = classifier_logits.argmax(dim=1)
            span_num_correct = input_ids.new_tensor(0)

            for index in range(batch_size):
                example = examples[example_indices[index].item()]
                gold_answers = frozenset(self._normalize_answer(a) for a in example.answers)
                passage_index = selected_passage_indices[index].item()
                _, start_index, end_index = self._compute_best_answer_spans(
                    input_ids[index, passage_index],
                    answer_mask[index, passage_index],
                    start_logits.view(batch_size, num_passages, -1)[index, passage_index],
                    end_logits.view(batch_size, num_passages, -1)[index, passage_index],
                    top_n=1,
                )[0]

                answer_text = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(input_ids[index, passage_index, start_index : end_index + 1])
                )
                if self._normalize_answer(answer_text) in gold_answers:
                    span_num_correct += 1

            ret["answer_num_correct"] = span_num_correct

        return ret

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._eval_epoch_end(outputs, "val_", len(self._val_dataset))

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._eval_epoch_end(outputs, "test_", len(self._test_dataset))

    def _eval_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], prefix: str, dataset_size: int,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        ret = {}
        for name in ("classifier_num_correct", "answer_num_correct"):
            if name in outputs[0]:
                key = name.split("_")[0] + "_acc"
                num_correct = torch.stack([x[name] for x in outputs]).sum().float()
                if self.use_ddp:
                    num_correct_list = [torch.empty_like(num_correct) for _ in range(dist.get_world_size())]
                    dist.all_gather(num_correct_list, num_correct)
                    num_correct = functools.reduce(operator.add, num_correct_list)
                ret[prefix + key] = num_correct / dataset_size
        return dict(log=ret)

    def configure_optimizers(self) -> Tuple[list, list]:
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_parameters, lr=self.hparams.learning_rate, correct_bias=False)
        num_training_steps = int(
            len(self._train_dataset)
            // (self.hparams.train_batch_size * self.trainer.num_gpus)
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )

        warmup_steps = int(self.hparams.warmup_proportion * num_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )

        return [self.optimizer], [dict(scheduler=self.scheduler, interval="step")]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser, root_dir: str) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument("--train_file", required=True, type=str)
        parser.add_argument("--validation_file", required=True, type=str)
        parser.add_argument("--test_file", required=True, type=str)
        parser.add_argument("--base_pretrained_model", default="bert-base-uncased")
        parser.add_argument("--eval_batch_size", default=64, type=int)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--max_answer_seq_length", default=10, type=int)
        parser.add_argument("--max_negatives", default=200, type=int)
        parser.add_argument("--max_non_gold_positives", default=10, type=int)
        parser.add_argument("--max_seq_length", default=350, type=int)
        parser.add_argument("--nq_gold_train_file", default=None, type=str)
        parser.add_argument("--nq_gold_validation_file", default=None, type=str)
        parser.add_argument("--nq_gold_test_file", default=None, type=str)
        parser.add_argument("--num_dataloader_workers", default=4, type=int)
        parser.add_argument("--num_eval_passages", default=100, type=int)
        parser.add_argument("--num_train_passages", default=24, type=int)
        parser.add_argument("--train_batch_size", default=16, type=int)
        parser.add_argument("--warmup_proportion", default=0.06, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)

        return parser

    def _compute_best_answer_spans(
        self,
        input_ids: torch.Tensor,
        answer_mask: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        top_n: int,
    ) -> List[Tuple[float, int, int]]:
        candidate_spans = [
            (start_logit.item() + end_logit.item(), i, i + j)
            for i, start_logit in enumerate(start_logits)
            for j, end_logit in enumerate(end_logits[i : i + self.hparams.max_answer_seq_length])
        ]
        candidate_spans = sorted(candidate_spans, key=lambda o: o[0], reverse=True)

        selected_spans = []

        def is_subword_id(token_id: int) -> bool:
            return self.tokenizer.convert_ids_to_tokens([token_id])[0].startswith("##")

        for score, start_index, end_index in candidate_spans:
            if start_index == 0 or end_index == 0:  # [CLS]
                continue

            if not all(answer_mask[start_index : end_index + 1]):
                continue

            if start_index > end_index:
                continue

            if any(
                start_index <= selected_start_index <= selected_end_index <= end_index
                or selected_start_index <= start_index <= end_index <= selected_end_index
                for _, selected_start_index, selected_end_index in selected_spans
            ):
                continue

            while (
                is_subword_id(input_ids[start_index].item()) and start_index > 0 and answer_mask[start_index - 1] == 1
            ):
                start_index -= 1

            while (
                is_subword_id(input_ids[end_index + 1].item())
                and end_index < len(answer_mask) - 1
                and answer_mask[end_index + 1] == 1
            ):
                end_index += 1

            selected_spans.append((score, start_index, end_index))
            if len(selected_spans) == top_n:
                break

        return selected_spans

    @staticmethod
    def _normalize_answer(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        text = text.lower()
        text = "".join(c for c in text if c not in frozenset(string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = " ".join(text.split())
        return text

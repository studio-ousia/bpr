import logging
from argparse import ArgumentParser

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.distributed import rank_zero_only
from bpr.reader import Reader

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--num_eval_passages", type=int, default=100)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument("--target_dataset", default="test", choices=["validation", "test"])
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARN, format="[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    )
    logging.getLogger("lightning").setLevel(logging.ERROR)

    # Reader.prepare_data = lambda self: None
    model = Reader.load_from_checkpoint(args.checkpoint_file)
    model.hparams.num_eval_passages = args.num_eval_passages
    model.hparams.eval_batch_size = args.eval_batch_size
    if args.target_dataset == "validation":
        model.hparams.test_file = model.hparams.validation_file
        model.hparams.nq_gold_test_file = model.hparams.nq_gold_validation_file

    trainer = Trainer.from_argparse_args(args)
    result = trainer.test(model)

    def report_results():
        print("result: %s" % result)

    rank_zero_only(report_results)()

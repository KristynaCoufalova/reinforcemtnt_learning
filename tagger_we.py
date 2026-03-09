#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.7.2")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        self._word_embedding = torch.nn.Embedding(len(train.words.string_vocab), args.we_dim)

        rnn_class = torch.nn.LSTM if args.rnn == "LSTM" else torch.nn.GRU
        self._word_rnn = rnn_class(input_size=args.we_dim, hidden_size=args.rnn_dim, batch_first=True, bidirectional=True)

        self._output_layer = torch.nn.Linear(args.rnn_dim, len(train.tags.string_vocab))

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._word_embedding(word_ids)

        lengths = (word_ids != MorphoDataset.PAD).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden, lengths, batch_first=True, enforce_sorted=False)

        packed, _ = self._word_rnn(packed)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)

        hidden = hidden[:, :, :hidden.shape[2] // 2] + hidden[:, :, hidden.shape[2] // 2:]
        hidden = self._output_layer(hidden).transpose(1, 2)
        return hidden

class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        word_ids = torch.tensor(self.dataset.words.string_vocab.indices(example["words"]), dtype=torch.long)
        tag_ids = torch.tensor(self.dataset.tags.string_vocab.indices(example["tags"]), dtype=torch.long)
        return word_ids, tag_ids

    def collate(self, batch):
        word_ids, tag_ids = zip(*batch)
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        return word_ids, tag_ids

def main(args: argparse.Namespace) -> dict[str, float]:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    model = Model(args, morpho.train)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD),
        metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=len(morpho.train.tags.string_vocab), ignore_index=MorphoDataset.PAD)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

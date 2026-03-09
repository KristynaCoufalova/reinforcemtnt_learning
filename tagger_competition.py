#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

import npfl138
npfl138.require_version("2425.7.2")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer
from npfl138.torch_datasets import Dataset


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")


class TaggerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return self.fc(unpacked)


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    morpho = MorphoDataset("czech_pdt")
    train_set = Dataset(morpho.train, batch_size=args.batch_size, shuffle_batches=True)
    dev_set = Dataset(morpho.dev, batch_size=args.batch_size)
    test_set = Dataset(morpho.test, batch_size=args.batch_size)

    model = TaggerModel(
        vocab_size=len(morpho.train.forms),
        embedding_dim=128,
        hidden_dim=256,
        output_dim=len(morpho.train.tags),
        pad_index=morpho.train.forms.pad_id
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=morpho.train.tags.pad_id)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_set:
            forms = torch.tensor(batch.forms, dtype=torch.long, device=device)
            tags = torch.tensor(batch.tags, dtype=torch.long, device=device)
            lengths = torch.tensor(batch.sentence_lens, dtype=torch.int64, device=device)

            optimizer.zero_grad()
            outputs = model(forms, lengths)
            outputs = outputs.view(-1, outputs.shape[-1])
            tags = tags.view(-1)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

        model.eval()
        accs = []
        with torch.no_grad():
            for batch in dev_set:
                forms = torch.tensor(batch.forms, dtype=torch.long, device=device)
                tags = torch.tensor(batch.tags, dtype=torch.long, device=device)
                lengths = torch.tensor(batch.sentence_lens, dtype=torch.int64, device=device)
                outputs = model(forms, lengths)
                predictions = outputs.argmax(dim=-1)
                mask = tags != morpho.train.tags.pad_id
                correct = (predictions == tags) & mask
                accs.append(correct.sum().item() / mask.sum().item())
        print(f"Epoch {epoch+1}, Dev Accuracy: {100 * np.mean(accs):.2f}%")

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        model.eval()
        with torch.no_grad():
            for batch in test_set:
                forms = torch.tensor(batch.forms, dtype=torch.long, device=device)
                lengths = torch.tensor(batch.sentence_lens, dtype=torch.int64, device=device)
                outputs = model(forms, lengths).cpu().numpy()
                for predicted_tags, words in zip(outputs.argmax(axis=-1), batch.forms):
                    for predicted_tag in predicted_tags[:len(words)]:
                        print(morpho.train.tags.string_vocab.string(predicted_tag), file=predictions_file)
                    print(file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

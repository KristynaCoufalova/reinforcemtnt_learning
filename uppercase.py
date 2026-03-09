#!/usr/bin/env python3
#6d87cb3f-bfa6-4769-ab07-f366cef38621
#5ccf1a4b-87c8-4985-a889-c6557b3e55a1
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.3.1")
from npfl138.datasets.uppercase_data import UppercaseData

# Set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=100, type=int, help="Number of most frequent chars to use.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--window", default=5, type=int, help="Window size.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--hidden_size", default=128, type=int, help="Hidden layer size.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Number of CPU threads.")

class BatchGenerator:
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor, batch_size: int, shuffle: bool):
        self._inputs = inputs
        self._outputs = outputs
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __len__(self):
        return (len(self._inputs) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        indices = torch.randperm(len(self._inputs)) if self._shuffle else torch.arange(len(self._inputs))
        while len(indices):
            batch = indices[:self._batch_size]
            indices = indices[self._batch_size:]
            yield self._inputs[batch], self._outputs[batch]

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self.embedding = torch.nn.Embedding(args.alphabet_size, 16)
        self.fc1 = torch.nn.Linear(16 * (2 * args.window + 1), args.hidden_size)
        self.fc2 = torch.nn.Linear(args.hidden_size, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = self.embedding(windows).view(windows.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).squeeze()
        return x

def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    uppercase_data = UppercaseData(args.window, args.alphabet_size, label_dtype=torch.float32)
    train = BatchGenerator(uppercase_data.train.windows, uppercase_data.train.labels, args.batch_size, shuffle=True)
    dev = BatchGenerator(uppercase_data.dev.windows, uppercase_data.dev.labels, args.batch_size, shuffle=False)
    test = BatchGenerator(uppercase_data.test.windows, uppercase_data.test.labels, args.batch_size, shuffle=False)

    model = Model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        correct, total = 0, 0
        for inputs, targets in train:
            optimizer.zero_grad()
            outputs = model(inputs).float()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
        train_acc = correct / total
        print(f"Epoch {epoch + 1}: Loss {epoch_loss:.4f}, Train Acc {train_acc:.4f}")

    os.makedirs(args.logdir, exist_ok=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test:
            outputs = model(inputs)
            preds = (outputs > 0.5).float().tolist()
            predictions.extend(preds)

    test_text = uppercase_data.test.text
    predicted_text = "".join([char.upper() if pred else char for char, pred in zip(test_text, predictions)])
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as f:
        f.write(predicted_text)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

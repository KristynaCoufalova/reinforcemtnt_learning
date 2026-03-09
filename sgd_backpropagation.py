import argparse
import datetime
import os
import re

import numpy as np
import torch
import torch.utils.tensorboard

import npfl138
npfl138.require_version("2425.2")
from npfl138 import MNIST

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class Model(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        self._W1 = torch.nn.Parameter(
            torch.randn(MNIST.C * MNIST.H * MNIST.W, args.hidden_layer) * 0.1,
            requires_grad=True,
        )
        self._b1 = torch.nn.Parameter(torch.zeros(args.hidden_layer))
        
        self._W2 = torch.nn.Parameter(
            torch.randn(args.hidden_layer, MNIST.LABELS) * 0.1, requires_grad=True
        )
        self._b2 = torch.nn.Parameter(torch.zeros(MNIST.LABELS))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(torch.float32) / 255.0
        inputs = inputs.view(inputs.shape[0], -1)
        hidden = torch.tanh(inputs @ self._W1 + self._b1)
        return hidden @ self._W2 + self._b2

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        self.train()
        for batch in dataset.batches(self._args.batch_size, shuffle=True):
            images = batch["images"].to(self._W1.device)
            labels = batch["labels"].to(self._W1.device, dtype=torch.int64)
            
            logits = self.forward(images)
            probabilities = torch.softmax(logits, dim=1)
            loss = -torch.log(probabilities[range(labels.shape[0]), labels]).mean()

            self.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                for param in [self._W1, self._b1, self._W2, self._b2]:
                    param -= self._args.learning_rate * param.grad

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        self.eval()
        correct = 0
        with torch.no_grad():
            for batch in dataset.batches(self._args.batch_size):
                logits = self.forward(batch["images"].to(self._W1.device))
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                correct += np.sum(predictions == batch["labels"].numpy())
        return correct / len(dataset)


def main(args: argparse.Namespace) -> tuple[float, float]:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    mnist = MNIST()
    writer = torch.utils.tensorboard.SummaryWriter(args.logdir)
    model = Model(args)
    
    if torch.cuda.is_available():
        model = model.to(device="cuda")
    elif torch.mps.is_available():
        model = model.to(device="mps")
    elif torch.xpu.is_available():
        model = model.to(device="xpu")
    
    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)
        dev_accuracy = model.evaluate(mnist.dev)
        print(f"Dev accuracy after epoch {epoch + 1} is {100 * dev_accuracy:.2f}", flush=True)
        writer.add_scalar("dev/accuracy", 100 * dev_accuracy, epoch + 1)
    
    test_accuracy = model.evaluate(mnist.test)
    print(f"Test accuracy after epoch {args.epochs} is {100 * test_accuracy:.2f}", flush=True)
    writer.add_scalar("test/accuracy", 100 * test_accuracy, args.epochs)
    return dev_accuracy, test_accuracy

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
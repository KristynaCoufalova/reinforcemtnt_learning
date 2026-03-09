#!/usr/bin/env python3
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
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        self._W1 = torch.nn.Parameter(
            torch.randn(MNIST.C * MNIST.H * MNIST.W, args.hidden_layer) * 0.1,
            requires_grad=True,  # This is the default.
        )
        self._b1 = torch.nn.Parameter(torch.zeros(args.hidden_layer))

        # TODO(sgd_backpropagation): Create the rest of the parameters:
        # - _W2, which is a parameter of size `[args.hidden_layer, MNIST.LABELS]`,
        #   initialized to `torch.randn` value with standard deviation 0.1,
        # - _b2, which is a parameter of size `[MNIST.LABELS]` initialized to zeros.
        self._W2 = torch.nn.Parameter(
            torch.randn(args.hidden_layer, MNIST.LABELS) * 0.1,
            requires_grad=True,
        )
        self._b2 = torch.nn.Parameter(torch.zeros(MNIST.LABELS))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO(sgd_backpropagation): Define the computation of the network. Notably:
        # - start by casting the input uint8 images to float32 using `.to(torch.float32)`,
        # - then divide the tensor by 255 to normalize it to the `[0, 1]` range,
        # - then reshape it to the shape `[inputs.shape[0], -1]`;
        #   the -1 is a wildcard which is computed so that the number
        #   of elements before and after the reshape is preserved,
        # - then multiply it by `self._W1` and then add `self._b1`,
        # - apply `torch.tanh`,
        # - finally, multiply the result by `self._W2` and then add `self._b2`.

        # TODO: In order to support manual gradient computation, you should
        # return not only the output layer, but also the hidden layer after applying
        # tanh, and the input layer after reshaping.
        inputs_float = inputs.to(torch.float32) / 255
        inputs_reshaped = inputs_float.reshape(inputs.shape[0], -1)
        hidden_preact = inputs_reshaped @ self._W1 + self._b1
        hidden = torch.tanh(hidden_preact)
        outputs = hidden @ self._W2 + self._b2
        return inputs_reshaped, hidden, outputs

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        self.train()
        for batch in dataset.batches(self._args.batch_size, shuffle=True):
            # Move data to the correct device
            images = batch["images"].to(self._W1.device)
            labels = batch["labels"].to(self._W1.device)
            
            # Forward pass
            inputs, hidden, logits = self(images)
            
            # Compute softmax probabilities
            probabilities = torch.softmax(logits, dim=1)
            
            # Create one-hot encoding of labels
            batch_size = labels.shape[0]
            one_hot = torch.zeros_like(probabilities)
            for i in range(batch_size):
                one_hot[i, labels[i]] = 1
            
            # Compute gradient of cross-entropy loss w.r.t. logits
            # This is the derivative of cross-entropy loss: p_i - y_i
            delta_output = probabilities - one_hot
            
            # Compute gradients for W2 and b2
            # grad_W2 = h^T @ delta_output
            grad_W2 = torch.zeros_like(self._W2)
            for i in range(batch_size):
                # Outer product for each sample in the batch
                grad_W2 += hidden[i].unsqueeze(1) @ delta_output[i].unsqueeze(0)
            grad_W2 /= batch_size
            
            # grad_b2 = sum(delta_output) / batch_size
            grad_b2 = delta_output.mean(dim=0)
            
            # Backpropagate to hidden layer
            # delta_hidden = (delta_output @ W2^T) * (1 - tanh^2(h))
            delta_hidden = torch.zeros_like(hidden)
            for i in range(batch_size):
                # For each sample, compute the backpropagated error
                back_error = torch.zeros_like(hidden[i])
                for j in range(self._W2.shape[1]):  # For each output class
                    for k in range(self._W2.shape[0]):  # For each hidden neuron
                        back_error[k] += delta_output[i, j] * self._W2[k, j]
                # Apply derivative of tanh
                delta_hidden[i] = back_error * (1 - hidden[i]**2)
            
            # Compute gradients for W1 and b1
            # grad_W1 = x^T @ delta_hidden
            grad_W1 = torch.zeros_like(self._W1)
            for i in range(batch_size):
                # Outer product for each sample in the batch
                grad_W1 += inputs[i].unsqueeze(1) @ delta_hidden[i].unsqueeze(0)
            grad_W1 /= batch_size
            
            # grad_b1 = sum(delta_hidden) / batch_size
            grad_b1 = delta_hidden.mean(dim=0)
            
            # Update parameters with SGD
            with torch.no_grad():
                self._W1 -= self._args.learning_rate * grad_W1
                self._b1 -= self._args.learning_rate * grad_b1
                self._W2 -= self._args.learning_rate * grad_W2
                self._b2 -= self._args.learning_rate * grad_b2

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        self.eval()
        with torch.no_grad():
            # Compute the accuracy of the model prediction
            correct = 0
            for batch in dataset.batches(self._args.batch_size):
                # TODO: Compute the logits of the batch images as in the training,
                # and then convert them to Numpy with `.numpy(force=True)`.
                images = batch["images"].to(self._W1.device)
                _, _, logits = self(images)

                # TODO(sgd_backpropagation): Evaluate how many batch examples were predicted
                # correctly and increase `correct` variable accordingly, assuming
                # the model predicts the class with the highest logit/probability.
                predictions = torch.argmax(logits, dim=1)
                labels = batch["labels"].to(self._W1.device)
                correct += (predictions == labels).sum().item()

            return correct / len(dataset)


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load raw data.
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = torch.utils.tensorboard.SummaryWriter(args.logdir)

    # Create the model
    model = Model(args)

    # Try using an accelerator if available.
    if torch.cuda.is_available():
        model = model.to(device="cuda")
    elif torch.mps.is_available():
        model = model.to(device="mps")
    elif torch.xpu.is_available():
        model = model.to(device="xpu")

    for epoch in range(args.epochs):
        # TODO(sgd_backpropagation): Run the `train_epoch` with `mnist.train` dataset
        model.train_epoch(mnist.train)

        # TODO(sgd_backpropagation): Evaluate the dev data using `evaluate` on `mnist.dev` dataset
        dev_accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * dev_accuracy), flush=True)
        writer.add_scalar("dev/accuracy", 100 * dev_accuracy, epoch + 1)

    # TODO(sgd_backpropagation): Evaluate the test data using `evaluate` on `mnist.test` dataset
    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy), flush=True)
    writer.add_scalar("test/accuracy", 100 * test_accuracy, epoch + 1)

    # Return dev and test accuracies for ReCodEx to validate.
    return dev_accuracy, test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
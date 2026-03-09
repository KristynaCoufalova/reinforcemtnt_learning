#!/usr/bin/env python3
import argparse

import numpy as np
import torch

import npfl138
npfl138.require_version("2425.5")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="5-3-2,10-3-2", type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the implementation.")

class Convolution:
    def __init__(
        self, filters: int, kernel_size: int, stride: int, input_shape: list[int], verify: bool,
    ) -> None:
        self._filters = filters
        self._kernel_size = kernel_size
        self._stride = stride
        self._verify = verify
        self._kernel = torch.nn.Parameter(torch.randn(kernel_size, kernel_size, input_shape[2], filters) * 0.1)
        self._bias = torch.nn.Parameter(torch.zeros(filters))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, in_h, in_w, in_c = inputs.shape
        k = self._kernel_size
        s = self._stride
        out_h = (in_h - k) // s + 1
        out_w = (in_w - k) // s + 1

        # Extract patches
        patches = torch.stack([
            inputs[:, i*s:i*s+k, j*s:j*s+k, :]
            for i in range(out_h) for j in range(out_w)
        ], dim=1)  # Shape: [B, out_h*out_w, k, k, in_c]

        # Reshape for matrix multiplication
        patches = patches.reshape(batch_size, out_h * out_w, -1)  # [B, OH*OW, K*K*in_c]
        kernel = self._kernel.reshape(-1, self._filters)  # [K*K*in_c, out_c]

        out = patches @ kernel  # [B, OH*OW, out_c]
        out = out + self._bias  # broadcast bias
        out = out.reshape(batch_size, out_h, out_w, self._filters)
        out = torch.relu(out)

        if self._verify:
            reference = torch.relu(torch.nn.functional.conv2d(
                inputs.movedim(-1, 1), self._kernel.permute(3, 2, 0, 1), self._bias, self._stride)).movedim(1, -1)
            np.testing.assert_allclose(out.detach().numpy(), reference.detach().numpy(), atol=1e-4,
                                       err_msg="Forward pass differs!")

        return out

    def backward(self, inputs: torch.Tensor, outputs: torch.Tensor, outputs_gradient: torch.Tensor):
        batch_size, in_h, in_w, in_c = inputs.shape
        k = self._kernel_size
        s = self._stride
        out_h = (in_h - k) // s + 1
        out_w = (in_w - k) // s + 1

        d_out = outputs_gradient * (outputs > 0)  # ReLU derivative
        d_out_flat = d_out.reshape(batch_size, out_h * out_w, self._filters)

        patches = torch.stack([
            inputs[:, i*s:i*s+k, j*s:j*s+k, :]
            for i in range(out_h) for j in range(out_w)
        ], dim=1)  # [B, OH*OW, k, k, in_c]
        patches_reshaped = patches.reshape(batch_size, out_h*out_w, -1)

        kernel_gradient = torch.einsum("bio,bik->ok", patches_reshaped, d_out_flat)
        kernel_gradient = kernel_gradient.reshape(self._kernel.shape)

        bias_gradient = d_out_flat.sum(dim=(0, 1))

        kernel_flat = self._kernel.reshape(-1, self._filters)
        d_input_patches = d_out_flat @ kernel_flat.T  # [B, OH*OW, K*K*in_c]
        d_input_patches = d_input_patches.reshape(batch_size, out_h, out_w, k, k, in_c)

        inputs_gradient = torch.zeros_like(inputs)
        for i in range(out_h):
            for j in range(out_w):
                inputs_gradient[:, i*s:i*s+k, j*s:j*s+k, :] += d_input_patches[:, i, j, :, :, :]

        if self._verify:
            with torch.enable_grad():
                inputs.requires_grad_(True)
                inputs.grad = self._kernel.grad = self._bias.grad = None
                reference = (outputs > 0) * torch.nn.functional.conv2d(
                    inputs.movedim(-1, 1), self._kernel.permute(3, 2, 0, 1), self._bias, self._stride).movedim(1, -1)
                reference.backward(gradient=outputs_gradient, inputs=[inputs, self._kernel, self._bias])
                for name, computed, reference in zip(
                        ["Bias", "Kernel", "Inputs"], [bias_gradient, kernel_gradient, inputs_gradient],
                        [self._bias.grad, self._kernel.grad, inputs.grad]):
                    np.testing.assert_allclose(computed.detach().numpy(), reference.detach().numpy(),
                                               atol=2e-4, err_msg=name + " gradient differs!")

        return inputs_gradient, [self._kernel, self._bias], [kernel_gradient, bias_gradient]


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convs = []
        for layer in args.cnn.split(","):
            filters, kernel_size, stride = map(int, layer.split("-"))
            self._convs.append(Convolution(filters, kernel_size, stride, input_shape, args.verify))
            input_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, filters]
        self._classifier = torch.nn.Linear(np.prod(input_shape), MNIST.LABELS)

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for batch in dataset.batches(self._args.batch_size, shuffle=True):
            hidden = batch["images"].to(torch.float32).movedim(1, -1) / 255
            conv_values = [hidden]
            for conv in self._convs:
                hidden = conv.forward(hidden)
                conv_values.append(hidden)
            hidden_flat = torch.flatten(hidden, 1)
            predictions = self._classifier(hidden_flat).softmax(dim=-1)
            one_hot_labels = torch.nn.functional.one_hot(batch["labels"].to(torch.int64), MNIST.LABELS)
            d_logits = (predictions - one_hot_labels) / len(batch["images"])
            variables = [self._classifier.bias, self._classifier.weight]
            gradients = [d_logits.sum(dim=0), d_logits.T @ hidden_flat]
            hidden_gradient = (d_logits @ self._classifier.weight).reshape(hidden.shape)
            for conv, inputs, outputs in reversed(list(zip(self._convs, conv_values[:-1], conv_values[1:]))):
                hidden_gradient, conv_variables, conv_gradients = conv.backward(inputs, outputs, hidden_gradient)
                variables.extend(conv_variables)
                gradients.extend(conv_gradients)
            for variable, gradient in zip(variables, gradients):
                variable -= self._args.learning_rate * gradient

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        total = correct = 0
        for batch in dataset.batches(self._args.batch_size):
            hidden = batch["images"].to(torch.float32).movedim(1, -1) / 255
            for conv in self._convs:
                hidden = conv.forward(hidden)
            hidden = torch.flatten(hidden, 1)
            predictions = self._classifier(hidden)
            correct += torch.sum(predictions.argmax(dim=-1) == batch["labels"])
            total += len(batch["labels"])
        return correct / total


def main(args: argparse.Namespace) -> tuple[float, float]:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()
    torch.set_grad_enabled(False)
    mnist = MNIST(sizes={"train": 5_000})
    model = Model(args)
    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)
        dev_accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * dev_accuracy))
    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy))
    return dev_accuracy, test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
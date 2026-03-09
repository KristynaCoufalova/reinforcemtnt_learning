#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as D

import npfl138
npfl138.require_version("2425.11")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--decoder_layers", default=[500, 500], type=int, nargs="+", help="Decoder layers.")
parser.add_argument("--encoder_layers", default=[500, 500], type=int, nargs="+", help="Encoder layers.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_size", default=None, type=int, help="Limit on the train set size.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
# If you add more arguments, ReCodEx will keep them with your default values.


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        return image, image  # return the image both as the input and the target


class VAE(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = lambda: torch.distributions.Normal(
            torch.zeros(args.z_dim, device=self.device),
            torch.ones(args.z_dim, device=self.device))

        # === ENCODER ===
        encoder_layers = []
        input_dim = MNIST.C * MNIST.H * MNIST.W
        encoder_layers.append(nn.Flatten())
        prev_dim = input_dim
        for units in args.encoder_layers:
            encoder_layers.append(nn.Linear(prev_dim, units))
            encoder_layers.append(nn.ReLU())
            prev_dim = units
        # Final output layer outputs 2*z_dim (mean and log_sd)
        encoder_layers.append(nn.Linear(prev_dim, 2 * args.z_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # === DECODER ===
        decoder_layers = []
        prev_dim = args.z_dim
        for units in args.decoder_layers:
            decoder_layers.append(nn.Linear(prev_dim, units))
            decoder_layers.append(nn.ReLU())
            prev_dim = units
        decoder_layers.append(nn.Linear(prev_dim, MNIST.C * MNIST.H * MNIST.W))
        decoder_layers.append(nn.Sigmoid())
        decoder_layers.append(nn.Unflatten(1, (MNIST.C, MNIST.H, MNIST.W)))
        self.decoder = nn.Sequential(*decoder_layers)

    def train_step(self, xs: tuple[torch.Tensor], y: torch.Tensor) -> dict[str, torch.Tensor]:
        images = xs[0]

        # === ENCODER FORWARD ===
        encoder_output = self.encoder(images)
        z_mean = encoder_output[:, :self._z_dim]
        z_log_sd = encoder_output[:, self._z_dim:]
        z_sd = torch.exp(z_log_sd)

        # === REPARAMETRIZATION TRICK ===
        q_z = D.Normal(z_mean, z_sd)
        z = q_z.rsample()

        # === DECODER FORWARD ===
        decoded_images = self.decoder(z)

        # === RECONSTRUCTION LOSS === (binary cross entropy)
        reconstruction_loss = F.binary_cross_entropy(
            decoded_images, images, reduction="mean"
        )

        # === LATENT LOSS === (KL divergence)
        prior = self._z_prior()
        latent_loss = D.kl.kl_divergence(q_z, prior).mean()

        # === TOTAL LOSS ===
        num_pixels = MNIST.C * MNIST.H * MNIST.W
        loss = reconstruction_loss * num_pixels + latent_loss * self._z_dim

        # === OPTIMIZATION STEP ===
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return the mean of the overall loss, and the current reconstruction and latent losses.
        loss = self.loss_tracker(loss)
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "latent_loss": latent_loss
        }

    def generate(self, epoch: int, logs: dict[str, float]) -> None:
        GRID = 20

        self.decoder.eval()
        with torch.no_grad(), torch.device(self.device):
            # Generate GRIDxGRID images.
            random_images = self.decoder(self._z_prior().sample([GRID * GRID]))

            # Generate GRIDxGRID interpolated images.
            if self._z_dim == 2:
                starts = torch.stack([-2 * torch.ones(GRID), torch.linspace(-2., 2., GRID)], -1)
                ends = torch.stack([2 * torch.ones(GRID), torch.linspace(-2., 2., GRID)], -1)
            else:
                starts, ends = self._z_prior().sample([2, GRID])
            interpolated_z = torch.cat(
                [starts[i] + (ends[i] - starts[i]) * torch.linspace(0., 1., GRID).unsqueeze(-1) for i in range(GRID)])
            interpolated_images = self.decoder(interpolated_z)

            grid = torch.cat([
                torch.cat([torch.cat(list(row), dim=2) for row in torch.chunk(random_images, GRID)], dim=1),
                torch.zeros([MNIST.C, MNIST.H * GRID, MNIST.W]),
                torch.cat([torch.cat(list(row), dim=2) for row in torch.chunk(interpolated_images, GRID)], dim=1),
            ], dim=2)
            self.get_tb_writer("train").add_image("images", grid, epoch)


def main(args: argparse.Namespace) -> float:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    mnist = MNIST(args.dataset, sizes={"train": args.train_size})
    train = TrainableDataset(mnist.train).dataloader(args.batch_size, shuffle=True, seed=args.seed)

    model = VAE(args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        logdir=args.logdir,
    )

    logs = model.fit(train, epochs=args.epochs, callbacks=[VAE.generate])

    return logs["train_loss"]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

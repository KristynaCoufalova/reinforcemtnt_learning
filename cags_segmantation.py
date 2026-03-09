#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2425.5")
from npfl138.datasets.cags import CAGS


# Model decoder for segmentation
class SegmentationDecoder(nn.Module):
    def __init__(self, input_channels=1280):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 256, 2, stride=2),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 14 -> 28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 28 -> 56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 56 -> 112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2),  # 112 -> 224
        )

    def forward(self, x):
        return self.decoder(x)


class Segmenter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.decoder = SegmentationDecoder(1280)

    def forward(self, x):
        x, _ = self.backbone.forward_intermediates(x)
        return self.decoder(x)

    def predict(self, dataloader, device):
        self.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                outputs = self(images)
                masks = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                preds.extend(masks)
        return preds


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=0, type=int)


def collate_fn_train(batch):
    return {
        "image": torch.stack([preprocessing(x["image"]) for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
    }

def collate_fn_dev(batch):
    return {
        "image": torch.stack([preprocessing(x["image"]) for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
    }

def collate_fn_test(batch):
    return {
        "image": torch.stack([preprocessing(x["image"]) for x in batch]),
    }


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global preprocessing
    cags = CAGS(decode_on_demand=False)
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(cags.train, batch_size=args.batch_size, shuffle=True, num_workers=args.threads,
                              collate_fn=collate_fn_train)
    dev_loader = DataLoader(cags.dev, batch_size=args.batch_size, num_workers=args.threads,
                            collate_fn=collate_fn_dev)
    test_loader = DataLoader(cags.test, batch_size=args.batch_size, num_workers=args.threads,
                             collate_fn=collate_fn_test)

    backbone = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0).to(device)
    model = Segmenter(backbone).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    metric = CAGS.MaskIoUMetric()

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        metric.reset()
        with torch.no_grad():
            for batch in dev_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                outputs = model(images)
                metric.update(torch.sigmoid(outputs), masks)
        print(f"Epoch {epoch+1}/{args.epochs}, Dev IoU: {metric.compute():.4f}")

    # Create prediction file
    os.makedirs(args.logdir, exist_ok=True)
    predictions = model.predict(test_loader, device)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        for mask in predictions:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

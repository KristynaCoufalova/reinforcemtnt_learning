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

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=17, type=int, help="Number of epochs.")
parser.add_argument("--freeze_epochs", default=5, type=int, help="Epochs to train with frozen backbone.")
parser.add_argument("--full_lr", default=1e-4, type=float, help="Learning rate for full model training.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

# Define model
class CAGSModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(1280, 34)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

    def predict(self, dataloader, device):
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                outputs = self(images)
                predictions.extend(outputs.cpu().numpy())
        return predictions

# Helper to keep BatchNorm layers in training mode
def set_batchnorm_training(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            m.train()

# Save test predictions
def save_predictions(predictions, path):
    with open(path, "w", encoding="utf-8") as f:
        for prediction in predictions:
            print(np.argmax(prediction), file=f)

# Main
def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    cags = CAGS()
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for sample in cags.train:
        sample["image"] = preprocessing(sample["image"])
    for sample in cags.dev:
        sample["image"] = preprocessing(sample["image"])
    for sample in cags.test:
        sample["image"] = preprocessing(sample["image"])

    # Dataloaders
    train_loader = DataLoader(cags.train, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(cags.dev, batch_size=args.batch_size)
    test_loader = DataLoader(cags.test, batch_size=args.batch_size)

    # Load pretrained backbone
    backbone = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)
    model = CAGSModel(backbone).to(device)

    # Freeze backbone but keep BatchNorm in training mode
    for param in model.backbone.parameters():
        param.requires_grad = False
    set_batchnorm_training(model.backbone)

    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()

        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.full_lr)
            print(f"Unfroze backbone and switched optimizer to full model with lr={args.full_lr}")

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on dev set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in dev_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}: Dev accuracy = {accuracy:.4f}")

        # Save predictions after each epoch
        epoch_path = os.path.join(args.logdir, f"epoch_{epoch + 1}_predictions.txt")
        save_predictions(model.predict(test_loader, device), epoch_path)

    # Final test prediction
    final_path = os.path.join(args.logdir, "cags_classification.txt")
    save_predictions(model.predict(test_loader, device), final_path)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

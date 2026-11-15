"""Train a ResNet18 classifier on the Brain Tumor MRI dataset.

Expected folder layout (Kaggle: "Brain Tumor MRI Dataset" by Masoud Nickparvar):
    data/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
    data/Testing/{glioma,meningioma,notumor,pituitary}/*.jpg

Usage:
    python train.py --data data --epochs 8
"""
import argparse

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets

from utils import CLASSES, build_model, get_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = datasets.ImageFolder(f"{args.data}/Training", get_transform(train=True))
    test_ds = datasets.ImageFolder(f"{args.data}/Testing", get_transform())
    assert train_ds.classes == CLASSES, f"Unexpected classes: {train_ds.classes}"

    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, args.batch, num_workers=2)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in test_dl:
                preds += model(x.to(device)).argmax(1).cpu().tolist()
                labels += y.tolist()
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        print(f"Epoch {epoch}/{args.epochs}  loss={total_loss / len(train_ds):.4f}  test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model.pt")
            best_preds, best_labels = preds, labels

    print(f"\nBest accuracy: {best_acc:.4f} (saved to model.pt)\n")
    print(classification_report(best_labels, best_preds, target_names=CLASSES))


if __name__ == "__main__":
    main()

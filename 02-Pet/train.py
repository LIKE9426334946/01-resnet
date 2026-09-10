import random

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import SimpleUNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0

    for images, masks in dataloader:
        # =========================
        # 1. Move to device
        # =========================

        images = images.to(device)
        masks = masks.to(device)

        # images:
        # (N, 3, H, W)
        #
        # masks:
        # (N, 1, H, W)

        # =========================
        # 2. Clear gradients
        # =========================

        optimizer.zero_grad()

        # =========================
        # 3. Forward
        # =========================

        logits = model(images)

        # logits:
        # (N, 1, H, W)

        # =========================
        # 4. Loss
        # =========================

        loss = criterion(
            logits,
            masks,
        )

        # =========================
        # 5. Backward
        # =========================

        loss.backward()

        # =========================
        # 6. Update parameters
        # =========================

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    return average_loss


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    threshold,
):
    model.eval()

    total_loss = 0.0

    all_tp = []
    all_fp = []
    all_fn = []
    all_tn = []

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        # =========================
        # 1. Forward
        # =========================

        logits = model(images)

        # =========================
        # 2. Loss
        # =========================

        loss = criterion(
            logits,
            masks,
        )

        total_loss += loss.item()

        # =========================
        # 3. Logits -> Probability
        # =========================

        probabilities = torch.sigmoid(logits)

        # =========================
        # 4. SMP get_stats
        # =========================
        #
        # probabilities:
        # (N, 1, H, W)
        #
        # masks:
        # (N, 1, H, W)

        tp, fp, fn, tn = smp.metrics.get_stats(
            probabilities,
            masks.long(),
            mode="binary",
            threshold=threshold,
        )

        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)
        all_tn.append(tn)

    # =========================
    # 合并所有 Batch
    # =========================

    tp = torch.cat(all_tp)
    fp = torch.cat(all_fp)
    fn = torch.cat(all_fn)
    tn = torch.cat(all_tn)

    # =========================
    # SMP Metrics
    # =========================

    iou = smp.metrics.iou_score(
        tp,
        fp,
        fn,
        tn,
        reduction="micro",
    )

    precision = smp.metrics.precision(
        tp,
        fp,
        fn,
        tn,
        reduction="micro",
    )

    recall = smp.metrics.recall(
        tp,
        fp,
        fn,
        tn,
        reduction="micro",
    )

    f1 = smp.metrics.f1_score(
        tp,
        fp,
        fn,
        tn,
        reduction="micro",
    )

    average_loss = total_loss / len(dataloader)

    return {
        "loss": average_loss,
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def main():
    # =========================
    # 1. Load config
    # =========================

    with open(
        "E:\\project\\week1\\02-Pet\\config.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    # =========================
    # 2. Seed
    # =========================

    seed = config["training"]["seed"]

    set_seed(seed)

    # =========================
    # 3. Device
    # =========================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # =========================
    # 4. DataLoader
    # =========================

    train_loader, val_loader = get_dataloaders(config)

    print(f"Train samples: " f"{len(train_loader.dataset)}")

    print(f"Validation samples: " f"{len(val_loader.dataset)}")

    # =========================
    # 5. Model
    # =========================

    model = SimpleUNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
    )

    model = model.to(device)

    # =========================
    # 6. Loss
    # =========================
    #
    # Binary segmentation
    #
    # 所以使用 BCEWithLogitsLoss
    #
    # 注意：
    # 模型最后不要加 sigmoid
    # BCEWithLogitsLoss 内部已经处理

    criterion = nn.BCEWithLogitsLoss()

    # =========================
    # 7. Optimizer
    # =========================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    # =========================
    # 8. Training
    # =========================

    epochs = config["training"]["epochs"]

    threshold = config["metrics"]["threshold"]

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {metrics['loss']:.4f} "
            f"| IoU: {metrics['iou']:.4f} "
            f"| Precision: {metrics['precision']:.4f} "
            f"| Recall: {metrics['recall']:.4f} "
            f"| F1: {metrics['f1']:.4f}"
        )

    # =========================
    # 9. Save Model
    # =========================

    torch.save(
        model.state_dict(),
        "simple_unet.pth",
    )

    print("Model saved: simple_unet.pth")


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class OxfordPetSegmentation:
    def __init__(
        self,
        root,
        split,
        image_size=128,
        download=True,
    ):
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )

        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]

        # =========================
        # 1. 调整图片大小
        # =========================

        image = F.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        # =========================
        # 2. 调整 Mask 大小
        # =========================
        # Mask 必须使用 NEAREST
        # 防止类别值被插值产生小数

        mask = F.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        # =========================
        # 3. Image -> Tensor
        # =========================

        image = F.to_tensor(image)

        # image:
        # (3, H, W)
        # pixel range: [0, 1]

        # =========================
        # 4. Mask -> Tensor
        # =========================

        mask = torch.as_tensor(
            __import__("numpy").array(mask),
            dtype=torch.long,
        )

        # Oxford-IIIT Pet 原始 Mask：
        #
        # 1 = pet
        # 2 = background
        # 3 = border
        #
        # 这里转换成二分类：
        #
        # pet + border -> 1
        # background   -> 0

        mask = (mask != 2).float()

        # 增加 channel 维度
        #
        # (H, W)
        #    ↓
        # (1, H, W)

        mask = mask.unsqueeze(0)

        return image, mask


def get_dataloaders(config):
    root = config["data"]["root"]
    image_size = config["data"]["image_size"]
    download = config["data"]["download"]

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    # 官方 trainval 数据
    train_dataset = OxfordPetSegmentation(
        root=root,
        split="trainval",
        image_size=image_size,
        download=download,
    )

    # 官方 test 数据
    # 这里为了代码简单，直接作为 validation dataset
    val_dataset = OxfordPetSegmentation(
        root=root,
        split="test",
        image_size=image_size,
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader

# CIFAR-10 dataset
# total 60000 32x32 images
# 50000 training images
# 10000 test images

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from model import resnet18


def main():
    # torchrun会为每个进程设置LOCAL_RANK
    local_rank = int(os.environ["LOCAL_RANK"])

    # Linux使用NCCL，Windows使用Gloo
    backend = "gloo" if os.name == "nt" else "nccl"

    # 初始化进程组
    dist.init_process_group(backend=backend)

    # 当前进程绑定到对应GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(23)
    breakpoint()

    # 只让主进程输出一次
    if rank == 0:
        print("starting executing!")
        print(f"GPU数量：{world_size}")

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # 只让rank 0下载数据，避免多个进程同时下载
    if rank == 0:
        CIFAR10(
            root="data",
            train=True,
            transform=transforms,
            download=True,
        )

    # 等待rank 0下载完成
    dist.barrier()

    training_data = CIFAR10(
        root="data",
        train=True,
        transform=transforms,
        download=False,
    )

    # 把训练数据分配给不同进程
    train_sampler = DistributedSampler(
        training_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=32,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True,
    )

    # 使用你在model.py中实现的ResNet-18
    model = resnet18(num_classes=10).to(device)

    # 使用DDP包装模型
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
    )

    num_epochs = 60

    for epoch in range(num_epochs):
        model.train()

        # 确保每个epoch使用不同的打乱顺序
        train_sampler.set_epoch(epoch)

        total_loss = 0.0
        total_samples = 0

        for images, labels in train_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        # 汇总所有GPU的损失和样本数量
        stats_device = device if backend == "nccl" else torch.device("cpu")

        stats = torch.tensor(
            [total_loss, total_samples],
            dtype=torch.float64,
            device=stats_device,
        )

        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        # 只让主进程打印，避免重复输出
        if rank == 0:
            average_loss = stats[0].item() / stats[1].item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], " f"Loss: {average_loss:.4f}")

    # 关闭DDP进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

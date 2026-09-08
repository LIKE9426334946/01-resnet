# CIFAR-10 dataset
# total 60000 32x32 iamges, 6000 images per class
# 50000 training images, 10000 test iamges

print("starting executing!")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.transforms import v2
from torchvision.datasets import CIFAR10

from model import resnet18

torch.manual_seed(23)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

training_data = CIFAR10(root="data", train=True, transform=transforms, download=True)
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

model = resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)

    print(f"Epoch [{epoch+1}/{num_epochs}]," f"Loss:{average_loss:.4f}")

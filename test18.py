import torch

from model import resnet18

# 1. 创建模型
model = resnet18()

# 2. 构造输入：[batch_size, channels, height, width]
x = torch.randn(2, 3, 224, 224)

# 3. 前向传播
output = model(x)

print(output.shape)

from pathlib import Path
import torch
import torch.nn as nn
import argparse
import csv
import json
import matplotlib.pyplot as plt

from torchvision.transforms import v2

base_dir = Path(__file__)
print(base_dir)
resolve_file = base_dir.resolve().parent
print(resolve_file)

img = torch.randint(0, 256, (1, 224, 224), dtype=torch.uint8)
mask = img.squeeze(0)
plt.imshow(mask, cmap="gray")
plt.axis("off")
plt.show()

image = images[0].cpu()
mask = masks[0].cpu().squeeze(0)

# Tensor:
# (3,H,W)
#
# matplotlib:
# (H,W,3)

image = image.permute(1, 2, 0)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Image")
plt.axis("off")


plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.show()

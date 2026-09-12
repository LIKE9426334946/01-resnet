# import torch
# import torch.nn as nn

# loss = nn.BCEWithLogitsLoss(reduction="none")
# input = torch.randn(2, 2)
# print(f"{input=}")

# target = torch.randint(0, 2, size=(2, 2), dtype=torch.float)
# print(f"{target=}")

# output = loss(input, target)
# print(f"{output=}")

import math
import os

import numpy as np

e = math.e

a = 1 / (1 + math.pow(e,-0.7195))
b = -(math.log(a))

print(a)
print(b)
print(os.name)
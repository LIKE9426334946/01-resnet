import numpy as np

label = np.array(
    [
        [0, 1, 2],
        [2, 1, 0],
        [1, 0, 2],
    ]
)

colormap = np.array(
    [
        [0, 0, 1],
        [255, 1, 0],
        [0, 255, 0],
    ],
    dtype=np.uint8,
)

color_image = colormap[label]

print(color_image)
print(color_image.shape)

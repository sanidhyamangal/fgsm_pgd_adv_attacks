"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import random  # for sampling

import numpy as np  # for ops
import torch
from torchvision.transforms.functional import rotate  # to rotate the image


def generated_rotated_image(image_batch, batch_size=64):
    new_image_batch = []
    x = np.hstack([0, 1, 2, 3] * batch_size)
    shuffle_idx = random.sample(range(4 * batch_size), 4 * batch_size)

    for i in range(image_batch.shape[0]):
        new_image_batch.extend([
            rotate(image_batch[i], 0).numpy(),
            rotate(image_batch[i], 1 * 90).numpy(),
            rotate(image_batch[i], 2 * 90).numpy(),
            rotate(image_batch[i], 3 * 90).numpy()
        ])

    new_image_batch = torch.FloatTensor(np.array(new_image_batch)[shuffle_idx])
    angles = torch.IntTensor(x)[shuffle_idx]

    return new_image_batch, angles

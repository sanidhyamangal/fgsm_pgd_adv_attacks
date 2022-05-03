"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import numpy as np  # for ops
import torch
from torchvision.transforms.functional import rotate  # to rotate the image


def generated_rotated_image(image_batch):
    new_image_batch = []
    rotations = np.random.randint(0, 4, size=image_batch.shape[0])

    for i in range(image_batch.shape[0]):
        new_image_batch.append(rotate(image_batch[i], rotations[i] * 90.0), )

    new_image_batch = torch.stack(new_image_batch)
    angles = torch.LongTensor(rotations)

    return new_image_batch, angles

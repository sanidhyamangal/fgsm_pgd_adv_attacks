"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn  # for nn based ops
from torch.optim import Adam
from torchvision import transforms  # for transforming
from torchvision.datasets import MNIST

from models import RotNetModel
from transforms import generated_rotated_image
from utils import DEVICE

loss_history = []
model = RotNetModel()
BATCH_SIZE = 16
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
# define dataloader

mnist_train = MNIST(root="./",
                    train=True,
                    download=True,
                    transform=transforms.Compose([transforms.ToTensor()]))
mnist_test = MNIST(root="./",
                   train=False,
                   download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))

train_dataloader = torch.utils.data.DataLoader(mnist_train,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(mnist_test,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              drop_last=True)

for epoch in range(1):
    for idx, batch in (train_dataloader):
        model.zero_grad()
        batch_images, batch_angles = generated_rotated_image(
            batch[0], batch_size=BATCH_SIZE)
        pred = model(batch_images.to(DEVICE()))
        loss = criterion(pred, batch_angles.to(DEVICE()))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().cpu().numpy())

        if idx % 100 == 0:
            print(f"Epoch {epoch}, Iteration: {idx},Loss: {loss_history[-1]}")

    model.eval()
    with torch.no_grad():
        _accuracy = []

        for test_batch in test_dataloader:
            batch_images, batch_angles = generated_rotated_image(
                test_batch[0], batch_size=BATCH_SIZE)
            pred = model(batch_images.to(DEVICE()))
            labels = torch.argmax(torch.sigmoid(pred), dim=1)

            correct = (labels.flatten() == batch_angles.to(
                DEVICE()).flatten()).sum().item()
            accuracy = correct / labels.shape[0]

            _accuracy.append(accuracy.detach().cpu().numpy())

        print(f"Epoch {epoch}, Val Accu: {np.mean(_accuracy)}")

plt.plot(loss_history)
plt.title("Rotation Pretraining Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("rotnet_pre_trained_1.png")

torch.save(model.feature_learner, "rotnet.pt")

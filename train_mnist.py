"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # for cmatrix development
import torch
import torch.nn as nn  # for nn based ops
from sklearn.metrics import confusion_matrix  # for building cmatrix
from torch.optim import Adam
from torchvision import transforms  # for transforming
from torchvision.datasets import MNIST  # for loading dataset

from models import CNN
from utils import DEVICE

loss_history = []
model = CNN("rotnet.pt", nclasses=10).to(DEVICE())
BATCH_SIZE = 64
optimizer = Adam(model.parameters(), lr=1e-4)
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

# training loop for the mnist data
for epoch in range(5):
    for idx, batch in enumerate(train_dataloader, 0):
        model.zero_grad()
        pred = model(batch[0].to(DEVICE()))
        loss = criterion(pred, batch[1].to(DEVICE()))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().cpu().numpy())

        # print the training logs for the epochs
        if idx % 100 == 0:
            print(f"Epoch {epoch}, Iteration: {idx},Loss: {loss_history[-1]}")

    # validation step after each epoch
    model.eval()
    with torch.no_grad():
        _accuracy = []

        for test_batch in test_dataloader:
            pred = model(batch[0].to(DEVICE()))
            labels = torch.argmax(torch.sigmoid(pred), dim=1)

            correct = (labels.flatten() == batch[1].to(
                DEVICE()).flatten()).sum().item()
            accuracy = correct / labels.shape[0]

            _accuracy.append(accuracy)

        print(f"Epoch {epoch}, Val Accu: {np.mean(_accuracy)}")

# plot the training loss for the training part
plt.plot(loss_history)
plt.title("MNIST Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("mnist_loss.png")

# save the trained model
torch.save(model, "mnist.pt")

# plot the confusion matrix
prediction_labels = []
gt = []

for test_batch in test_dataloader:
    pred = model(batch[0].to(DEVICE()))
    labels = torch.argmax(torch.sigmoid(pred), dim=1)
    prediction_labels.extend(labels.detach().cpu())
    gt.extend(batch[1])

    if len(gt) % BATCH_SIZE == 3:
        break

c_matrix = confusion_matrix(
    torch.stack(prediction_labels).numpy(),
    torch.stack(gt).numpy())
sns.heatmap(c_matrix, annot=True)
plt.savefig("mnist_heatmap.png")

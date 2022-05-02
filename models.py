"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import torch
import torch.nn as nn  # for neural network ops


class RotNetModel(nn.Module):
    """Model to train the rotnet model"""
    def __init__(self, activation=nn.ReLU) -> None:
        super().__init__()

        # define the activation, model arch for feature learner aka encoder
        self.activation = activation()
        model_arch = [
            nn.Conv2d(1, out_channels=16, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(16, out_channels=32, kernel_size=3, stride=2, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, out_channels=64, kernel_size=3, stride=2, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]

        self.feature_learner = nn.Sequential(*model_arch)

        # define the classifier model which would be later pruned
        self.classifier = nn.Sequential(nn.LazyLinear(out_features=512),
                                        self.activation, nn.Dropout(0.4),
                                        nn.Linear(512, 512), self.activation,
                                        nn.Dropout(0.4), nn.Linear(512, 4))

    def forward(self, x):
        z = self.feature_learner(x)
        z = torch.reshape(z, shape=(z.shape[0], -1))
        z = self.classifier(z)

        return z


class CNN(nn.Module):
    def __init__(self,
                 path_feature_learning: str,
                 nclasses: int,
                 activation=nn.ReLU) -> None:
        super().__init__()

        self.activation = activation()
        # use the pretrained feature learner to train the cnn based classifier
        self.feature_learner = torch.load(path_feature_learning,
                                          map_location="cpu")
        # define the feature learner in eval model
        self.feature_learner.eval()

        # freeze the model params for the training loop
        for p in self.feature_learner.parameters():
            p.requires_grad = False

        # define the mlp based classifier for classifying the model
        self.classifier = nn.Sequential(nn.LazyLinear(out_features=256),
                                        self.activation, nn.Dropout(0.4),
                                        nn.Linear(256, 256), self.activation,
                                        nn.Dropout(0.4),
                                        nn.Linear(256, nclasses))

    def forward(self, x):
        z = self.feature_learner(x)
        z = torch.reshape(z, shape=(z.shape[0], -1))
        z = self.classifier(z)

        return z

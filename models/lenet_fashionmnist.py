import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from config import ckp_dir, device


class LeNet(torch.nn.Module):
    """Network architecture from: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model,
                train_data: torchvision.datasets,
                test_data: torchvision.datasets,
                device: torch.device,
                epochs: int = 5,
                criterion: torch.nn = torch.nn.CrossEntropyLoss(),
                evaluate: bool = False):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):

        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Evaluate model!
        test_acc = -1.
        if evaluate:
            predictions, labels = evaluate_model(model, test_data, device)
            test_acc = np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.cpu().numpy())

        print(f"Epoch {epoch + 1}/{epochs} - test accuracy: {(100 * test_acc):.2f}% and CE loss {loss.item():.2f}")

    return model


def evaluate_model(model, data, device):
    """Evaluate torch model."""
    model.eval()
    logits = torch.Tensor().to(device)
    targets = torch.LongTensor().to(device)

    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            logits = torch.cat([logits, model(images)])
            targets = torch.cat([targets, labels])

    return torch.nn.functional.softmax(logits, dim=1), targets


def load_model(train_loader, test_loader):
    ckp_path = os.path.join(ckp_dir, 'lenet_fashionmnist.pt')
    # Load model architecture.
    model = LeNet()

    if os.path.exists(ckp_path):
        print("Load trained LeNet(on Fashion-MNIST)...")
        model.load_state_dict(torch.load(ckp_path, map_location=device))
        return model
    else:
        print("Train LeNet(on Fashion-MNIST)...")
        model = train_model(model=model.to(device),
                            train_data=train_loader,
                            test_data=test_loader,
                            device=device,
                            epochs=5,
                            criterion=torch.nn.CrossEntropyLoss().to(device),
                            evaluate=True)

        # Model to GPU and eval mode.
        model.to(device)
        model.eval()

        # Check test set performance.
        predictions, labels = evaluate_model(model, test_loader, device)
        test_acc = np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.cpu().numpy())
        print(f"Model test accuracy: {(100 * test_acc):.2f}%")

        torch.save(model.state_dict(), ckp_path)

        exit()

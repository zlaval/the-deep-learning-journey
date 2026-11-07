import random

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}\n device: {device}")

BATCH_SIZE = 32


class TrainData:
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    test_data: datasets.FashionMNIST
    class_names: list

    def load_train_data(self):
        train_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
            target_transform=None
        )

        self.test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        self.train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=BATCH_SIZE, shuffle=False)
        self.class_names = train_data.classes


class CNNModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

    def make_predictions(self, data: list):
        pred_probs = []
        self.eval()
        with torch.inference_mode():
            for sample in data:
                sample = torch.unsqueeze(sample, dim=0).to(device)
                pred_logit = self(sample)
                pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
                pred_probs.append(pred_prob.cpu())
        return torch.stack(pred_probs)


class Train:

    def __init__(self, model: CNNModel, data_loader: TrainData):
        self.model = model
        self.data_loader = data_loader
        self.test_dataloader = data_loader.test_dataloader
        self.train_dataloader = data_loader.train_dataloader
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    def train(self):
        torch.manual_seed(42)
        epochs = 5
        for epoch in range(epochs):
            print(f"Epoch: {epoch}\n---------")
            self.train_step()
            self.test_step()

    def train_step(self):
        train_loss, train_acc = 0, 0
        self.model.to(device)
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss
            train_acc += self.accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss /= len(self.train_dataloader)
        train_acc /= len(self.train_dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    def test_step(self):
        test_loss, test_acc = 0, 0
        self.model.to(device)
        self.model.eval()
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                X, y = X.to(device), y.to(device)
                test_pred = self.model(X)
                test_loss += self.loss_fn(test_pred, y)
                test_acc += self.accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            test_loss /= len(self.test_dataloader)
            test_acc /= len(self.test_dataloader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

    def accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc


def main():
    data_loader = TrainData()
    data_loader.load_train_data()

    model = CNNModel(input_shape=1,
                     hidden_units=10,
                     output_shape=len(data_loader.class_names)).to(device)

    train = Train(model=model, data_loader=data_loader)
    train.train()

    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(data_loader.test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    pred_probs = model.make_predictions(test_samples)
    pred_classes = pred_probs.argmax(dim=1)
    print(f"Predicted labels: {pred_classes}")



if __name__ == "__main__":
    main()

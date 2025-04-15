
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import time
import os

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck=False, dropout_rate=0.0):
        super(DenseLayer, self).__init__()
        inter_channels = 4 * growth_rate if bottleneck else in_channels
        self.bottleneck = bottleneck

        if bottleneck:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
            in_channels = inter_channels

        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        out = x
        if self.bottleneck:
            out = self.conv1(self.relu1(self.bn1(out)))
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.dropout:
            out = self.dropout(out)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck=False, dropout_rate=0.0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck, dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super(TransitionLayer, self).__init__()
        out_channels = int(in_channels * compression)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        return self.pool(x)

class DenseNet(nn.Module):
    def __init__(self, config):
        super(DenseNet, self).__init__()

        self.growth_rate = config.get('growth_rate', 12)
        self.num_blocks = config.get('num_blocks', 3)
        self.layers_per_block = config.get('layers_per_block', 4)
        self.init_channels = config.get('init_channels', 24)
        self.compression = config.get('compression', 0.5)
        self.dropout_rate = config.get('dropout_rate', 0.0)
        self.bottleneck = config.get('bottleneck', False)
        self.num_classes = config.get('num_classes', 10)

        self.init_conv = nn.Conv2d(3, self.init_channels, kernel_size=3, padding=1, bias=False)
        channels = self.init_channels
        self.features = nn.Sequential()

        for i in range(self.num_blocks):
            block = DenseBlock(self.layers_per_block, channels, self.growth_rate,
                               bottleneck=self.bottleneck, dropout_rate=self.dropout_rate)
            self.features.add_module(f'dense_block_{i + 1}', block)
            channels += self.layers_per_block * self.growth_rate
            if i != self.num_blocks - 1:
                trans = TransitionLayer(channels, compression=self.compression)
                self.features.add_module(f'transition_{i + 1}', trans)
                channels = int(channels * self.compression)

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, self.num_classes)

        self.train_accs, self.val_accs, self.losses = [], [], []

    def forward(self, x):
        x = self.init_conv(x)
        x = self.features(x)
        x = self.relu(self.bn(x))
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(x)

    def fit(self, train_loader, val_loader, optimizer, criterion, device,
            num_epochs=50, patience=5, verbose=True):

        best_model = copy.deepcopy(self.state_dict())
        best_acc = 0.0
        no_improve = 0

        self.train_accs, self.val_accs, self.losses = [], [], []

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = self(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_train += yb.size(0)
                correct_train += (predicted == yb).sum().item()

            train_acc = correct_train / total_train
            self.train_accs.append(train_acc)
            self.losses.append(running_loss / len(train_loader))

            val_acc = self.evaluate(val_loader, device)
            self.val_accs.append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1:02d} | Loss: {self.losses[-1]:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(self.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"🔁 Early stopping at epoch {epoch+1}")
                    break

        print(f"🏁 Best Val Accuracy: {best_acc:.4f}")

    def evaluate(self, loader, device):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = self(xb)
                _, predicted = torch.max(outputs, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        return correct / total

    def visualize_training(self, save_path="training_plot.png"):
        epochs = range(1, len(self.train_accs) + 1)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.losses, label='Train Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, label='Train Acc')
        plt.plot(epochs, self.val_accs, label='Val Acc')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train vs Val Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
        plt.close()

    def save_model(self, path="best_model.pth"):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="best_model.pth"):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

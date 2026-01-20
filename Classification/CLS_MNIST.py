import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

# ---------------------------------------------------------
# 0. Utilities
# ---------------------------------------------------------
def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(True)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


# ---------------------------------------------------------
# 1. 改为分类器的 ConvNet（保留你原来的 encoder 结构）
# ---------------------------------------------------------
class ConvClassifier(nn.Module):
    def __init__(self, latent_dim=32, activation="relu"):
        super(ConvClassifier, self).__init__()

        act = get_activation(activation)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # output: 8×14×14
            act,
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # output: 16×7×7
            act,
            nn.Flatten()
        )

        # Feature dimension = 32 × 7 × 7 = 1568
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)
        self.classifier = nn.Linear(latent_dim, 10)  # 输出10类

    def forward(self, x):
        h = self.encoder(x)
        z = self.fc(h)
        logits = self.classifier(z)
        return logits


# ---------------------------------------------------------
# 2. Main
# ---------------------------------------------------------
if __name__ == '__main__':

    class Config:
        batch_size = 128
        num_epochs = 50
        lr = 1e-3
        weight_decay = 1e-5
        latent_dim = 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        activation = "relu"
        save_dir = "./results_classifier"

    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 3. Dataset: MNIST
    # ---------------------------------------------------------
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False
    )

    # ---------------------------------------------------------
    # 4. Model + Optimizer
    # ---------------------------------------------------------
    model = ConvClassifier(cfg.latent_dim, cfg.activation).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    # ---------------------------------------------------------
    # 5. Training
    # ---------------------------------------------------------
    for epoch in range(cfg.num_epochs):

        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] Loss={total_loss/len(train_loader):.4f}")

        # -----------------------------------------------------
        # Evaluate every epoch
        # -----------------------------------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(cfg.device)
                labels = labels.to(cfg.device)

                pred = model(imgs).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"  Test Accuracy: {acc:.4f}")

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------
    torch.save(model, f"{cfg.save_dir}/classifier_final.pth")
    print("Training completed.")

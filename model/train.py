from time import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import QuadraticRootPredictor
from utils.data_loaders import SimpleDatasetLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuadraticRootPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
num_epochs = 300

train_dataset = SimpleDatasetLoader("data/x_train.csv", "data/y_train.csv")
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")


def save_model(model, path="weight/model_weights.pth"):
    torch.save(model.state_dict(), path)
    print("Model saved to", path)


if __name__ == "__main__":
    s = time()
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    print(f"Training time: {time() - s:.2f} seconds")
    save_model(model)

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import QuadraticRootPredictor
from utils.data_loaders import SimpleDatasetLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuadraticRootPredictor().to(device)
model_path = "weight/model_weights.pth"



def load_model(model, path=model_path):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))


test_dataset = SimpleDatasetLoader("data/x_test.csv", "data/y_test.csv")
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")


if __name__ == "__main__":
    criterion = nn.MSELoss()
    load_model(model)
    test_model(model, test_loader, criterion)

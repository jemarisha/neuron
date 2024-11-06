import torch

from model import QuadraticRootPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuadraticRootPredictor().to(device)
model_path = "weight/model_weights.pth"


def load_model(model, path=model_path):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    print("Model loaded from", path)


def predict_roots(model, a, b, c):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([[a, b, c]], dtype=torch.float32).to(device)
        predictions = model(inputs)
        return predictions[0].cpu().numpy()


load_model(model)

# x^2 - 5x + 6 = 0 (ожидаемые корни: x1 = 3, x2 = 2)
a, b, c = 1, -5, 6
predicted_roots = predict_roots(model, a, b, c)
print(f"Predicted roots for equation {a}x^2 + ({b})x + ({c}) = 0: x1 = {predicted_roots[0]:.2f}, x2 = {predicted_roots[1]:.2f}")

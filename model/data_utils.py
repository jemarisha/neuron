import os

import pandas as pd
import torch


def generate_data(x_csv_train="data/x_train.csv",
                  y_csv_train="data/y_train.csv",
                  x_csv_test="data/x_test.csv",
                  y_csv_test="data/y_test.csv",
                  data_size=100000):
    os.makedirs(os.path.dirname(x_csv_train), exist_ok=True)

    a = torch.randint(1, 10, (data_size,)).float()
    b = torch.randint(-40, 40, (data_size,)).float()
    c = torch.randint(-20, 20, (data_size,)).float()

    def find_roots(a, b, c):
        discriminant = b ** 2 - 4 * a * c
        root1 = (-b + torch.sqrt(discriminant)) / (2 * a)
        root2 = (-b - torch.sqrt(discriminant)) / (2 * a)
        return root1, root2

    discriminant = b ** 2 - 4 * a * c
    valid_indices = discriminant >= 0
    a, b, c = a[valid_indices], b[valid_indices], c[valid_indices]
    x1, x2 = find_roots(a, b, c)

    x_data = pd.DataFrame({'a': a.numpy(), 'b': b.numpy(), 'c': c.numpy()})
    y_data = pd.DataFrame({'x1': x1.numpy(), 'x2': x2.numpy()})

    train_size = int(0.8 * len(x_data))
    x_data_train, x_data_test = x_data[:train_size], x_data[train_size:]
    y_data_train, y_data_test = y_data[:train_size], y_data[train_size:]

    x_data_train.to_csv(x_csv_train, index=False)
    y_data_train.to_csv(y_csv_train, index=False)
    x_data_test.to_csv(x_csv_test, index=False)
    y_data_test.to_csv(y_csv_test, index=False)


if __name__ == "__main__":
    generate_data()

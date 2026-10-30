"""Drawing helpers for PyTorch Episode 4.

These functions only draw pictures. 
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def show_paths(dataset, batch_sizes=(1, 32, 10_000), updates=300, lr=0.1):
    """Run the race again, note w and b after every update, and draw each path on the loss map.

    Same start (w = 0, b = 0), same seed and same learning rate as the race in the notebook.
    """
    x, y = dataset.tensors
    xs, ys = x.numpy().ravel().astype(np.float64), y.numpy().ravel().astype(np.float64)
    W, B = np.meshgrid(np.linspace(-0.3, 3.8, 160), np.linspace(-0.3, 2.8, 160))
    # mean((w*x + b - y)^2) for every (w, b) on the grid, from a few averages of the data
    loss_map = (W**2 * np.mean(xs**2) + B**2 + np.mean(ys**2) + 2 * W * B * np.mean(xs)
                - 2 * W * np.mean(xs * ys) - 2 * B * np.mean(ys))

    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(5 * len(batch_sizes), 4.6), sharey=True)
    for ax, bs in zip(axes, batch_sizes):
        torch.manual_seed(0)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        model = nn.Linear(1, 1)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        path = [(0.0, 0.0)]
        while len(path) <= updates:
            for xb, yb in loader:
                loss = nn.functional.mse_loss(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                path.append((model.weight.item(), model.bias.item()))
                if len(path) > updates:
                    break
        ws, bs_ = zip(*path)
        ax.contour(W, B, loss_map, levels=np.geomspace(0.03, 8, 14), colors="#999999", linewidths=0.6)
        ax.plot(ws, bs_, "-", lw=1, color="#d33333")
        ax.plot(ws[-1], bs_[-1], "o", color="#d33333")
        ax.plot(3, 2, "*", ms=14, color="#2a9d2a")
        name = "full batch (10,000)" if bs == len(x) else f"batch size {bs}"
        ax.set(title=f"{name}: first {updates} updates", xlabel="w")
    axes[0].set_ylabel("b")
    fig.tight_layout()
    plt.show()


def show_predictions(model, X, title):
    """Colour every point by the class the model predicts (blue = 0, red = 1)."""
    with torch.no_grad():
        predicted = (model(X) > 0).float().flatten()
    share = predicted.mean().item()
    plt.figure(figsize=(5.5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=predicted, cmap="coolwarm", vmin=0, vmax=1, s=6)
    plt.title(f"{title}: model says class 1 for {share:.0%} of points")
    plt.tight_layout()
    plt.show()

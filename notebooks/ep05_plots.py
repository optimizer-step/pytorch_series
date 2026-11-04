"""Drawing helpers for PyTorch Episode 5.

These functions only draw pictures. Everything that learns is in the notebook.
Keep this file next to the notebook, so `from ep05_plots import ...` finds it.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

COLORS = ["#3b7fc4", "#e8862a", "#3a9d5d", "#8e6cc9", "#c9a227"]
LETTERS = "ABCDE"


def show_probs(probs, options):
    """Bar chart of the probabilities, with the percentage on top of each bar."""
    probs = [float(p) for p in probs]
    plt.figure(figsize=(6, 3.8))
    bars = plt.bar(options, probs, color=COLORS[:len(options)])
    for bar, p in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.1%}", ha="center", fontsize=13)
    plt.ylim(0, 1.1)
    plt.ylabel("probability")
    plt.title(f"After softmax: they add up to {sum(probs):.0%}")
    plt.tight_layout()
    plt.show()


def show_penalty():
    """The cross-entropy penalty, -log(p), where p is how sure the model was of the right answer."""
    p = np.linspace(0.005, 1, 400)
    plt.figure(figsize=(7, 4.2))
    plt.plot(p, -np.log(p), color="#16181d", lw=2)
    # where each label sits, in points from its dot (the two on the right would overlap)
    offsets = {0.99: (-4, 34), 0.9: (-14, 14), 0.5: (8, 4), 0.26: (8, 4), 0.1: (8, 4), 0.01: (8, 4)}
    for q, offset in offsets.items():
        plt.plot(q, -math.log(q), "o", color=COLORS[1] if q == 0.26 else COLORS[0], ms=8)
        plt.annotate(f"{q:.0%} → {-math.log(q):.2f}", (q, -math.log(q)), textcoords="offset points",
                     xytext=offset, fontsize=10, ha="right" if q >= 0.9 else "left")
    plt.xlabel("how sure the model was of the RIGHT answer")
    plt.ylabel("penalty = −log(p)")
    plt.title("Sure and right: almost free. Sure and wrong: very expensive.")
    plt.ylim(0, 5.4)
    plt.tight_layout()
    plt.show()


def show_spirals(X, y):
    """Scatter the points, one colour per class."""
    plt.figure(figsize=(5, 5))
    for c in range(int(y.max()) + 1):
        pts = X[y == c]
        plt.scatter(pts[:, 0], pts[:, 1], s=4, color=COLORS[c], label=f"class {c} ({LETTERS[c]})")
    plt.legend(loc="upper right", fontsize=8, markerscale=3)
    plt.gca().set_aspect("equal")
    plt.title(f"{len(y):,} points, {int(y.max()) + 1} classes")
    plt.tight_layout()
    plt.show()


def show_regions(model, X, y, title):
    """Colour the background by the class the model picks, and draw the points on top."""
    k = int(y.max()) + 1
    g = np.linspace(-1.15, 1.15, 300)
    gx, gy = np.meshgrid(g, g)
    grid = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        picked = model(grid).argmax(dim=1).numpy().reshape(gx.shape)
        correct = (model(X).argmax(dim=1) == y).float().mean().item()
    plt.figure(figsize=(5, 5))
    plt.contourf(gx, gy, picked, levels=np.arange(k + 1) - 0.5, cmap=ListedColormap(COLORS[:k]), alpha=0.25)
    for c in range(k):
        pts = X[y == c]
        plt.scatter(pts[:, 0], pts[:, 1], s=5, color=COLORS[c])
    plt.gca().set_aspect("equal")
    plt.title(f"{title}: {correct:.1%} correct")
    plt.tight_layout()
    plt.show()

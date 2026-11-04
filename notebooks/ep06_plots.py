"""Plotting helpers for PyTorch Episode 6: drawing only, no training."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

BLUE, ORANGE = "#3b7fc4", "#e8862a"


def _points(X, y, wrong):
    cls = y.flatten()
    for c, colour in ((0, BLUE), (1, ORANGE)):
        pts = X[cls == c]
        plt.scatter(pts[:, 0], pts[:, 1], s=28, color=colour, edgecolor="white", lw=0.5, zorder=3)
    if wrong is not None:
        bad = X[wrong]
        plt.scatter(bad[:, 0], bad[:, 1], s=170, facecolors="none", edgecolors="#16181d", lw=1.6, zorder=4)


def show_book(X, y, wrong):
    """The practice book: 100 points, with the misprinted answers circled."""
    plt.figure(figsize=(6, 4.4))
    _points(X, y, wrong)
    plt.title(f"Practice book: {len(y)} questions, {len(wrong)} misprinted answers (circled)")
    plt.tight_layout()
    plt.show()


def show_history(history):
    """Practice-book loss and mock-test loss after every epoch, with the best mock test marked."""
    practice, mock = history["practice"], history["mock"]
    epochs = np.arange(1, len(mock) + 1)
    best = int(np.argmin(mock)) + 1
    plt.figure(figsize=(8, 4.4))
    plt.plot(epochs, practice, color=BLUE, lw=2, label="practice book (train)")
    plt.plot(epochs, mock, color=ORANGE, lw=2, label="mock test (validation)")
    plt.axvline(best, color="#16181d", ls="--", lw=1)
    plt.plot(best, mock[best - 1], "o", color="#16181d", ms=8, zorder=5)
    plt.annotate(f"best mock test: epoch {best}", (best, mock[best - 1]), textcoords="offset points",
                 xytext=(10, 28), fontsize=11)
    plt.axvspan(best, len(mock), color="#d33333", alpha=0.06)
    plt.text(len(mock) * 0.97, max(mock) * 0.93, "learning the misprints", ha="right", color="#b02a2a", fontsize=11)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0, max(max(mock), max(practice)) * 1.05)
    plt.legend(loc="upper left")
    plt.title("Practice keeps improving. The mock test doesn't.")
    plt.tight_layout()
    plt.show()


def show_sizes(results):
    """Practice-book and lab-exam accuracy for every model size, from (width, knobs, practice, lab) rows."""
    widths, knobs, practice, lab = zip(*results)
    best = int(np.argmax(lab))
    plt.figure(figsize=(8, 4.4))
    plt.plot(knobs, practice, "o-", color=BLUE, lw=2, label="practice book")
    plt.plot(knobs, lab, "o-", color=ORANGE, lw=2, label="lab exam (1,000 new questions)")
    for k, w, a in zip(knobs, widths, lab):
        plt.annotate(str(w), (k, a), textcoords="offset points", xytext=(0, -17), ha="center", fontsize=10)
    plt.plot(knobs[best], lab[best], "o", color="#16181d", ms=10, zorder=5)
    plt.annotate(f"best: {widths[best]} neurons", (knobs[best], lab[best]), textcoords="offset points",
                 xytext=(0, 14), ha="center", fontsize=11)
    plt.text(knobs[0], 1.03, "too small", color="#555b66", fontsize=11)
    plt.text(knobs[-1], 1.03, "too big", color="#b02a2a", fontsize=11, ha="right")
    plt.xscale("log")
    plt.ylim(0.4, 1.08)
    plt.xlabel("knobs (log scale) · numbers = neurons per layer")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.title("Too small, just right, too big")
    plt.tight_layout()
    plt.show()


def show_boundary(model, X, y, wrong=None, title=""):
    """Colour the background by the model's answer (in exam mode), then draw the practice book on top."""
    g = np.linspace(-2.0, 3.0, 300), np.linspace(-1.6, 2.1, 240)
    gx, gy = np.meshgrid(*g)
    grid = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        says = (model(grid) > 0).float().numpy().reshape(gx.shape)
        practice = ((model(X) > 0).float() == y).float().mean().item()
    model.train(was_training)
    plt.figure(figsize=(6, 4.4))
    plt.contourf(gx, gy, says, levels=[-0.5, 0.5, 1.5], cmap=ListedColormap([BLUE, ORANGE]), alpha=0.22)
    _points(X, y, wrong)
    plt.title(f"{title}: {practice:.0%} on the practice book")
    plt.tight_layout()
    plt.show()
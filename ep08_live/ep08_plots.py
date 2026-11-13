"""Plotting helpers for PyTorch Episode 8.

load_drawing() turns a drawing into a 28 x 28 photo in the Fashion-MNIST style,
open_in_paint() opens a drawing in Paint (a blank one if asked), and act() and
the reveal helpers log the time of each big moment to Videos/clip_marks/ep08.csv.
"""
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageFilter

BLUE, ORANGE, RED, GREEN = "#3b7fc4", "#e8862a", "#d33333", "#3a9d5d"
GOLD, GREY, INK, PALE = "#f2b705", "#8a919c", "#16181d", "#c9d6e8"
DARK, NEON, HOT = "#0f1115", "#39d98a", "#ff5a5a"

_MARKS = Path.home() / "Videos" / "clip_marks" / "ep08.csv"
_QUIET = os.environ.get("OS_QUIET") == "1"  # test runs: no marks, no pauses


def _mark(moment):
    """Notes the clock time of a big moment (one line in a CSV file)."""
    if _QUIET:
        return
    try:
        _MARKS.parent.mkdir(parents=True, exist_ok=True)
        new = not _MARKS.exists()
        with open(_MARKS, "a", encoding="utf-8") as f:
            if new:
                f.write("time,moment\n")
            f.write(f"{datetime.now().isoformat(timespec='seconds')},{moment}\n")
    except OSError:
        pass


def _pause(seconds):
    if not _QUIET:
        time.sleep(seconds)


def _frames(n):
    return 1 if _QUIET else n


def _as_float(photo):
    """uint8 photos (0 to 255) become floats (0 to 1), on the CPU."""
    photo = photo.detach().cpu()
    return photo.float() / 255 if photo.dtype == torch.uint8 else photo.float()


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def _device(model):
    return next(model.parameters()).device


def _show(fig, handle=None):
    """Shows a figure, or replaces the one already on screen (handle)."""
    if handle is None:
        handle = display(fig, display_id=True)
    else:
        handle.update(fig)
    plt.close(fig)
    return handle


def _ease(t):
    return 1 - (1 - t) ** 3


# ---------------------------------------------------------------- acts and reveals

def act(number, title, line=""):
    """A full-width card that opens an act."""
    _mark(f"ACT {number} · {title}")
    fig = plt.figure(figsize=(16, 4.2), facecolor=DARK)
    fig.add_axes([0, 0, 1, 1]).axis("off")
    fig.text(0.5, 0.70, f"ACT {number}", ha="center", color=GOLD, fontsize=30, fontweight="bold")
    fig.text(0.5, 0.40, title.upper(), ha="center", color="white", fontsize=46, fontweight="bold")
    if line:
        fig.text(0.5, 0.14, line, ha="center", color=PALE, fontsize=19)
    _show(fig)


def reveal(label, value, note="", start=0.0, percent=True, colour=NEON):
    """One big number, counting up in the middle of a dark card."""
    _mark(f"REVEAL · {label}")
    handle = None
    n = _frames(30)
    for i in range(1, n + 1):
        v = start + (value - start) * _ease(i / n)
        fig = plt.figure(figsize=(16, 5.6), facecolor=DARK)
        fig.add_axes([0, 0, 1, 1]).axis("off")
        fig.text(0.5, 0.80, label, ha="center", color=PALE, fontsize=24)
        text = f"{v:.1%}" if percent else f"{int(round(v)):,}"
        fig.text(0.5, 0.36, text, ha="center", va="center", color=colour, fontsize=120,
                 fontweight="bold")
        if note and i == n:
            fig.text(0.5, 0.07, note, ha="center", color="white", fontsize=19)
        handle = _show(fig, handle)
        _pause(0.045)


def reveal_board(rows, title, target=None):
    """Bars that grow together: rows = [(name, value, colour), ...], values from 0 to 1."""
    _mark(f"REVEAL · {title}")
    handle = None
    n = _frames(30)
    for i in range(1, n + 1):
        t = _ease(i / n)
        fig, ax = plt.subplots(figsize=(16, 1.6 + 1.25 * len(rows)), facecolor=DARK)
        ax.set_facecolor(DARK)
        for k, (name, value, colour) in enumerate(rows):
            ax.barh(k, value * t, color=colour, height=0.62)
            ax.text(value * t + 0.008, k, f"{value * t:.1%}", va="center", color="white",
                    fontsize=30, fontweight="bold")
            ax.text(-0.01, k, name, va="center", ha="right", color="white", fontsize=20)
        if target is not None:
            ax.axvline(target, color=GOLD, ls="--", lw=2)
        ax.set_xlim(0, 1.18)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.axis("off")
        ax.set_title(title, color=PALE, fontsize=22, pad=14)
        fig.subplots_adjust(left=0.27, right=0.97, top=0.80, bottom=0.05)
        handle = _show(fig, handle)
        _pause(0.045)


# ---------------------------------------------------------------- photos

def show_photos(images, labels=None, classes=None, titles=None, cmap="gray_r"):
    """Up to 10 photos in a row, numbered, named, or with your own titles."""
    images = images[:10]
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(1.75 * n, 2.75 if labels is not None else 2.35),
                             squeeze=False)
    for i, ax in enumerate(axes[0]):
        img = _as_float(images[i]).reshape(28, 28)
        ax.imshow(img, cmap=cmap, vmin=0, vmax=max(img.max().item(), 1e-6))
        if titles is not None:
            ax.set_title(titles[i], fontsize=11)
        elif labels is not None:
            k = int(labels[i])
            ax.set_title(f"{k}\n{classes[k]}" if classes else str(k), fontsize=11)
        else:
            ax.set_title(str(i + 1), fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_chances(model, photo, classes, label=None):
    """The photo, and the model's chance for each of the 10 shelves."""
    photo = _as_float(photo).reshape(28, 28)
    with torch.no_grad():
        scores = model(photo.reshape(1, 1, 28, 28).to(_device(model)))
        chances = torch.softmax(scores, dim=1)[0].cpu()
    pick = int(chances.argmax())
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 2.3]})
    a.imshow(photo, cmap="gray_r", vmin=0, vmax=max(photo.max().item(), 1e-6))
    a.axis("off")
    if label is not None:
        k = int(label)
        mark = "right shelf" if k == pick else "wrong shelf"
        a.set_title(f"label: {classes[k]}\n({mark})", fontsize=13, color=GREEN if k == pick else RED)
    colours = [BLUE if k == pick else PALE for k in range(10)]
    b.barh(range(10), chances.tolist(), color=colours)
    b.set_yticks(range(10), classes, fontsize=11)
    b.invert_yaxis()
    b.set_xlim(0, 1.14)
    for k in range(10):
        b.text(chances[k].item() + 0.01, k, f"{chances[k].item():.1%}", va="center", fontsize=10)
    b.set_title(f"model says: {classes[pick]}, {chances[pick].item():.1%} sure", fontsize=14)
    b.set_xlabel("chance (softmax of the 10 scores)", fontsize=11)
    for side in ("top", "right"):
        b.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()


def show_slides(photo, shift, steps=7):
    """One photo moved 0, 1, 2... pixels to the right by the shift function given."""
    photo = _as_float(photo).reshape(1, 1, 28, 28)
    fig, axes = plt.subplots(1, steps, figsize=(1.9 * steps, 2.6))
    for k, ax in enumerate(axes):
        moved = shift(photo, k)[0, 0]
        ax.imshow(moved, cmap="gray_r", vmin=0, vmax=1)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 28, 28, fill=False, edgecolor=GREY, lw=1))
        ax.set_title(f"{k} pixel{'s' if k != 1 else ''} right", fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- the small window

def show_window(photo, kernel, feature_map, title="One small window, slid over the whole photo"):
    """The 3 x 3 window sweeping the photo, and the map it fills in, row by row."""
    photo = _as_float(photo).reshape(28, 28)
    k = kernel.detach().cpu().reshape(3, 3)
    fmap = feature_map.detach().cpu().reshape(28, 28)
    limit = max(fmap.abs().max().item(), 1e-6)
    stops = list(range(0, 28, 2)) if not _QUIET else [27]
    handle = None
    for r in stops + [27]:
        c = 13
        fig, (a, b, m) = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1, 0.62, 1]})
        a.imshow(photo, cmap="gray_r", vmin=0, vmax=1)
        if r < 27:
            a.add_patch(plt.Rectangle((c - 1.5, r - 1.5), 3, 3, fill=False, edgecolor=RED, lw=3))
        a.set_title("the photo", fontsize=14)
        a.axis("off")
        b.imshow(k, cmap="RdBu_r", vmin=-max(k.abs().max().item(), 1e-6), vmax=max(k.abs().max().item(), 1e-6))
        for i in range(3):
            for j in range(3):
                b.text(j, i, f"{k[i, j].item():g}", ha="center", va="center", fontsize=20, fontweight="bold")
        b.set_title("the window: 9 numbers", fontsize=14)
        _clean(b)
        shown = fmap.clone()
        if r < 27:
            shown[r + 1:] = 0
        m.imshow(shown, cmap="RdBu_r", vmin=-limit, vmax=limit)
        m.set_title("what the window found", fontsize=14)
        m.axis("off")
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        handle = _show(fig, handle)
        _pause(0.08)


def show_pool(feature_map):
    """A feature map, and the same map after max pooling: the biggest of every 2 x 2."""
    fmap = feature_map.detach().cpu().reshape(28, 28)
    pooled = torch.nn.functional.max_pool2d(fmap.reshape(1, 1, 28, 28), 2)[0, 0]
    limit = max(fmap.abs().max().item(), 1e-6)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.2))
    a.imshow(fmap, cmap="RdBu_r", vmin=-limit, vmax=limit)
    a.set_title("before pooling: 28 x 28", fontsize=14)
    b.imshow(pooled, cmap="RdBu_r", vmin=-limit, vmax=limit)
    b.set_title("after pooling: 14 x 14\nthe biggest of every 2 x 2 block", fontsize=14)
    for ax in (a, b):
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_cnn(model, compare=None):
    """The CNN as a row of boxes: what goes in, what comes out of each layer, and its knobs."""
    device = _device(model)
    x = torch.zeros(1, 1, 28, 28, device=device)
    stages = [("photo", tuple(x.shape[1:]), 0)]
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            knobs = sum(p.numel() for p in layer.parameters())
            stages.append((type(layer).__name__, tuple(x.shape[1:]), knobs))
    fig, ax = plt.subplots(figsize=(16, 4.6))
    n = len(stages)
    for i, (name, shape, knobs) in enumerate(stages):
        colour = {"Conv2d": BLUE, "MaxPool2d": ORANGE, "Linear": GREEN}.get(name, GREY)
        ax.add_patch(plt.Rectangle((i - 0.42, 0.25), 0.84, 0.5, color=colour, alpha=0.18))
        ax.add_patch(plt.Rectangle((i - 0.42, 0.25), 0.84, 0.5, fill=False, edgecolor=colour, lw=2))
        ax.text(i, 0.62, name, ha="center", fontsize=12, fontweight="bold", color=colour)
        ax.text(i, 0.44, " x ".join(str(s) for s in shape), ha="center", fontsize=11, color=INK)
        if knobs:
            ax.text(i, 0.13, f"{knobs:,} knobs", ha="center", fontsize=11, color=colour, fontweight="bold")
        if i < n - 1:
            ax.annotate("", xy=(i + 0.58, 0.5), xytext=(i + 0.42, 0.5),
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.5))
    total = sum(p.numel() for p in model.parameters())
    title = f"{total:,} knobs in all"
    if compare:
        title += f"   |   Episode 7's shop model: {compare:,}"
    ax.set_title(title, fontsize=16)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 0.85)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- training

_history = {"epoch": [], "practice": [], "mock": []}


def live_chart(epoch, practice, mock, patience=5):
    """Redraws the practice-book and mock-test curves after every epoch."""
    if epoch == 1:
        for values in _history.values():
            values.clear()
    _history["epoch"].append(epoch)
    _history["practice"].append(practice)
    _history["mock"].append(mock)
    epochs, practices, mocks = _history["epoch"], _history["practice"], _history["mock"]
    best = max(range(len(mocks)), key=lambda i: (mocks[i], -i))   # first best mock test
    bad = epoch - epochs[best]
    clear_output(wait=True)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(epochs, practices, "-o", color=BLUE, lw=2.2, label="practice book (training set)")
    ax.plot(epochs, mocks, "-o", color=ORANGE, lw=2.2, label="mock test (validation set)")
    ax.scatter([epochs[best]], [mocks[best]], s=380, marker="*", color=GOLD, edgecolor=INK,
               zorder=5, label=f"best mock test so far: {mocks[best]:.3f} (epoch {epochs[best]})")
    ax.set_xlim(0.5, max(20, epoch + 1) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    low = min(min(practices), min(mocks))
    ax.set_ylim(min(0.8, low - 0.02), 1.0)
    ax.set_xlabel("epoch", fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    if bad >= patience:
        status = f"{patience} bad mock tests in a row: STOP, go back to epoch {epochs[best]}"
    else:
        status = "bad mock tests in a row: " + "●" * bad + "○" * (patience - bad)
    ax.set_title(f"epoch {epoch}   |   practice {practice:.3f}   |   mock test {mock:.3f}\n{status}",
                 fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    display(fig)
    plt.close(fig)


_board = {"handle": None}


def leaderboard(results, target, target_knobs, final=False):
    """The race to the bottom: every model's board exam against the score to beat.
    results = [(knobs, score), ...]. The newest bar grows in."""
    if len(results) == 1:
        _board["handle"] = None
    if final:
        _mark("REVEAL · race to the bottom")
    n = _frames(20)
    for i in range(1, n + 1):
        t = _ease(i / n)
        fig, ax = plt.subplots(figsize=(16, 2.2 + 1.1 * (len(results) + 1)), facecolor=DARK)
        ax.set_facecolor(DARK)
        rows = [(f"Episode 7 · {target_knobs:,} knobs", target, GREY)]
        for knobs, score in results:
            rows.append((f"CNN · {knobs:,} knobs", score, NEON if score > target else HOT))
        for k, (name, score, colour) in enumerate(rows):
            grow = t if k == len(rows) - 1 and k > 0 else 1.0
            ax.barh(k, score * grow, color=colour, height=0.62)
            ax.text(score * grow + 0.004, k, f"{score * grow:.1%}", va="center", color="white",
                    fontsize=24, fontweight="bold")
            ax.text(0.795, k, name, va="center", ha="right", color="white", fontsize=18)
        ax.axvline(target, color=GOLD, ls="--", lw=2, zorder=0)
        ax.set_xlim(0.80, 0.94)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.axis("off")
        winners = [kn for kn, s in results if s > target]
        title = "The race to the bottom: how small can a CNN be, and still win?"
        if final and winners:
            title = f"Smallest winner: {min(winners):,} knobs, {target_knobs / min(winners):.0f}x smaller"
        ax.set_title(title, color=PALE, fontsize=20, pad=12)
        fig.subplots_adjust(left=0.30, right=0.97, top=0.82, bottom=0.04)
        _board["handle"] = _show(fig, _board["handle"])
        _pause(0.04)


# ---------------------------------------------------------------- answer sheets

def show_shift_curve(curves):
    """Mock-test accuracy as every photo slides 0, 1, 2... pixels. curves = {name: [scores]}."""
    colours = [RED, BLUE, GREEN, ORANGE]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for (name, scores), colour, dy in zip(curves.items(), colours, [-0.07, 0.035, 0.035, 0.035]):
        ks = list(range(len(scores)))
        ax.plot(ks, scores, "-o", color=colour, lw=2.6, ms=8, label=name)
        for k, v in zip(ks, scores):
            ax.text(k, v + dy, f"{v:.0%}", ha="center", color=colour, fontsize=11, fontweight="bold")
    ax.axhline(0.1, color=GREY, ls="--", lw=1.2)
    ax.text(len(scores) - 1, 0.115, "pure guessing: 10%", ha="right", color=GREY, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(scores)))
    ax.set_xlabel("pixels moved to the right", fontsize=12)
    ax.set_ylabel("mock test accuracy", fontsize=12)
    ax.set_title("The 3-pixel rematch", fontsize=15)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_shelf_compare(scores_a, scores_b, classes, names=("Episode 7", "CNN")):
    """Shelf by shelf, two models side by side, hardest shelf at the top."""
    order = sorted(range(10), key=lambda k: scores_b[k])
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ys = np.arange(10)
    ax.barh(ys - 0.2, [scores_a[k] for k in order], height=0.38, color=PALE, label=names[0])
    ax.barh(ys + 0.2, [scores_b[k] for k in order], height=0.38, color=BLUE, label=names[1])
    for y, k in zip(ys, order):
        ax.text(scores_b[k] + 0.006, y + 0.2, f"{scores_b[k]:.1%}", va="center", fontsize=10, color=BLUE)
        ax.text(scores_a[k] + 0.006, y - 0.2, f"{scores_a[k]:.1%}", va="center", fontsize=10, color=GREY)
    ax.set_yticks(ys, [classes[k] for k in order], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.06)
    ax.set_xlabel("photos put on the right shelf (board exam)", fontsize=12)
    gains = [(scores_b[k] - scores_a[k], k) for k in range(10)]
    gain, k = max(gains)
    ax.set_title(f"Biggest gain: {classes[k]}, +{gain * 100:.1f} points", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- inside the CNN

def show_filters(weight, title="The 16 windows the CNN taught itself (layer 1)"):
    """Each 3 x 3 filter of a Conv2d layer: red = bigger weight, blue = negative."""
    w = weight.detach().cpu()[:, 0]
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
    for i, ax in enumerate(axes.flat):
        limit = max(w[i].abs().max().item(), 1e-6)
        ax.imshow(w[i], cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"filter {i}", fontsize=11)
        _clean(ax)
    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.show()


def show_maps(model, photo, classes, title=None):
    """What each layer-1 window finds in one photo, plus what the model says."""
    photo = _as_float(photo).reshape(1, 1, 28, 28).to(_device(model))
    with torch.no_grad():
        maps = torch.relu(model[0](photo))[0].cpu()
        chances = torch.softmax(model(photo), dim=1)[0].cpu()
    pick = int(chances.argmax())
    fig = plt.figure(figsize=(16, 6.2))
    big = fig.add_axes([0.01, 0.12, 0.24, 0.72])
    big.imshow(photo[0, 0].cpu(), cmap="gray_r", vmin=0, vmax=1)
    big.set_title(f"model says: {classes[pick]}\n{chances[pick]:.1%} sure", fontsize=15)
    big.axis("off")
    for i in range(16):
        ax = fig.add_axes([0.28 + (i % 8) * 0.09, 0.50 - (i // 8) * 0.42, 0.085, 0.36])
        ax.imshow(maps[i], cmap="magma")
        ax.set_title(f"filter {i}", fontsize=10)
        _clean(ax)
    fig.suptitle(title or "What the 16 windows see in this photo (bright = found it here)", fontsize=15)
    plt.show()


def show_where_it_looks(photo, sure, label, classes, base, size=8):
    """Black tape slides over the photo. Where hiding a spot hurts most = where the model looks.
    sure[r, c] = the score for the right shelf with the tape's corner at row r, column c;
    base = the score with no tape."""
    _mark(f"REVEAL · where it looks · {classes[int(label)]}")
    photo = _as_float(photo).reshape(28, 28)
    sure = sure.detach().cpu()
    n = sure.shape[0]
    heat = torch.zeros(28, 28)
    count = torch.zeros(28, 28)
    drop = (base - sure).clamp(min=0)
    for r in range(n):
        for c in range(n):
            heat[r:r + size, c:c + size] += drop[r, c]
            count[r:r + size, c:c + size] += 1
    heat = heat / count.clamp(min=1)
    positions = [(r, c) for r in range(0, n, 3) for c in range(0, n, 3)]
    if _QUIET:
        positions = positions[-1:]
    handle = None
    flat = [(rr, cc) for rr in range(n) for cc in range(n)]
    for step in range(len(positions) + 1):
        fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=DARK)
        if step < len(positions):
            r, c = positions[step]
            taped = photo.clone()
            taped[r:r + size, c:c + size] = 0
            a.imshow(taped, cmap="gray", vmin=0, vmax=1)
            a.add_patch(plt.Rectangle((c - 0.5, r - 0.5), size, size, fill=False, edgecolor=GOLD, lw=3))
            a.set_title(f"tape here: {classes[int(label)]} score {sure[r, c]:.1f} "
                        f"(no tape: {base:.1f})", color="white", fontsize=16)
            partial = torch.full((n, n), float("nan"))
            upto = flat.index((r, c)) + 1
            for rr, cc in flat[:upto]:
                partial[rr, cc] = drop[rr, cc]
            b.imshow(partial, cmap="inferno", vmin=0, vmax=max(drop.max().item(), 1e-6))
            b.set_title("how much the score drops", color="white", fontsize=16)
        else:
            a.imshow(photo, cmap="gray", vmin=0, vmax=1)
            a.set_title(f"{classes[int(label)]}: score {base:.1f} with no tape", color="white", fontsize=16)
            b.imshow(photo, cmap="gray", vmin=0, vmax=1)
            b.imshow(heat, cmap="inferno", alpha=0.65)
            b.set_title("where it looks: bright = hiding it hurts most", color="white", fontsize=16)
        for ax in (a, b):
            ax.axis("off")
        fig.tight_layout()
        handle = _show(fig, handle)
        _pause(0.05)


def watch(photo, step, chance, wanted, style="dream", handle=None):
    """Redraws a photo that is being changed step by step. Returns the handle for the next call."""
    if handle is None:
        _mark(f"REVEAL · {style} · {wanted}")
    img = _as_float(photo).reshape(28, 28)
    fig = plt.figure(figsize=(16, 6.2), facecolor=DARK)
    ax = fig.add_axes([0.33, 0.06, 0.34, 0.78])
    ax.imshow(img, cmap="magma" if style == "dream" else "gray", vmin=0, vmax=1)
    ax.axis("off")
    fig.text(0.5, 0.92, f"asking for: {wanted}   |   step {step}", ha="center", color=PALE, fontsize=20)
    fig.text(0.70, 0.45, f"{chance:.1%}", color=NEON if chance > 0.5 else HOT, fontsize=64,
             fontweight="bold", va="center")
    fig.text(0.70, 0.30, "sure", color=PALE, fontsize=22)
    return _show(fig, handle)


def show_static(static, called, sure, classes):
    """TV static: a few of the noise photos, and how many of 10,000 landed on each shelf."""
    _mark("REVEAL · TV static")
    counts = torch.bincount(called.cpu(), minlength=10)
    handle = None
    n = _frames(30)
    for i in range(1, n + 1):
        t = _ease(i / n)
        fig = plt.figure(figsize=(16, 6.4), facecolor=DARK)
        for j in range(6):
            ax = fig.add_axes([0.01 + (j % 3) * 0.105, 0.50 - (j // 3) * 0.42, 0.10, 0.36])
            ax.imshow(_as_float(static[j]).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        ax = fig.add_axes([0.47, 0.10, 0.50, 0.72])
        ax.set_facecolor(DARK)
        top = int(counts.argmax())
        for k in range(10):
            v = counts[k].item() * t
            ax.barh(k, v, color=NEON if k == top else PALE, height=0.7)
            ax.text(v + 80, k, f"{int(round(v)):,}", va="center", color="white", fontsize=14, fontweight="bold")
        ax.set_yticks(range(10), classes, color="white", fontsize=13)
        ax.invert_yaxis()
        ax.set_xlim(0, 10000 * 1.12)
        ax.set_xticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        fig.text(0.5, 0.92, f"10,000 photos of TV static. The CNN calls {int(counts[top] * t):,} of them "
                 f"'{classes[top]}'. 'I don't know': 0", ha="center", color="white", fontsize=19)
        if i == n:
            fig.text(0.72, 0.02, f"average: {sure.mean().item():.1%} sure", ha="center", color=GOLD,
                     fontsize=16)
        handle = _show(fig, handle)
        _pause(0.045)


def show_fooled(photos, chances, classes):
    """One noise photo per shelf, each pushed until the model is sure."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.4), facecolor=DARK, layout="constrained")
    for k, ax in enumerate(axes.flat):
        ax.imshow(_as_float(photos[k]).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{classes[k]}\n{chances[k]:.2%} sure", color=NEON, fontsize=14)
        ax.axis("off")
    fig.suptitle("Ten photos of static. The CNN is sure of every one.", color="white", fontsize=18)
    plt.show()


def show_attack_curve(budgets, curves):
    """Board-exam accuracy when every pixel may move by up to budget (out of 255)."""
    colours = [BLUE, RED, GREEN]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for (name, scores), colour in zip(curves.items(), colours):
        ax.plot(budgets, scores, "-o", color=colour, lw=2.6, ms=8, label=name)
        for b, v in zip(budgets, scores):
            ax.text(b, v + 0.03, f"{v:.1%}", ha="center", color=colour, fontsize=11, fontweight="bold")
    ax.axhline(0.1, color=GREY, ls="--", lw=1.2)
    ax.text(budgets[-1], 0.115, "pure guessing: 10%", ha="right", color=GREY, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(budgets, [f"{b}/255" for b in budgets])
    ax.set_xlabel("how far each pixel may move (out of 255)", fontsize=12)
    ax.set_ylabel("board exam accuracy", fontsize=12)
    ax.set_title("Tiny changes, a huge drop", fontsize=15)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def flicker(model, original, changed, classes, label, times=6):
    """The real photo and the changed one, swapped back and forth: spot the difference."""
    _mark("REVEAL · flicker")
    device = _device(model)
    pics = [_as_float(original).reshape(1, 1, 28, 28), _as_float(changed).reshape(1, 1, 28, 28)]
    with torch.no_grad():
        chances = [torch.softmax(model(p.to(device)), dim=1)[0].cpu() for p in pics]
    names = ["the real photo", "the sneaky copy"]
    handle = None
    order = [0, 1] * (1 if _QUIET else times)
    for step, j in enumerate(order + ["both"]):
        fig = plt.figure(figsize=(16, 6.4), facecolor=DARK)
        if j == "both":
            for col, jj in enumerate((0, 1)):
                ax = fig.add_axes([0.04 + col * 0.31, 0.08, 0.28, 0.72])
                ax.imshow(pics[jj][0, 0], cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                pick = int(chances[jj].argmax())
                ax.set_title(f"{names[jj]}\n{classes[pick]} · {chances[jj][pick]:.1%}",
                             color=NEON if pick == int(label) else HOT, fontsize=17)
            ax = fig.add_axes([0.68, 0.08, 0.28, 0.72])
            diff = (pics[1] - pics[0])[0, 0]
            ax.imshow(diff, cmap="RdBu_r", vmin=-0.06, vmax=0.06)
            ax.set_title(f"the change, made visible\nbiggest: {diff.abs().max() * 255:.0f} of 255",
                         color="white", fontsize=17)
            ax.axis("off")
        else:
            ax = fig.add_axes([0.33, 0.06, 0.34, 0.78])
            ax.imshow(pics[j][0, 0], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            pick = int(chances[j].argmax())
            fig.text(0.5, 0.92, names[j], ha="center", color=PALE, fontsize=20)
            fig.text(0.70, 0.45, classes[pick], color=NEON if pick == int(label) else HOT, fontsize=40,
                     fontweight="bold", va="center")
            fig.text(0.70, 0.32, f"{chances[j][pick]:.1%} sure", color="white", fontsize=24)
        handle = _show(fig, handle)
        _pause(0.6)


# ---------------------------------------------------------------- a drawing of your own

def open_in_paint(path="my_drawing.png", blank=True):
    """Opens the drawing in Paint. blank=True starts from a clean white page first.
    Draw, then press Ctrl+S: Paint saves to the same file, with no dialog."""
    path = Path(path)
    if _QUIET:
        return
    if blank or not path.exists():
        Image.new("RGB", (560, 560), "white").save(path)
    subprocess.Popen(["mspaint", str(path.resolve())])
    print("Paint is open. Draw, then Ctrl+S, then come back here.")


def load_drawing(path, show=True):
    """A drawing becomes a 28 x 28 photo, loosely following the steps the Fashion-MNIST paper describes:
    trim the empty border, shrink the longest side to 28, sharpen, centre it on a
    28 x 28 square, flip dark and light."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(white, img)
    grey = img.convert("L")
    a = np.asarray(grey, dtype=np.float32)
    corners = np.array([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]])
    background = float(np.median(corners))
    ink = np.abs(a - background) > 0.05 * 255                 # 5% away from the corner colour
    if not ink.any():
        raise ValueError("The drawing looks empty: draw something darker on a light background.")
    rows, cols = np.where(ink)
    box = (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)
    item = grey.crop(box)
    w, h = item.size
    scale = 28 / max(w, h)
    item = item.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    item = item.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=0))
    canvas = Image.new("L", (28, 28), int(round(background)))
    canvas.paste(item, ((28 - item.size[0]) // 2, (28 - item.size[1]) // 2))
    photo = torch.tensor(np.asarray(canvas, dtype=np.float32) / 255)
    if background > 127:                                     # light background: flip
        photo = 1 - photo
    photo = (photo / max(photo.max().item(), 1e-6)).clamp(0, 1)
    if show:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.8))
        a1.imshow(grey, cmap="gray")
        a1.set_title("your drawing", fontsize=13)
        a1.axis("off")
        a2.imshow(photo, cmap="gray_r", vmin=0, vmax=1)
        a2.set_title("what the model gets: 28 x 28", fontsize=13)
        a2.axis("off")
        plt.tight_layout()
        plt.show()
    return photo

"""Plotting helpers for PyTorch Episode 10.

act() and the reveal helpers log the time of each big moment to
Videos/clip_marks/ep10.csv.
"""
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from matplotlib.colors import ListedColormap

BLUE, ORANGE, RED, GREEN = "#3b7fc4", "#e8862a", "#d33333", "#3a9d5d"
GOLD, GREY, INK, PALE = "#f2b705", "#8a919c", "#16181d", "#c9d6e8"
DARK, NEON, HOT, SKY = "#0f1115", "#39d98a", "#ff5a5a", "#6cb6ff"
DIM = "#2a2f38"

_MARKS = Path.home() / "Videos" / "clip_marks" / "ep10.csv"
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


def _dark(ax):
    ax.set_facecolor(DARK)
    ax.tick_params(colors=PALE, labelsize=12)
    for side in ax.spines.values():
        side.set_visible(False)


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
    """One big number, counting up (or down) in the middle of a dark card."""
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
        fig.subplots_adjust(left=0.30, right=0.97, top=0.80, bottom=0.05)
        handle = _show(fig, handle)
        _pause(0.045)


# ---------------------------------------------------------------- the two shapes

def _linear_widths(model):
    return [m.out_features for m in model if isinstance(m, torch.nn.Linear)]


def _knobs(model):
    return sum(p.numel() for p in model.parameters())


def show_depth(shallow, deep):
    """Two networks side by side: one box per Linear layer, as wide as its neurons."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.2), facecolor=DARK)
    for ax, model, name, colour in ((axes[0], shallow, "Episode 7 · 3 layers", SKY),
                                    (axes[1], deep, f"Today · {len(_linear_widths(deep))} layers", HOT)):
        ax.set_facecolor(DARK)
        widths = _linear_widths(model)
        n = len(widths)
        gap = 0.35
        h = 0.84 / (n + (n - 1) * gap)
        y = 0.10
        ax.text(0.5, 0.035, "photo · 784 pixels", ha="center", color=PALE, fontsize=14)
        for k, w in enumerate(widths):
            width = 0.12 + 0.76 * w / 128
            ax.add_patch(plt.Rectangle((0.5 - width / 2, y), width, h, color=colour, alpha=0.9))
            label = f"{w} shelves" if k == n - 1 else f"{w} neurons"
            if n <= 4 or k in (0, n - 2, n - 1):
                ax.text(0.5, y + h / 2, label, ha="center", va="center", color=INK,
                        fontsize=15 if n <= 4 else 9, fontweight="bold")
            y += h * (1 + gap)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(f"{name}\n{_knobs(model):,} knobs", color="white", fontsize=20, pad=8)
    fig.tight_layout()
    _show(fig)


# ---------------------------------------------------------------- the race chart

_race = {"curves": {}, "epochs": {}, "handle": None, "current": None}
_EVERY = 6400                     # photos between two mock tests (100 batches of 64)


def _race_colour(name):
    if "BatchNorm" in name:
        return NEON
    if "3 layers" in name:
        return SKY
    if "SGD" in name:
        return GOLD
    return HOT


def new_race():
    """Clears the race chart: the next study() starts on an empty chart."""
    _race["curves"].clear()
    _race["epochs"].clear()
    _race["current"] = None


def race_chart(name, curve, epochs=5):
    """Mock-test accuracy while a model studies, redrawn every 100 batches.
    Earlier runs stay on the chart, faint, so the new run races them."""
    if name != _race["current"] or len(curve) == 1:      # a new run starts
        _race["current"], _race["handle"] = name, None
        _race["curves"].pop(name, None)
        _mark(f"RACE · {name}")
    _race["curves"][name] = list(curve)
    _race["epochs"][name] = epochs
    right = max(_race["epochs"].values()) * 50 + 6        # thousands of photos
    fig, ax = plt.subplots(figsize=(12.5, 6.6), facecolor=DARK)
    _dark(ax)
    for epoch in range(1, 6):
        if epoch * 50 < right:
            ax.axvline(epoch * 50, color=DIM, lw=1, zorder=0)
            ax.text(epoch * 50 - 1.5, 0.985, f"epoch {epoch}", color=GREY, fontsize=10, ha="right",
                    va="top")
    for other, values in _race["curves"].items():
        x = [(k + 1) * _EVERY / 1000 for k in range(len(values))]
        now = other == name
        colour = _race_colour(other)
        label = f"{other}: {values[-1]:.1%}" if now else f"{other} · best {max(values):.1%}"
        ax.plot(x, values, color=colour, lw=3.4 if now else 1.8, alpha=1.0 if now else 0.45,
                label=label)
        if now:
            ax.scatter([x[-1]], [values[-1]], s=110, color=colour, zorder=5)
    ax.set_xlim(0, right)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("photos studied (thousands)", color=PALE, fontsize=13)
    ax.set_ylabel("mock test", color=PALE, fontsize=13)
    ax.grid(axis="y", color=DIM, lw=1)
    ax.set_title(f"{name}  ·  {len(curve) * _EVERY:,} photos  ·  mock test {curve[-1]:.1%}"
                 f"  ·  best {max(curve):.1%}", color="white", fontsize=17, pad=12)
    ax.legend(loc="lower right", facecolor=DARK, edgecolor=GREY, labelcolor="white", fontsize=12)
    fig.tight_layout()
    _race["handle"] = _show(fig, _race["handle"])


# ---------------------------------------------------------------- inside the network

def show_xray(push, title, compare=None, names=("20 layers", "20 layers + BatchNorm")):
    """One bar per layer, on a log scale: the average push each layer's knobs get.
    The bars appear from layer 20 down to layer 1, the way backward() travels."""
    panels = [(list(push), names[0], HOT)]
    if compare is not None:
        panels.append((list(compare), names[1], NEON))
    _mark(f"XRAY · {title}")
    low = min(min(p) for p, _, _ in panels)
    high = max(max(p) for p, _, _ in panels)
    n = len(panels[0][0])
    frames = _frames(n)
    handle = None
    for f in range(1, frames + 1):
        shown = n if frames == 1 else f
        fig, axes = plt.subplots(1, len(panels), figsize=(8.2 * len(panels), 7.6), facecolor=DARK,
                                 squeeze=False)
        for ax, (values, label, colour) in zip(axes[0], panels):
            _dark(ax)
            for k in range(n - shown, n):
                ax.barh(k + 1, values[k], color=colour, height=0.72)
            ax.set_xscale("log")
            ax.set_xlim(low / 4, high * 40)
            ax.set_ylim(0.3, n + 0.7)
            ax.set_yticks([1, 5, 10, 15, n], ["layer 1", "layer 5", "layer 10", "layer 15",
                                              f"layer {n}"])
            if shown == n:
                for k in (0, n - 1):
                    ax.text(values[k] * 1.6, k + 1, f"{values[k]:.1e}", va="center", color="white",
                            fontsize=15, fontweight="bold")
                ratio = values[-1] / values[0]
                if ratio >= 10:
                    line = f"layer 1 gets 1/{ratio:,.0f} of layer {n}'s push"
                else:
                    line = f"layer 1 gets {1 / ratio:.1f}x layer {n}'s push"
                ax.set_xlabel(line, color=GOLD, fontsize=15, labelpad=10)
            ax.set_title(label, color="white", fontsize=19, pad=10)
        fig.suptitle(title, color=PALE, fontsize=18)
        fig.tight_layout()
        handle = _show(fig, handle)
        _pause(0.12)


def show_picks(counts, classes, title):
    """How many photos the model put on each shelf."""
    counts = [int(c) for c in counts]
    top = max(range(len(counts)), key=lambda k: counts[k])
    total = sum(counts)
    _mark(f"REVEAL · {title}")
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=DARK)
    _dark(ax)
    ax.bar(range(len(counts)), counts, color=[HOT if k == top else GREY for k in range(len(counts))])
    for k, c in enumerate(counts):
        ax.text(k, c + total * 0.012, f"{c:,}", ha="center", color="white", fontsize=14,
                fontweight="bold" if k == top else "normal")
    ax.set_xticks(range(len(counts)), classes, rotation=25, color=PALE, fontsize=12)
    ax.set_yticks([])
    ax.set_ylim(0, total * 1.12)
    ax.set_title(f"{title}\n{counts[top]:,} of {total:,} photos: {classes[top]}", color="white",
                 fontsize=18, pad=10)
    fig.tight_layout()
    _show(fig)


def show_brain(alive, title, compare=None, compare_title=""):
    """Every hidden neuron as one square: lit if it fires for at least one photo,
    dark if it never fires. The dark ones switch off one by one."""
    panels = [(alive, title)]
    if compare is not None:
        panels.append((compare, compare_title))
    _mark(f"BRAIN · {title}")
    rng = np.random.default_rng(0)
    grids = []
    for grid, name in panels:
        grid = np.asarray(grid.cpu() if torch.is_tensor(grid) else grid, dtype=bool)
        dead = np.argwhere(~grid)
        grids.append((grid, name, dead[rng.permutation(len(dead))]))
    frames = _frames(28)
    cmap = ListedColormap([DIM, NEON])
    handle = None
    for f in range(1, frames + 1):
        t = _ease(f / frames)
        fig, axes = plt.subplots(1, len(grids), figsize=(8.4 * len(grids), 5.4), facecolor=DARK,
                                 squeeze=False)
        for ax, (grid, name, dead) in zip(axes[0], grids):
            k = int(round(t * len(dead)))
            shown = np.ones(grid.shape)
            for r, c in dead[:k]:
                shown[r, c] = 0
            ax.imshow(shown, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([0, grid.shape[0] - 1], ["layer 1", f"layer {grid.shape[0]}"], color=PALE,
                          fontsize=12)
            for side in ax.spines.values():
                side.set_visible(False)
            colour = HOT if len(dead) else NEON
            ax.set_title(f"{name}\n{k:,} of {grid.size:,} neurons never fire", color=colour,
                         fontsize=18, pad=10)
        fig.patch.set_facecolor(DARK)
        fig.tight_layout()
        handle = _show(fig, handle)
        _pause(0.05)


# ---------------------------------------------------------------- BatchNorm, up close

def show_bn(before, after, name="layer 2, neuron 0"):
    """One neuron's numbers for a batch of photos, before and after BatchNorm."""
    b = before.detach().float().cpu().numpy()
    a = after.detach().float().cpu().numpy()
    _mark("BATCHNORM · one neuron")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), facecolor=DARK)
    for ax, values, label, colour in ((axes[0], b, f"before · {name}", HOT),
                                      (axes[1], a, "after BatchNorm", NEON)):
        _dark(ax)
        ax.hist(values, bins=16, color=colour)
        ax.axvline(values.mean(), color=GOLD, ls="--", lw=2)
        ax.set_yticks([])
        ax.set_title(f"{label}\naverage {values.mean():.3f}  ·  spread {values.std():.3f}",
                     color="white", fontsize=17, pad=10)
    fig.suptitle(f"The same {len(b)} photos, one neuron", color=PALE, fontsize=16)
    fig.tight_layout()
    _show(fig)


def show_company(photo, rows, classes, label):
    """One photo, and the shelf the model picks for it in each situation.
    rows = [(situation, the 10 raw scores), ...]"""
    _mark("REVEAL · same photo, different company")
    img = photo.detach().float().cpu().reshape(28, 28)
    picks = []
    for name, scores in rows:
        chances = torch.softmax(scores.detach().float().cpu(), dim=0)
        k = int(chances.argmax())
        picks.append((name, k, chances[k].item()))
    handle = None
    steps = _frames(len(rows))
    for s in range(1, steps + 1):
        shown = len(rows) if steps == 1 else s
        fig = plt.figure(figsize=(16, 5.2), facecolor=DARK)
        ax = fig.add_axes([0.015, 0.10, 0.20, 0.72])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.text(0.115, 0.88, f"photo · {classes[label]}", ha="center", color="white", fontsize=17)
        for j, (name, k, sure) in enumerate(picks[:shown]):
            x = 0.33 + j * 0.24
            right = k == label
            fig.text(x, 0.80, name, ha="center", color=PALE, fontsize=15)
            fig.text(x, 0.50, classes[k], ha="center", color=NEON if right else HOT, fontsize=34,
                     fontweight="bold")
            fig.text(x, 0.30, f"{sure:.1%} sure", ha="center", color="white", fontsize=19)
            fig.text(x, 0.14, "right" if right else "wrong", ha="center",
                     color=NEON if right else HOT, fontsize=15)
        handle = _show(fig, handle)
        _pause(1.2)


# ---------------------------------------------------------------- the lab

def show_epochs(curves, title="Mock test after every epoch"):
    """The mock test after each epoch: a healthy run against buggy ones.
    curves = {name: [score after epoch 1, epoch 2, ...]}"""
    colours = [NEON, HOT, GOLD, SKY]
    _mark(f"REVEAL · {title}")
    n = max(len(v) for v in curves.values())
    frames = _frames(n)
    handle = None
    for f in range(1, frames + 1):
        shown = n if frames == 1 else f
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK)
        _dark(ax)
        for (name, values), colour in zip(curves.items(), colours):
            part = values[:shown]
            x = list(range(1, len(part) + 1))
            ax.plot(x, part, "-o", color=colour, lw=3.4, ms=12, label=name)
            ax.text(x[-1] + 0.1, part[-1], f"{part[-1]:.1%}", color=colour, fontsize=20,
                    va="center", fontweight="bold")
        ax.set_xlim(0.6, n + 0.9)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(1, n + 1), [f"epoch {e}" for e in range(1, n + 1)], fontsize=13)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=DIM, lw=1)
        ax.set_title(title, color="white", fontsize=19, pad=12)
        ax.legend(loc="lower left", facecolor=DARK, edgecolor=GREY, labelcolor="white", fontsize=13)
        fig.tight_layout()
        handle = _show(fig, handle)
        _pause(0.7)


def show_twins(weight, title, compare=None, compare_title="healthy start"):
    """Layer 1's neurons, each drawn as the 28 x 28 picture of its 784 weights."""
    panels = []
    if compare is not None:
        panels.append((compare, compare_title))
    panels.append((weight, title))
    _mark(f"TWINS · {title}")
    fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 5.6), facecolor=DARK,
                             squeeze=False)
    for ax, (w, name) in zip(axes[0], panels):
        w = w.detach().float().cpu()
        rows = w.reshape(len(w), -1)
        distinct = torch.unique(rows, dim=0).shape[0]
        pics = w.reshape(-1, 28, 28)[:128]
        n = len(pics)
        cols = 16
        lines = (n + cols - 1) // cols
        mosaic = np.full((lines * 29 + 1, cols * 29 + 1), np.nan)
        for j, pic in enumerate(pics):
            r, c = divmod(j, cols)
            p = pic.numpy()
            span = max(abs(p).max(), 1e-12)
            mosaic[1 + r * 29:1 + r * 29 + 28, 1 + c * 29:1 + c * 29 + 28] = p / span
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad(DARK)
        ax.imshow(mosaic, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.axis("off")
        colour = NEON if distinct > 1 else HOT
        ax.set_title(f"{name}\n{distinct:,} different neuron{'s' if distinct != 1 else ''} of {len(rows):,}",
                     color=colour, fontsize=18, pad=10)
    fig.tight_layout()
    _show(fig)


def checklist(items, title="5 one-line bugs · no error message"):
    """The closing card: one line per bug."""
    _mark("CHECKLIST")
    height = 2.0 + 0.95 * len(items)
    fig = plt.figure(figsize=(16, height), facecolor=DARK)
    fig.add_axes([0, 0, 1, 1]).axis("off")
    fig.text(0.5, 1 - 0.8 / height, title.upper(), ha="center", color=GOLD, fontsize=30,
             fontweight="bold")
    for k, item in enumerate(items):
        y = 1 - (1.75 + 0.95 * k) / height
        fig.text(0.07, y, f"{k + 1}", color=GOLD, fontsize=28, fontweight="bold", va="center")
        fig.text(0.11, y, item, color="white", fontsize=23, va="center", family="monospace")
    _show(fig)

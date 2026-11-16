"""Plotting helpers for PyTorch Episode 11.

act() and the reveal helpers log the time of each big moment to
Videos/clip_marks/ep11.csv.
"""
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display

GOLD, GREY, INK, PALE = "#f2b705", "#8a919c", "#16181d", "#c9d6e8"
DARK, NEON, HOT, SKY = "#0f1115", "#39d98a", "#ff5a5a", "#6cb6ff"
DIM, VIOLET = "#2a2f38", "#b48cff"

_MARKS = Path.home() / "Videos" / "clip_marks" / "ep11.csv"
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


def _grey(photo):
    return photo.detach().float().cpu().reshape(28, 28).numpy()


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


def reveal_board(rows, title, target=None, target_label=""):
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
            if target_label and i == n:
                ax.text(target, -0.52, f"{target_label}: {target:.1%}", color=GOLD, fontsize=15,
                        ha="center", va="bottom")
        ax.set_xlim(0, 1.18)
        ax.set_ylim(len(rows) - 0.4, -0.75 if target is not None else -0.6)
        ax.axis("off")
        ax.set_title(title, color=PALE, fontsize=22, pad=14)
        fig.subplots_adjust(left=0.30, right=0.97, top=0.80, bottom=0.05)
        handle = _show(fig, handle)
        _pause(0.045)


# ---------------------------------------------------------------- photos

def show_photos(photos, labels, classes, title, cols=10):
    """A grid of shop photos, with each photo's shelf under it."""
    n = len(photos)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.45 * cols, 1.62 * rows + 0.9), facecolor=DARK,
                             squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < n:
            ax.imshow(_grey(photos[k]), cmap="gray", vmin=0, vmax=1)
            if rows <= 3:
                ax.set_title(classes[int(labels[k])], color=PALE, fontsize=9, pad=2)
    fig.suptitle(title, color="white", fontsize=18)
    fig.tight_layout()
    _show(fig)


def show_adapter(photo, adapted, mean, std):
    """One photo before and after the adapter: 28 x 28 grey -> the size and colours ImageNet used."""
    a = adapted.detach().float().cpu()[0]
    rgb = (a * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)).clamp(0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), facecolor=DARK)
    axes[0].imshow(_grey(photo), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("our photo · 28 x 28 · 1 grey channel", color="white", fontsize=17, pad=10)
    axes[1].imshow(rgb.permute(1, 2, 0).numpy())
    axes[1].set_title(f"what ResNet18 gets · {a.shape[1]} x {a.shape[2]} · 3 colour channels",
                      color="white", fontsize=17, pad=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GREY)
    fig.tight_layout()
    _show(fig)


def show_guesses(photo, guesses, title):
    """One photo and the borrowed brain's top guesses: guesses = [(name, chance), ...]."""
    _mark(f"REVEAL · {title}")
    handle = None
    steps = _frames(len(guesses))
    for s in range(1, steps + 1):
        shown = len(guesses) if steps == 1 else s
        fig = plt.figure(figsize=(16, 5.2), facecolor=DARK)
        ax = fig.add_axes([0.02, 0.08, 0.24, 0.76])
        ax.imshow(_grey(photo), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.text(0.14, 0.89, title, ha="center", color="white", fontsize=17)
        for j, (name, chance) in enumerate(guesses[:shown]):
            y = 0.72 - j * 0.24
            fig.text(0.32, y, f"{j + 1}.", color=GOLD, fontsize=30, fontweight="bold", va="center")
            fig.text(0.37, y, f'"{name}"', color="white" if j else NEON, fontsize=34, fontweight="bold",
                     va="center")
            fig.text(0.95, y, f"{chance:.1%}", color=PALE, fontsize=26, va="center", ha="right")
        handle = _show(fig, handle)
        _pause(0.9)


_WEARABLE = {"jersey", "sweatshirt", "cardigan", "jean", "miniskirt", "suit", "trench coat", "gown", "kimono",
             "abaya", "poncho", "pajama", "fur coat", "lab coat", "academic gown", "overskirt", "hoopskirt",
             "swimming trunks", "maillot", "bikini", "sarong", "apron", "military uniform", "running shoe",
             "sandal", "clog", "Loafer", "cowboy boot", "sock", "purse", "backpack", "mailbag"}


def show_shelf_guesses(rows, title):
    """Per shop shelf, ImageNet's most common guess: rows = [(shelf, guess, count, total), ...].
    Green when the guess is something you could wear or carry, red when it isn't."""
    _mark(f"REVEAL · {title}")
    fig = plt.figure(figsize=(16, 1.3 + 0.62 * len(rows)), facecolor=DARK)
    fig.add_axes([0, 0, 1, 1]).axis("off")
    h = fig.get_size_inches()[1]
    fig.text(0.5, 1 - 0.55 / h, title, ha="center", color=PALE, fontsize=20)
    for k, (shelf, guess, count, total) in enumerate(rows):
        y = 1 - (1.25 + 0.62 * k) / h
        colour = NEON if guess in _WEARABLE else HOT
        fig.text(0.20, y, shelf, ha="right", color="white", fontsize=19, va="center")
        fig.text(0.24, y, "->", ha="center", color=GREY, fontsize=19, va="center")
        fig.text(0.28, y, f'"{guess}"', color=colour, fontsize=21, fontweight="bold", va="center")
        fig.text(0.86, y, f"{count:,} of {total:,}", ha="right", color=PALE, fontsize=17, va="center")
    _show(fig)


# ---------------------------------------------------------------- the race chart

_race = {"curves": {}, "handle": None, "current": None, "right": 0, "target": None}


def _race_colour(name):
    if "lr 0.01" in name:
        return GOLD
    if "old head" in name:
        return VIOLET
    if "borrowed" in name:
        return NEON
    if "blank" in name:
        return HOT
    return SKY


def new_race():
    """Clears the race chart: the next study() starts on an empty chart, with no target line."""
    _race["curves"].clear()
    _race["current"] = None
    _race["right"] = 0
    _race["target"] = None


def race_target(value, label):
    """A gold line on the race chart: the score to beat."""
    _race["target"] = (value, label)


def race_chart(name, seen, curve, total):
    """Mock-test accuracy while a model studies. seen = photos studied at each point.
    Earlier runs stay on the chart, faint, so the new run races them."""
    if name != _race["current"] or len(curve) == 1:      # a new run starts
        _race["current"], _race["handle"] = name, None
        _race["curves"].pop(name, None)
        _mark(f"RACE · {name}")
    _race["curves"][name] = (list(seen), list(curve))
    _race["right"] = max(_race["right"], total)
    right = _race["right"] * 1.03
    fig, ax = plt.subplots(figsize=(12.5, 6.6), facecolor=DARK)
    _dark(ax)
    if _race["target"] is not None:
        target, target_label = _race["target"]
        ax.axhline(target, color=GOLD, ls="--", lw=2, zorder=1)
        ax.text(right * 0.01, target + 0.015, f"{target_label}: {target:.1%}", color=GOLD, fontsize=13)
    for other, (xs, values) in _race["curves"].items():
        now = other == name
        colour = _race_colour(other)
        label = f"{other}: {values[-1]:.1%}" if now else f"{other} · best {max(values):.1%}"
        ax.plot(xs, values, color=colour, lw=3.4 if now else 1.8, alpha=1.0 if now else 0.45,
                label=label, marker="o" if now else None, ms=7)
        if now:
            ax.scatter([xs[-1]], [values[-1]], s=110, color=colour, zorder=5)
    ax.set_xlim(0, right)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("photos studied (the same practice photos, again and again)", color=PALE, fontsize=13)
    ax.set_ylabel("mock test", color=PALE, fontsize=13)
    ax.grid(axis="y", color=DIM, lw=1)
    ax.set_title(f"{name}  ·  {seen[-1]:,} photos studied  ·  mock test {curve[-1]:.1%}"
                 f"  ·  best {max(curve):.1%}", color="white", fontsize=17, pad=12)
    ax.legend(loc="lower right", facecolor=DARK, edgecolor=GREY, labelcolor="white", fontsize=12)
    fig.tight_layout()
    _race["handle"] = _show(fig, _race["handle"])


# ---------------------------------------------------------------- inside the borrowed brain

def show_filters(borrowed, blank):
    """conv1's 64 windows (7 x 7, 3 colours): borrowed from ImageNet, and a blank brain's."""
    _mark("FILTERS · borrowed vs blank")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6), facecolor=DARK)
    for ax, w, name in ((axes[0], borrowed, "borrowed · learned from ImageNet"),
                        (axes[1], blank, "blank · random start")):
        w = w.detach().float().cpu()
        mosaic = np.ones((8 * 8 + 9, 8 * 8 + 9, 3)) * 0.06
        for j in range(min(64, len(w))):
            r, c = divmod(j, 8)
            f = w[j]
            f = (f - f.min()) / max(float(f.max() - f.min()), 1e-12)
            mosaic[1 + r * 9:1 + r * 9 + 7, 1 + c * 9:1 + c * 9 + 7] = f.permute(1, 2, 0).numpy()
        ax.imshow(mosaic, interpolation="nearest")
        ax.axis("off")
        ax.set_title(name, color="white", fontsize=19, pad=10)
    fig.suptitle("Layer 1: 64 windows of 7 x 7 pixels, in 3 colours", color=PALE, fontsize=17)
    fig.tight_layout()
    _show(fig)


def show_stages(photo, maps, names):
    """One photo, and what each stage of the borrowed brain keeps: the strongest 4 maps per stage."""
    _mark("STAGES · one photo through the brain")
    fig, axes = plt.subplots(4, len(maps) + 1, figsize=(3.2 * (len(maps) + 1), 12.4), facecolor=DARK)
    for r in range(4):
        ax = axes[r, 0]
        ax.axis("off")
        if r == 0:
            ax.imshow(_grey(photo), cmap="gray", vmin=0, vmax=1)
            ax.set_title("photo", color="white", fontsize=15)
    for c, (m, name) in enumerate(zip(maps, names), start=1):
        m = m.detach().float().cpu()[0]
        strongest = m.mean(dim=(1, 2)).argsort(descending=True)[:4]
        for r, k in enumerate(strongest):
            ax = axes[r, c]
            ax.imshow(m[k].numpy(), cmap="magma", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{name}\n{m.shape[0]} maps of {m.shape[1]} x {m.shape[2]}", color="white",
                             fontsize=14)
    fig.tight_layout()
    _show(fig)


def show_knobs(frozen, trained, title="Freeze the brain, train only the head"):
    """One long bar: the frozen knobs and the trained ones."""
    _mark(f"KNOBS · {title}")
    total = frozen + trained
    fig, ax = plt.subplots(figsize=(16, 3.6), facecolor=DARK)
    ax.set_facecolor(DARK)
    ax.barh(0, frozen, color=SKY, height=0.5)
    ax.barh(0, max(trained, total * 0.004), left=frozen, color=NEON, height=0.5)
    ax.text(frozen / 2, 0, f"frozen · {frozen:,}", ha="center", va="center", color=INK, fontsize=22,
            fontweight="bold")
    ax.text(total * 1.005, 0, f"trained · {trained:,}", va="center", color=NEON, fontsize=22,
            fontweight="bold")
    ax.set_xlim(0, total * 1.35)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    ax.set_title(f"{title}  ·  {trained / total:.2%} of the knobs learn", color="white", fontsize=19, pad=8)
    fig.tight_layout()
    _show(fig)


# ---------------------------------------------------------------- the lab

def show_epochs(curves, title="Mock test while the head studies", every=1):
    """The mock test every few epochs: a healthy run against a buggy one.
    curves = {name: [score after epoch every, 2 * every, ...]}"""
    colours = [NEON, HOT, GOLD, SKY, VIOLET]
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
        ax.set_xticks(range(1, n + 1), [f"epoch {e * every}" for e in range(1, n + 1)], fontsize=13)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=DIM, lw=1)
        ax.set_title(title, color="white", fontsize=19, pad=12)
        ax.legend(loc="lower left", facecolor=DARK, edgecolor=GREY, labelcolor="white", fontsize=13)
        fig.tight_layout()
        handle = _show(fig, handle)
        _pause(0.7)


def checklist(items, title="Transfer learning · 5 silent mistakes"):
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

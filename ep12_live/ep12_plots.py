"""Plotting helpers for PyTorch Episode 12.


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

_MARKS = Path.home() / "Videos" / "clip_marks" / "ep12.csv"
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


def _device(model):
    return next(model.parameters()).device


def _answer(model, photo):
    """The model's shelf and how sure it is, for one photo (exam mode)."""
    model.eval()
    with torch.no_grad():
        chances = torch.softmax(model(photo.reshape(1, 1, 28, 28).to(_device(model))), dim=1)[0].cpu()
    k = int(chances.argmax())
    return k, float(chances[k])


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


def scorecard(scores, title="The same CNN, four tests"):
    """Grouped bars that grow together. scores = {model name: {test name: value}}: one group per test."""
    _mark(f"REVEAL · {title}")
    names = list(scores)
    tests = list(scores[names[0]])
    colours = [HOT, NEON, SKY, GOLD]
    handle = None
    n = _frames(30)
    for i in range(1, n + 1):
        t = _ease(i / n)
        fig, ax = plt.subplots(figsize=(16, 7.2), facecolor=DARK)
        _dark(ax)
        width = 0.8 / len(names)
        for m, (name, colour) in enumerate(zip(names, colours)):
            xs = np.arange(len(tests)) + (m - (len(names) - 1) / 2) * width
            values = [scores[name][test] * t for test in tests]
            ax.bar(xs, values, width=width * 0.92, color=colour, label=name)
            for x, v in zip(xs, values):
                ax.text(x, v + 0.02, f"{v:.1%}", ha="center", color="white", fontsize=17, fontweight="bold")
        ax.set_xticks(range(len(tests)), tests, fontsize=17, color="white")
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=DIM, lw=1)
        ax.set_title(title, color="white", fontsize=22, pad=14)
        ax.legend(loc="upper right", facecolor=DARK, edgecolor=GREY, labelcolor="white", fontsize=15)
        fig.tight_layout()
        handle = _show(fig, handle)
        _pause(0.045)


# ---------------------------------------------------------------- photos and their changes

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
            if labels is not None:
                ax.set_title(classes[int(labels[k])], color=PALE, fontsize=9, pad=2)
    fig.suptitle(title, color="white", fontsize=18)
    fig.tight_layout()
    _show(fig)


def show_changes(model, versions, label, classes, title="One photo, four ways"):
    """The same photo changed in different ways, each with the model's answer.
    versions = [(name, photo), ...]; green when the answer is right, red when it's wrong."""
    _mark(f"CHANGES · {title}")
    fig, axes = plt.subplots(1, len(versions), figsize=(4 * len(versions), 5.6), facecolor=DARK)
    for ax, (name, photo) in zip(np.atleast_1d(axes), versions):
        k, sure = _answer(model, photo)
        ax.imshow(_grey(photo), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GREY)
        ax.set_title(name, color=PALE, fontsize=17, pad=10)
        ax.set_xlabel(f"{classes[k]}\n{sure:.1%} sure", color=NEON if k == int(label) else HOT,
                      fontsize=20, fontweight="bold", labelpad=10)
    fig.suptitle(title, color="white", fontsize=21)
    fig.tight_layout()
    _show(fig)


def show_augmented(photo, augment, count=16, title="One practice photo, 16 different lessons"):
    """One photo, and count new versions of it made by augment (the same function the study loop uses)."""
    batch = photo.reshape(1, 1, 28, 28).repeat(count, 1, 1, 1)
    versions = augment(batch).detach().cpu()
    cols = 8
    rows = (count + cols - 1) // cols + 1
    fig = plt.figure(figsize=(16, 2.25 * rows + 0.6), facecolor=DARK)
    ax = fig.add_axes([0.42, 1 - 2.0 / (2.25 * rows + 0.6), 0.16, 1.8 / (2.25 * rows + 0.6)])
    ax.imshow(_grey(photo), cmap="gray", vmin=0, vmax=1)
    ax.set_title("the photo on disk", color=GOLD, fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GOLD)
    for k in range(count):
        r, c = divmod(k, cols)
        h = 2.25 * rows + 0.6
        a = fig.add_axes([0.02 + c * 0.122, 1 - (2.6 + 2.2 * (r + 1)) / h, 0.105, 1.95 / h])
        a.imshow(_grey(versions[k]), cmap="gray", vmin=0, vmax=1)
        a.set_xticks([])
        a.set_yticks([])
        for side in a.spines.values():
            side.set_color(DIM)
    fig.text(0.5, 0.01, title, ha="center", color="white", fontsize=18)
    _show(fig)


def show_mirror(model, photos, labels, classes, title="Shoes, facing the other way"):
    """Each photo above its mirror image, with the model's answer under both."""
    _mark(f"REVEAL · {title}")
    n = len(photos)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 6.4), facecolor=DARK, squeeze=False)
    for c in range(n):
        for r, photo in enumerate((photos[c], torch.flip(photos[c], dims=[-1]))):
            ax = axes[r, c]
            k, sure = _answer(model, photo)
            ax.imshow(_grey(photo), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(GREY)
            ax.set_xlabel(f"{classes[k]}\n{sure:.0%}", color=NEON if k == int(labels[c]) else HOT,
                          fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("as stored", color=PALE, fontsize=14)
    axes[1, 0].set_ylabel("mirrored", color=PALE, fontsize=14)
    fig.suptitle(title, color="white", fontsize=20)
    fig.tight_layout()
    _show(fig)


# ---------------------------------------------------------------- charts

_race = {"curves": {}, "handle": None, "current": None, "right": 0, "target": None}


def _race_colour(name):
    if "plain" in name:
        return HOT
    if "sneaky" in name or "practice" in name:
        return VIOLET
    if "augment" in name:
        return NEON
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


def show_shift_curve(steps, curves, title="Slide every photo: mock test"):
    """Accuracy as every photo slides left (minus) or right (plus). curves = {name: [score per step]}."""
    _mark(f"REVEAL · {title}")
    colours = [HOT, NEON, SKY, GOLD]
    handle = None
    frames = _frames(len(steps))
    for f in range(1, frames + 1):
        shown = len(steps) if frames == 1 else f
        fig, ax = plt.subplots(figsize=(13, 6.4), facecolor=DARK)
        _dark(ax)
        for (name, scores), colour in zip(curves.items(), colours):
            ax.plot(steps[:shown], scores[:shown], "-o", color=colour, lw=3.2, ms=10, label=name)
        ax.axhline(0.1, color=GREY, ls="--", lw=1.2)
        ax.text(steps[-1], 0.115, "pure guessing: 10%", ha="right", color=GREY, fontsize=12)
        ax.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(steps, [f"{s:+d}" if s else "0" for s in steps], fontsize=13)
        ax.set_xlabel("pixels moved (minus = left, plus = right)", color=PALE, fontsize=14)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=DIM, lw=1)
        ax.set_title(title, color="white", fontsize=20, pad=12)
        ax.legend(loc="lower center", facecolor=DARK, edgecolor=GREY, labelcolor="white", fontsize=14)
        fig.tight_layout()
        handle = _show(fig, handle)
        _pause(0.25)


def show_dial(pads, normal, slid, title="How far may a practice photo slide?"):
    """Two lines against the augmentation dial: the normal mock test and the slid one."""
    _mark(f"REVEAL · {title}")
    handle = None
    frames = _frames(len(pads))
    for f in range(1, frames + 1):
        shown = len(pads) if frames == 1 else f
        fig, ax = plt.subplots(figsize=(13, 6.4), facecolor=DARK)
        _dark(ax)
        for values, colour, name in ((normal, NEON, "normal photos"), (slid, SKY, "photos slid 3 pixels")):
            ax.plot(pads[:shown], values[:shown], "-o", color=colour, lw=3.4, ms=12, label=name)
            ax.text(pads[shown - 1] + 0.3, values[shown - 1], f"{values[shown - 1]:.1%}", color=colour,
                    fontsize=18, fontweight="bold", va="center")
        ax.set_xlim(min(pads) - 0.8, max(pads) + 2.6)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(pads, [f"{p} px" for p in pads], fontsize=14)
        ax.set_xlabel("how far each practice photo may slide", color=PALE, fontsize=14)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=DIM, lw=1)
        ax.set_title(title, color="white", fontsize=20, pad=12)
        ax.legend(loc="lower left", facecolor=DARK, edgecolor=GREY, labelcolor="white", fontsize=14)
        fig.tight_layout()
        handle = _show(fig, handle)
        _pause(0.5)


# ---------------------------------------------------------------- the lab: sneaky changes

def show_attack_curve(budgets, curves, title="Tiny changes, a huge drop"):
    """Board-exam accuracy when every pixel may move by up to budget (out of 255)."""
    _mark(f"REVEAL · {title}")
    colours = [HOT, NEON, SKY]
    fig, ax = plt.subplots(figsize=(13, 6.4), facecolor=DARK)
    _dark(ax)
    for (name, scores), colour in zip(curves.items(), colours):
        ax.plot(budgets, scores, "-o", color=colour, lw=3.2, ms=10, label=name)
        for b, v in zip(budgets, scores):
            ax.text(b, v + 0.035, f"{v:.1%}", ha="center", color=colour, fontsize=14, fontweight="bold")
    ax.axhline(0.1, color=GREY, ls="--", lw=1.2)
    ax.text(budgets[-1], 0.115, "pure guessing: 10%", ha="right", color=GREY, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(budgets, [f"{b}/255" for b in budgets], fontsize=13)
    ax.set_xlabel("how far each pixel may move (out of 255)", color=PALE, fontsize=14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=DIM, lw=1)
    ax.set_title(title, color="white", fontsize=20, pad=12)
    ax.legend(loc="upper right", facecolor=DARK, edgecolor=GREY, labelcolor="white", fontsize=14)
    fig.tight_layout()
    _show(fig)


def flicker(model, original, changed, classes, label, times=6):
    """The real photo and the changed one, swapped back and forth: spot the difference."""
    _mark("REVEAL · flicker")
    pics = [original.detach().float().cpu().reshape(1, 1, 28, 28),
            changed.detach().float().cpu().reshape(1, 1, 28, 28)]
    answers = [_answer(model, p) for p in pics]
    names = ["the real photo", "the sneaky copy"]
    handle = None
    order = [0, 1] * (1 if _QUIET else times)
    for j in order + ["both"]:
        fig = plt.figure(figsize=(16, 6.4), facecolor=DARK)
        if j == "both":
            for col, jj in enumerate((0, 1)):
                ax = fig.add_axes([0.04 + col * 0.31, 0.08, 0.28, 0.72])
                ax.imshow(pics[jj][0, 0], cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                k, sure = answers[jj]
                ax.set_title(f"{names[jj]}\n{classes[k]} · {sure:.1%}",
                             color=NEON if k == int(label) else HOT, fontsize=17)
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
            k, sure = answers[j]
            fig.text(0.5, 0.92, names[j], ha="center", color=PALE, fontsize=20)
            fig.text(0.70, 0.45, classes[k], color=NEON if k == int(label) else HOT, fontsize=40,
                     fontweight="bold", va="center")
            fig.text(0.70, 0.32, f"{sure:.1%} sure", color="white", fontsize=24)
        handle = _show(fig, handle)
        _pause(0.6)


def checklist(items, title="Data augmentation · the checklist"):
    """The closing card: one line per rule."""
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

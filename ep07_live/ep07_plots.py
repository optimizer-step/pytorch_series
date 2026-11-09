"""Plotting helpers for PyTorch Episode 7: drawing only, no training.

load_drawing() is the one exception: it turns a drawing (PNG or JPG) into a
28 x 28 photo in the Fashion-MNIST style, so the shop model can look at it.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import torch
from IPython.display import clear_output, display
from PIL import Image, ImageFilter

BLUE, ORANGE, RED, GREEN = "#3b7fc4", "#e8862a", "#d33333", "#3a9d5d"
GOLD, GREY, INK, PALE = "#f2b705", "#8a919c", "#16181d", "#c9d6e8"


def _as_float(photo):
    """uint8 photos (0 to 255) become floats (0 to 1)."""
    photo = photo.detach()
    return photo.float() / 255 if photo.dtype == torch.uint8 else photo.float()


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


# ---------------------------------------------------------------- photos

def show_photos(images, labels=None, classes=None, titles=None):
    """Up to 10 photos in a row, numbered 1, 2, 3... or named when labels are given."""
    images = images[:10]
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(1.75 * n, 2.75 if labels is not None else 2.35),
                             squeeze=False)
    for i, ax in enumerate(axes[0]):
        img = _as_float(images[i])
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=max(img.max().item(), 1e-6))
        if titles is not None:
            ax.set_title(titles[i], fontsize=12)
        elif labels is not None:
            k = int(labels[i])
            ax.set_title(f"{k}\n{classes[k]}" if classes else str(k), fontsize=11)
        else:
            ax.set_title(str(i + 1), fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_shelves(images, labels, classes, per_shelf=10):
    """One row per shelf: the first photos of each kind."""
    fig, axes = plt.subplots(10, per_shelf, figsize=(1.05 * per_shelf + 2.2, 11.5))
    for k in range(10):
        picks = (labels == k).nonzero().flatten()[:per_shelf]
        for j, ax in enumerate(axes[k]):
            ax.imshow(_as_float(images[picks[j]]), cmap="gray_r", vmin=0, vmax=1)
            _clean(ax)
        axes[k, 0].set_ylabel(f"{k} · {classes[k]}", rotation=0, ha="right", va="center",
                              fontsize=12, labelpad=10)
    plt.suptitle("The 10 shelves: the first 10 photos of each", fontsize=15, y=0.995)
    plt.tight_layout()
    plt.show()


def show_numbers(photo, top=9, left=8, size=10):
    """The photo, and a 10 x 10 corner of it with every pixel's number written in."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})
    a.imshow(photo, cmap="gray_r", vmin=0, vmax=255)
    a.add_patch(plt.Rectangle((left - 0.5, top - 0.5), size, size, fill=False, edgecolor=RED, lw=2.5))
    a.set_title("What you see: 28 x 28 pixels", fontsize=13)
    a.axis("off")
    crop = photo[top:top + size, left:left + size]
    b.imshow(crop, cmap="gray_r", vmin=0, vmax=255)
    for r in range(size):
        for c in range(size):
            v = int(crop[r, c])
            b.text(c, r, v, ha="center", va="center", fontsize=9, color="white" if v > 130 else INK)
    b.set_title("What the model sees (red box): just numbers, 0 to 255", fontsize=13)
    b.set_xticks([])
    b.set_yticks([])
    for side in b.spines.values():
        side.set_edgecolor(RED)
        side.set_linewidth(2.5)
    plt.tight_layout()
    plt.show()


def show_unroll(photo, rows=(7, 14, 21)):
    """The photo, and the same photo as one long row of 784 pixels."""
    photo = _as_float(photo)
    colours = [RED, ORANGE, BLUE]
    fig = plt.figure(figsize=(16, 6))
    top = fig.add_axes([0.39, 0.40, 0.22, 0.56])
    top.imshow(photo, cmap="gray_r", vmin=0, vmax=1)
    for r, c in zip(rows, colours):
        top.add_patch(plt.Rectangle((-0.5, r - 0.5), 28, 1, fill=False, edgecolor=c, lw=2.5))
        top.text(28.3, r, f"row {r}", color=c, fontsize=12, va="center", fontweight="bold")
    top.set_title("28 rows of 28 pixels", fontsize=13)
    top.axis("off")
    strip = fig.add_axes([0.03, 0.14, 0.94, 0.11])
    strip.imshow(photo.reshape(1, 784), cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    for r, c in zip(rows, colours):
        strip.add_patch(plt.Rectangle((r * 28 - 0.5, -0.5), 28, 1, fill=False, edgecolor=c, lw=2.5))
        strip.text(r * 28 + 13.5, -0.75, f"row {r} = pixels {r * 28} to {r * 28 + 27}",
                   color=c, fontsize=11, ha="center", va="bottom", fontweight="bold")
    strip.set_yticks([])
    strip.set_xticks(range(0, 785, 112))
    strip.set_xlabel("the same photo, unrolled: 28 rows taped end to end = 784 numbers in one row",
                     fontsize=12)
    plt.show()


def show_barcodes(images, labels, classes, per_shelf=12, title=None):
    """Every photo as one row of 784 pixels, 12 photos per shelf, stacked."""
    fig, axes = plt.subplots(10, 1, figsize=(16, 10.5))
    for k, ax in enumerate(axes):
        picks = (labels == k).nonzero().flatten()[:per_shelf]
        rows = _as_float(images[picks]).reshape(len(picks), 784)
        ax.imshow(rows, cmap="gray_r", aspect="auto", vmin=0, vmax=1)
        _clean(ax)
        ax.set_ylabel(classes[k], rotation=0, ha="right", va="center", fontsize=12, labelpad=8)
    axes[-1].set_xticks(range(0, 785, 112))
    axes[-1].set_xlabel("pixel 0 to pixel 783: each line is one photo, unrolled", fontsize=12)
    plt.suptitle(title or "What the model gets: every photo as a barcode of 784 numbers",
                 fontsize=15, y=0.995)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- the model

def show_network(model, classes=None, shown=10):
    """The model's Linear layers as columns of neurons, with the knobs between them."""
    layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    sizes = [layers[0].in_features] + [m.out_features for m in layers]
    names = ["pixels in"] + ["neurons"] * (len(layers) - 1) + ["scores out"]
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ys = [np.linspace(0.88, 0.12, min(n, shown)) for n in sizes]
    for i in range(len(layers)):
        for a in ys[i]:
            for b in ys[i + 1]:
                ax.plot([i, i + 1], [a, b], color=GREY, alpha=0.13, lw=0.8, zorder=1)
        knobs = layers[i].weight.numel() + layers[i].bias.numel()
        ax.text(i + 0.5, 0.965, f"{knobs:,} knobs", ha="center", fontsize=13,
                color=ORANGE, fontweight="bold")
    for i, (n, y) in enumerate(zip(sizes, ys)):
        colour = GREY if i == 0 else (GREEN if i == len(sizes) - 1 else BLUE)
        ax.scatter([i] * len(y), y, s=300, color=colour, zorder=2, edgecolor="white", lw=1.5)
        ax.text(i, 0.02, f"{n:,} {names[i]}", ha="center", fontsize=13, fontweight="bold", color=colour)
        if n > shown:
            ax.text(i + 0.07, 0.5, f"+{n - shown:,}\nmore", fontsize=11, color=colour, va="center")
    if classes is not None and sizes[-1] == len(classes):
        for y, name in zip(ys[-1], classes):
            ax.text(len(sizes) - 1 + 0.08, y, name, fontsize=11, va="center", color=INK)
    total = sum(p.numel() for p in model.parameters())
    first = layers[0].weight.numel() + layers[0].bias.numel()
    ax.set_title(f"{total:,} knobs in all, {first / total:.0%} of them in the first layer", fontsize=15)
    ax.set_xlim(-0.4, len(sizes) - 1 + 0.75)
    ax.set_ylim(-0.03, 1.0)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_chances(model, photo, classes, label=None):
    """The photo, and the model's chance for each of the 10 shelves."""
    photo = _as_float(photo).reshape(28, 28)
    with torch.no_grad():
        chances = torch.softmax(model(photo.reshape(1, 28, 28)), dim=1)[0]
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


# ---------------------------------------------------------------- the answer sheet

def _picked(model, X):
    with torch.no_grad():
        scores = model(X)
    return scores.argmax(dim=1), torch.softmax(scores, dim=1).max(dim=1).values


def show_shelf_scores(scores, classes):
    """How often each shelf gets the right answer: best at the top, the worst in red."""
    scores = [float(s) for s in scores]
    order = sorted(range(len(scores)), key=lambda k: scores[k])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    colours = [RED if i == 0 else BLUE for i in range(len(order))]
    ax.barh(range(len(order)), [scores[k] for k in order], color=colours)
    ax.set_yticks(range(len(order)), [classes[k] for k in order], fontsize=12)
    for i, k in enumerate(order):
        ax.text(scores[k] + 0.006, i, f"{scores[k]:.1%}", va="center", fontsize=11)
    ax.set_xlim(0, 1.09)
    ax.set_xlabel("photos put on the right shelf", fontsize=12)
    ax.set_title(f"Hardest shelf: {classes[order[0]]}, right only {scores[order[0]]:.1%} of the time",
                 fontsize=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()


def show_confusion(model, X, y, classes):
    """10 x 10 table: rows = the right shelf, columns = the shelf the model chose."""
    picked, _ = _picked(model, X)
    table = torch.zeros(10, 10, dtype=torch.long)
    for right, said in zip(y, picked):
        table[right, said] += 1
    off = table.clone().fill_diagonal_(0)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.imshow(off, cmap="Oranges")
    for r in range(10):
        for c in range(10):
            v = int(table[r, c])
            if r == c:
                ax.text(c, r, v, ha="center", va="center", fontsize=10.5, color=BLUE, fontweight="bold")
            elif v:
                ax.text(c, r, v, ha="center", va="center", fontsize=10.5,
                        color="white" if v > off.max() * 0.6 else INK)
    ax.set_xticks(range(10), classes, rotation=40, ha="right", fontsize=11)
    ax.set_yticks(range(10), classes, fontsize=11)
    ax.set_xlabel("the shelf the model chose", fontsize=12)
    ax.set_ylabel("the right shelf", fontsize=12)
    worst = divmod(int(off.argmax()), 10)
    ax.set_title(f"Biggest mix-up: {classes[worst[0]]} put on the {classes[worst[1]]} shelf, "
                 f"{int(off.max())} times", fontsize=13)
    plt.tight_layout()
    plt.show()


def show_mistakes(model, X, y, classes):
    """The 9 wrong answers the model was most sure about."""
    picked, sure = _picked(model, X)
    wrong = (picked != y).nonzero().flatten()
    worst = wrong[sure[wrong].argsort(descending=True)[:9]]
    fig, axes = plt.subplots(1, 9, figsize=(17, 3.1))
    for n, (ax, i) in enumerate(zip(axes, worst)):
        ax.imshow(X[i], cmap="gray_r")
        ax.set_title(f"{n + 1}\nlabel: {classes[y[i]]}\nmodel: {classes[picked[i]]}\n{sure[i]:.1%} sure",
                     fontsize=9.5)
        ax.axis("off")
    plt.suptitle(f"Wrong, and very sure: the 9 most confident mistakes out of {len(wrong):,}", fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- a drawing of your own

def load_drawing(path, show=True):
    """A drawing becomes a 28 x 28 photo, in the steps the Fashion-MNIST paper used:
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


# ---------------------------------------------------------------- research lab

def show_neurons(weights, n=24, title=None, fires=None):
    """The first n neurons of a layer: each neuron's 784 numbers drawn back as a 28 x 28 picture.
    With fires (how often each neuron switches on), the dead ones are marked in red."""
    w = weights.detach()[:n].reshape(-1, 28, 28)
    cols = 8
    rows = (len(w) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.85, rows * 2.0 + 0.8), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(w):
            limit = max(w[i].abs().max().item(), 1e-6)
            ax.imshow(w[i], cmap="RdBu_r", vmin=-limit, vmax=limit)
            dead = fires is not None and fires[i] == 0
            ax.set_title(f"neuron {i}" + (" · DEAD" if dead else ""), fontsize=10,
                         color=RED if dead else INK, fontweight="bold" if dead else "normal")
            if dead:
                ax.add_patch(plt.Rectangle((-0.5, -0.5), 28, 28, fill=False, edgecolor=RED, lw=3))
    if title is None and fires is not None:
        dead_all = int((fires == 0).sum())
        title = (f"{dead_all} of {len(fires)} neurons are dead: they never switch on, "
                 f"for any photo (red boxes)")
    plt.suptitle(title or "What studying changed in each neuron: red = weight went up, "
                          "blue = weight went down", fontsize=14)
    plt.tight_layout()
    plt.show()


def show_slides(photo, shift, steps=7):
    """One photo moved 0, 1, 2... pixels to the right by the shift function given."""
    photo = _as_float(photo).reshape(1, 28, 28)
    fig, axes = plt.subplots(1, steps, figsize=(1.9 * steps, 2.6))
    for k, ax in enumerate(axes):
        moved = shift(photo, k)[0]
        ax.imshow(moved, cmap="gray_r", vmin=0, vmax=1)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 28, 28, fill=False, edgecolor=GREY, lw=1))
        ax.set_title(f"{k} pixel{'s' if k != 1 else ''} right", fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_shift_curve(plain, shifted=None):
    """Mock-test accuracy as every photo slides 0, 1, 2... pixels to the right."""
    ks = list(range(len(plain)))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(ks, plain, "-o", color=RED, lw=2.6, ms=8, label="our model")
    below = shifted is not None                              # two curves: red numbers go underneath
    for k, v in zip(ks, plain):
        ax.text(k, v - 0.065 if below else v + 0.03, f"{v:.0%}", ha="center", color=RED,
                fontsize=11, fontweight="bold")
    if shifted is not None:
        ax.plot(ks, shifted, "-o", color=BLUE, lw=2.6, ms=8, label="studied with shifted photos")
        for k, v in zip(ks, shifted):
            ax.text(k, v + 0.03, f"{v:.0%}", ha="center", color=BLUE, fontsize=11, fontweight="bold")
    ax.axhline(0.1, color=GREY, ls="--", lw=1.2)
    ax.text(ks[-1], 0.115, "pure guessing: 10%", ha="right", color=GREY, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    ax.set_xlabel("pixels moved to the right", fontsize=12)
    ax.set_ylabel("mock test accuracy", fontsize=12)
    ax.set_title("Slide the photo, and the model gets lost" if shifted is None else
                 "Practice with slid photos helps, but only for slides like the ones it practised",
                 fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_data_curve(results):
    """Mock-test accuracy against how many practice photos the model studied."""
    sizes = [n for n, _ in results]
    scores = [s for _, s in results]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(sizes, scores, "-o", color=GREEN, lw=2.6, ms=9)
    for n, s in results:
        ax.text(n, s + 0.012, f"{s:.1%}", ha="center", fontsize=12, fontweight="bold", color=GREEN)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
    ax.set_ylim(min(scores) - 0.05, 1.0)
    ax.set_xlabel("photos in the practice book (each step is 10 times more)", fontsize=12)
    ax.set_ylabel("mock test accuracy", fontsize=12)
    ax.set_title("More photos, better assistant: but every 10x buys less", fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

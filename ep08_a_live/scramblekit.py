"""Race helpers for scramble.py.

race(): trains the same model on the normal photos and on the scrambled photos, one epoch each in turn, with two
live bars, a timer and each model's exam score after every epoch.
slide(): slides every exam photo a few pixels to the right and scores a model again.
Every run is appended to scramble_log.txt, and race() saves both models to scramble_run.pt.

Same recipe as Episode 7: Fashion-MNIST scaled to 0-1, the first 50,000 training photos, the 10,000 test photos as
the exam, torch.manual_seed(42) before each model, Adam (lr 0.001), batches of 64 reshuffled every epoch, 12 epochs,
on the CPU. Each model shuffles with its own copy of torch's random stream, so the two runs never change each other's
batches.
"""
import copy
import os
import platform
import sys
import time
from datetime import datetime

import torch
from torch import nn

os.system("")  # ANSI colours and cursor moves in the Windows console
sys.stdout.reconfigure(encoding="utf-8")  # the bars' block characters
YEL, GRN, RED, DIM, BOLD, RST = "\033[93m", "\033[92m", "\033[91m", "\033[90m", "\033[1m", "\033[0m"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "scramble_log.txt")
RUN = os.path.join(HERE, "scramble_run.pt")
DATA = [os.path.join(HERE, "data"), os.path.join(HERE, "..", "ep07_live", "data")]
SEED, LR, BATCH = 42, 0.001, 64
NAMES = ("NORMAL PHOTOS", "SCRAMBLED PHOTOS")


def fashion_mnist():
    """Fashion-MNIST from the local files (never downloads): the 50,000 photos to learn from and their labels, then
    the 10,000-photo exam and its labels. Pixels scaled from 0-255 to 0-1."""
    from torchvision import datasets
    root = next((r for r in DATA if os.path.isdir(os.path.join(r, "FashionMNIST", "raw"))), None)
    if root is None:
        raise SystemExit("Fashion-MNIST not found: put data\\FashionMNIST\\raw next to scramblekit.py")
    train = datasets.FashionMNIST(root, train=True, download=False)
    exam = datasets.FashionMNIST(root, train=False, download=False)
    X = train.data.float() / 255
    return X[:50000], train.targets[:50000], exam.data.float() / 255, exam.targets


def accuracy(model, X, y):
    with torch.no_grad():
        picked = model(X).argmax(dim=1)
    return round((picked == y).float().mean().item(), 3)


class _Runner:
    """One model, its optimizer, and its own copy of the random stream a DataLoader would shuffle with."""

    def __init__(self, name, new_model, X, X_exam):
        self.name, self.X, self.X_exam = name, X, X_exam
        torch.manual_seed(SEED)
        self.model = new_model()
        self.start = copy.deepcopy(self.model.state_dict())
        self.rng = torch.Generator()
        self.rng.set_state(torch.get_rng_state())
        self.opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.done, self.scores = 0, []

    def batches(self, y):
        n = len(self.X)
        torch.empty((), dtype=torch.int64).random_(generator=self.rng)  # the DataLoader's own draw
        seed = int(torch.empty((), dtype=torch.int64).random_(generator=self.rng).item())  # its sampler's seed
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            yield self.X[idx], y[idx]


def race(new_model, normal, scrambled, labels, epochs=12):
    """Train new_model() on normal=(photos, exam photos) and on scrambled=(photos, exam photos), one epoch each in
    turn, with the live bars. labels=(labels, exam labels). Returns the two trained models."""
    y, y_exam = labels
    runners = [_Runner(NAMES[0], new_model, *normal), _Runner(NAMES[1], new_model, *scrambled)]
    loss_fn = nn.CrossEntropyLoss()
    per_epoch = -(-len(y) // BATCH)
    print(f"{DIM}Fashion-MNIST · {len(y):,} photos to learn from · same model, same seed · {epochs} epochs · "
          f"CPU{RST}\n\n\n", end="")
    t0 = time.perf_counter()

    def draw():
        out = ["\033[3F"]
        for r in runners:
            frac = r.done / (epochs * per_epoch)
            bar = "█" * int(frac * 20) + "░" * (20 - int(frac * 20))
            ep = r.done // per_epoch + (1 if r.done % per_epoch else 0)
            score = f"{r.scores[-1]:.1%}" if r.scores else "--"
            out.append(f"\033[K{BOLD}{r.name:<17}{RST} {bar}  {DIM}epoch{RST} {ep:>2}/{epochs}  "
                       f"{DIM}exam{RST} {YEL}{BOLD}{score:>6}{RST}\n")
        out.append(f"\033[K{DIM}time{RST} {YEL}{BOLD}{time.perf_counter() - t0:6.1f} s{RST}\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    last = 0.0
    for _ in range(epochs):
        for r in runners:
            for xb, yb in r.batches(y):
                prediction = r.model(xb)
                loss = loss_fn(prediction, yb)
                r.opt.zero_grad()
                loss.backward()
                r.opt.step()
                r.done += 1
                now = time.perf_counter()
                if now - last > 0.05:
                    draw()
                    last = now
            r.scores.append(accuracy(r.model, r.X_exam, y_exam))
            draw()
    secs = time.perf_counter() - t0
    draw()

    n, s = runners[0].scores[-1], runners[1].scores[-1]
    print(f"{GRN}{BOLD}DONE{RST}  {epochs} epochs each · {secs:.1f} s")
    print(f"{BOLD}EXAM{RST} {DIM}· 10,000 photos neither model ever saw{RST}")
    print(f"  {NAMES[0]:<17} {YEL}{BOLD}{n:.1%}{RST}")
    print(f"  {NAMES[1]:<17} {YEL}{BOLD}{s:.1%}{RST}   {DIM}{abs(n - s) * 100:.1f} points apart{RST}")

    main = sys.modules.get("__main__")
    torch.save({"order": getattr(main, "order", None), "start": runners[0].start,
                "normal": runners[0].model.state_dict(), "scrambled": runners[1].model.state_dict(),
                "scores": {r.name: r.scores for r in runners}, "secs": secs}, RUN)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  race  epochs={epochs}  secs={secs:.1f}  normal={n}  "
                f"scrambled={s}  normal_by_epoch={','.join(map(str, runners[0].scores))}  "
                f"scrambled_by_epoch={','.join(map(str, runners[1].scores))}  | torch {torch.__version__} | "
                f"{torch.get_num_threads()} threads | {platform.processor()}\n")
    return runners[0].model, runners[1].model


def slide(model, X, y, pixels=2):
    """Slide every exam photo `pixels` to the right, as Episode 7 did, and score the model again."""
    moved = torch.roll(X, shifts=pixels, dims=2)
    moved[:, :, :pixels] = 0  # blank the columns that wrapped round
    before, after = accuracy(model, X, y), accuracy(model, moved, y)
    print(f"\n{BOLD}SLIDE{RST} every exam photo {pixels} pixel{'' if pixels == 1 else 's'} to the right")
    print(f"  {NAMES[0]:<17} {YEL}{BOLD}{before:.1%}{RST}  →  {RED}{BOLD}{after:.1%}{RST}")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  slide  pixels={pixels}  before={before}  after={after}\n")

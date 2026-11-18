# N-BEATS: a deep learning model (neural network) that we train ourselves, written in plain PyTorch.
#
# Idea (Oreshkin et al., 2020): a stack of small fully connected blocks. Every block looks at the
# input window (the last 104 weeks), makes a guess of the next 8 weeks (forecast) and also a guess of
# the input itself (backcast). The backcast is removed from the input, so the next block only works on
# what is still not explained. All block forecasts are added together.
#
# One global model learns from all 29 series together. Every input window is divided by its own
# average, so cheap and expensive products look the same to the network. It predicts 3 quantiles
# (10%, 50%, 90%) with the quantile (pinball) loss, so we also get an 80% range.
#
# Training is fast (a few seconds on a laptop), so like the other models we train a new network at
# every origin, only on the history up to that origin (no leakage). The last 8 weeks of that history
# are used to stop training early when the network stops getting better (early stopping).

import numpy as np
import pandas as pd
import torch
from torch import nn

FREQ = "W-MON"
INPUT_SIZE = 104       # look back 2 years
QUANTILES = [0.1, 0.5, 0.9]
N_BLOCKS = 4
HIDDEN = 256
EPOCHS = 30            # passes over the training windows
BATCH = 256
LR = 1e-3
VAL_WEEKS = 8          # last 8 weeks of the history are used to stop training early
PATIENCE = 4
REFIT_EVERY = 1        # train a new network at every origin
SEED = 1


class Block(nn.Module):
    def __init__(self, input_size, h, n_q):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_size, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.backcast = nn.Linear(HIDDEN, input_size)
        self.forecast = nn.Linear(HIDDEN, h * n_q)

    def forward(self, x):
        z = self.body(x)
        return self.backcast(z), self.forecast(z)


class NBeats(nn.Module):
    def __init__(self, input_size, h, n_q):
        super().__init__()
        self.h, self.n_q = h, n_q
        self.blocks = nn.ModuleList([Block(input_size, h, n_q) for _ in range(N_BLOCKS)])

    def forward(self, x):
        residual = x
        total = 0
        for block in self.blocks:
            back, fore = block(residual)
            residual = residual - back
            total = total + fore
        return total.view(-1, self.h, self.n_q)


def quantile_loss(pred, y):
    # pred: (batch, h, 3), y: (batch, h)
    loss = 0
    for i, q in enumerate(QUANTILES):
        diff = y - pred[:, :, i]
        loss = loss + torch.maximum(q * diff, (q - 1) * diff).mean()
    return loss


def make_windows(values, h, start, end):
    # all windows (input of 104 weeks, target of h weeks) whose target ends before index `end`
    xs, ys = [], []
    for t in range(max(INPUT_SIZE, start), end - h + 1):
        x = values[t - INPUT_SIZE:t]
        y = values[t:t + h]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        level = x.mean()
        if level <= 0:
            continue
        xs.append(x / level)
        ys.append(y / level)
    return xs, ys


def series_arrays(hist):
    return {uid: g.sort_values("ds")["y"].to_numpy(dtype="float32") for uid, g in hist.groupby("unique_id")}


def train(hist, h):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    arrays = series_arrays(hist)
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for v in arrays.values():
        n = len(v)
        x, y = make_windows(v, h, 0, n - VAL_WEEKS)        # training windows
        tr_x += x
        tr_y += y
        x, y = make_windows(v, h, n - VAL_WEEKS - h + 1, n)  # validation windows at the end
        va_x += x
        va_y += y
    tr_x, tr_y = torch.tensor(np.array(tr_x)), torch.tensor(np.array(tr_y))
    va_x, va_y = torch.tensor(np.array(va_x)), torch.tensor(np.array(va_y))

    model = NBeats(INPUT_SIZE, h, len(QUANTILES))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(tr_x))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            opt.zero_grad()
            loss = quantile_loss(model(tr_x[idx]), tr_y[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = quantile_loss(model(va_x), va_y).item()
        # early stopping: keep the best model, stop when it does not improve for PATIENCE epochs
        if val < best:
            best, bad = val, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, hist, h):
    rows = []
    for uid, g in hist.groupby("unique_id"):
        g = g.sort_values("ds")
        v = g["y"].to_numpy(dtype="float32")
        x = v[-INPUT_SIZE:]
        if len(x) < INPUT_SIZE:
            x = np.concatenate([np.full(INPUT_SIZE - len(x), x[0]), x])
        level = x.mean()
        with torch.no_grad():
            out = model(torch.tensor(x / level).unsqueeze(0))[0].numpy() * level
        dates = pd.date_range(g["ds"].iloc[-1] + pd.Timedelta(weeks=1), periods=h, freq=FREQ)
        rows.append(pd.DataFrame({
            "unique_id": uid, "ds": dates,
            "y_pred": out[:, 1], "q10": out[:, 0], "q50": out[:, 1], "q90": out[:, 2],
        }))
    return pd.concat(rows, ignore_index=True)


def make_nbeats(refit_every=REFIT_EVERY):
    state = {"model": None, "calls": 0, "fits": 0}

    def nbeats(hist, h):
        if state["model"] is None or state["calls"] % refit_every == 0:
            state["model"] = train(hist, h)
            state["fits"] += 1
        state["calls"] += 1
        return predict(state["model"], hist, h)

    nbeats.state = state
    return nbeats

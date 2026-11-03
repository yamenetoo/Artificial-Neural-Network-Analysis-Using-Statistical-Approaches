"""E7: PyTorch networks on image data (MNIST 5k sample): deep MLP and CNN; audit + compression baselines."""
import time, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from common import *
import annstat as A
from mlxtend.data import mnist_data
from sklearn.model_selection import train_test_split
torch.set_num_threads(2)
X, y = mnist_data(); X = (X / 255.0).astype("float32"); y = y.astype(int)

def make(kind, seed):
    torch.manual_seed(seed)
    if kind == "mlp3":
        net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        hid, head, k = net[5], net[6], 64
    else:
        net = nn.Sequential(nn.Unflatten(1, (1, 28, 28)), nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3), nn.ReLU(),
                            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16 * 5 * 5, 128), nn.ReLU(), nn.Linear(128, 10))
        hid, head, k = net[9], net[10], 128
    return net, hid, head, k

def train(net, Xtr, ytr, epochs=25):
    opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-5)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]; opt.zero_grad()
            nn.functional.cross_entropy(net(Xt[idx]), yt[idx]).backward(); opt.step()

rows, audit = [], []
methods = ("lsc_qr", "lsc_leverage", "lsc_random", "obs", "thinet", "merge", "weight_norm", "cluster_mean", "pca")
for kind in ("mlp3", "cnn"):
    for seed in range(8):
        t0 = time.time()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=seed, stratify=y)
        net, hid, head, k = make(kind, seed); train(net, Xtr, ytr)
        a = A.from_torch(net, hid, head, classes=np.arange(10))
        lin_in = [m for m in net if isinstance(m, nn.Linear)][-2]
        inc = (lin_in.weight.detach().numpy().T, lin_in.bias.detach().numpy())
        ks = [k // 4, k // 2]
        df = A.reduction_path(a, Xtr, ytr, Xte, yte, ks, methods, incoming=inc, seed=seed)
        df["model"], df["seed"], df["width"] = kind, seed, k
        rows.append(df)
        if seed < 3:
            rep = A.StatisticalTuner(a, seed=0).analyze(Xtr[:1500], ytr[:1500], Xte, yte, n_boot=300, n_perm=20, input_tests_on=False, surrogate_on=False)
            r = rep.redundancy
            audit.append({"model": kind, "seed": seed, "k": k, "test_acc": float(np.mean(a.predict(Xte) == yte)),
                          "effective_rank": r["effective_rank"], "participation_ratio": r["participation_ratio"], "dead": r["n_dead"],
                          "assoc_sig": int(rep.class_separation.significant.sum()), "ablation_sig": int(rep.ablation.significant.sum())})
        print(kind, seed, round(time.time() - t0, 1), "s", flush=True)
pd.concat(rows).to_csv(f"{RES}/e7_torch_raw.csv", index=False)
pd.DataFrame(audit).to_csv(f"{RES}/e7_torch_audit.csv", index=False)

"""E9: CIFAR-10 (PNG copy of the original data, from a GitHub mirror) with a VGG-style CNN in PyTorch."""
import os, sys, time, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
from common import *
import annstat as A
torch.set_num_threads(max(1, (os.cpu_count() or 2)))
DATA = os.environ.get("ANNSTAT_DATA", "./data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
def load():
    """CIFAR-10 through torchvision (downloads on first use; works in Colab)."""
    from torchvision.datasets import CIFAR10
    tr, te = CIFAR10(DATA, train=True, download=True), CIFAR10(DATA, train=False, download=True)
    return np.asarray(tr.data), np.asarray(tr.targets), np.asarray(te.data), np.asarray(te.targets)
Xtr_all, ytr_all, Xte, yte = load()
mean, std = Xtr_all.reshape(-1, 3).mean(0) / 255, Xtr_all.reshape(-1, 3).std(0) / 255
def prep(X): return torch.tensor(((X / 255.0 - mean) / std).transpose(0, 3, 1, 2), dtype=torch.float32)
print("data", Xtr_all.shape, Xte.shape, flush=True)
def make(width=256):
    net = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
        nn.Linear(128 * 4 * 4, width), nn.ReLU(), nn.Linear(width, 10))
    return net, net[15], net[16]   # ReLU after the 256-unit dense layer, final Linear
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 20000; EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 12; SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
print("device", DEV, "n_train", NTR, "epochs", EPOCHS, "seeds", SEEDS, flush=True)
methods = ("lsc_qr", "lsc_leverage", "lsc_random", "obs", "thinet", "merge", "weight_norm")
Xte_t = prep(Xte[:5000]); yte5 = yte[:5000]
rows, audit = [], []
for seed in range(SEEDS):
    t0 = time.time(); rng = np.random.default_rng(seed); torch.manual_seed(seed)
    idx = rng.choice(len(ytr_all), NTR, replace=False)
    Xt, yt = prep(Xtr_all[idx]), torch.tensor(ytr_all[idx])
    net, hid, head = make(); net.to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(NTR)
        for i in range(0, NTR, 128):
            b = perm[i:i + 128]; opt.zero_grad(); nn.functional.cross_entropy(net(Xt[b].to(DEV)), yt[b].to(DEV)).backward(); opt.step()
        if ep % 4 == 3: print(" seed", seed, "epoch", ep + 1, round(time.time() - t0), "s", flush=True)
    net.cpu(); a = A.from_torch(net, hid, head, classes=np.arange(10), batch_size=1000)
    fit = rng.choice(NTR, 6000, replace=False); Xf, yf = Xt[fit].numpy(), yt[fit].numpy()
    lin_in = [m for m in net if isinstance(m, nn.Linear)][-2]; inc = (lin_in.weight.detach().numpy().T, lin_in.bias.detach().numpy())
    df = A.reduction_path(a, Xf, yf, Xte_t.numpy(), yte5, [32, 64, 128], methods, incoming=inc, seed=seed)
    df["model"], df["seed"], df["width"] = "cifar_cnn", seed, 256; rows.append(df)
    rep = A.StatisticalTuner(a, seed=0).analyze(Xf[:2000], yf[:2000], Xte_t.numpy(), yte5, n_boot=300, n_perm=20, input_tests_on=False, surrogate_on=False)
    r = rep.redundancy
    audit.append({"seed": seed, "k": 256, "test_acc": float(np.mean(a.predict(Xte_t.numpy()) == yte5)), "effective_rank": r["effective_rank"],
                  "participation_ratio": r["participation_ratio"], "dead": r["n_dead"], "assoc_sig": int(rep.class_separation.significant.sum()),
                  "ablation_sig": int(rep.ablation.significant.sum())})
    pd.concat(rows).to_csv(f"{RES}/e9_cifar_raw.csv", index=False); pd.DataFrame(audit).to_csv(f"{RES}/e9_cifar_audit.csv", index=False)
    print("seed", seed, "done", round(time.time() - t0), "s", audit[-1], flush=True)

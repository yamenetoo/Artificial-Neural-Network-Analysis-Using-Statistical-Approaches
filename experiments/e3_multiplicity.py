"""E3: false discoveries when 50 neurons are tested at a raw alpha=0.05 vs BH / Holm."""
import numpy as np, pandas as pd
from common import *
import annstat as A
rng = np.random.default_rng(7)
R, k, k_rel = 400, 50, 10
rows = []
for n in (60, 150, 300):
    fwer_raw = fwer_holm = 0; fdp_raw = []; fdp_bh = []; pw_raw = []; pw_bh = []
    for _ in range(R):
        y = rng.integers(0, 3, n)
        H = rng.normal(size=(n, k))
        H[:, :k_rel] += 0.5 * y[:, None]               # first 10 neurons carry class signal
        sep = A.class_separation_tests(H, y)
        null = np.arange(k) >= k_rel
        raw = sep.p_value.values < .05; bh = sep.p_adj.values < .05
        holm = A.multitest.holm(sep.p_value.values) < .05
        fwer_raw += (raw & null).any(); fwer_holm += (holm & null).any()
        fdp_raw.append((raw & null).sum() / max(raw.sum(), 1)); fdp_bh.append((bh & null).sum() / max(bh.sum(), 1))
        pw_raw.append(raw[~null].mean()); pw_bh.append(bh[~null].mean())
    rows.append({"n": n, "P(any false positive) raw": fwer_raw / R, "P(any false positive) Holm": fwer_holm / R,
                 "FDR raw": np.mean(fdp_raw), "FDR BH": np.mean(fdp_bh), "power raw": np.mean(pw_raw), "power BH": np.mean(pw_bh)})
df = pd.DataFrame(rows); df.to_csv(f"{RES}/e3_multiplicity.csv", index=False); print(df.round(3))

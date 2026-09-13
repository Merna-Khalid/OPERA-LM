"""EXPLORATORY follow-up to depth_survival.py: MLP (nonlinear) probes for
the mid-range recency dead zone.

D3 found byte t-k linearly decodable from the final Fenwick readout only
for k<=4 (k=8 at untrained level). Two readings: (a) the information is
absent; (b) it is present but not linearly accessible -- plausible given
D5's rank collapse (PR ~27/512). A 2-layer MLP probe distinguishes them.

Probes on the incumbent checkpoint (L=2, bytes T1024): sources P_1 and
S_out_1, targets byte t-k for k in {4,8,16} plus next byte t+1 (head
calibration; ceiling = model argmax acc). Train 80% of seqs, test 20%,
AdamW, 4 epochs, batch 8192, MPS.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.depth_survival import (build_model, capture,  # noqa: E402
                                        load_corpus, make_batch_full,
                                        tokens_of)

T = 1024
CKPT = os.path.join('runs_reprs', 'rx_fgate_wd',
                    'opera_v8_0_pe-none_rotfree_amp_foreach_cmp-default_'
                    'gpudata_aux0.25_opt-muon_mfg_mwd0.01.pt')
LAGS = (4, 8, 16)


def mlp_acc(Xtr, ytr, Xte, yte, C, device, seed=0, epochs=4):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(),
                        nn.Linear(256, C)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    Xtr_t = Xtr.float().to(device)
    ytr_t = ytr.to(device)
    Xte_t = Xte.float().to(device)
    yte_t = yte.to(device)
    n = Xtr_t.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 8192):
            j = perm[i:i + 8192]
            loss = nn.functional.cross_entropy(net(Xtr_t[j]), ytr_t[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (net(Xte_t).argmax(-1) == yte_t).float().mean().item()


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    data, meta = load_corpus()
    seqs = data[1][:128]
    model = build_model(2, CKPT)
    model.to(device)
    t0 = time.time()
    cap = capture(model, seqs, device)
    ids = np.concatenate([make_batch_full(seqs[i:i + 16], T)[0].numpy()
                          for i in range(0, len(seqs), 16)])
    lens = np.concatenate([make_batch_full(seqs[i:i + 16], T)[1].numpy()
                           for i in range(0, len(seqs), 16)])
    cut = 102
    res = {'argmax_acc': cap['argmax_acc']}
    for src in ('P_1', 'S_out_1'):
        for k in LAGS:
            Xtr, ytr = tokens_of(cap[src][:cut], ids[:cut], lens[:cut],
                                 min_t=k, shift=-k)
            Xte, yte = tokens_of(cap[src][cut:], ids[cut:], lens[cut:],
                                 min_t=k, shift=-k)
            acc = mlp_acc(Xtr, ytr, Xte, yte, 259, device)
            res[f'{src}_lag{k}_mlp'] = acc
            print(f"  {src} lag {k:>2}: MLP {acc:.3f}")
        Xtr, ytr = tokens_of(cap[src][:cut], ids[:cut], lens[:cut],
                             shift=+1)
        Xte, yte = tokens_of(cap[src][cut:], ids[cut:], lens[cut:],
                             shift=+1)
        acc = mlp_acc(Xtr, ytr, Xte, yte, 259, device)
        res[f'{src}_next_mlp'] = acc
        print(f"  {src} next  : MLP {acc:.3f} "
              f"(head argmax ceiling {cap['argmax_acc']:.3f})")
    import json
    out = os.path.join('runs_reprs', 'mlp_probe_rx_fgate_wd.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()

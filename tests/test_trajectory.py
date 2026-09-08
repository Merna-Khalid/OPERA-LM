"""Fast unit test for opera_lm.trajectory: metric shapes, finiteness,
and exact position alignment on a tiny random-init OPERA model."""
import torch

from opera_lm.model import OperaSpinorFenwickTree
from opera_lm.garden_path import prefix_states, surprisal, _restore
from opera_lm.trajectory import trajectory_metrics, analyze


def _tiny_model():
    torch.manual_seed(0)
    return OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                  num_layers=2, pe_mode='none',
                                  fold_mode='left').eval()


def test_trajectory_metrics_shapes_and_alignment():
    model = _tiny_model()
    B, T = 6, 12
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, 101, (B, T), generator=g)
    lengths = torch.full((B,), T, dtype=torch.long)

    emb = _restore(model)
    e0 = emb(ids).detach()
    states = prefix_states(model, e0, ids, lengths)     # [B, T, D]
    sup = surprisal(model, ids, lengths, e0)            # [B, T-1]
    _restore(model)
    assert model.word_emb is emb, "word_emb left patched"

    m = trajectory_metrics(states)
    step, turn, extrap = m['step'], m['turn'], m['extrap']

    # shapes: step at t in [0,T-2], turn/extrap at t in [1,T-2]
    assert step.shape == (B, T - 1)
    assert turn.shape == (B, T - 2)
    assert extrap.shape == (B, T - 2)

    # finiteness and sign invariants
    for t in (step, turn, extrap):
        assert torch.isfinite(t).all()
    assert (step >= 0).all()
    assert (turn >= -1e-6).all() and (turn <= 2 + 1e-6).all()
    assert (extrap >= 0).all()

    # exact alignment: step[t] = ||f_{t+1} - f_t||
    manual_step = (states[:, 1:] - states[:, :-1]).norm(dim=-1)
    assert torch.allclose(step, manual_step, atol=1e-5)

    # turn[:, k] is the direction change at arrival position t = k+1:
    # 1 - cos(step vector t-1, step vector t)
    d = states[:, 1:] - states[:, :-1]
    cos = ((d[:, :-1] * d[:, 1:]).sum(-1)
           / (d[:, :-1].norm(dim=-1) * d[:, 1:].norm(dim=-1)))
    assert torch.allclose(turn, 1 - cos, atol=1e-5)

    # surprisal aligned to the same arrival positions: [B, T-1], t+1
    assert sup.shape == (B, T - 1)
    assert torch.isfinite(sup).all() and (sup > 0).all()


def test_analyze_runs_and_aggregate_r_bounded():
    model = _tiny_model()
    B, T = 8, 10
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1, 101, (B, T), generator=g)
    lengths = torch.full((B,), T, dtype=torch.long)
    emb = _restore(model)
    e0 = emb(ids).detach()
    states = prefix_states(model, e0, ids, lengths)
    sup = surprisal(model, ids, lengths, e0)
    _restore(model)

    m = trajectory_metrics(states)
    embn = e0[:, 1:].norm(dim=-1)
    freq = torch.zeros_like(sup)          # constant control: degenerate,
                                          # exercises the skip path
    res = analyze(m['step'], m['turn'], m['extrap'], sup, embn, freq=freq,
                  label='tiny')
    assert res['n_seqs'] == B
    for name, row in res['correlations_vs_surprisal'].items():
        r = row['agg_r']
        assert r != r or abs(r) <= 1.0, f"{name} agg r out of range"
    assert len(res['pooled_corr']['matrix']) == 6     # incl. log_freq

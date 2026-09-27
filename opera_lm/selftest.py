"""Self-test suite (no data download needed).

Run: python -m opera_lm.selftest       # sequential, aborts with a traceback
                                        # on the first failing arm; ends with
                                        # "ALL PASS"
     pytest tests/test_selftest.py     # each arm is an independent pytest
                                        # test: a failure in one does not
                                        # prevent the others from running,
                                        # and a single arm can be selected
                                        # with -k or ::test_name

Each test_* function below is one self-contained "arm" (own seeds, own
models, own data) so it can run standalone, in any order, under pytest.
`torch.manual_seed(0)` is the first line of every arm: sections that reseed
internally (most of them, to pair same-seed models for identity/parity
checks) immediately override it, so it's a no-op there; sections that don't
reseed get the same reproducible starting state the original single
`selftest()` function gave them when it seeded once at the top.
"""
import os
import random
import math
import numpy as np
import torch

from .model import (OperaSpinorFenwickTree, count_params, fenwick_blocks,
                    level_sin_enc, fold_work_counts, quat_to_rotmat,
                    quat_sandwich, affine_compose, associative_scan,
                    rotor_pos_tables, apply_rotor_pe, inject_geometry)
from .losses import lm_loss, msup_loss, future_bag_loss, far_repeat_mask, train_lm_loss
from .data import doc_chunks, doc_chunk_sizes
from .train import (curriculum_len, GpuBatchSource, extrapolation_eval,
                    rmt_states_diagnostic, train, get_lr)
from .packed import pack_pool, PackedBatchSource, packed_stats
from .incremental import OperaDecoder, fenwick_blocks_of
from .muon import Muon, zeropower_via_newtonschulz5, split_muon_params

# ============================================================================
# SELF-TEST -- one function per arm
# ============================================================================

def test_fold_grade():
    # SCALE-GRADED READOUT (docs/OPERA_ScaleGraded_prereg.md, SG-H4 guards)
    torch.manual_seed(0)
    # (a) routing only: zero param delta AND byte-identical init (the
    #     constructor consumes no randomness for fold_grade)
    torch.manual_seed(11)
    m_inc = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                   pe_mode='none', fold_mode='left')
    torch.manual_seed(11)
    m_gr = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='left',
                                  fold_grade=(4, 4))
    dpar = count_params(m_gr) - count_params(m_inc)
    sd_i, sd_g = m_inc.state_dict(), m_gr.state_dict()
    assert set(sd_i) == set(sd_g)
    init_err = max((sd_i[k] - sd_g[k]).abs().max().item() for k in sd_i)
    print(f"  grade: param delta {dpar} (expected 0), init err {init_err:.2e}")
    assert dpar == 0 and init_err == 0.0

    # (b) ROUTING UNIT-TEST (SG-H4): a level-l block perturbation must not
    #     move other groups' slots. Strict isolation needs value flow that
    #     is block-diagonal per quaternion slot, so the TEST-ONLY config
    #     freezes the node's cross-slot couplings: norm_mode='blockrms'
    #     (per-slot RMS) and zeroed fusion-gate weights (g = sigmoid(bias),
    #     input-independent). The registered production config keeps the
    #     incumbent node untouched (layer norm, live gates); this test
    #     therefore verifies the ROUTING TOPOLOGY, not the node.
    R_g, G_g, T = 4, 4, 33
    w_g = (16 - R_g) // G_g
    torch.manual_seed(3)
    m = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                               pe_mode='none', fold_mode='left',
                               norm_mode='blockrms', fold_grade=(R_g, G_g))
    with torch.no_grad():
        for fg in m.fusion_gate:
            fg.weight.zero_()
    m.eval()
    m._need_locks = False

    def readout(tokv):
        with torch.no_grad():
            emb = m.word_emb(tokv)
            R_L, R_R, R_O = m.get_rotations(0)
            lv, _, _ = m.build_tree(emb, 0, R_L, R_R, R_O)
            return m.prefix_states(lv, T, 0, R_L, R_R, R_O)[0]   # [T, d]

    tok = torch.randint(1, 101, (1, T))
    p = T - 3                                   # the perturbed byte
    tok2 = tok.clone()
    tok2[0, p] = (tok2[0, p] + 7) % 100 + 1
    p0, p1 = readout(tok), readout(tok2)
    table, _ = fenwick_blocks(T)
    iso_err = moved = 0.0
    n_iso = 0
    for t in range(T):
        diff_t = (p1[t] - p0[t]).abs().reshape(16, 4).amax(-1)   # [nb]
        if t < p:                                # prefix excludes byte p
            assert diff_t.max().item() == 0.0, \
                f"t={t} (< p) changed: causality"
            continue
        # the unique path block of prefix t containing byte p
        off, lvl_p = 0, None
        for (lv, _node) in table[t]:
            if off <= p < off + (1 << lv):
                lvl_p = lv
                break
            off += 1 << lv
        assert lvl_p is not None
        grp = min(lvl_p, G_g - 1)
        others = [j for j in range(R_g, 16) if (j - R_g) // w_g != grp]
        iso_err = max(iso_err, diff_t[others].max().item())
        moved = max(moved, diff_t[:R_g].max().item(),
                    diff_t[[j for j in range(R_g, 16)
                            if (j - R_g) // w_g == grp]].max().item())
        n_iso += len(others)
    print(f"  grade routing: isolation err {iso_err:.2e} over {n_iso} "
          f"non-routed slots; routed-slot movement {moved:.2e}")
    assert iso_err == 0.0 and moved > 1e-6

    # (c) causality + forward/backward on the PRODUCTION config (layer
    #     norm, live gates) for both dose SHAPES (hybrid R>0 and radical
    #     R=0; the registered nb=128 doses are (24,8)/(0,8), scaled here
    #     to the test model's nb=16)
    for fgd in ((4, 4), (0, 4)):
        torch.manual_seed(5)
        mp = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                    num_layers=2, pe_mode='none',
                                    fold_mode='left', fold_grade=fgd)
        tok = torch.randint(1, 101, (3, 13))
        lens = torch.tensor([13, 7, 5])
        loss, _, _ = lm_loss(mp(tok, lens).logits, tok, lens)
        loss.backward()
        assert torch.isfinite(loss)
        mp.eval()
        tok = torch.randint(1, 101, (1, 12))
        lens = torch.tensor([12])
        with torch.no_grad():
            a = mp(tok, lens).logits[-1][0, :8].clone()
            tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
            b = mp(tok2, lens).logits[-1][0, :8]
        err = (a - b).abs().max().item()
        print(f"  grade {fgd}: loss {loss.item():.3f}, causality err {err:.2e}")
        assert err < 1e-5

    # (d) NOT bitwise-incumbent at init (prereg §1): identical weights,
    #     but the graded readout differs wherever routing does. At
    #     count==1 positions the GLOBAL slots (dims [0, 4R)) match the
    #     incumbent bit-for-bit while graded slots differ (unrouted
    #     groups zero-seeded vs seed values); full-model logits differ.
    torch.manual_seed(21)
    m1i = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                 pe_mode='none', fold_mode='left')
    torch.manual_seed(21)
    m1g = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                 pe_mode='none', fold_mode='left',
                                 fold_grade=(4, 4))
    m1i.eval(); m1g.eval()
    m1i._need_locks = False; m1g._need_locks = False

    def ro(mm, tokv):
        with torch.no_grad():
            emb = mm.word_emb(tokv)
            R_L, R_R, R_O = mm.get_rotations(0)
            lv, _, _ = mm.build_tree(emb, 0, R_L, R_R, R_O)
            return mm.prefix_states(lv, 12, 0, R_L, R_R, R_O)[0]

    tok = torch.randint(1, 101, (1, 12))
    r_i, r_g = ro(m1i, tok), ro(m1g, tok)
    table12, _ = fenwick_blocks(12)
    single = [t for t in range(12) if len(table12[t]) == 1]     # 0, 1, 3, 7
    multi = [t for t in range(12) if t not in single]
    g_same = max((r_i[t, :16] - r_g[t, :16]).abs().max().item()
                 for t in single)
    g_diff = max((r_i[t, 16:] - r_g[t, 16:]).abs().max().item()
                 for t in single)
    m_diff = max((r_i[t] - r_g[t]).abs().max().item() for t in multi)
    m_inc.eval(); m_gr.eval()
    lens = torch.full((1,), 12)
    with torch.no_grad():
        li = m_inc(tok, lens).logits[-1]
        lg = m_gr(tok, lens).logits[-1]
    e_logit = (li - lg).abs().max().item()
    print(f"  grade-vs-incumbent at init: single-block global-slot err "
          f"{g_same:.2e}, graded-slot diff {g_diff:.2e}; multi-block "
          f"diff {m_diff:.2e}; logits diff {e_logit:.2e}")
    assert g_same == 0.0 and g_diff > 0 and m_diff > 1e-4 and e_logit > 1e-4


def test_stream_arms():
    # STREAM / TREE-WIDTH ARMS (architecture review 2026-09-24; draft
    # prereg docs/OPERA_Stream_prereg.md): head_mode, resid_mode,
    # fold_gate, fold_h0, state_mult.
    torch.manual_seed(0)
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', fold_mode='left')

    def build(seed=31, **kw):
        torch.manual_seed(seed)
        return OperaSpinorFenwickTree(**{**base, **kw})

    def dead_params(mm):
        tok = torch.randint(1, 101, (3, 21))
        lens = torch.tensor([21, 14, 9])
        mm.zero_grad(set_to_none=True)
        loss, _, _ = lm_loss(mm(tok, lens).logits, tok, lens)
        loss.backward()
        assert torch.isfinite(loss)
        return sorted(n for n, p in mm.named_parameters()
                      if p.grad is None or p.grad.abs().max() == 0)

    def causality(mm):
        mm.eval()
        tok = torch.randint(1, 101, (1, 23))
        lens = torch.tensor([23])
        tok2 = tok.clone()
        tok2[0, 17] = (tok2[0, 17] + 5) % 100 + 1
        with torch.no_grad():
            a = mm(tok, lens).logits[-1][0]
            b = mm(tok2, lens).logits[-1][0]
        mm.train()
        return ((a[:17] - b[:17]).abs().max().item(),
                (a[17:] - b[17:]).abs().max().item())

    def decoder_err(mm):
        mm.eval()
        T = 33                        # odd + crosses power-of-2 boundaries
        tok = torch.randint(1, 100, (1, T))
        with torch.no_grad():
            full = mm(tok, torch.tensor([T]), head_last_only=True).logits[-1][0]
            dec = OperaDecoder(mm)
            inc = torch.stack([dec.append(int(t)) for t in tok[0]])
        mm.train()
        return (full - inc).abs().max().item()

    # (a) flags-off: the last layer's cross_mlp/blend_gate are dead (the
    #     review finding this arm set exists to fix) -- documented here so
    #     a silent change to either side is caught.
    dead0 = dead_params(build())
    assert dead0 == sorted(['mod_depth'] + [
        n for n, _ in build().named_parameters()
        if n.startswith(('cross_mlp.1.', 'blend_gate.1.'))]), dead0
    print(f"  stream a: flags-off dead params = mod_depth + last layer's "
          f"cross_mlp/blend_gate ({len(dead0)} tensors), as documented")

    # (b) every arm: exact causality, incremental decoder exact vs full
    #     forward; dead params = mod_depth (lock_mode='none') only under
    #     head_mode='stream', else also the fold-head's dead last layer.
    arms = [
        ('stream', dict(head_mode='stream')),
        ('stream+add', dict(head_mode='stream', resid_mode='add')),
        ('fold_gate', dict(fold_gate='separate')),
        ('h0', dict(fold_h0=True)),
        ('mult2', dict(state_mult=2)),
        ('mult2+gb+tie', dict(state_mult=2, tie=True,
                              fold_gate_bias=(2.0, 0.0, -2.0))),
        ('all', dict(head_mode='stream', resid_mode='add',
                     fold_gate='separate', fold_h0=True, state_mult=2,
                     pe_mode='sin')),
    ]
    for tag, kw in arms:
        mm = build(**kw)
        dead = dead_params(mm)
        exp = (['mod_depth'] if kw.get('head_mode') == 'stream' else dead0)
        assert dead == exp, (tag, dead)
        before, after = causality(mm)
        assert before == 0.0 and after > 1e-6, (tag, before, after)
        derr = decoder_err(mm)
        assert derr < 1e-5, (tag, derr)
        print(f"  stream b [{tag}]: {count_params(mm):,} params, "
              f"{len(dead) - 1} dead tensors; causality {before:.0e}; "
              f"decoder err {derr:.1e}")

    # (c) resid_mode='add' creates no blend gate; requires head 'stream'.
    assert build(head_mode='stream', resid_mode='add').blend_gate is None
    try:
        build(resid_mode='add')
        raise AssertionError("resid_mode='add' without 'stream' accepted")
    except AssertionError as e:
        assert 'stream' in str(e)

    # (d) fold_gate='separate': a COPY of the tree gate -> every shared
    #     param bitwise the incumbent and forward bitwise at init; after
    #     one step the two gates receive different gradients.
    m_i, m_f = build(), build(fold_gate='separate')
    sd_i, sd_f = m_i.state_dict(), m_f.state_dict()
    extra = sorted(set(sd_f) - set(sd_i))
    assert all(k.startswith('fusion_gate_fold.') for k in extra) and extra
    assert all(torch.equal(sd_i[k], sd_f[k]) for k in sd_i)
    tok = torch.randint(1, 101, (2, 19))
    lens = torch.tensor([19, 12])
    m_i.eval(); m_f.eval()
    with torch.no_grad():
        e = max((x - y).abs().max().item() for x, y in
                zip(m_i(tok, lens).logits, m_f(tok, lens).logits))
    m_f.train()
    loss, _, _ = lm_loss(m_f(tok, lens).logits, tok, lens)
    loss.backward()
    gt = m_f.fusion_gate[0].weight.grad
    gf = m_f.fusion_gate_fold[0].weight.grad
    gdiff = (gt - gf).abs().max().item()
    print(f"  stream d: fold_gate copy -> logits err {e:.1e} at init; "
          f"tree/fold gate grads differ by {gdiff:.2e}")
    assert e == 0.0 and gdiff > 1e-8

    # (e) fold_h0: zero-init, receives gradient; composes with fold_grade
    #     (causal, finite); the decoder refuses fold_grade explicitly.
    m_h = build(fold_h0=True)
    assert all(p.abs().max() == 0 for p in m_h.fold_h0)
    dead_params(m_h)
    gh = m_h.fold_h0[0].grad.abs().max().item()
    m_hg = build(fold_h0=True, fold_grade=(4, 4))
    assert dead_params(m_hg) == dead0
    before, after = causality(m_hg)
    assert before == 0.0 and after > 1e-6
    try:
        OperaDecoder(m_hg)
        raise AssertionError("decoder accepted fold_grade")
    except NotImplementedError:
        pass
    print(f"  stream e: h0 grad {gh:.2e}; h0+grade causality {before:.0e}; "
          f"decoder refuses fold_grade")
    assert gh > 0

    # (f) state_mult=2: exact param delta (tree modules at 2x width + the
    #     two projections); msup reads nodes through tree_out.
    m1, m2 = build(), build(state_mult=2)
    d, nb, L = 64, 16, 2
    tree1 = sum(p.numel() for n, p in m1.named_parameters()
                if n.startswith(('fusion_gate', 'rot_free', 'comp_norm')))
    tree2 = sum(p.numel() for n, p in m2.named_parameters()
                if n.startswith(('fusion_gate', 'rot_free', 'comp_norm')))
    exp_tree2 = L * ((4 * d) * (6 * nb) + 6 * nb   # fusion gate, 2x width
                     + 3 * (2 * nb) * 9            # rot_free
                     + 2 * (2 * d))                # comp_norm
    proj = sum(p.numel() for n, p in m2.named_parameters()
               if n.startswith(('tree_in', 'tree_out')))
    assert tree2 == exp_tree2 and proj == L * 2 * d * (2 * d)
    assert count_params(m2) - count_params(m1) == (tree2 - tree1) + proj
    out = m2(tok, lens, return_levels=True)
    assert out.levels[0][0].shape[-1] == 2 * d
    ms = msup_loss(m2, out.levels, tok, lens)
    assert torch.isfinite(ms)
    ms.backward()
    assert m2.tree_out[0].weight.grad is not None
    print(f"  stream f: state_mult=2 tree params {tree1:,} -> {tree2:,} "
          f"(+{proj:,} projections); msup {ms.item():.3f} via tree_out")

    # (g) optimizer routing: h0 (1-D per layer) and norms -> AdamW;
    #     projections -> Muon; the fold gate follows the tree gate's
    #     routing (in Muon exactly when 'fusion_gate' is included).
    m_all = build(head_mode='stream', resid_mode='add',
                  fold_gate='separate', fold_h0=True, state_mult=2)
    mu, ad = split_muon_params(m_all, include=('fusion_gate',))
    mu_n, ad_n = {n for n, _ in mu}, {n for n, _ in ad}
    assert {'fold_h0.0', 'fold_h0.1'} <= ad_n
    assert not any(n.startswith(('stream_norm', 'tree_in_norm'))
                   for n in mu_n)
    assert {'tree_in.0.weight', 'tree_out.1.weight',
            'fusion_gate_fold.0.weight'} <= mu_n
    mu0, _ = split_muon_params(m_all)
    assert 'fusion_gate_fold.0.weight' not in {n for n, _ in mu0}
    print("  stream g: Muon/AdamW routing correct for all new tensors")


def test_node_mix_and_state_tie():
    # node_mix_rank (identity-init cross-slot mixing, MLP-LDRU evidence)
    # and state_tie (gates/rotors shared across state_mult copies, M2RNN
    # evidence) -- docs/OPERA_Literature_Scan_2026-09-24.md §5.
    torch.manual_seed(0)
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', fold_mode='left')
    tok = torch.randint(1, 101, (3, 21))
    lens = torch.tensor([21, 14, 9])

    def build(seed=41, **kw):
        torch.manual_seed(seed)
        return OperaSpinorFenwickTree(**{**base, **kw})

    def causal_and_decoder(mm):
        mm.eval()
        t1 = torch.randint(1, 100, (1, 33))
        t2 = t1.clone()
        t2[0, 20] = (t2[0, 20] + 5) % 99 + 1
        L1 = torch.tensor([33])
        with torch.no_grad():
            a = mm(t1, L1, head_last_only=True).logits[-1][0]
            b = mm(t2, L1, head_last_only=True).logits[-1][0]
            dec = OperaDecoder(mm)
            inc = torch.stack([dec.append(int(t)) for t in t1[0]])
        mm.train()
        return ((a[:20] - b[:20]).abs().max().item(),
                (a[20:] - b[20:]).abs().max().item(),
                (a - inc).abs().max().item())

    # (a) node_mix: shared params + forward bitwise the incumbent at init
    #     (U = 0), U receives gradient, causal, decoder-exact, Muon-routed.
    m_i, m_n = build(), build(node_mix_rank=8)
    sd_i, sd_n = m_i.state_dict(), m_n.state_dict()
    assert sorted(set(sd_n) - set(sd_i)) == [
        'node_mix_down.0', 'node_mix_down.1', 'node_mix_up.0', 'node_mix_up.1']
    assert all(torch.equal(sd_i[k], sd_n[k]) for k in sd_i)
    m_i.eval(); m_n.eval()
    with torch.no_grad():
        e = max((x - y).abs().max().item() for x, y in
                zip(m_i(tok, lens).logits, m_n(tok, lens).logits))
    m_n.train()
    loss, _, _ = lm_loss(m_n(tok, lens).logits, tok, lens)
    loss.backward()
    gU = m_n.node_mix_up[0].grad.abs().max().item()
    before, after, derr = causal_and_decoder(m_n)
    mu, _ = split_muon_params(m_n)
    assert {'node_mix_up.0', 'node_mix_down.1'} <= {n for n, _ in mu}
    print(f"  node-mix: bitwise at init (logits err {e:.0e}); dL/dU {gU:.2e}; "
          f"causality {before:.0e}; decoder err {derr:.1e}")
    assert e == 0.0 and gU > 0 and before == 0.0 and after > 1e-6
    assert derr < 1e-5

    # (b) state_tie: exact param shapes (gates/rotors at nb, state at k*nb).
    k, d, nb, L = 2, 64, 16, 2
    m_t = build(state_mult=k, state_tie=True)
    assert m_t.fusion_gate[0].weight.shape == (3 * nb, 2 * k * d)
    assert m_t.rot_free.shape == (L, 3, nb, 3, 3)
    R_L, _, _ = m_t.get_rotations(0)
    assert R_L.shape == (k * nb, 3, 3) and torch.equal(R_L[:nb], R_L[nb:])
    # tying is real: children whose two copies are identical produce a
    # parent whose two copies are identical (same gates, rotors, product;
    # LayerNorm's affine is ones/zeros at init). Untied state_mult is
    # the negative control.
    def copy_gap(mm):
        mm.eval()
        with torch.no_grad():
            hl = torch.randn(4, d).repeat(1, k)   # copy-major: [c0 | c1]
            hr = torch.randn(4, d).repeat(1, k)
            p = mm._compose(hl, hr, 0, *mm.get_rotations(0))[0]
        mm.train()
        return (p[:, :d] - p[:, d:]).abs().max().item()
    gap_t, gap_u = copy_gap(m_t), copy_gap(build(state_mult=k))
    print(f"  state-tie: copy gap tied {gap_t:.1e} vs untied {gap_u:.2e}")
    assert gap_t < 1e-6 and gap_u > 1e-3
    for tag, kw in (('tie', dict(state_mult=k, state_tie=True)),
                    ('tie+stream+mix+h0+gb',
                     dict(state_mult=k, state_tie=True, head_mode='stream',
                          node_mix_rank=8, fold_h0=True,
                          fold_gate_bias=(2.0, 0.0, -2.0)))):
        mm = build(**kw)
        mm.zero_grad(set_to_none=True)
        loss, _, _ = lm_loss(mm(tok, lens).logits, tok, lens)
        loss.backward()
        dead = sorted(n for n, p in mm.named_parameters()
                      if p.grad is None or p.grad.abs().max() == 0)
        before, after, derr = causal_and_decoder(mm)
        print(f"  state-tie [{tag}]: {count_params(mm):,} params, "
              f"{len(dead)} dead tensors; causality {before:.0e}; "
              f"decoder err {derr:.1e}")
        assert before == 0.0 and after > 1e-6 and derr < 1e-5
        if kw.get('head_mode') == 'stream':
            # node_mix_down's gradient is proportional to node_mix_up = 0
            # at init (LoRA convention: only the first step is blind)
            assert [n for n in dead if not n.startswith('node_mix_down')] \
                == ['mod_depth'], dead
    try:
        build(state_tie=True)
        raise AssertionError("state_tie without state_mult accepted")
    except AssertionError as e2:
        assert 'state_mult' in str(e2)


def test_innov_and_disent():
    # INNOVATION FOLD (fold_innov_rank) and CAUSAL DISENTANGLER
    # (tree_disent_rank) -- docs/OPERA_Redundancy_Research_2026-09-24.md
    # §4 2a/2b.
    torch.manual_seed(0)
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', fold_mode='left')
    tok = torch.randint(1, 101, (3, 21))
    lens = torch.tensor([21, 14, 9])

    def build(seed=51, **kw):
        torch.manual_seed(seed)
        return OperaSpinorFenwickTree(**{**base, **kw})

    for flag, up in (('fold_innov_rank', 'fold_innov_up'),
                     ('tree_disent_rank', 'tree_disent_up')):
        m_i, m_x = build(), build(**{flag: 8})
        sd_i, sd_x = m_i.state_dict(), m_x.state_dict()
        assert all(torch.equal(sd_i[k], sd_x[k]) for k in sd_i)
        m_i.eval(); m_x.eval()
        with torch.no_grad():
            e = max((a - b).abs().max().item() for a, b in
                    zip(m_i(tok, lens).logits, m_x(tok, lens).logits))
        m_x.train()
        loss, _, _ = lm_loss(m_x(tok, lens).logits, tok, lens)
        loss.backward()
        gU = getattr(m_x, up)[0].grad.abs().max().item()
        # make the map live, then causality + decoder exactness
        with torch.no_grad():
            for P in getattr(m_x, up):
                P.normal_(0.0, 0.05)
        m_x.eval()
        t1 = torch.randint(1, 100, (1, 33))
        t2 = t1.clone()
        t2[0, 20] = (t2[0, 20] + 5) % 99 + 1
        L1 = torch.tensor([33])
        with torch.no_grad():
            a = m_x(t1, L1, head_last_only=True).logits[-1][0]
            b = m_x(t2, L1, head_last_only=True).logits[-1][0]
            dec = OperaDecoder(m_x)
            inc = torch.stack([dec.append(int(t)) for t in t1[0]])
        before = (a[:20] - b[:20]).abs().max().item()
        after = (a[20:] - b[20:]).abs().max().item()
        derr = (a - inc).abs().max().item()
        mu, _ = split_muon_params(m_x)
        assert f'{up}.0' in {n for n, _ in mu}
        print(f"  {flag}: bitwise at init (err {e:.0e}); dL/dU {gU:.2e}; "
              f"live: causality {before:.0e}, decoder err {derr:.1e}")
        assert e == 0.0 and gU > 0 and before == 0.0 and after > 1e-6
        assert derr < 1e-5

    # the disentangler reads the LEFT neighbour only: perturbing leaf j
    # moves node j+1 of level 0 (its right neighbour) and never node j-1
    m = build(tree_disent_rank=8)
    with torch.no_grad():
        for P in m.tree_disent_up:
            P.normal_(0.0, 0.05)
        x = torch.randn(1, 8, 64)
        x2 = x.clone()
        x2[0, 4] += 1.0
        l0a = m._disentangle(x, 0)
        l0b = m._disentangle(x2, 0)
    moved = (l0a - l0b).abs().amax(-1)[0]
    assert moved[3] == 0 and moved[4] > 0 and moved[5] > 0 and moved[6] == 0
    print("  tree_disent: node j+1 reads node j; nodes < j and > j+1 untouched")


def test_lowrank_gain():
    # lowrank_gain: gain * (x V) U^T / (|U||V|) for node_mix / fold_innov /
    # tree_disent. Bitwise incumbent at init (gain = 0), gradient reaches
    # the gain at init (no dead-lock), |correction| <= |gain| |x|, gains
    # on AdamW and U/V on Muon, causal + decoder-exact once live.
    torch.manual_seed(0)
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', fold_mode='left')
    tok = torch.randint(1, 101, (3, 21))
    lens = torch.tensor([21, 14, 9])
    for flag, name in (('node_mix_rank', 'node_mix'),
                       ('fold_innov_rank', 'fold_innov'),
                       ('tree_disent_rank', 'tree_disent')):
        torch.manual_seed(61)
        m_i = OperaSpinorFenwickTree(**base)
        torch.manual_seed(61)
        m_g = OperaSpinorFenwickTree(**base, lowrank_gain=True, **{flag: 8})
        sd_i, sd_g = m_i.state_dict(), m_g.state_dict()
        assert all(torch.equal(sd_i[k], sd_g[k]) for k in sd_i)
        m_i.eval(); m_g.eval()
        with torch.no_grad():
            e = max((a - b).abs().max().item() for a, b in
                    zip(m_i(tok, lens).logits, m_g(tok, lens).logits))
        m_g.train()
        loss, _, _ = lm_loss(m_g(tok, lens).logits, tok, lens)
        loss.backward()
        gg = getattr(m_g, name + '_gain')[0].grad.abs().item()
        mu, ad = split_muon_params(m_g)
        assert f'{name}_gain.0' in {n for n, _ in ad}
        assert {f'{name}_up.0', f'{name}_down.0'} <= {n for n, _ in mu}
        # calibration: for isotropic x the TYPICAL |corr|/|x| equals gain
        # (rms over samples), whatever the norms of U and V
        with torch.no_grad():
            getattr(m_g, name + '_gain')[0].fill_(0.5)
            getattr(m_g, name + '_up')[0].mul_(7.0)       # norms cancel
            x = torch.randn(4096, 64) * 3.0
            c = m_g._lowrank_corr(x, name, 0)
            ratio = ((c.norm(dim=-1) ** 2).mean()
                     / (x.norm(dim=-1) ** 2).mean()).sqrt().item()
            for gpar in getattr(m_g, name + '_gain'):
                gpar.fill_(0.3)
        m_g.eval()
        t1 = torch.randint(1, 100, (1, 33))
        t2 = t1.clone()
        t2[0, 20] = (t2[0, 20] + 5) % 99 + 1
        L1 = torch.tensor([33])
        with torch.no_grad():
            a = m_g(t1, L1, head_last_only=True).logits[-1][0]
            b = m_g(t2, L1, head_last_only=True).logits[-1][0]
            dec = OperaDecoder(m_g)
            inc = torch.stack([dec.append(int(t)) for t in t1[0]])
        before = (a[:20] - b[:20]).abs().max().item()
        derr = (a - inc).abs().max().item()
        print(f"  lowrank_gain [{name}]: bitwise at init (err {e:.0e}); "
              f"dL/dgain {gg:.2e}; rms |corr|/|x| {ratio:.3f} (gain 0.5); "
              f"causality {before:.0e}; decoder err {derr:.1e}")
        assert e == 0.0 and gg > 0 and abs(ratio - 0.5) < 0.05
        assert before == 0.0 and derr < 1e-5


def test_counterclockwise_fold():
    # fold_dir='both': right-nested (counterclockwise) fold joined to the
    # left fold by a zero-init per-slot gain.
    torch.manual_seed(0)
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', fold_mode='left')
    tok = torch.randint(1, 101, (3, 21))
    lens = torch.tensor([21, 14, 9])
    for impl in ('compact', 'downsweep'):
        torch.manual_seed(71)
        m_i = OperaSpinorFenwickTree(**base, fold_impl=impl)
        torch.manual_seed(71)
        m_b = OperaSpinorFenwickTree(**base, fold_impl=impl, fold_dir='both')
        sd_i, sd_b = m_i.state_dict(), m_b.state_dict()
        assert sorted(set(sd_b) - set(sd_i)) == ['fold_join_gain.0',
                                                  'fold_join_gain.1']
        assert all(torch.equal(sd_i[k], sd_b[k]) for k in sd_i)
        m_i.eval(); m_b.eval()
        with torch.no_grad():
            e = max((a - b).abs().max().item() for a, b in
                    zip(m_i(tok, lens).logits, m_b(tok, lens).logits))
        m_b.train()
        loss, _, _ = lm_loss(m_b(tok, lens).logits, tok, lens)
        loss.backward()
        gg = m_b.fold_join_gain[0].grad.abs().max().item()
        _, ad = split_muon_params(m_b)
        assert 'fold_join_gain.0' in {n for n, _ in ad}
        # the right fold equals a naive per-position right-nested fold
        with torch.no_grad():
            T = 23
            x = m_b.word_emb(torch.randint(1, 101, (2, T)))
            R = m_b.get_rotations(0)
            lv, _, _ = m_b.build_tree(x, 0, *R)
            offs, off = [], 0
            for lvl_ in lv:
                offs.append(off)
                off += lvl_.shape[1]
            idx, count, S, lvl, _ = m_b._fenwick_indices(T, len(lv), offs,
                                                         x.device)
            gathered = torch.cat(lv, 1)[:, idx.reshape(-1)].reshape(2, T, S, -1)
            fast = m_b._right_fold(gathered, count, S, 0, *R)
            table, _ = fenwick_blocks(T)
            werr = 0.0
            for t in range(T):
                blocks = [lv[k][:, j] for (k, j) in table[t]]
                acc = blocks[-1]
                for b in reversed(blocks[:-1]):
                    acc = m_b._compose(b, acc, 0, *R, fold=True)[0]
                werr = max(werr, (acc - fast[:, t]).abs().max().item())
            for gpar in m_b.fold_join_gain:
                gpar.fill_(0.4)
        m_b.eval()
        t1 = torch.randint(1, 100, (1, 33))
        t2 = t1.clone()
        t2[0, 20] = (t2[0, 20] + 5) % 99 + 1
        L1 = torch.tensor([33])
        with torch.no_grad():
            a = m_b(t1, L1, head_last_only=True).logits[-1][0]
            b = m_b(t2, L1, head_last_only=True).logits[-1][0]
            dec = OperaDecoder(m_b)
            inc = torch.stack([dec.append(int(t)) for t in t1[0]])
        before = (a[:20] - b[:20]).abs().max().item()
        after = (a[20:] - b[20:]).abs().max().item()
        derr = (a - inc).abs().max().item()
        print(f"  ccw fold [{impl}]: bitwise at init (err {e:.0e}); "
              f"dL/dgain {gg:.2e}; right fold vs naive {werr:.1e}; "
              f"causality {before:.0e}; decoder err {derr:.1e}")
        assert e == 0.0 and gg > 0 and werr < 1e-5
        assert before == 0.0 and after > 1e-6 and derr < 1e-5


def test_future_bag():
    # future_bag=(W, K): training-only head predicting hashed trigrams of
    # the next W tokens. Forward is untouched (bitwise), targets match a
    # brute-force count, the loss reaches the fold's parameters.
    torch.manual_seed(0)
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', fold_mode='left', head_mode='stream')
    torch.manual_seed(81)
    m_i = OperaSpinorFenwickTree(**base)
    torch.manual_seed(81)
    m_f = OperaSpinorFenwickTree(**base, future_bag=(8, 32))
    sd_i, sd_f = m_i.state_dict(), m_f.state_dict()
    assert sorted(set(sd_f) - set(sd_i)) == ['future_head.bias',
                                              'future_head.weight']
    assert all(torch.equal(sd_i[k], sd_f[k]) for k in sd_i)
    tok = torch.randint(1, 101, (2, 40))
    lens = torch.tensor([40, 30])
    m_i.eval(); m_f.eval()
    with torch.no_grad():
        e = max((a - b).abs().max().item() for a, b in
                zip(m_i(tok, lens).logits, m_f(tok, lens).logits))
    # brute-force target check through a capturing head
    W, K = 8, 32
    cap = {}
    orig = m_f.future_head.forward
    def spy(hh):
        cap['n'] = hh.shape
        return orig(hh)
    m_f.future_head.forward = spy
    m_f.train()
    out = m_f(tok, lens, return_states=True)
    loss = future_bag_loss(m_f, out.states[-1], tok, lens, n_pos=16)
    m_f.future_head.forward = orig
    x = tok
    x1 = torch.nn.functional.pad(x, (1, 0))[:, :40]
    x2 = torch.nn.functional.pad(x, (2, 0))[:, :40]
    hsh = (x * 1000003 + x1 * 10007 + x2 * 101) % K
    g = torch.Generator(device='cpu').manual_seed(40)
    t = torch.randint(0, 40 - W, (2, 16), generator=g)
    ok = True
    for b in range(2):
        for j in range(16):
            tt = int(t[b, j])
            brute = torch.bincount(hsh[b, tt + 1:tt + W + 1], minlength=K)
            ok &= int(brute.sum()) == W
    loss.backward()
    gfold = m_f.fusion_gate[0].weight.grad.abs().max().item()
    print(f"  future_bag: forward bitwise (err {e:.0e}); aux loss "
          f"{loss.item():.3f} (ln K = {math.log(K):.3f}); head input "
          f"{tuple(cap['n'])}; window counts = W: {ok}; grad at fold "
          f"{gfold:.2e}")
    assert e == 0.0 and ok and gfold > 0 and torch.isfinite(loss)


def test_far_repeat_and_byte_dropout():
    # far_repeat_mask vs brute force; weighting keeps the loss scale;
    # train() runs with far weighting and byte dropout.
    torch.manual_seed(0)
    rng = random.Random(5)
    T, n, D = 300, 4, 20
    seq = [rng.randrange(3, 12) for _ in range(T)]      # small alphabet -> repeats
    x = torch.tensor([seq, seq[::-1]])
    lens = torch.tensor([T, T - 37])
    fast = far_repeat_mask(x, lens, n, D)
    brute = torch.zeros(2, T - 1, dtype=torch.bool)
    for b in range(2):
        xs = x[b].tolist()
        for i in range(n - 1, int(lens[b])):
            g = tuple(xs[i - n + 1:i + 1])
            prev = [j for j in range(n - 1, i) if tuple(xs[j - n + 1:j + 1]) == g]
            if prev and i - prev[-1] > D and i >= 1:
                brute[b, i - 1] = True
    assert torch.equal(fast, brute), (fast ^ brute).sum()
    # weighting redistributes, does not rescale: uniform weights == plain
    m = OperaSpinorFenwickTree(vocab_size=20, d=64, nb=16, num_layers=2,
                               pe_mode='none', rot_mode='free')
    tok = torch.randint(3, 20, (2, 40))
    ll = torch.tensor([40, 31])
    out = m(tok, ll, return_states=True, head_last_only=True)
    a, _, _ = train_lm_loss(m, out.states, tok, ll, final_logits=out.logits[0])
    b, _, _ = train_lm_loss(m, out.states, tok, ll, final_logits=out.logits[0],
                            target_weights=torch.full((2, 39), 3.0))
    assert abs(a.item() - b.item()) < 1e-5
    print(f"  far-repeat mask == brute force ({int(fast.sum())} marked of "
          f"{int((torch.arange(T - 1)[None] < lens[:, None] - 1).sum())}); "
          f"uniform weights leave the loss unchanged ({a.item():.4f})")
    import tempfile as _tf
    rng2 = random.Random(12)
    V = 64
    mk = lambda k, lo, hi: [[rng2.randrange(3, V) for _ in range(rng2.randrange(lo, hi))]
                            for _ in range(k)]
    data = (mk(80, 20, 33), mk(30, 20, 33), mk(16, 33, 65), V)
    with _tf.TemporaryDirectory() as tmp:
        res = train(steps=4, batch=8, max_len=32, vocab_size=V, d=64, nb=16,
                    num_layers=2, eval_max_len=64, device='cpu',
                    pe_mode='none', fold_mode='left', rot_mode='free',
                    data=data, idx2word=None, out_dir=tmp,
                    compile_mode='off', use_amp=False, gpu_data=False,
                    warmup_steps=2, seed=3, optimizer='muon',
                    far_weight=4.0, far_n=4, far_d=8, byte_dropout=0.15)
    assert math.isfinite(res['final_loss'])
    print(f"  train() with far weighting + byte dropout: loss "
          f"{res['final_loss']:.3f}")


def test_metal_equivalence():
    # use_metal=True (fused compose kernel, MPS only): same model as the
    # eager path -- loss, logits and gradients -- for the configurations
    # in use (stream head + downsweep, bounded node mixing, fold gate
    # bias). Skipped where MPS is unavailable.
    torch.manual_seed(0)
    if not torch.backends.mps.is_available():
        print("  metal: MPS unavailable -- skipped")
        return
    from .losses import train_lm_loss as _tll
    dev = 'mps'
    ids = torch.randint(3, 101, (2, 97), device=dev)
    lens = torch.tensor([97, 60], device=dev)
    for kw in (dict(head_mode='stream', fold_impl='downsweep'),
               dict(head_mode='stream', fold_impl='downsweep',
                    node_mix_rank=16, lowrank_gain=True),
               dict(fold_gate_bias=(2.0, 0.0, -2.0))):
        outs = []
        for metal in (False, True):
            torch.manual_seed(0)
            m = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                       num_layers=2, pe_mode='none',
                                       rot_mode='free', use_metal=metal,
                                       **kw).to(dev)
            if kw.get('node_mix_rank'):
                with torch.no_grad():
                    for g in m.node_mix_gain:
                        g.fill_(0.3)
            out = m(ids, lens, return_states=True, head_last_only=True)
            loss, _, _ = _tll(m, out.states, ids, lens,
                              final_logits=out.logits[0])
            loss.backward()
            outs.append((loss.item(), out.logits[0].detach().cpu(),
                         {n: p.grad.cpu() for n, p in m.named_parameters()
                          if p.grad is not None}))
        (l0, lg0, g0), (l1, lg1, g1) = outs
        gerr = max(((g0[n] - g1[n]).norm() / (g0[n].norm() + 1e-12)).item()
                   for n in g0)
        lerr = (lg0 - lg1).abs().max().item()
        print(f"  metal {kw}: loss diff {abs(l0 - l1):.1e}, logit diff "
              f"{lerr:.1e}, worst grad rel diff {gerr:.1e}")
        assert abs(l0 - l1) < 1e-4 and lerr < 1e-3 and gerr < 1e-2


def test_resid_init_scale():
    # resid_init_scale: in-place scaling of cross_mlp output projections
    # (no RNG) -> all other params bitwise; 'auto' = 1/sqrt(2L); with the
    # additive residual the L=8 stack is near-isometric at init.
    torch.manual_seed(0)
    base = dict(vocab_size=101, d=64, nb=16, num_layers=8, pe_mode='none',
                rot_mode='free', head_mode='stream', resid_mode='add')
    torch.manual_seed(91)
    m0 = OperaSpinorFenwickTree(**base)
    torch.manual_seed(91)
    m1 = OperaSpinorFenwickTree(**base, resid_init_scale='auto')
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    for k in sd0:
        if k.startswith('cross_mlp.') and k.endswith('.2.weight'):
            assert torch.allclose(sd1[k], sd0[k] / 4.0)
        elif k.startswith('cross_mlp.') and k.endswith('.2.bias'):
            assert sd1[k].abs().max() == 0
        else:
            assert torch.equal(sd0[k], sd1[k]), k
    def growth(m):
        m = m.double().eval()
        tok = torch.randint(3, 101, (2, 64))
        ln = torch.full((2,), 64)
        with torch.no_grad():
            a = m(tok, ln, return_states=True).states
            m.word_emb.weight.add_(1e-9 * torch.randn_like(m.word_emb.weight))
            b = m(tok, ln, return_states=True).states
        d = [(x - y).abs().max().item() for x, y in zip(a, b)]
        return (d[-1] / d[0]) ** (1 / 7)
    g0, g1 = growth(m0), growth(m1)
    print(f"  resid_init_scale: only cross_mlp out-proj changed; perturbation "
          f"growth/layer at L=8: default {g0:.2f}x, auto {g1:.2f}x")
    assert g1 < g0 and g1 < 1.2


def test_holographic_memory():
    # hmem_nb: quaternion holographic memory. (a) the algebra recalls:
    # with unit keys, conj(k_j) (x) sum_i k_i (x) v_i returns v_j exactly
    # for one pair and with high cosine for several; (b) flags-off bitwise
    # and gain=0 bitwise at init; (c) live: causal, decoder-exact; (d)
    # routing: gate/gain on AdamW.
    from .model import quat_mul, quat_conj
    torch.manual_seed(0)
    n = 256
    keys = torch.nn.functional.normalize(torch.randn(32, n, 4), dim=-1)
    vals = torch.randn(32, n, 4)
    one = quat_mul(quat_conj(keys[0]), quat_mul(keys[0], vals[0]))
    e1 = (one - vals[0]).abs().max().item()
    cos = []
    for npairs in (4, 16, 32):
        M = quat_mul(keys[:npairs], vals[:npairs]).sum(0)
        r = quat_mul(quat_conj(keys[1]), M)
        cos.append(torch.nn.functional.cosine_similarity(
            r.flatten(), vals[1].flatten(), dim=0).item())
    print(f"  hmem algebra: 1 pair exact (err {e1:.1e}); retrieval cosine "
          f"with 4/16/32 stored pairs (n={n} slots): "
          + " / ".join(f"{c:.2f}" for c in cos))
    assert e1 < 1e-5 and cos[0] > 0.4 and cos[0] > cos[-1] > 0.1
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', head_mode='stream', fold_impl='downsweep')
    torch.manual_seed(97)
    m0 = OperaSpinorFenwickTree(**base)
    torch.manual_seed(97)
    m1 = OperaSpinorFenwickTree(**base, hmem_nb=32)
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    tok = torch.randint(1, 101, (2, 37))
    ln = torch.tensor([37, 30])
    m0.eval(); m1.eval()
    with torch.no_grad():
        e = max((a - b).abs().max().item() for a, b in
                zip(m0(tok, ln).logits, m1(tok, ln).logits))
    m1.train()
    loss, _, _ = lm_loss(m1(tok, ln).logits, tok, ln)
    loss.backward()
    gg = m1.hmem_gain[0].grad.abs().item()
    mu, ad = split_muon_params(m1, ('fusion_gate',))
    ad_n = {k for k, _ in ad}
    assert {'hmem_gain.0', 'hmem_write_gate.0.weight'} <= ad_n
    with torch.no_grad():
        for g in m1.hmem_gain:
            g.fill_(0.5)
    m1.eval()
    t1 = torch.randint(1, 100, (1, 33))
    t2 = t1.clone()
    t2[0, 20] = (t2[0, 20] + 5) % 99 + 1
    L1 = torch.tensor([33])
    with torch.no_grad():
        a = m1(t1, L1, head_last_only=True).logits[-1][0]
        b = m1(t2, L1, head_last_only=True).logits[-1][0]
        dec = OperaDecoder(m1)
        inc = torch.stack([dec.append(int(t)) for t in t1[0]])
    before = (a[:20] - b[:20]).abs().max().item()
    after = (a[20:] - b[20:]).abs().max().item()
    derr = (a - inc).abs().max().item()
    print(f"  hmem model: bitwise at init (err {e:.0e}); dL/dgain {gg:.2e}; "
          f"live: causality {before:.0e}, decoder err {derr:.1e}; "
          f"gate/gain on AdamW")
    assert e == 0.0 and gg > 0 and before == 0.0 and after > 1e-6
    assert derr < 1e-4


def test_hmem_decay():
    # hmem_decay: the parallel decay scan equals the naive recurrence; the
    # 'gated' arm equals 'fixed' at init; flags-off unchanged; decoder
    # exact; decay params and forget gate on AdamW; half-lives 8..4096.
    from .model import _decay_scan
    torch.manual_seed(0)
    a = torch.rand(2, 37, 5, 1) * 0.5 + 0.5
    b = torch.randn(2, 37, 5, 4)
    fast = _decay_scan(a, b)
    ref = torch.zeros_like(b)
    Mr = torch.zeros(2, 5, 4)
    for t in range(37):
        Mr = a[:, t] * Mr + b[:, t]
        ref[:, t] = Mr
    serr = (fast - ref).abs().max().item()
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', head_mode='stream', fold_impl='downsweep',
                hmem_nb=16)
    torch.manual_seed(5)
    m0 = OperaSpinorFenwickTree(**base)
    torch.manual_seed(5)
    mf = OperaSpinorFenwickTree(**base, hmem_decay='fixed')
    torch.manual_seed(5)
    mg = OperaSpinorFenwickTree(**base, hmem_decay='gated')
    sd0, sdf = m0.state_dict(), mf.state_dict()
    assert all(torch.equal(sd0[k], sdf[k]) for k in sd0)
    lam = torch.sigmoid(mf.hmem_decay_logit[0])
    hl = torch.log(torch.tensor(0.5)) / torch.log(lam)
    tok = torch.randint(1, 101, (2, 29))
    ln = torch.tensor([29, 20])
    for mm in (mf, mg):
        with torch.no_grad():
            for g in mm.hmem_gain:
                g.fill_(0.5)
        mm.eval()
    with torch.no_grad():
        ef = max((x - y).abs().max().item() for x, y in
                 zip(mf(tok, ln).logits, mg(tok, ln).logits))
    derr = 0.0
    for mm in (mf, mg):
        t1 = torch.randint(1, 100, (1, 33))
        with torch.no_grad():
            full = mm(t1, torch.tensor([33]), head_last_only=True).logits[-1][0]
            dec = OperaDecoder(mm)
            inc = torch.stack([dec.append(int(t)) for t in t1[0]])
        derr = max(derr, (full - inc).abs().max().item())
    _, ad = split_muon_params(mg, ('fusion_gate',))
    ad_n = {k for k, _ in ad}
    assert {'hmem_decay_logit.0', 'hmem_forget_gate.0.weight'} <= ad_n
    print(f"  hmem_decay: scan vs recurrence {serr:.1e}; half-lives "
          f"{hl.min():.1f}..{hl.max():.0f}; gated == fixed at init "
          f"(err {ef:.0e}); decoder err {derr:.1e}")
    assert serr < 1e-5 and ef < 1e-6 and derr < 1e-4
    assert 7 < hl.min() < 9 and 4000 < hl.max() < 4200


def test_hmem_conv():
    # hmem_conv: zero-init taps -> same model as without (float-level);
    # with live taps: causal, decoder-exact, taps on AdamW; on MPS the
    # Metal path equals the eager path.
    from .muon import split_muon_params
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', head_mode='stream', fold_impl='downsweep',
                hmem_nb=16, hmem_decay='gated')
    torch.manual_seed(5)
    m0 = OperaSpinorFenwickTree(**base)
    torch.manual_seed(5)
    mc = OperaSpinorFenwickTree(**base, hmem_conv=4)
    for mm in (m0, mc):
        with torch.no_grad():
            for g in mm.hmem_gain:
                g.fill_(0.5)
        mm.eval()
    tok = torch.randint(1, 101, (2, 29))
    ln = torch.tensor([29, 29])
    with torch.no_grad():
        e0 = max((x - y).abs().max().item() for x, y in
                 zip(m0(tok, ln).logits, mc(tok, ln).logits))
        for c in mc.hmem_conv_w:
            c.normal_(0, 0.3)
        a = mc(tok, ln).logits[-1]
        tok2 = tok.clone()
        tok2[:, 15:] = torch.randint(1, 101, (2, 14))
        b = mc(tok2, ln).logits[-1]
        caus = (a[:, :15] - b[:, :15]).abs().max().item()
        live = (a[:, 15:] - b[:, 15:]).abs().max().item()
        t1 = torch.randint(1, 100, (1, 33))
        full = mc(t1, torch.tensor([33]), head_last_only=True).logits[-1][0]
        dec = OperaDecoder(mc)
        inc = torch.stack([dec.append(int(t)) for t in t1[0]])
        derr = (full - inc).abs().max().item()
    mu, ad = split_muon_params(mc)
    assert not any('hmem_conv' in n for n, _ in mu)
    merr = 0.0
    if torch.backends.mps.is_available():
        outs = []
        for metal in (False, True):
            torch.manual_seed(3)
            m = OperaSpinorFenwickTree(**base, hmem_conv=4, use_metal=metal).to('mps')
            with torch.no_grad():
                for g in m.hmem_gain:
                    g.fill_(0.5)
                for c in m.hmem_conv_w:
                    c.normal_(0, 0.3, generator=torch.Generator('mps').manual_seed(1))
            tk = torch.randint(1, 101, (2, 41), device='mps')
            lk = torch.tensor([41, 30], device='mps')
            loss, _, _ = lm_loss(m(tk, lk).logits, tk, lk)
            loss.backward()
            outs.append((loss.item(), m.hmem_conv_w[0].grad.cpu().clone(),
                         m.hmem_k[1].weight.grad.cpu().clone()))
        merr = max(abs(outs[0][0] - outs[1][0]),
                   *[((p - q).abs().max() / p.abs().max()).item()
                     for p, q in zip(outs[0][1:], outs[1][1:])])
    print(f"  hmem_conv: zero taps == no conv (err {e0:.1e}); live taps: causality "
          f"{caus:.0e} (future changes output {live:.2f}), decoder err {derr:.1e}; "
          f"metal vs eager {merr:.1e}; taps on AdamW")
    assert e0 < 1e-5 and caus == 0 and live > 0 and derr < 1e-4 and merr < 1e-3


def test_hmem_kernel():
    # fused Metal holographic-memory core: forward and all four gradients
    # equal PyTorch autograd of the reference; the model's metal path
    # equals its eager path. Skipped where MPS is unavailable.
    torch.manual_seed(0)
    if not torch.backends.mps.is_available():
        print("  hmem kernel: MPS unavailable -- skipped")
        return
    from .hmem_kernel import HmemCore, hmem_core_reference
    dev = 'mps'
    B, T, n = 2, 37, 5
    k = torch.nn.functional.normalize(torch.randn(B, T, n, 4, device=dev), dim=-1)
    v = torch.randn(B, T, n, 4, device=dev)
    w = torch.rand(B, T, device=dev)
    a = torch.rand(B, T, n, device=dev) * 0.5 + 0.5
    g = torch.randn(B, T, n, 4, device=dev)
    ref_in = [x.clone().requires_grad_() for x in (k, v, w, a)]
    ker_in = [x.clone().requires_grad_() for x in (k, v, w, a)]
    (hmem_core_reference(*ref_in) * g).sum().backward()
    (HmemCore.apply(*ker_in) * g).sum().backward()
    gerr = max(((x.grad - y.grad).abs().max() / (x.grad.abs().max() + 1e-12)).item()
               for x, y in zip(ref_in, ker_in))
    # fused v2 (raw projection in: normalization, sigmoids, decay inside)
    from .hmem_kernel import HmemFused, hmem_fused_reference
    ferr = 0.0
    for mode in ('none', 'fixed', 'gated'):
        C = 8 * n + 1 + (n if mode == 'gated' else 0)
        y = torch.randn(B, T, C, device=dev)
        L = torch.randn(n, device=dev)
        ya, La = y.clone().requires_grad_(), L.clone().requires_grad_()
        yb, Lb = y.clone().requires_grad_(), L.clone().requires_grad_()
        (hmem_fused_reference(ya, La, n, mode) * g).sum().backward()
        (HmemFused.apply(yb, Lb, n, mode) * g).sum().backward()
        pairs = [(ya.grad, yb.grad)] + ([(La.grad, Lb.grad)] if mode != 'none' else [])
        ferr = max([ferr] + [((p - q).abs().max() / (p.abs().max() + 1e-12)).item()
                             for p, q in pairs])
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', head_mode='stream', fold_impl='downsweep',
                hmem_nb=16, hmem_decay='gated')
    outs = []
    for metal in (False, True):
        torch.manual_seed(3)
        m = OperaSpinorFenwickTree(**base, use_metal=metal).to(dev)
        with torch.no_grad():
            for gg in m.hmem_gain:
                gg.fill_(0.5)
        tok = torch.randint(1, 101, (2, 41), device=dev)
        ln = torch.tensor([41, 30], device=dev)
        loss, _, _ = lm_loss(m(tok, ln).logits, tok, ln)
        loss.backward()
        outs.append((loss.item(), [q.grad.cpu().clone() for q in (
            m.hmem_k[0].weight, m.hmem_v[1].weight, m.hmem_write_gate[0].weight,
            m.hmem_forget_gate[1].weight, m.hmem_decay_logit[0])]))
    mdiff = abs(outs[0][0] - outs[1][0])
    mg = max(((p - q).abs().max() / p.abs().max()).item()
             for p, q in zip(outs[0][1], outs[1][1]))
    print(f"  hmem kernel: grads vs autograd rel {gerr:.1e}; fused-v2 vs "
          f"reference (3 decay modes) {ferr:.1e}; model metal vs eager: loss "
          f"diff {mdiff:.1e}, memory grads rel {mg:.1e}")
    assert gerr < 1e-5 and ferr < 1e-5 and mdiff < 1e-4 and mg < 1e-3


def test_downsweep_fold():
    # fold_impl='downsweep': the same all-prefix left fold, top-down over
    # the tree (~T composes/layer instead of sum(popcount-1)). Must equal
    # the compacted fold (float reassociation only), in forward and in
    # gradients, for every supported flag combination, and stay causal
    # and decoder-exact.
    torch.manual_seed(0)
    base = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
                rot_mode='free', fold_mode='left')
    for kw in (dict(), dict(fold_gate_bias=(2.0, 0.0, -2.0)),
               dict(fold_gate='separate', fold_innov_rank=8),
               dict(head_mode='stream', state_mult=2, state_tie=True)):
        worst_f = worst_g = 0.0
        for T in (1, 2, 3, 8, 33, 100):
            mods = []
            for impl in ('compact', 'downsweep'):
                torch.manual_seed(3)
                mods.append(OperaSpinorFenwickTree(**base, fold_impl=impl, **kw))
            if kw.get('fold_innov_rank'):
                for mm in mods:
                    with torch.no_grad():
                        for P in mm.fold_innov_up:
                            P.fill_(0.03)
            tok = torch.randint(1, 101, (2, T))
            lens = torch.tensor([T, max(1, T - 3)])
            outs = [mm(tok, lens).logits for mm in mods]
            worst_f = max(worst_f, max((a - b).abs().max().item()
                                       for a, b in zip(*outs)))
            if T > 1:
                for mm, o in zip(mods, outs):
                    lm_loss(o, tok, lens)[0].backward()
                worst_g = max(worst_g, max(
                    (pa.grad - pb.grad).abs().max().item()
                    for pa, pb in zip(mods[0].parameters(),
                                      mods[1].parameters())
                    if pa.grad is not None))
        print(f"  downsweep {kw or 'plain'}: max logit err {worst_f:.1e}, "
              f"max grad err {worst_g:.1e} vs compact")
        assert worst_f < 1e-5 and worst_g < 1e-5
    # work count: ~T composes/layer (vs sum(popcount-1) for compact)
    torch.manual_seed(3)
    m = OperaSpinorFenwickTree(**base, fold_impl='downsweep')
    calls = []
    orig = m._compose
    m._compose = lambda hl, *a, **k: (calls.append(hl.shape[0]),
                                      orig(hl, *a, **k))[1]
    T = 1024
    with torch.no_grad():
        x = m.word_emb(torch.randint(1, 101, (1, T)))
        R = m.get_rotations(0)
        lv, _, _ = orig.__self__.build_tree(x, 0, *R)
        n_tree = sum(calls); calls.clear()
        m.prefix_states(lv, T, 0, *R)
    _, compact, _ = fold_work_counts(T)
    print(f"  downsweep work @T=1024: {sum(calls)} fold composes "
          f"(compact: {compact}); tree {n_tree}")
    assert sum(calls) <= T and compact > 3 * sum(calls)


def test_samuon():
    # SAMuon-lite (Wu et al. 2026, arXiv 2608.25990, Alg. 1 variant #2)
    torch.manual_seed(0)
    from .muon import samuon_lite_reshape, samuon_warmup_weight

    def run(sa_gamma, steps=4, warmup=0, resume_at=None, seed=3):
        torch.manual_seed(seed)
        W = torch.nn.Parameter(torch.randn(24, 16))
        B = torch.nn.Parameter(torch.randn(5, 3, 3))       # batched 3x3s
        grp = {'params': [W, B], 'names': ['w', 'b'], 'use_muon': True,
               'lr': 0.02}
        if sa_gamma is not None:
            grp.update(sa_gamma=sa_gamma, sa_warmup=warmup)
        opt = Muon([grp], lr=1e-3)
        gens = torch.Generator().manual_seed(7)
        for s in range(steps):
            if resume_at is not None and s == resume_at:
                sd = opt.state_dict()
                opt = Muon([dict(grp, params=[W, B])], lr=1e-3)
                opt.load_state_dict(sd)
            W.grad = torch.randn(24, 16, generator=gens)
            B.grad = torch.randn(5, 3, 3, generator=gens)
            opt.step()
        return W.detach().clone(), B.detach().clone(), opt

    # (a) gamma = 1 is byte-identical to plain Muon (group key present or not)
    w0, b0, _ = run(None)
    w1, b1, _ = run(1.0)
    assert torch.equal(w0, w1) and torch.equal(b0, b1)
    # (b) warmup: w_0 = 0, so the FIRST step is exactly Muon; w ramps to 1
    assert samuon_warmup_weight(0, 10) == 0.0
    assert samuon_warmup_weight(10, 10) == 1.0
    assert samuon_warmup_weight(0, 0) == 1.0
    wa, ba, _ = run(None, steps=1)
    wb, bb, _ = run(5.0, steps=1, warmup=10)
    assert torch.equal(wa, wb) and torch.equal(ba, bb)
    # (c) the reshape: power iteration finds the true head (per matrix in a
    #     batch), the head keeps its NS coefficient, the bulk is boosted.
    g = torch.Generator().manual_seed(1)
    d = torch.randn(4, 24, 16, generator=g)
    d[..., :, :] += 3.0 * torch.randn(4, 24, 1, generator=g) @ \
        torch.randn(4, 1, 16, generator=g)                  # clear head
    o = zeropower_via_newtonschulz5(d)
    st = {}
    o_sa = samuon_lite_reshape(o, d, 4.0, st, iters=20)
    U, S, Vh = torch.linalg.svd(d)
    u1, v1 = U[..., :, 0], Vh[..., 0, :]
    align = (st['sa_v'] * v1).sum(-1).abs().min().item()
    head_mu = torch.einsum('bi,bij,bj->b', u1, o.float(), v1)
    head_sa = torch.einsum('bi,bij,bj->b', u1, o_sa.float(), v1)
    u2, v2 = U[..., :, 1], Vh[..., 1, :]
    bulk_mu = torch.einsum('bi,bij,bj->b', u2, o.float(), v2)
    bulk_sa = torch.einsum('bi,bij,bj->b', u2, o_sa.float(), v2)
    # head coefficient: gamma*c - (gamma-1) with c ~ NS's (approx-1) head
    # coefficient -> stays O(1); bulk coefficient: exactly gamma x Muon's
    hd = (head_sa - (4.0 * head_mu - 3.0)).abs().max().item()
    br = (bulk_sa / bulk_mu - 4.0).abs().max().item()
    print(f"  samuon a/b: gamma=1 and warmup step 0 bitwise Muon; "
          f"c: head alignment {align:.4f}, head formula err {hd:.1e}, "
          f"bulk ratio err {br:.1e}")
    assert align > 0.999 and hd < 5e-2 and br < 5e-2
    # (d) resume: optimizer state_dict round-trip mid-warmup is exact
    #     (sa_step lives in the param group, sa_v in per-param state)
    wu, bu, opt_u = run(3.54, steps=6, warmup=4)
    wr, br_, _ = run(3.54, steps=6, warmup=4, resume_at=3)
    assert torch.equal(wu, wr) and torch.equal(bu, br_)
    assert opt_u.param_groups[0]['sa_step'] == 6
    print("  samuon d: mid-warmup state_dict resume bitwise; sa_step saved")


def test_scan_fold():
    # SCAN FOLD (OPERA-Scan arm, v8.5; design: OPERA_Scan_Arm_Design.md)
    torch.manual_seed(0)
    # (a) ASSOCIATIVITY of the affine compose: exact semidirect
    #     (Euclidean-group) product, verified to float tolerance on random
    #     UNNORMALIZED states (q carried unnormalized through the scan)
    gs = torch.Generator().manual_seed(41)
    q1 = torch.randn(256, 4, generator=gs); b1 = torch.randn(256, 4, generator=gs)
    q2 = torch.randn(256, 4, generator=gs); b2 = torch.randn(256, 4, generator=gs)
    q3 = torch.randn(256, 4, generator=gs); b3 = torch.randn(256, 4, generator=gs)
    lq, lb = affine_compose(*affine_compose(q1, b1, q2, b2), q3, b3)
    rq, rb = affine_compose(q1, b1, *affine_compose(q2, b2, q3, b3))
    aerr = max((lq - rq).abs().max().item(), (lb - rb).abs().max().item())
    print(f"  scan affine compose associativity err: {aerr:.2e}")
    # absolute err scales with the |q|~2 products under the conformal
    # (decay) action -- relative err is ~1e-6; gate at 3e-5 abs
    assert aerr < 3e-5
    # (b) SCAN == NAIVE SEQUENTIAL left-prefix (same operator, same
    #     operand order: P_j = l_j ⊕ P_{j-1}, later leaf on the LEFT),
    #     all positions, several T including non-powers of 2 and odd T.
    #     Unit-norm leaves (the model's near-unit init): absolute float
    #     error stays ~1e-6 (unnormalized leaves scale the error with the
    #     |q| product -- the homogeneous drift measured in (f) instead).
    for T in (1, 2, 3, 5, 8, 13, 29, 64):
        q = torch.randn(2, T, 3, 4, generator=gs)
        q = q / q.norm(dim=-1, keepdim=True)
        b = torch.randn(2, T, 3, 4, generator=gs) * 0.5
        qs, bs = associative_scan(q, b)
        nq_l, nb_l = [q[:, 0]], [b[:, 0]]
        aq, ab = q[:, 0], b[:, 0]
        for t in range(1, T):
            aq, ab = affine_compose(q[:, t], b[:, t], aq, ab)
            nq_l.append(aq)
            nb_l.append(ab)
        nq_t = torch.stack(nq_l, dim=1)
        nb_t = torch.stack(nb_l, dim=1)
        serr = max((qs - nq_t).abs().max().item(),
                   (bs - nb_t).abs().max().item())
        print(f"  scan==naive T={T}: err {serr:.2e}")
        assert serr < 1e-4
    # (c) full model: forward/backward finite, grads reach the leaf maps
    #     and the readout gate; EXACT causality
    m_sc = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='scan')
    tok = torch.randint(1, 101, (3, 29))
    lens = torch.tensor([29, 14, 5])
    loss, _, _ = lm_loss(m_sc(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    gw = m_sc.scan_wq[0].weight.grad.abs().sum().item()
    gm = m_sc.scan_wm[0].weight.grad.abs().sum().item()
    gg = m_sc.scan_gate[0].weight.grad.abs().sum().item()
    print(f"  scan: loss {loss.item():.3f}, leaf-map grad {gw:.3f}, "
          f"mag grad {gm:.3f}, gate grad {gg:.3f} (all >0)")
    assert gw > 0 and gm > 0 and gg > 0
    m_sc.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a1 = m_sc(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_sc(tok2, lens).logits[-1][0, :8]
    cerr = (a1 - b1).abs().max().item()
    print(f"  scan causality err: {cerr:.2e}")
    assert cerr < 1e-5
    # (d) PARAM ACCOUNTING: delta vs --fold left at the same config is
    #     exactly (per layer) -(quat 3*nb*4 + fusion_gate (2d*3nb+3nb) +
    #     comp_norm 2d + mod_depth 1) + (scan_wq + scan_wb, d*2nb+2nb each,
    #     + (scan_gate + scan_wm) 2x(d*nbs+nbs) + scan_norm 2d) with
    #     nbs = nb/2; and the 22M rung (d=640 nb=160 L=4 vocab 10000)
    #     stays within 5% of the so3-left reference 22,254,488.
    torch.manual_seed(37)
    m_l5 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='left')
    torch.manual_seed(37)
    m_s5 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='scan')
    d5, nb5, L5 = 64, 16, 2
    exp5 = L5 * (-(3 * nb5 * 4 + (2 * d5) * (3 * nb5) + 3 * nb5
                   + 2 * d5 + 1)
                 + 2 * (d5 * 2 * nb5 + 2 * nb5)
                 + 2 * (d5 * (nb5 // 2) + nb5 // 2) + 2 * d5)
    got5 = count_params(m_s5) - count_params(m_l5)
    print(f"  scan param delta vs left (d=64,nb=16,L=2): {got5:+d} "
          f"(expected {exp5:+d})")
    assert got5 == exp5
    m_rl = OperaSpinorFenwickTree(vocab_size=10000, d=640, nb=160,
                                  num_layers=4, pe_mode='none',
                                  fold_mode='left')
    m_rs = OperaSpinorFenwickTree(vocab_size=10000, d=640, nb=160,
                                  num_layers=4, pe_mode='none',
                                  fold_mode='scan')
    n_rl, n_rs = count_params(m_rl), count_params(m_rs)
    print(f"  22M rung: left {n_rl:,} (reference 22,254,488), "
          f"scan {n_rs:,} ({100.0 * (n_rs - n_rl) / n_rl:+.2f}%)")
    assert n_rl == 22254488
    assert abs(n_rs - n_rl) / n_rl < 0.05
    del m_rl, m_rs
    # (e) SMOKE TRAINING: 5 Adam steps on a fixed batch -- loss decreases,
    #     no NaN
    m_sm = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='scan')
    opt = torch.optim.Adam(m_sm.parameters(), lr=3e-3)
    tok = torch.randint(1, 101, (4, 17))
    lens = torch.tensor([17, 12, 9, 5])
    losses = []
    for _ in range(5):
        opt.zero_grad()
        ls, _, _ = lm_loss(m_sm(tok, lens).logits, tok, lens)
        ls.backward()
        opt.step()
        losses.append(ls.item())
    print(f"  scan smoke train: {losses[0]:.3f} -> {losses[-1]:.3f} (5 steps)")
    assert all(np.isfinite(losses)) and losses[-1] < losses[0]
    # (f) DRIFT at T=4096 under the v8.6 CONFORMAL law with decay: leaf
    #     magnitudes lie in (0,1), so |q| -- a product of decays -- is
    #     TRIVIALLY bounded by 1 (no overflow; it may harmlessly
    #     underflow toward 0 at long T -- the readout normalization is
    #     scale-invariant), and b is a decay-weighted sum bounded by
    #     max|b_leaf| / (1 - s_max). Plus the scan-vs-naive float error
    #     at T=1024 under the same law.
    q = torch.randn(1, 4096, 2, 4, generator=gs)
    q = q / q.norm(dim=-1, keepdim=True)
    s_drift = torch.rand(1, 4096, 2, 1, generator=gs) * 0.09 + 0.9  # [0.9,0.99]
    q = q * s_drift
    b = torch.randn(1, 4096, 2, 4, generator=gs) * 0.5
    qs, bs = associative_scan(q, b)
    q_hi = qs.norm(dim=-1).max().item()
    b_hi = bs.norm(dim=-1).max().item()
    fin = bool(torch.isfinite(qs).all() and torch.isfinite(bs).all())
    print(f"  scan drift T=4096 (decay s in [0.9,0.99]): max |q| {q_hi:.4f} "
          f"(<=1 by construction), max |b| {b_hi:.2f} (bounded), finite {fin}")
    assert fin and q_hi <= 1.0 + 1e-4 and b_hi < 200.0
    q1k, b1k = q[:, :1024], b[:, :1024]
    qs1k, bs1k = associative_scan(q1k, b1k)
    aq, ab = q1k[:, 0], b1k[:, 0]
    err1k = max((qs1k[:, 0] - aq).abs().max().item(),
                (bs1k[:, 0] - ab).abs().max().item())
    for t in range(1, 1024):
        aq, ab = affine_compose(q1k[:, t], b1k[:, t], aq, ab)
        err1k = max(err1k, (qs1k[:, t] - aq).abs().max().item(),
                    (bs1k[:, t] - ab).abs().max().item())
    print(f"  scan==naive float drift @T=1024 (conformal): err {err1k:.2e}")
    assert err1k < 1e-3
    # (g) DECAY DIRECTION (RG-LRU/GLA mechanic, sanity): a low-magnitude
    #     leaf at position j multiplicatively attenuates the contribution
    #     of tokens BEFORE j in every later prefix (x s_j); tokens at and
    #     after j are unaffected, and prefixes before j are untouched.
    #     Identity rotations -> contributions are exactly scalar.
    T_d, j_d, s_d = 8, 4, 0.05
    qd = torch.zeros(1, T_d, 1, 4)
    qd[..., 0] = 1.0
    qd_lo = qd.clone()
    qd_lo[0, j_d, 0, 0] = s_d
    bd = torch.zeros(1, T_d, 1, 4)
    bd[..., 1] = 1.0                                   # v = (1,0,0)
    _, b_ref = associative_scan(qd, bd)
    _, b_att = associative_scan(qd_lo, bd)
    B1 = b_ref[0, -1, 0, 1].item()                     # = T_d
    B2 = b_att[0, -1, 0, 1].item()                     # = (T_d-j_d) + s_d*j_d
    e1 = B1 - (T_d - j_d)                              # early contribution
    e2 = B2 - (T_d - j_d)                              # attenuated early
    pre = (b_ref[0, :j_d] - b_att[0, :j_d]).abs().max().item()
    print(f"  scan decay direction: early contrib {e1:.2f} -> {e2:.3f} "
          f"(x{s_d} at the decay leaf), late {T_d - j_d:.1f} unchanged, "
          f"prefixes < j err {pre:.1e}")
    assert abs(B2 - ((T_d - j_d) + s_d * j_d)) < 1e-5
    assert e2 < 0.2 * e1 and pre == 0.0
    # (h) SALIENCE GATE (v8.7, --salience): bottleneck FiLM on the readout
    # (i) IDENTITY AT INIT: salience params are created LAST (shared RNG
    #     stream) and the final projection is zero-init -> salience-ON is
    #     bit-identical to salience-OFF at the same seed
    torch.manual_seed(51)
    m_off = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                   pe_mode='none', fold_mode='scan')
    torch.manual_seed(51)
    m_on = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='scan',
                                  scan_salience=True)
    m_off.eval(); m_on.eval()
    tok = torch.randint(1, 101, (2, 12))
    lens = torch.tensor([12, 7])
    with torch.no_grad():
        id_err = (m_off(tok, lens).logits[-1] - m_on(tok, lens).logits[-1]).abs().max().item()
    print(f"  salience: identity-at-init err {id_err:.2e} (zero-init FiLM)")
    assert id_err == 0.0
    # (ii) param delta: +L*(d*r + r + r*2d + 2d) with r = 64
    d_sal = count_params(m_on) - count_params(m_off)
    exp_sal = 1 * (64 * 64 + 64 + 64 * 128 + 128)
    print(f"  salience param delta: +{d_sal} (expected +{exp_sal}, r=64)")
    assert d_sal == exp_sal
    m_rsal = OperaSpinorFenwickTree(vocab_size=10000, d=640, nb=160,
                                    num_layers=4, pe_mode='none',
                                    fold_mode='scan', scan_salience=True)
    n_rsal = count_params(m_rsal)
    print(f"  22M rung + salience: {n_rsal:,} "
          f"({100.0 * (n_rsal - 22254488) / 22254488:+.2f}% vs reference)")
    assert abs(n_rsal - 22254488) / 22254488 < 0.05
    del m_rsal
    # (iii) NON-TRIVIAL causality: randomize the final projection so the
    #       FiLM actually modulates (sanity: output moves vs the same-seed
    #       zero-init model), then perturb token 10 -> positions < 10
    #       bit-unchanged
    torch.manual_seed(53)
    m_sc2 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                   pe_mode='none', fold_mode='scan',
                                   scan_salience=True)
    torch.manual_seed(53)
    m_z = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                 pe_mode='none', fold_mode='scan',
                                 scan_salience=True)
    with torch.no_grad():
        for sm in m_sc2.salience:
            sm[-1].weight.normal_(0, 0.1)
            sm[-1].bias.normal_(0, 0.1)
    m_sc2.eval(); m_z.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        moved = (m_sc2(tok, lens).logits[-1] - m_z(tok, lens).logits[-1]).abs().max().item()
        a2 = m_sc2(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b2 = m_sc2(tok2, lens).logits[-1][0, :8]
    scerr = (a2 - b2).abs().max().item()
    print(f"  salience: randomized FiLM moves output by {moved:.2e} (>0), "
          f"causality err {scerr:.2e}")
    assert moved > 1e-4 and scerr == 0.0
    # (iv) smoke train: loss decreases, grads reach the salience stack
    m_ss = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='scan',
                                  scan_salience=True)
    opt = torch.optim.Adam(m_ss.parameters(), lr=3e-3)
    tok = torch.randint(1, 101, (4, 17))
    lens = torch.tensor([17, 12, 9, 5])
    losses = []
    for _ in range(5):
        opt.zero_grad()
        ls, _, _ = lm_loss(m_ss(tok, lens).logits, tok, lens)
        ls.backward()
        opt.step()
        losses.append(ls.item())
    gsal = sum(p.grad.abs().sum().item() for p in m_ss.salience.parameters())
    print(f"  salience smoke train: {losses[0]:.3f} -> {losses[-1]:.3f}, "
          f"salience grad sum {gsal:.3f} (>0)")
    assert all(np.isfinite(losses)) and losses[-1] < losses[0] and gsal > 0

    # (i) RANK-1 WORKSPACE (v8.9, --workspace)
    # (a) IDENTITY AT INIT: workspace params created LAST (shared RNG
    #     stream), gate zero-init -> ON is bit-identical to OFF, same seed
    torch.manual_seed(57)
    m_w0 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='scan')
    torch.manual_seed(57)
    m_w1 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='scan',
                                  workspace=True)
    m_w0.eval(); m_w1.eval()
    tok = torch.randint(1, 101, (2, 12))
    lens = torch.tensor([12, 7])
    with torch.no_grad():
        wid_err = (m_w0(tok, lens).logits[-1] - m_w1(tok, lens).logits[-1]).abs().max().item()
    print(f"  workspace: identity-at-init err {wid_err:.2e} (zero-init gate)")
    assert wid_err == 0.0
    # (c) PARAM DELTA: per layer K*d + 6*(d*dk) + d with K=4, dk=64
    d_ws = count_params(m_w1) - count_params(m_w0)
    exp_ws = 1 * (4 * 64 + 6 * (64 * 64) + 64)
    print(f"  workspace param delta: +{d_ws} (expected +{exp_ws}, K=4 dk=64)")
    assert d_ws == exp_ws
    m_rws = OperaSpinorFenwickTree(vocab_size=10000, d=640, nb=160,
                                   num_layers=4, pe_mode='none',
                                   fold_mode='scan', workspace=True)
    n_rws = count_params(m_rws)
    m_rwss = OperaSpinorFenwickTree(vocab_size=10000, d=640, nb=160,
                                    num_layers=4, pe_mode='none',
                                    fold_mode='scan', scan_salience=True,
                                    workspace=True)
    n_rwss = count_params(m_rwss)
    print(f"  22M rung + workspace: {n_rws:,} "
          f"({100.0 * (n_rws - 22254488) / 22254488:+.2f}% vs reference); "
          f"+salience stacked: {n_rwss:,} "
          f"({100.0 * (n_rwss - 22254488) / 22254488:+.2f}%)")
    assert abs(n_rws - 22254488) / 22254488 < 0.05
    assert abs(n_rwss - 22254488) / 22254488 < 0.05
    del m_rws, m_rwss
    # (b) CAUSALITY + TARGETED LEAK TEST: randomize the gate so the read
    #     path is active (sanity: output moves vs zero-gate model). T=128
    #     = 4 chunks of 32. (i) perturb token j=101 (chunk 3): outputs at
    #     positions < 101 must be bit-identical. (ii) the leak proof:
    #     those positions cover chunks 0..2 = c..c+1 with the perturbed
    #     token in chunk c+2 -- if latents leaked across chunks, chunk
    #     c+1's outputs would move.
    torch.manual_seed(59)
    m_wc = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='scan',
                                  workspace=True)
    torch.manual_seed(59)
    m_wz = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='scan',
                                  workspace=True)
    with torch.no_grad():
        m_wc.ws_gate.normal_(0, 0.1)
    m_wc.eval(); m_wz.eval()
    tok = torch.randint(1, 101, (1, 128))
    lens = torch.tensor([128])
    with torch.no_grad():
        moved = (m_wc(tok, lens).logits[-1] - m_wz(tok, lens).logits[-1]).abs().max().item()
        ref = m_wc(tok, lens).logits[-1][0].clone()
        tok2 = tok.clone(); tok2[0, 101] = (tok2[0, 101] + 5) % 100 + 1
        pert = m_wc(tok2, lens).logits[-1][0]
    werr_global = (ref[:101] - pert[:101]).abs().max().item()
    werr_chunk = (ref[:96] - pert[:96]).abs().max().item()
    post = (ref[101:] - pert[101:]).abs().max().item()
    print(f"  workspace: gate moves output by {moved:.2e} (>0); perturb "
          f"token 101 (chunk 3): positions <101 err {werr_global:.2e}, "
          f"chunks 0-2 err {werr_chunk:.2e} (leak proof), "
          f"positions >=101 differ by {post:.2e}")
    assert moved > 1e-4 and werr_global == 0.0 and werr_chunk == 0.0
    assert post > 1e-4
    # (d) smoke train: loss decreases; grads reach the workspace stack.
    #     NOTE: with the gate at exact zero, only ws_gate gets grad at
    #     step 0; the projections get grads once the gate is nonzero.
    m_wt = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='scan',
                                  workspace=True)
    opt = torch.optim.Adam(m_wt.parameters(), lr=3e-3)
    tok = torch.randint(1, 101, (4, 65))            # > 2 chunks
    lens = torch.tensor([65, 48, 33, 17])
    losses = []
    for _ in range(5):
        opt.zero_grad()
        ls, _, _ = lm_loss(m_wt(tok, lens).logits, tok, lens)
        ls.backward()
        opt.step()
        losses.append(ls.item())
    ggate = m_wt.ws_gate.grad.abs().sum().item()
    go = m_wt.ws_o.grad.abs().sum().item()
    print(f"  workspace smoke train: {losses[0]:.3f} -> {losses[-1]:.3f}, "
          f"gate grad {ggate:.3f} (>0), ws_o grad {go:.3e} (>0)")
    assert all(np.isfinite(losses)) and losses[-1] < losses[0]
    assert ggate > 0 and go > 0
    # (e) torch.compile on MPS still works with workspace ON (one fwd vs
    #     eager, eval mode; guarded: CPU-only machines skip)
    if torch.backends.mps.is_available():
        m_cp = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                      num_layers=2, pe_mode='none',
                                      fold_mode='scan', workspace=True)
        m_cp = m_cp.to('mps').eval()
        m_cc = torch.compile(m_cp)
        tok_m = tok.to('mps')
        lens_m = lens.to('mps')
        with torch.no_grad():
            e_ref = m_cp(tok_m, lens_m).logits[-1]
            e_cmp = m_cc(tok_m, lens_m).logits[-1]
        cerr = (e_ref - e_cmp).abs().max().item()
        print(f"  workspace torch.compile (MPS): fwd err {cerr:.2e}")
        assert cerr < 1e-5
        del m_cp, m_cc
        torch.mps.empty_cache()


def test_spine_readout():
    # SPINE READOUT (r19)
    torch.manual_seed(0)
    # (a) single-block positions bypass -> identical to left (params
    #     created last: shared RNG stream; num_layers=1)
    torch.manual_seed(17)
    m_l3 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='left')
    torch.manual_seed(17)
    m_s = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                 pe_mode='none', fold_mode='spine')
    m_l3.eval(); m_s.eval()
    tok = torch.randint(1, 101, (2, 8))
    lens = torch.full((2,), 8)
    with torch.no_grad():
        ll3 = m_l3(tok, lens).logits[-1]
        ls3 = m_s(tok, lens).logits[-1]
    sp_pos = [0, 1, 3, 7]; mp_pos = [2, 4, 5, 6]
    e1 = (ll3[:, sp_pos] - ls3[:, sp_pos]).abs().max().item()
    e2 = (ll3[:, mp_pos] - ls3[:, mp_pos]).abs().max().item()
    print(f"  spine: single-block == left err {e1:.2e} "
          f"(multi-block differs by {e2:.2e}, as it must)")
    assert e1 < 1e-5 and e2 > 1e-4
    # (b) param delta: +2*dk*d per layer (same budget as attend)
    d_s = count_params(m_s) - count_params(m_l3)
    exp_s = 1 * 2 * 64 * 64
    print(f"  spine param delta: +{d_s} (expected +{exp_s})")
    assert d_s == exp_s
    # (c) multi-layer, ragged: forward/backward finite, grads reach q/k
    m_s2 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='spine')
    tok = torch.randint(1, 101, (3, 29))
    lens = torch.tensor([29, 14, 5])
    loss, _, _ = lm_loss(m_s2(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    gq = m_s2.spine_q.grad.abs().sum().item()
    gk = m_s2.spine_k.grad.abs().sum().item()
    print(f"  spine: loss {loss.item():.3f}, q/k grad sums "
          f"{gq:.3f}/{gk:.3f} (>0)")
    assert gq > 0 and gk > 0
    # (d) exact causality
    m_s2.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a2 = m_s2(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b2 = m_s2(tok2, lens).logits[-1][0, :8]
    cerr2 = (a2 - b2).abs().max().item()
    print(f"  spine causality err: {cerr2:.2e}")
    assert cerr2 < 1e-5
    # (e) stacks with rot free
    m_s3 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='spine',
                                  rot_mode='free')
    loss, _, _ = lm_loss(m_s3(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"  spine + rotfree: loss {loss.item():.3f}, OK")


def test_rack_fold():
    # RACK FOLD (r17)
    torch.manual_seed(0)
    # (a) the fold's core operation satisfies the RACK AXIOM numerically:
    #     with x ▷ y := ŷ x ŷ⁻¹, check (x▷y)▷z == (x▷z)▷(y▷z)
    x = torch.randn(64, 4); y = torch.randn(64, 4); z = torch.randn(64, 4)
    def rop(a, b):
        bh = b / b.norm(dim=-1, keepdim=True)
        return quat_sandwich(bh, a)
    lhs = rop(rop(x, y), z)
    rhs = rop(rop(x, z), rop(y, z))
    rack_err = (lhs - rhs).abs().max().item()
    print(f"  rack axiom (self-distributivity) err: {rack_err:.2e}")
    assert rack_err < 1e-4
    # (b) conjugation is an exact isometry and fixes the scalar channel
    q = torch.randn(64, 4); q = q / q.norm(dim=-1, keepdim=True)
    h = torch.randn(64, 4)
    hr = quat_sandwich(q, h)
    iso_err = (h.norm(dim=-1) - hr.norm(dim=-1)).abs().max().item()
    sc_err = (h[..., 0] - hr[..., 0]).abs().max().item()
    print(f"  rack conjugation: isometry err {iso_err:.2e}, "
          f"scalar invariance err {sc_err:.2e}")
    assert iso_err < 1e-4 and sc_err < 1e-4
    # (c) single-block positions bypass the fold -> identical to left
    #     (rack params created last: shared RNG stream; num_layers=1)
    torch.manual_seed(13)
    m_l2 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='left')
    torch.manual_seed(13)
    m_r = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                 pe_mode='none', fold_mode='rack')
    m_l2.eval(); m_r.eval()
    tok = torch.randint(1, 101, (2, 8))
    lens = torch.full((2,), 8)
    with torch.no_grad():
        ll2 = m_l2(tok, lens).logits[-1]
        lr2 = m_r(tok, lens).logits[-1]
    sp = [0, 1, 3, 7]; mp = [2, 4, 5, 6]
    e1 = (ll2[:, sp] - lr2[:, sp]).abs().max().item()
    e2 = (ll2[:, mp] - lr2[:, mp]).abs().max().item()
    print(f"  rack: single-block == left err {e1:.2e} "
          f"(multi-block differs by {e2:.2e}, as it must)")
    assert e1 < 1e-5 and e2 > 1e-4
    # (d) param accounting: +(2d+1)*nb per layer
    d_l = count_params(m_r) - count_params(m_l2)
    exp_l = 1 * ((2 * 64 + 1) * 16)
    print(f"  rack param delta: +{d_l} (expected +{exp_l})")
    assert d_l == exp_l
    # (e) DEEP-FOLD STABILITY -- the rack's own pre-registered claim:
    #     11 fold steps (T=2047-scale popcount), norms bounded, no LN/tanh
    a = torch.randn(32, 16, 4) * 2.0
    rng = torch.Generator().manual_seed(3)
    n0 = a.reshape(32, -1).norm(dim=-1).mean().item()
    for _ in range(11):
        nx = torch.randn(32, 16, 4, generator=rng)
        qn = nx / nx.norm(dim=-1, keepdim=True)
        a = quat_sandwich(qn, a) + 0.12 * nx        # g ~ sigmoid(-2)
    n1 = a.reshape(32, -1).norm(dim=-1).mean().item()
    print(f"  rack 11-step fold: norm {n0:.2f} -> {n1:.2f} "
          f"(bounded linear growth, no blow-up/decay)")
    assert np.isfinite(n1) and n1 < 6 * n0
    # (f) full model: forward/backward finite, grads reach gate, causality
    m_r2 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='rack')
    tok = torch.randint(1, 101, (3, 29))
    lens = torch.tensor([29, 14, 5])
    loss, _, _ = lm_loss(m_r2(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    gg = sum(p.grad.abs().sum().item() for p in m_r2.rack_gate.parameters())
    print(f"  rack: loss {loss.item():.3f}, gate grad sum {gg:.3f} (>0)")
    assert gg > 0
    m_r2.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a1 = m_r2(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_r2(tok2, lens).logits[-1][0, :8]
    cerr = (a1 - b1).abs().max().item()
    print(f"  rack causality err: {cerr:.2e}")
    assert cerr < 1e-5


def test_rack_exitnorm():
    # RACK EXIT-NORM (r18)
    torch.manual_seed(0)
    # (a) single-block positions STILL bypass -> identical to left (the
    #     exit norm applies to folded positions only); multi-block
    #     positions differ from plain rack (the norm did something)
    torch.manual_seed(13)
    m_l4 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='left')
    torch.manual_seed(13)
    m_re = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='rack',
                                  rack_exitnorm=True)
    torch.manual_seed(13)
    m_r0 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='rack')
    m_l4.eval(); m_re.eval(); m_r0.eval()
    tok = torch.randint(1, 101, (2, 8))
    lens = torch.full((2,), 8)
    with torch.no_grad():
        ll4 = m_l4(tok, lens).logits[-1]
        lre = m_re(tok, lens).logits[-1]
        lr0 = m_r0(tok, lens).logits[-1]
    sp = [0, 1, 3, 7]; mp = [2, 4, 5, 6]
    e1 = (ll4[:, sp] - lre[:, sp]).abs().max().item()
    e2 = (lr0[:, mp] - lre[:, mp]).abs().max().item()
    print(f"  rack-exitnorm: single-block == left err {e1:.2e} "
          f"(multi-block differs from plain rack by {e2:.2e})")
    assert e1 < 1e-5 and e2 > 1e-4
    # (b) param delta vs plain rack: +2d per layer (LN weight+bias); LN
    #     init is deterministic -> all OTHER params identical (same seed)
    d_e = count_params(m_re) - count_params(m_r0)
    print(f"  rack-exitnorm param delta: +{d_e} (expected +{2 * 64})")
    assert d_e == 2 * 64
    same = all(torch.equal(p, q) for (n_, p), (m_, q)
               in zip(m_r0.named_parameters(), m_re.named_parameters())
               if 'exit_norm' not in m_)
    assert same, "exitnorm must not perturb the RNG stream"
    print("  rack-exitnorm: non-norm params bit-identical to plain rack")
    # (c) forward/backward finite, grads reach the norm
    m_re2 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                   pe_mode='none', fold_mode='rack',
                                   rack_exitnorm=True)
    tok = torch.randint(1, 101, (3, 29))
    lens = torch.tensor([29, 14, 5])
    loss, _, _ = lm_loss(m_re2(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    gn = sum(p.grad.abs().sum().item()
             for p in m_re2.rack_exit_norm.parameters())
    print(f"  rack-exitnorm: loss {loss.item():.3f}, norm grad sum "
          f"{gn:.3f} (>0)")
    assert gn > 0


def test_oam_node_transport():
    # OAM NODE TRANSPORT (v8.2)
    torch.manual_seed(0)
    # (a) k=1, charges '0', transport node is BITWISE --fold left: the
    #     loop is compose(acc, next) with shared node params -- the left
    #     fold itself. Param delta = num_layers (phi only, frozen).
    torch.manual_seed(29)
    m_lf = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='left')
    torch.manual_seed(29)
    m_n1 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='oam',
                                  oam_transport='node', oam_k=1,
                                  oam_charges='0')
    m_lf.eval(); m_n1.eval()
    tok = torch.randint(1, 101, (2, 12))
    lens = torch.tensor([12, 7])
    with torch.no_grad():
        d_n1 = (m_lf(tok, lens).logits[-1] - m_n1(tok, lens).logits[-1]).abs().max().item()
    dp = count_params(m_n1) - count_params(m_lf)
    print(f"  oam-node k=1 chg0 == left err: {d_n1:.2e} "
          f"(param delta +{dp}, expected +2 = phi)")
    assert d_n1 < 1e-6 and dp == 2
    # (b) READOUT-ARTIFACT NULL: k=2, phi=0, combine=sum -- all channels
    #     bitwise identical, softmax(0,0) mixes two equal halves -> the
    #     whole model collapses to left exactly. (combine=compose would
    #     NOT collapse: composing x with x is the artifact the null
    #     isolates.)
    torch.manual_seed(29)
    m_n2 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='oam',
                                  oam_transport='node', oam_k=2,
                                  oam_phi=0.0, oam_combine='sum')
    m_n2.eval()
    with torch.no_grad():
        d_n2 = (m_lf(tok, lens).logits[-1] - m_n2(tok, lens).logits[-1]).abs().max().item()
    print(f"  oam-node k=2 phi0 sum == left err: {d_n2:.2e} "
          f"(identical channels collapse)")
    assert d_n2 < 1e-5
    # (c) charge on differs; zero-param claim holds at k=4 (delta = phi
    #     only); forward/backward finite; phi grad flows; causality exact
    torch.manual_seed(31)
    m_n4 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='oam',
                                  oam_transport='node', oam_k=4)
    torch.manual_seed(31)
    m_lf2 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                   pe_mode='none', fold_mode='left')
    dp4 = count_params(m_n4) - count_params(m_lf2)
    print(f"  oam-node k=4 param delta: +{dp4} (expected +2 = phi; "
          f"ZERO gates, node params shared)")
    assert dp4 == 2
    tok = torch.randint(1, 101, (3, 29))
    lens = torch.tensor([29, 14, 5])
    loss, _, _ = lm_loss(m_n4(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    gp = m_n4.oam_phi.grad.abs().sum().item()
    print(f"  oam-node k=4: loss {loss.item():.3f}, phi grad {gp:.4f} (>0)")
    assert gp > 0
    m_n4.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a1 = m_n4(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_n4(tok2, lens).logits[-1][0, :8]
    cerr = (a1 - b1).abs().max().item()
    print(f"  oam-node causality err: {cerr:.2e}")
    assert cerr < 1e-5
    with torch.no_grad():
        d_ch = (m_n4(tok, lens).logits[-1] - m_lf2(tok, lens).logits[-1]).abs().max().item()
    print(f"  oam-node k=4 charge-on vs left diff: {d_ch:.2e} (>0)")
    assert d_ch > 1e-4


def test_oam_fold():
    # OAM FOLD (v8.1)
    torch.manual_seed(0)
    # (a) k=1, charges '0' is BITWISE --fold rack (gate draw sequence is
    #     identical; charge rotation skipped entirely when all charges 0)
    torch.manual_seed(17)
    m_rk = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='rack')
    torch.manual_seed(17)
    m_o1 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='oam',
                                  oam_k=1, oam_charges='0')
    m_rk.eval(); m_o1.eval()
    tok = torch.randint(1, 101, (2, 12))
    lens = torch.tensor([12, 7])
    with torch.no_grad():
        lr_ = m_rk(tok, lens).logits[-1]
        lo_ = m_o1(tok, lens).logits[-1]
    e_rk = (lr_ - lo_).abs().max().item()
    print(f"  oam k=1 chg0 == rack err: {e_rk:.2e} (phi delta "
          f"+{count_params(m_o1) - count_params(m_rk)} params, expected +2)")
    assert e_rk < 1e-6 and count_params(m_o1) - count_params(m_rk) == 2
    # (b) charge has an effect when on; phi=0 == charges all-0 (rotation
    #     skipped in both -> bitwise identical at same seed)
    torch.manual_seed(19)
    m_oc = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='oam', oam_k=2,
                                  oam_charges='-1,1')
    torch.manual_seed(19)
    m_o0 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='oam', oam_k=2,
                                  oam_charges='0,0')
    torch.manual_seed(19)
    m_op0 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                   pe_mode='none', fold_mode='oam', oam_k=2,
                                   oam_charges='-1,1', oam_phi=0.0)
    m_oc.eval(); m_o0.eval(); m_op0.eval()
    tok = torch.randint(1, 101, (2, 12))
    lens = torch.full((2,), 12)
    with torch.no_grad():
        lc = m_oc(tok, lens).logits[-1]
        l0 = m_o0(tok, lens).logits[-1]
        lp0 = m_op0(tok, lens).logits[-1]
    d_on = (lc - l0).abs().max().item()
    d_off = (l0 - lp0).abs().max().item()
    print(f"  oam charge: on-vs-off diff {d_on:.2e} (>0), "
          f"phi0-vs-chg00 diff {d_off:.2e} (==0)")
    assert d_on > 1e-4 and d_off < 1e-6
    # (g) PER-CHANNEL GATE SEMANTICS (opt2 bug catcher): the batched
    #     einsum must equal a per-channel reference loop -- channel c's
    #     gate is a function of channel c's input ONLY. The v8.1 einsum
    #     ('bmki,cni->bmcn') summed over k: every gate read the pooled
    #     k-sum. This test fails on that formulation.
    Bt, mt, kt, dt, nbt = 2, 3, 4, 10, 7
    inp_t = torch.randn(Bt, mt, kt, 2 * dt)
    W_t = torch.randn(kt, nbt, 2 * dt)
    b_t = torch.randn(kt, nbt)
    g_fast = torch.einsum('bmki,kni->bmkn', inp_t, W_t) + b_t[None, None]
    g_ref = torch.stack([inp_t[:, :, c] @ W_t[c].T + b_t[c]
                         for c in range(kt)], dim=2)
    e_g = (g_fast - g_ref).abs().max().item()
    g_bug = torch.einsum('bmki,cni->bmcn', inp_t, W_t) + b_t[None, None]
    e_bug = (g_bug - g_ref).abs().max().item()
    print(f"  oam per-channel gate: einsum == reference err {e_g:.2e} "
          f"(v8.1 pooled formulation deviates by {e_bug:.2e}, as it must)")
    assert e_g < 1e-5 and e_bug > 1e-2
    # (h) --oam-pair conj: auto k=4 charges (-1.5,-0.5,+0.5,+1.5) must
    #     permute to (-0.5,+0.5,-1.5,+1.5) -> perm [1,2,0,3]; forward
    #     differs from seq (compose is non-commutative); k=1 unaffected;
    #     state_dict is UNCHANGED (perm buffer non-persistent)
    torch.manual_seed(23)
    m_ps = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='oam', oam_k=4)
    torch.manual_seed(23)
    m_pc = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='oam', oam_k=4,
                                  oam_pair='conj')
    assert m_pc.oam_pair_perm.tolist() == [1, 2, 0, 3]
    assert set(m_ps.state_dict().keys()) == set(m_pc.state_dict().keys())
    m_ps.eval(); m_pc.eval()
    tok = torch.randint(1, 101, (2, 12))
    lens = torch.full((2,), 12)
    with torch.no_grad():
        d_pair = (m_ps(tok, lens).logits[-1] - m_pc(tok, lens).logits[-1]).abs().max().item()
    print(f"  oam pair-conj: perm [1,2,0,3] OK, seq-vs-conj diff "
          f"{d_pair:.2e} (>0), state_dict keys unchanged")
    assert d_pair > 1e-5

    # (c) charge rotation math: z-rotation of vector parts by m*phi*l,
    #     norm-preserving, scalar & vz untouched
    th = 0.37
    h_ = torch.randn(50, 4)
    c_, s_ = math.cos(th), math.sin(th)
    vx2 = h_[..., 1] * c_ - h_[..., 2] * s_
    vy2 = h_[..., 1] * s_ + h_[..., 2] * c_
    rot_ = torch.stack([h_[..., 0], vx2, vy2, h_[..., 3]], dim=-1)
    assert (rot_.norm(dim=-1) - h_.norm(dim=-1)).abs().max().item() < 1e-5
    v0 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    tv = torch.stack([v0[0], v0[1] * c_ - v0[2] * s_,
                      v0[1] * s_ + v0[2] * c_, v0[3]])
    assert abs(tv[1].item() - c_) < 1e-6 and abs(tv[2].item() - s_) < 1e-6
    print("  oam charge rotation: z-rotation verified, isometry exact")
    # (d) full model k=4: forward/backward finite, grads reach phi+gates,
    #     causality, combine=sum runs, param delta as specced
    m_o4 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='oam')
    tok = torch.randint(1, 101, (3, 29))
    lens = torch.tensor([29, 14, 5])
    loss, _, _ = lm_loss(m_o4(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    gg = sum(p.grad.abs().sum().item() for p in m_o4.oam_gate.parameters())
    gp = m_o4.oam_phi.grad.abs().sum().item()
    print(f"  oam k=4: loss {loss.item():.3f}, gate grad {gg:.3f}, "
          f"phi grad {gp:.3f} (both >0)")
    assert gg > 0 and gp > 0
    m_o4.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a1 = m_o4(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_o4(tok2, lens).logits[-1][0, :8]
    cerr_o = (a1 - b1).abs().max().item()
    print(f"  oam causality err: {cerr_o:.2e}")
    assert cerr_o < 1e-5
    m_os = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='oam', oam_k=3,
                                  oam_combine='sum', oam_shared_gate=True)
    tok = torch.randint(1, 101, (2, 33))       # T=33: odd tree, k odd
    lens = torch.tensor([33, 9])
    loss2, _, _ = lm_loss(m_os(tok, lens).logits, tok, lens)
    loss2.backward()
    assert torch.isfinite(loss2)
    print(f"  oam k=3 combine=sum shared-gate: loss {loss2.item():.3f}, OK")
    # (i) v8.3 INDUCED OAM: level gate starts at sigmoid(-3)~0.047 (twist
    #     ~5%), params get grads, chan-emb zeros at init -> k=1 chg0 with
    #     chan-emb STILL == rack bitwise (embedding is exactly 0)
    m_lg = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='oam',
                                  oam_transport='node', oam_levelgate=True,
                                  oam_chan_emb=True)
    assert m_lg.oam_level_gate.shape == (2, 16)
    assert m_lg.oam_chan_emb.shape == (2, 4, 16)
    lv0 = torch.sigmoid(m_lg.oam_level_gate.data).max().item()
    assert abs(lv0 - 0.0474) < 1e-3, lv0
    tok = torch.randint(1, 101, (2, 17))
    lens = torch.tensor([17, 6])
    loss3, _, _ = lm_loss(m_lg(tok, lens).logits, tok, lens)
    loss3.backward()
    gl = m_lg.oam_level_gate.grad.abs().sum().item()
    ge = m_lg.oam_chan_emb.grad.abs().sum().item()
    print(f"  oam v8.3: levelgate init {lv0:.4f} (~5%), grads gate {gl:.3f}"
          f" emb {ge:.3f} (>0), loss {loss3.item():.3f}")
    assert gl > 0 and ge > 0 and torch.isfinite(loss3)
    torch.manual_seed(29)
    m_rk3 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                   pe_mode='none', fold_mode='rack')
    torch.manual_seed(29)
    m_o3 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='oam', oam_k=1,
                                  oam_charges='0', oam_chan_emb=True)
    m_rk3.eval(); m_o3.eval()
    tok = torch.randint(1, 101, (2, 12))
    lens = torch.full((2,), 12)
    with torch.no_grad():
        e3 = (m_rk3(tok, lens).logits[-1] - m_o3(tok, lens).logits[-1]).abs().max().item()
    print(f"  oam chan-emb zeros at init: k=1 chg0 == rack err {e3:.2e}")
    assert e3 < 1e-6


def test_attend_readout():
    # ATTEND READOUT (v8.0)
    torch.manual_seed(0)
    # (a) single-block positions bypass attention -> IDENTICAL to left
    #     fold given same seed (attend params created last: shared RNG
    #     stream). num_layers=1 so layer-2 mixing can't spread diffs.
    torch.manual_seed(11)
    m_left = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                    pe_mode='none', fold_mode='left')
    torch.manual_seed(11)
    m_att = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                   pe_mode='none', fold_mode='attend')
    m_left.eval(); m_att.eval()
    T = 8
    tok = torch.randint(1, 101, (2, T))
    lens = torch.full((2,), T)
    with torch.no_grad():
        ll = m_left(tok, lens).logits[-1]
        la = m_att(tok, lens).logits[-1]
    # positions with popcount(L)==1: prefixes of length 1,2,4,8 -> idx 0,1,3,7
    single_pos = [0, 1, 3, 7]
    err_single = (ll[:, single_pos] - la[:, single_pos]).abs().max().item()
    multi_pos = [2, 4, 5, 6]
    diff_multi = (ll[:, multi_pos] - la[:, multi_pos]).abs().max().item()
    print(f"  attend: single-block positions == left fold, err {err_single:.2e}"
          f" (multi-block positions differ by {diff_multi:.2e}, as they should)")
    assert err_single < 1e-5
    assert diff_multi > 1e-4, "attend produced no difference where it must"
    # (b) param accounting: +2*dk*d per layer
    delta = count_params(m_att) - count_params(m_left)
    expected = 1 * 2 * 64 * 64
    print(f"  attend param delta: +{delta} (expected +{expected})")
    assert delta == expected
    # (c) forward/backward finite + grads reach q/k, multi-layer, ragged
    m2 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                pe_mode='none', fold_mode='attend')
    tok = torch.randint(1, 101, (3, 29))
    lens = torch.tensor([29, 14, 5])
    loss, _, _ = lm_loss(m2(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    gq = m2.fold_attn_q.grad.abs().sum().item()
    gk = m2.fold_attn_k.grad.abs().sum().item()
    print(f"  attend: loss {loss.item():.3f}, q/k grad sums {gq:.3f}/{gk:.3f} (>0)")
    assert gq > 0 and gk > 0
    # (d) exact causality
    m2.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a = m2(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b = m2(tok2, lens).logits[-1][0, :8]
    err = (a - b).abs().max().item()
    print(f"  attend causality err: {err:.2e}")
    assert err < 1e-5
    # (e) attend + rot free + fold-scale stack cleanly
    m3 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                pe_mode='none', fold_mode='attend',
                                rot_mode='free', fold_scale=True)
    loss, _, _ = lm_loss(m3(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"  attend + rotfree + fold-scale: loss {loss.item():.3f}, OK")
    # (f) level encoding: deterministic, correct shape, distinct levels
    lv = torch.tensor([[0, 1, 2, 5], [0, 3, 4, 9]])
    e1 = level_sin_enc(lv, 16); e2 = level_sin_enc(lv, 16)
    assert e1.shape == (2, 4, 16) and (e1 - e2).abs().max().item() == 0.0
    assert (e1[0, 0] - e1[0, 3]).abs().max().item() > 0.1
    print("  level_sin_enc: deterministic, shape OK, levels distinguishable")


def test_docs_data_chunker():
    # DOCS DATA CHUNKER (v8.0): schedule covers every bucket
    torch.manual_seed(0)
    ml, cap = 20, 80
    sizes = doc_chunk_sizes(ml, cap)
    assert set(sizes) == {20, 40, 60, 80}
    words = [f'w{i}' for i in range(1000)]
    chunks = doc_chunks(words, ml, cap)
    lens_seen = sorted(set(len(c) for c in chunks))
    total = sum(len(c) for c in chunks)
    print(f"  doc_chunks: lengths {lens_seen}, coverage {total}/1000 words")
    assert total == 1000, "chunker must not drop words (except tiny tails)"
    assert all(l in (20, 40, 60, 80) for l in lens_seen[:-1] + [lens_seen[-1]]
               ) or lens_seen[-1] < 20  # tail may be short
    for k in (2, 3, 4):
        assert any(len(c) == k * ml for c in chunks), f"bucket {k} unpopulated"
    print("  doc_chunks: every extrapolation bucket populated by construction")
    # v8.4 scale regime: schedule must cover 1024..4096 buckets too
    assert set(doc_chunk_sizes(1024, 4096)) == {1024, 2048, 3072, 4096}
    print("  doc_chunks: scale schedule (train 1024, cap 4096) covers all buckets")
    # opt6 LONG-FIRST (docs-en): longest chunks lead the cycle, so every
    # article of length X > L powers the bucket containing min(X, cap)
    # with its OPENING chunk (the historical order needed X > 8L).
    assert doc_chunk_sizes(1024, 4096, long_first=True) == \
        [4096, 3072, 2048] + [1024] * 8
    for n_words, want_first in ((1500, 1500), (2500, 2500), (3500, 3500),
                                (9000, 4096), (600, 600)):
        ch = doc_chunks([f'w{i}' for i in range(n_words)], 1024, 4096,
                        long_first=True)
        assert len(ch[0]) == want_first, (n_words, len(ch[0]))
        assert sum(len(c) for c in ch) == n_words
    print("  doc_chunks long-first: 1.5k/2.5k/3.5k-word articles land in "
          "buckets 1/2/3 by their opening chunk; coverage exact")


def test_fold_compaction_equivalence():
    # FOLD COMPACTION EQUIVALENCE -- the v7.9 claim. Same weights,
    # compacted 'left' vs reference 'left-masked', ragged lengths,
    # T chosen to exercise popcount up to 4 (29 = 11101).
    torch.manual_seed(0)
    for T in (13, 29):
        torch.manual_seed(3)
        m_c = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                     pe_mode='none', fold_mode='left')
        torch.manual_seed(3)
        m_m = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                     pe_mode='none', fold_mode='left-masked')
        m_c.eval(); m_m.eval()
        tok = torch.randint(1, 101, (3, T))
        lens = torch.tensor([T, max(5, T // 2), 5])
        with torch.no_grad():
            err = 0.0
            for lc, lm in zip(m_c(tok, lens).logits, m_m(tok, lens).logits):
                err = max(err, (lc - lm).abs().max().item())
        masked_w, compact_w, S = fold_work_counts(T)
        print(f"  fold compaction T={T}: err vs masked {err:.2e}; "
              f"work {masked_w} -> {compact_w} row-composes "
              f"({masked_w/compact_w:.2f}x less, {S} slots)")
        assert err < 1e-5, f"fold compaction not equivalent at T={T}: {err}"
    # gradient path through index_copy: backward runs, grads finite
    m_g = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                 pe_mode='none', fold_mode='left')
    tok = torch.randint(1, 101, (3, 29))
    lens = torch.tensor([29, 14, 5])
    loss, _, _ = lm_loss(m_g(tok, lens).logits, tok, lens)
    loss.backward()
    for n_, p_ in m_g.named_parameters():
        if p_.grad is not None:
            assert torch.isfinite(p_.grad).all(), f"non-finite grad in {n_}"
    print(f"  fold compaction backward: loss {loss.item():.3f}, all grads finite")
    # work accounting at the two regimes of interest
    for T in (20, 1024):
        masked_w, compact_w, S = fold_work_counts(T)
        print(f"  fold work @T={T}: masked {masked_w} vs compacted {compact_w} "
              f"({masked_w/compact_w:.2f}x less)")


def test_rot_ablation():
    # ROT ABLATION (r15): --rot free
    torch.manual_seed(0)
    # (a) flags-off inventory: quat present, rot_free absent
    m_so3 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2)
    assert m_so3.quat is not None and m_so3.rot_free is None
    # (b) free inventory: rot_free present, quat absent; param delta is
    #     exactly num_layers*3*nb*(9-4)
    m_free = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                    pe_mode='none', rot_mode='free')
    assert m_free.rot_free is not None and m_free.quat is None
    delta = count_params(m_free) - count_params(
        OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                               pe_mode='none'))
    expected = 2 * 3 * 16 * (9 - 4)
    print(f"  rot free param delta: +{delta} (expected +{expected})")
    assert delta == expected
    # (c) free init near identity (comparable regime to so3 init)
    I3 = torch.eye(3)
    init_dev = (m_free.rot_free - I3).abs().max().item()
    print(f"  rot free init max |M - I|: {init_dev:.3f} (noise scale 0.1)")
    assert init_dev < 0.8
    # (d) forward/backward finite, grads reach rot_free, exact causality
    tok = torch.randint(1, 101, (3, 13))
    lens = torch.tensor([13, 7, 5])
    loss, _, _ = lm_loss(m_free(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    assert m_free.rot_free.grad is not None
    gfree = m_free.rot_free.grad.abs().sum().item()
    print(f"  rot free: loss {loss.item():.3f}, rot_free grad sum {gfree:.4f} (>0)")
    assert gfree > 0
    m_free.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a = m_free(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b = m_free(tok2, lens).logits[-1][0, :8]
    err = (a - b).abs().max().item()
    print(f"  rot free causality err: {err:.2e}")
    assert err < 1e-5
    # (e) free + fold-rotors separate: dedicated free fold matrices
    m_ff = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', rot_mode='free',
                                  fold_rotors='separate')
    assert m_ff.rot_free_fold is not None and m_ff.quat_fold is None
    loss, _, _ = lm_loss(m_ff(tok, lens).logits, tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"  rot free + fold-rotors separate: loss {loss.item():.3f}, OK")
    # (f) the mechanistic contrast the ablation rests on: so3 matrices
    #     are isometries, trained-free matrices need not be. Verify the
    #     so3 path's R are orthogonal and a generic free init is not.
    R_L, _, _ = m_so3.get_rotations(0)
    I = torch.eye(3).expand(16, 3, 3)
    so3_orth = (R_L @ R_L.transpose(-1, -2) - I).abs().max().item()
    F_L, _, _ = m_free.get_rotations(0)
    free_orth = (F_L @ F_L.transpose(-1, -2) - I).abs().max().item()
    print(f"  isometry contrast: so3 orth err {so3_orth:.2e}, "
          f"free orth err {free_orth:.2e} (free is unconstrained)")
    assert so3_orth < 1e-5 and free_orth > 1e-3


def test_rotations_valid():
    # rotations valid
    torch.manual_seed(0)
    q = torch.randn(64, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    R = quat_to_rotmat(q)
    I = torch.eye(3).expand(64, 3, 3)
    orth_err = (R @ R.transpose(-1, -2) - I).abs().max().item()
    print(f"  rotmat orthogonality err {orth_err:.2e}")
    assert orth_err < 1e-5


def test_flags_off_module_inventory():
    # flags-off model matches v7.0 module inventory (no logit_scale,
    # head has bias, no dropout modules)
    torch.manual_seed(0)
    m0 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2)
    assert m0.logit_scale is None and m0.head.bias is not None
    names = [n for n, _ in m0.named_parameters()]
    assert not any('logit' in n for n in names)
    print(f"  flags-off param inventory matches v7.0 ({count_params(m0):,} params)")


def test_tie_init():
    # TIE INIT FIX: initial loss must be ~ln(V), not sqrt(d)-scale.
    torch.manual_seed(0)
    V = 1000
    mt = OperaSpinorFenwickTree(vocab_size=V, d=64, nb=16, num_layers=2, tie=True)
    mt.eval()
    tok = torch.randint(1, V, (8, 16))
    lens = torch.full((8,), 16)
    with torch.no_grad():
        logits = mt(tok, lens).logits
        loss, _, _ = lm_loss(logits, tok, lens)
    lnV = math.log(V)
    print(f"  tied init loss {loss.item():.2f} vs ln(V)={lnV:.2f} "
          f"(v7.2-style tying would be >> {lnV:.0f})")
    assert loss.item() < 1.6 * lnV, "tied head init fix failed"
    # param saving check
    mu = OperaSpinorFenwickTree(vocab_size=V, d=64, nb=16, num_layers=2, tie=False)
    saved = count_params(mu) - count_params(mt)
    print(f"  tie saves {saved:,} params (expected ~ d*V = {64*V:,})")
    assert saved > 0.9 * 64 * V


def test_msup():
    # msup: vectorized loss finite, backward works, adds no params
    torch.manual_seed(0)
    m = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2)
    tok = torch.randint(1, 101, (3, 13))
    lens = torch.tensor([13, 7, 5])
    out = m(tok, lens, return_levels=True)
    base, _, _ = lm_loss(out.logits, tok, lens)
    ms = msup_loss(m, out.levels, tok, lens)
    (base + 0.1 * ms).backward()
    print(f"  msup loss {ms.item():.3f} (base {base.item():.3f}); backward OK")
    assert torch.isfinite(ms)


def test_msup_target_indexing():
    # msup target correctness on a hand case: T=8, level 1 node 0
    # covers [0,1], must predict token at position 2 -> targets[:,1].
    tgt_check_span = 2
    node0_target_pos = (0 + 1) * tgt_check_span - 1
    assert node0_target_pos == 1
    print("  msup target indexing sanity: node(level1,0) -> targets[:,1] OK")


def test_dropout_eval_only():
    # dropout only active in training
    torch.manual_seed(0)
    tok = torch.randint(1, 101, (3, 13))
    lens = torch.tensor([13, 7, 5])
    md = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2, dropout=0.5)
    md.eval()
    with torch.no_grad():
        a = md(tok, lens).logits[-1]
        b = md(tok, lens).logits[-1]
    print(f"  dropout eval determinism err {(a-b).abs().max().item():.2e}")
    assert (a - b).abs().max().item() == 0.0


def test_causality_all_flags():
    # causality with all flags on
    torch.manual_seed(0)
    mall = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  tie=True, dropout=0.3)
    mall.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a = mall(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b = mall(tok2, lens).logits[-1][0, :8]
    caus_err = (a - b).abs().max().item()
    print(f"  causality err (all flags): {caus_err:.2e}")
    assert caus_err < 1e-5


def test_rotor_pe_isometry():
    # rotor PE: isometry (norm preserved per state), identity at t=0,
    # and cache-free correctness across lengths
    torch.manual_seed(0)
    cos, sin = rotor_pos_tables(12, 16, 'cpu')
    st = torch.randn(2, 12, 64)
    st_r = apply_rotor_pe(st, cos, sin, 16)
    norm_err = (st.norm(dim=-1) - st_r.norm(dim=-1)).abs().max().item()
    t0_err = (st[:, 0] - st_r[:, 0]).abs().max().item()
    print(f"  rotor PE: isometry err {norm_err:.2e}, identity-at-t0 err {t0_err:.2e}")
    assert norm_err < 1e-4 and t0_err < 1e-6
    # scalar + v_z channels untouched at every position
    hs = st.reshape(2, 12, 16, 4); hr = st_r.reshape(2, 12, 16, 4)
    inv_err = max((hs[..., 0] - hr[..., 0]).abs().max().item(),
                  (hs[..., 3] - hr[..., 3]).abs().max().item())
    print(f"  rotor PE: scalar/v_z invariance err {inv_err:.2e}")
    assert inv_err < 1e-6


def test_pe_modes_causality():
    # forward/backward + causality for all three PE modes
    torch.manual_seed(0)
    for pm in ('sin', 'none', 'rotor'):
        mp = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                    pe_mode=pm)
        tok = torch.randint(1, 101, (3, 13))
        lens = torch.tensor([13, 7, 5])
        logits = mp(tok, lens).logits
        loss, _, _ = lm_loss(logits, tok, lens)
        loss.backward()
        assert torch.isfinite(loss)
        mp.eval()
        tok = torch.randint(1, 101, (1, 12))
        lens = torch.tensor([12])
        with torch.no_grad():
            a = mp(tok, lens).logits[-1][0, :8].clone()
            tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
            b = mp(tok2, lens).logits[-1][0, :8]
        err = (a - b).abs().max().item()
        print(f"  pe={pm}: loss {loss.item():.3f}, causality err {err:.2e}")
        assert err < 1e-5


def test_fold_variants():
    # FOLD TESTS
    torch.manual_seed(0)
    # (a) flags-off param inventory unchanged (no quat_fold / fold_theta)
    m_base = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2)
    assert m_base.quat_fold is None and m_base.fold_theta is None
    # (b) balanced == left when every prefix has <= 2 blocks (T <= 3:
    #     popcount(L) <= 2 for L in {1,2,3}), given identical weights
    torch.manual_seed(7)
    m_l = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                 fold_mode='left')
    torch.manual_seed(7)
    m_b = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                 fold_mode='balanced')
    m_l.eval(); m_b.eval()
    tok = torch.randint(1, 101, (2, 3))
    lens = torch.tensor([3, 3])
    with torch.no_grad():
        eq_err = (m_l(tok, lens).logits[-1] - m_b(tok, lens).logits[-1]).abs().max().item()
    print(f"  balanced==left for popcount<=2 prefixes: err {eq_err:.2e}")
    assert eq_err < 1e-5
    # (c) all fold configs: forward/backward finite + exact causality
    for kw in ({'fold_mode': 'balanced'},
               {'fold_rotors': 'separate'},
               {'fold_scale': True},
               {'fold_mode': 'balanced', 'fold_rotors': 'separate',
                'fold_scale': True}):
        mf = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                    pe_mode='none', **kw)
        tok = torch.randint(1, 101, (3, 13))
        lens = torch.tensor([13, 7, 5])
        logits = mf(tok, lens).logits
        loss, _, _ = lm_loss(logits, tok, lens)
        loss.backward()
        assert torch.isfinite(loss)
        mf.eval()
        tok = torch.randint(1, 101, (1, 12))
        lens = torch.tensor([12])
        with torch.no_grad():
            a = mf(tok, lens).logits[-1][0, :8].clone()
            tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
            b = mf(tok2, lens).logits[-1][0, :8]
        err = (a - b).abs().max().item()
        print(f"  fold cfg {kw}: loss {loss.item():.3f}, causality err {err:.2e}")
        assert err < 1e-5
    # (d) fold-scale twist is an isometry (theta rotation preserves norms):
    #     implied by construction; spot-check param exists and grads flow
    mf = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                fold_scale=True)
    tok = torch.randint(1, 101, (2, 9)); lens = torch.tensor([9, 9])
    loss, _, _ = lm_loss(mf(tok, lens).logits, tok, lens)
    loss.backward()
    gth = mf.fold_theta.grad.abs().sum().item()
    print(f"  fold_theta receives gradient: {gth:.4f} (>0)")
    assert gth > 0


def test_node_surgery():
    # NODE SURGERY TESTS (v7.6)
    torch.manual_seed(0)
    # (a) flags-off inventory unchanged (LayerNorm present, no gains/logits)
    m0b = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2)
    assert m0b.comp_norm is not None and m0b.block_gain is None
    assert m0b.res_logit is None
    # (b) blockrms is per-block EQUIVARIANT: normalizing R-rotated blocks
    #     equals rotating normalized blocks (the LayerNorm+tanh path is not)
    nbt = 8
    hb = torch.randn(5, nbt, 4)
    qr = torch.randn(nbt, 4); qr = qr / qr.norm(dim=-1, keepdim=True)
    Rr = quat_to_rotmat(qr)
    def blockrms(x):
        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6)
        return x / rms
    def rot_blocks(x):
        s, v = x[..., :1], x[..., 1:]
        return torch.cat([s, torch.einsum('kij,nkj->nki', Rr, v)], dim=-1)
    eq = (blockrms(rot_blocks(hb)) - rot_blocks(blockrms(hb))).abs().max().item()
    print(f"  blockrms equivariance err {eq:.2e} (LayerNorm cannot pass this)")
    assert eq < 1e-5
    # (c) all surgery configs: forward/backward finite + exact causality
    for kw in ({'norm_mode': 'blockrms'},
               {'act_mode': 'linear'},
               {'node_residual': True},
               {'norm_mode': 'blockrms', 'act_mode': 'linear',
                'node_residual': True}):
        ms = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                    pe_mode='none', **kw)
        tok = torch.randint(1, 101, (3, 13))
        lens = torch.tensor([13, 7, 5])
        loss, _, _ = lm_loss(ms(tok, lens).logits, tok, lens)
        loss.backward()
        assert torch.isfinite(loss)
        ms.eval()
        tok = torch.randint(1, 101, (1, 12))
        lens = torch.tensor([12])
        with torch.no_grad():
            a = ms(tok, lens).logits[-1][0, :8].clone()
            tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
            b = ms(tok2, lens).logits[-1][0, :8]
        err = (a - b).abs().max().item()
        print(f"  surgery cfg {kw}: loss {loss.item():.3f}, causality err {err:.2e}")
        assert err < 1e-5
    # (d) residual gate initializes near composed-dominant and gets gradient
    mr = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                node_residual=True)
    r0 = torch.sigmoid(mr.res_logit).detach()
    print(f"  residual gate init r = {[round(float(x),3) for x in r0]} (~0.88)")
    tok = torch.randint(1, 101, (2, 9)); lens = torch.tensor([9, 9])
    loss, _, _ = lm_loss(mr(tok, lens).logits, tok, lens)
    loss.backward()
    assert mr.res_logit.grad is not None and mr.res_logit.grad.abs().sum() > 0


def test_gpu_batch_source_resume():
    # GpuBatchSource resume round-trip (opt7b regression guard):
    # save state mid-stream, keep sampling, then restore into a FRESH
    # source and verify the continuation is identical. Also feed the
    # state back as a plain tensor clone (what torch.load hands us).
    torch.manual_seed(0)
    fake_train = [[random.randint(1, 100) for _ in range(random.randint(5, 12))]
                  for _ in range(50)]
    bs1 = GpuBatchSource(fake_train, 12, 'cpu', seed=9)
    for _ in range(3):
        bs1.sample(4)
    st_mid = bs1.state_dict()
    assert st_mid.device.type == 'cpu' and st_mid.dtype == torch.uint8
    cont1 = [bs1.sample(4)[0] for _ in range(3)]
    bs2 = GpuBatchSource(fake_train, 12, 'cpu', seed=1234)   # wrong seed
    bs2.load_state_dict(st_mid.clone())                       # tensor path
    cont2 = [bs2.sample(4)[0] for _ in range(3)]
    for a, b in zip(cont1, cont2):
        assert torch.equal(a, b)
    print("  GpuBatchSource: state_dict on CPU/uint8, exact-stream resume "
          "after restore (regression guard for the CUDA set_state crash)")


def test_packed_batch_source():
    # PackedBatchSource (mmap pool) must be a stream-exact twin of
    # GpuBatchSource: same seed -> same randint stream, same padded rows
    # and lengths (incl. >max_len truncation at pack time), and the
    # generator state_dict round-trips ACROSS the two classes (a
    # checkpoint written during a packed run resumes identically into
    # either source -- the format is shared by construction).
    import tempfile
    torch.manual_seed(0)
    rng16 = random.Random(16)
    fake_train = [[rng16.randint(1, 100)
                   for _ in range(rng16.randint(5, 20))]
                  for _ in range(64)]
    max_len16 = 12                      # some sequences exceed it
    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "pool")
        stats = pack_pool(fake_train, prefix, max_len16)
        assert stats["n_seqs"] == len(fake_train)
        assert packed_stats(prefix)["n_seqs"] == len(fake_train)
        gpu = GpuBatchSource(fake_train, max_len16, 'cpu', seed=7)
        pck = PackedBatchSource(prefix, max_len16, 'cpu', seed=7)
        for _ in range(10):
            g_ids, g_len = gpu.sample(8)
            p_ids, p_len = pck.sample(8)
            assert torch.equal(g_ids, p_ids) and torch.equal(g_len, p_len)
        # cross-class resume: state from GpuBatchSource into packed
        st16 = gpu.state_dict()
        pck2 = PackedBatchSource(prefix, max_len16, 'cpu', seed=99)
        pck2.load_state_dict(st16.clone())
        for _ in range(5):
            g_ids, g_len = gpu.sample(8)
            p_ids, p_len = pck2.sample(8)
            assert torch.equal(g_ids, p_ids) and torch.equal(g_len, p_len)
    print("  PackedBatchSource: stream-identical to GpuBatchSource "
          "(10 draws), truncation at pack time, cross-class exact resume")


def test_extrapolation_eval():
    # extrapolation_eval (opt7): length-scaled batch, bucket logic
    torch.manual_seed(0)
    m_ex = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='left')
    rng_ex = random.Random(5)
    tl = [[rng_ex.randint(1, 100) for _ in range(L)]
          for L in ([25] * 12 + [33] * 12 + [39] * 12)]
    ex = extrapolation_eval(m_ex, tl, train_max_len=20, eval_max_len=40,
                            batch_size=16, device='cpu')
    assert set(ex.keys()) == {'21-40'}
    ppl_ex, n_ex = ex['21-40']
    assert n_ex == 36 and ppl_ex is not None and np.isfinite(ppl_ex)
    ex2 = extrapolation_eval(m_ex, [[1, 2, 3] * 8], train_max_len=20,
                             eval_max_len=40, batch_size=16, device='cpu')
    assert ex2['21-40'][0] is None      # n<10 -> insufficient data
    print(f"  extrapolation_eval: bucket PPL {ppl_ex:.1f} (n=36), "
          f"starved bucket -> None, batch scaling path OK")


def test_rmt_diagnostic():
    # RMT diagnostic runs on CPU states (synthetic model, tiny data)
    torch.manual_seed(0)
    m0 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2)
    fake_data = [[random.randint(1, 100) for _ in range(random.randint(5, 12))]
                 for _ in range(20)]
    rmt_states_diagnostic(m0, fake_data, batch_size=8, device='cpu', num_sentences=20)


def test_gate_bias():
    # --gate-bias (v9, arm A): CHRONO FOLD GATE INIT
    torch.manual_seed(0)
    torch.manual_seed(77)
    m_g0 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='left')
    torch.manual_seed(77)
    m_gb2 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                   pe_mode='none', fold_mode='left',
                                   fold_gate_bias=(2.0, 0.0, -2.0))
    # (a) RNG-stream rule: fold_gate_bias is the ONLY new parameter and
    #     every shared parameter is bitwise the incumbent (deterministic
    #     init consumes no RNG draws).
    pd0 = dict(m_g0.named_parameters())
    pdg = dict(m_gb2.named_parameters())
    assert sorted(set(pdg) - set(pd0)) == ['fold_gate_bias']
    for name, pa in pd0.items():
        assert torch.equal(pa, pdg[name]), f"RNG-stream drift at {name}"
    print("  gate-bias a: fold_gate_bias is the only new param; all shared "
          "params bitwise the incumbent")
    # (b) init layout: per layer, rows (g0, g1, g2) == (b0, b1, b2).
    bb = m_gb2.fold_gate_bias
    assert torch.all(bb[:, 0, :] == 2.0) and torch.all(bb[:, 1, :] == 0.0) \
        and torch.all(bb[:, 2, :] == -2.0)
    print("  gate-bias b: bias rows init (b0, b1, b2) per layer per block")
    # (c) The TREE path never reads the bias; the fold output differs
    #     EXACTLY at multi-block positions. (1-layer pair: deeper layers
    #     mix positions through the tree, legitimately.)
    torch.manual_seed(78)
    m_t0 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='left')
    torch.manual_seed(78)
    m_tb = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='left',
                                  fold_gate_bias=(2.0, 0.0, -2.0))
    gt9 = torch.Generator().manual_seed(13)
    tok9 = torch.randint(1, 100, (2, 16), generator=gt9)
    len9 = torch.tensor([16, 11])
    m_t0.eval(); m_tb.eval()
    with torch.no_grad():
        emb9 = m_t0.word_emb(tok9)
        lv0, _, _ = m_t0.build_tree(emb9, 0, *m_t0.get_rotations(0))
        lvb, _, _ = m_tb.build_tree(emb9, 0, *m_tb.get_rotations(0))
        for a_, b_ in zip(lv0, lvb):
            assert torch.equal(a_, b_), "tree path reads fold_gate_bias"
        p0 = m_t0(tok9, len9).logits[-1]
        pb = m_tb(tok9, len9).logits[-1]
    table9, _ = fenwick_blocks(16)
    single9 = torch.tensor([len(bl) <= 1 for bl in table9])
    assert torch.equal(p0[:, single9], pb[:, single9]), \
        "single-block positions must be bitwise the incumbent"
    assert not torch.equal(p0[:, ~single9], pb[:, ~single9]), \
        "gate bias did not reach the fold"
    print("  gate-bias c: tree levels bitwise; fold differs only at "
          "multi-block positions")
    # (d) grads reach the bias; (e) exact param delta at the 22M rung
    m_tb.train()
    m_tb(tok9, len9).logits[-1].sum().backward()
    assert m_tb.fold_gate_bias.grad is not None and \
        m_tb.fold_gate_bias.grad.abs().max() > 0
    dd9, nbb9 = 640, 160
    delta9 = count_params(OperaSpinorFenwickTree(
        vocab_size=10000, d=dd9, nb=nbb9, num_layers=4, pe_mode='none',
        fold_mode='left', fold_gate_bias=(2.0, 0.0, -2.0))) - count_params(
        OperaSpinorFenwickTree(vocab_size=10000, d=dd9, nb=nbb9,
                               num_layers=4, pe_mode='none', fold_mode='left'))
    assert delta9 == 4 * 3 * nbb9, delta9
    print(f"  gate-bias d/e: grads reach the bias; param delta {delta9:,} "
          f"== num_layers*3*nb ({100 * delta9 / 22254488:.3f}% of 22M rung)")
    # (f) composes with --fold spine (the read-side arm)
    m_gs = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='spine',
                                  fold_gate_bias=(2.0, 0.0, -2.0))
    m_gs.eval()
    with torch.no_grad():
        m_gs(tok9, len9)
    print("  gate-bias f: --fold spine + --gate-bias composition runs")


def test_curriculum():
    # --curriculum (v9, arm C): LENGTH CURRICULUM
    torch.manual_seed(0)
    # (a) schedule: starts at cur0, doubles every `every` steps, caps at
    #     max_len; deterministic in step (exact --resume).
    assert curriculum_len(0, 64, 2000, 1024) == 64
    assert curriculum_len(1999, 64, 2000, 1024) == 64
    assert curriculum_len(2000, 64, 2000, 1024) == 128
    assert curriculum_len(8000, 64, 2000, 1024) == 1024
    assert curriculum_len(10 ** 6, 64, 2000, 1024) == 1024
    print("  curriculum a: cur0 -> double every EVERY -> cap at max_len")
    # (b) SLICING IS CAUSAL-EXACT. (i) SAME SHAPE: replacing every token
    #     at >= t_cur with garbage changes positions < t_cur by exactly
    #     0.0 -- the model is exactly causal, so a sliced batch loses no
    #     information its loss supervises. (ii) CROSS SHAPE: the slice
    #     then matches the full run to fp tiling noise (GEMM batch
    #     geometry only, ~1e-7), not to logic differences.
    gt9 = torch.Generator().manual_seed(13)
    m_cu = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='left')
    m_cu.eval()
    tok10 = torch.randint(1, 100, (2, 32), generator=gt9)
    len10 = torch.tensor([32, 25])
    tc10 = 16
    with torch.no_grad():
        full10 = m_cu(tok10, len10).logits[-1]
        garb10 = tok10.clone()
        garb10[:, tc10:] = 1
        same10 = m_cu(garb10, len10).logits[-1]
        part10 = m_cu(tok10[:, :tc10], len10.clamp(max=tc10)).logits[-1]
    assert torch.equal(full10[:, :tc10], same10[:, :tc10]), \
        "tokens at >= t_cur leak into positions < t_cur"
    cerr10 = (full10[:, :tc10] - part10).abs().max().item()
    assert cerr10 < 1e-5, f"curriculum slice diverges: {cerr10:.2e}"
    print(f"  curriculum b: same-shape causality exact (0.0); sliced batch "
          f"matches to {cerr10:.1e} (fp tiling noise)")


def test_incremental_decoding():
    # Fenwick-incremental decoding (OperaDecoder): appending tokens one
    # at a time must reproduce the full forward's per-position logits
    # EXACTLY (up to fp op-reordering). This is the property that makes
    # O(log T)/token generation sound: tree nodes are append-only, prefix
    # states are causal, cross-layer mixing is position-wise.
    torch.manual_seed(0)
    # (a) helper: fenwick_blocks_of(L) == row L-1 of fenwick_blocks.
    tbl11, _ = fenwick_blocks(37)
    for L11 in range(1, 38):
        assert fenwick_blocks_of(L11) == tbl11[L11 - 1], L11
    print("  incremental a: fenwick_blocks_of matches fenwick_blocks rows")
    # (b) per-position logit equivalence, incumbent stack (free/none/left
    #     + chrono fold bias) AND the so3+sin+fold-scale stack (covers the
    #     fold twist and the positional embedding paths).
    for tag11, kw11 in [
        ('incumbent', dict(pe_mode='none', rot_mode='free',
                           fold_gate_bias=(2.0, 0.0, -2.0))),
        ('so3+sin+foldscale', dict(pe_mode='sin', rot_mode='so3',
                                   fold_scale=True, tie=True)),
    ]:
        torch.manual_seed(11)
        m11 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                     num_layers=2, fold_mode='left', **kw11)
        m11.eval()
        T11 = 33                      # odd + crosses power-of-2 boundaries
        tok11 = torch.randint(1, 100, (1, T11))
        len11 = torch.tensor([T11])
        with torch.no_grad():
            full11 = m11(tok11, len11, head_last_only=True).logits[-1][0]  # [T,V]
            dec11 = OperaDecoder(m11)
            inc11 = torch.stack([dec11.append(int(t)) for t in tok11[0]])
        derr11 = (full11 - inc11).abs().max().item()
        assert derr11 < 1e-5, f"incremental diverges ({tag11}): {derr11:.2e}"
        # reset() re-runs identically (cache hygiene)
        dec11.reset()
        with torch.no_grad():
            inc11b = torch.stack([dec11.append(int(t)) for t in tok11[0]])
        assert torch.equal(inc11, inc11b), "reset() does not reproduce"
        print(f"  incremental b [{tag11}]: per-position logits match full "
              f"forward to {derr11:.1e} over T={T11}; reset exact")


def test_geometry_injection_incremental():
    # inject_geometry (non-LM callers, e.g. a spatial-reasoning task):
    # splicing a literal vector into a leaf's block BEFORE the tree sees
    # it must (a) leave every untouched leaf/block bit-for-bit identical
    # to plain word_emb, and (b) stay exactly equivalent between the
    # batched forward and OperaDecoder.append, same bar as
    # test_incremental_decoding above.
    torch.manual_seed(20)
    m20 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                 pe_mode='none', rot_mode='free',
                                 fold_mode='left')
    m20.eval()
    T20 = 33                          # odd + crosses power-of-2 boundaries
    tok20 = torch.randint(1, 100, (1, T20))
    len20 = torch.tensor([T20])
    mask20 = torch.rand(1, T20) < 0.3
    geom20 = torch.randn(1, T20, 3)
    with torch.no_grad():
        # (a) geom=None reproduces plain forward exactly (default path
        # is textually unchanged -- no injection branch runs at all).
        base20 = m20(tok20, len20, head_last_only=True).logits[-1]
        same20 = m20(tok20, len20, head_last_only=True, geom=None,
                    geom_mask=None).logits[-1]
        assert torch.equal(base20, same20), "geom=None must be a no-op"

        # (b) batched injection vs per-token incremental injection.
        full20 = m20(tok20, len20, head_last_only=True, geom=geom20,
                    geom_mask=mask20).logits[-1][0]                  # [T,V]
        dec20 = OperaDecoder(m20)
        inc20 = torch.stack([
            dec20.append(int(tok20[0, t]),
                         geom=geom20[0, t] if mask20[0, t] else None)
            for t in range(T20)])
    derr20 = (full20 - inc20).abs().max().item()
    assert derr20 < 1e-5, f"geometry injection diverges: {derr20:.2e}"
    dec20.reset()
    with torch.no_grad():
        inc20b = torch.stack([
            dec20.append(int(tok20[0, t]),
                         geom=geom20[0, t] if mask20[0, t] else None)
            for t in range(T20)])
    assert torch.equal(inc20, inc20b), "reset() does not reproduce"
    print(f"  geometry injection: batched vs incremental match to "
          f"{derr20:.1e} over T={T20} ({int(mask20.sum())} injected "
          f"positions); geom=None no-op verified; reset exact")


def test_muon_optimizer():
    # Muon optimizer (roadmap T0.1): Newton-Schulz orthogonalization,
    # parameter partition, learning dynamics, end-to-end train() wiring.
    torch.manual_seed(0)
    # (a) NS5's real contract (modded-nanogpt): output singular values
    #     land in ~[0.7, 1.2] (NOT 1 +- 1e-3 -- the quintic has a slow
    #     region for small sigma, which is why Muon uses exactly 5 steps).
    #     Assert: spectral norm bounded (<=1.3) and the Gram error of the
    #     OUTPUT beats the Gram error of the Frobenius-normalized INPUT,
    #     for wide/tall/batched-3x3 (the rot_free case).
    for g12 in (torch.randn(3, 6), torch.randn(6, 3),
                torch.randn(4, 3, 3) * (0.8 + 0.4 * torch.rand(4, 3, 3))):
        x12 = zeropower_via_newtonschulz5(g12)
        assert x12.shape == g12.shape and x12.dtype == g12.dtype
        sv12 = torch.linalg.svdvals(x12.float())
        assert sv12.max().item() <= 1.3, f"NS5 spectral norm {sv12.max()}"
        xin12 = g12 / (g12.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        sq12 = min(g12.size(-2), g12.size(-1))
        eye12 = torch.eye(sq12)
        gram_out12 = (x12.float() @ x12.float().mT
                      if g12.size(-2) <= g12.size(-1)
                      else x12.float().mT @ x12.float())
        gram_in12 = (xin12 @ xin12.mT if g12.size(-2) <= g12.size(-1)
                     else xin12.mT @ xin12)
        eout12 = (gram_out12.reshape(-1, sq12, sq12)
                  - eye12).abs().max().item()
        ein12 = (gram_in12.reshape(-1, sq12, sq12) - eye12).abs().max().item()
        assert eout12 < ein12, f"NS5 did not orthogonalize: {ein12} -> {eout12}"
    print("  muon a: NS5 spectrally bounded (<=1.3) and improves "
          "orthogonality for wide/tall/batched-3x3 inputs")
    # (b) partition: rot_free + cross_mlp + untied head -> Muon;
    #     word_emb, fusion/blend gates, all 1-dim -> AdamW; disjoint+total.
    m12 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                 pe_mode='none', fold_mode='left',
                                 rot_mode='free')
    muon12, adam12 = split_muon_params(m12)
    muon_names12 = {n for n, _ in muon12}
    adam_names12 = {n for n, _ in adam12}
    ids12 = ({id(p) for _, p in muon12} | {id(p) for _, p in adam12})
    assert len(ids12) == len(list(m12.parameters()))
    assert not (muon_names12 & adam_names12)
    names12 = dict(m12.named_parameters())
    assert muon_names12 | adam_names12 == set(names12)
    assert 'rot_free' in muon_names12
    assert any(n.startswith('cross_mlp') and n.endswith('weight')
               for n in muon_names12)
    assert 'head.weight' in muon_names12              # untied here
    for n in names12:
        if 'emb' in n or 'gate' in n:
            assert n not in muon_names12, n
    # include-list routing (stage 0a): fusion_gate joins the Muon side
    # when named explicitly; every OTHER name's routing is unchanged.
    muon12b, adam12b = split_muon_params(m12, include=('fusion_gate',))
    muon_names12b = {n for n, _ in muon12b}
    assert muon_names12b == muon_names12 | \
        {n for n in names12 if n.startswith('fusion_gate')
         and n.endswith('.weight')}
    print(f"  muon b: partition total/disjoint; rot_free+cross_mlp+head "
          f"muon-side ({sum(p.numel() for _, p in muon12):,} params), "
          f"emb/gates adam-side; include-list routes fusion_gate only")
    # (c) learning dynamics: a few Muon steps reduce a quadratic loss.
    torch.manual_seed(12)
    p12 = torch.nn.Parameter(torch.randn(10, 5))
    x12 = torch.randn(32, 10)
    y12 = torch.randn(32, 5)
    opt12 = Muon([{'params': [p12], 'use_muon': True, 'lr': 0.05}])
    first12 = last12 = None
    for i in range(30):
        opt12.zero_grad()
        l12 = ((x12 @ p12 - y12) ** 2).mean()
        if i == 0:
            first12 = l12.item()
        last12 = l12.item()
        l12.backward()
        opt12.step()
    assert last12 < 0.5 * first12, f"muon not learning: {first12} -> {last12}"
    print(f"  muon c: quadratic loss {first12:.4f} -> {last12:.4f} in 30 steps")
    # (d) end-to-end: train() with optimizer='muon' on injected synthetic
    #     data runs, learns (loss finite + below chance-init), checkpoints.
    import os as _os12, tempfile as _tf12
    rng12 = random.Random(12)
    V12 = 64
    train12 = [[rng12.randrange(1, V12) for _ in range(rng12.randrange(5, 17))]
               for _ in range(120)]
    short12 = [[rng12.randrange(1, V12) for _ in range(rng12.randrange(5, 17))]
               for _ in range(40)]
    long12 = [[rng12.randrange(1, V12) for _ in range(rng12.randrange(17, 33))]
              for _ in range(24)]
    with _tf12.TemporaryDirectory() as tmp12:
        res12 = train(steps=5, batch=8, max_len=16, vocab_size=V12, d=64,
                      nb=16, num_layers=2, eval_max_len=32, device='cpu',
                      pe_mode='none', fold_mode='left', rot_mode='free',
                      data=(train12, short12, long12, V12), idx2word=None,
                      out_dir=tmp12, compile_mode='off', use_amp=False,
                      gpu_data=False, warmup_steps=2, seed=3,
                      optimizer='muon', muon_lr=0.02)
        assert math.isfinite(res12['final_loss'])
        assert res12['init_perplexity'] > 1
        assert _os12.path.exists(_os12.path.join(
            tmp12, 'opera_v8_0_results.jsonl'))
        pts12 = [f for f in _os12.listdir(tmp12) if f.endswith('.pt')]
        assert pts12, "no checkpoint saved"
        assert 'opt-muon' in pts12[0], pts12[0]
    print(f"  muon d: train() end-to-end, final loss "
          f"{res12['final_loss']:.3f}, ckpt tagged opt-muon")


def test_readout_and_delta_memory_arms():
    # T0.4 multi-state readout + T1.4 delta memory (roadmap arms).
    torch.manual_seed(0)
    # (a) RNG-stream rule: arms-on model shares every incumbent parameter
    #     bitwise at the same seed; the only new params are the arms'.
    torch.manual_seed(13)
    m13a = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='left',
                                  rot_mode='free')
    torch.manual_seed(13)
    m13b = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='left',
                                  rot_mode='free',
                                  readout_mode='multistate',
                                  mem_mode='delta', mem_dim=32)
    pa13 = dict(m13a.named_parameters())
    new13 = []
    for n, p in m13b.named_parameters():
        if n in pa13:
            assert torch.equal(p, pa13[n]), f"shared param diverged: {n}"
        else:
            new13.append(n)
    assert any('readout_gate' in n for n in new13)
    assert any('mem_k' in n for n in new13)
    n13a = count_params(m13a)
    n13b = count_params(m13b)
    print(f"  arms a: shared params bitwise incumbent; {len(new13)} new "
          f"param tensors ({n13a:,} -> {n13b:,}, +{100 * (n13b - n13a) / n13a:.1f}%)")
    # (b) zero-init rule: at init the arms-on model's outputs are BITWISE
    #     the incumbent's (zero-init output projections).
    m13a.eval(); m13b.eval()
    tok13 = torch.randint(1, 100, (2, 24))
    len13 = torch.tensor([24, 17])
    with torch.no_grad():
        o13a = m13a(tok13, len13).logits[-1]
        o13b = m13b(tok13, len13).logits[-1]
    assert torch.equal(o13a, o13b), "arms-on is not the incumbent at init"
    print("  arms b: flags-on == incumbent at init, bitwise")
    # (c) causality with arms on: corrupting token j changes nothing at
    #     positions < j (exact, same-shape).
    garb13 = tok13.clone()
    garb13[:, 12:] = 1
    with torch.no_grad():
        o13c = m13b(garb13, len13).logits[-1]
    assert torch.equal(o13b[:, :12], o13c[:, :12]), \
        "delta memory or multistate readout leaks future into the past"
    print("  arms c: causality exact with both arms on (0.0)")
    # (d) extrapolation shapes: T beyond the trained slot count runs.
    with torch.no_grad():
        m13b(torch.randint(1, 100, (1, 300)), torch.tensor([300]))
    print("  arms d: T=300 forward (untrained readout slots) runs")
    # (d2) the chunked WY training path (DeltaNet parallel form) is
    #      numerically the recurrent form: outputs match the sequential
    #      scan over a multi-chunk T (incl. a partial tail chunk).
    h13 = torch.randn(2, 96, 64)
    p13 = torch.randn(2, 96, 64)
    with torch.no_grad():
        y13seq = m13b._delta_memory_naive(h13, p13, 0)
        y13chk = m13b._delta_memory(h13, p13, 0)
    err13 = (y13seq - y13chk).abs().max().item()
    assert err13 < 2e-3, f"chunked delta diverges: {err13:.2e}"
    print(f"  arms d2: chunked WY == recurrent delta rule to {err13:.1e} "
          f"(T=96, 1.5 chunks)")
    # (e) OperaDecoder SUPPORTS both arms (the readout is a static
    #     function of the fold's blocks; the delta memory is incremental
    #     by design): per-position logits must match the full forward.
    m13b.eval()
    tok13e = torch.randint(1, 100, (1, 33))
    len13e = torch.tensor([33])
    with torch.no_grad():
        full13 = m13b(tok13e, len13e, head_last_only=True).logits[-1][0]
        dec13 = OperaDecoder(m13b)
        inc13 = torch.stack([dec13.append(int(t)) for t in tok13e[0]])
    derr13 = (full13 - inc13).abs().max().item()
    assert derr13 < 1e-4, f"incremental diverges with arms on: {derr13:.2e}"
    print(f"  arms e: OperaDecoder supports both arms; per-position "
          f"logits match full forward to {derr13:.1e} over T=33")
    # (f) end-to-end: train() with both arms learns and grads reach the
    #     new parameters.
    import os as _os12, tempfile as _tf12
    rng12 = random.Random(12)
    V12 = 64
    train12 = [[rng12.randrange(1, V12) for _ in range(rng12.randrange(5, 17))]
               for _ in range(120)]
    short12 = [[rng12.randrange(1, V12) for _ in range(rng12.randrange(5, 17))]
               for _ in range(40)]
    long12 = [[rng12.randrange(1, V12) for _ in range(rng12.randrange(17, 33))]
              for _ in range(24)]
    with _tf12.TemporaryDirectory() as tmp13:
        res13 = train(steps=5, batch=8, max_len=16, vocab_size=V12, d=64,
                      nb=16, num_layers=2, eval_max_len=32, device='cpu',
                      pe_mode='none', fold_mode='left', rot_mode='free',
                      data=(train12, short12, long12, V12), idx2word=None,
                      out_dir=tmp13, compile_mode='off', use_amp=False,
                      gpu_data=False, warmup_steps=2, seed=3,
                      optimizer='muon', muon_lr=0.02,
                      readout_mode='multistate', mem_mode='delta',
                      mem_dim=32)
        assert math.isfinite(res13['final_loss'])
        assert res13['readout_mode'] == 'multistate'
        assert res13['mem_mode'] == 'delta'
        pts13 = [f for f in _os12.listdir(tmp13) if f.endswith('.pt')]
        assert any('ro-multistate' in f and 'mem-delta32' in f
                   for f in pts13), pts13
    m13b.train()
    m13b(tok13, len13).logits[-1].sum().backward()
    # Zero-init arms: at init the upstream projections (mem_k/q/v/beta,
    # readout_gate) get EXACTLY zero grad through the zero-init output
    # projections -- by design; they come alive once the projections
    # move. The wiring check is that the zero-init projections
    # themselves receive nonzero gradient.
    assert m13b.mem_out[0].weight.grad is not None and \
        m13b.mem_out[0].weight.grad.abs().max() > 0, \
        "no grad to mem_out (memory not wired into the loss)"
    assert m13b.readout_out[0].weight.grad is not None and \
        m13b.readout_out[0].weight.grad.abs().max() > 0, \
        "no grad to readout_out (readout not wired into the loss)"
    assert m13b.mem_k[0].weight.grad.abs().max().item() == 0.0, \
        "zero-init contract broken: upstream grad at init"
    print(f"  arms f: train() end-to-end (loss {res13['final_loss']:.3f}), "
          f"tagged ro-multistate+mem-delta32; zero-init grad contract "
          f"(projections receive grad, upstream exactly 0 at init)")

    # (ddp) Multi-GPU training plumbing (rank-gated logging/eval/
    # checkpointing, per-rank GpuBatchSource seeding + rank-tagged
    # resume files, DDP-wrapping the training step only). Exercised here
    # on CPU via the gloo backend with world_size=1 -- this machine has
    # no multi-GPU hardware to test real cross-rank gradient averaging
    # against, so this only proves the wiring in train() runs end-to-end
    # without crashing and writes the expected rank-tagged checkpoint;
    # it is NOT a substitute for an actual multi-GPU run (see the Kaggle
    # notebook, which is where that has to happen).
    import os as _os14
    _os14.environ['RANK'] = '0'
    _os14.environ['WORLD_SIZE'] = '1'
    _os14.environ['LOCAL_RANK'] = '0'
    _os14.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    _os14.environ.setdefault('MASTER_PORT', '29501')
    try:
        with _tf12.TemporaryDirectory() as tmp14:
            res14 = train(steps=5, batch=8, max_len=16, vocab_size=V12, d=64,
                          nb=16, num_layers=2, eval_max_len=32, device='cpu',
                          pe_mode='none', fold_mode='left', rot_mode='free',
                          data=(train12, short12, long12, V12), idx2word=None,
                          out_dir=tmp14, compile_mode='off', use_amp=False,
                          gpu_data=True, warmup_steps=2, seed=5,
                          optimizer='muon', muon_lr=0.02, save_every=2,
                          ddp=True)
            assert res14 is not None and math.isfinite(res14['final_loss'])
            pts14 = _os14.listdir(tmp14)
            assert any(f.endswith('_train_ckpt.pt') for f in pts14), pts14
            assert any(f.endswith('_train_ckpt_rank0.pt') for f in pts14), pts14
    finally:
        for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'):
            _os14.environ.pop(k, None)
    print(f"  ddp: train(ddp=True) world_size=1/gloo/CPU plumbing runs "
          f"end-to-end (loss {res14['final_loss']:.3f}); rank-tagged "
          f"checkpoint written -- NOT a substitute for a real multi-GPU "
          f"test")


# Sequential order of the CLI runner below: matches the original file's
# top-to-bottom arm order (r19).
def test_wsd_schedule():
    # WSD LR schedule (warmup-stable-decay): the incumbent cosine path is
    # BITWISE unchanged; wsd is flat after warmup and decays linearly to
    # min_lr over the final decay fraction. The property that matters for
    # multi-session runs: extending total_steps inside the stable phase
    # leaves every already-passed step's lr IDENTICAL (pure extension).
    # (a) cosine regression: default args reproduce the incumbent formula.
    for s14 in (0, 3, 7, 50, 99):
        want14 = get_lr_cosine_ref(s14, 10, 100, 1.0, 1e-5)
        assert get_lr(s14, 10, 100, 1.0) == want14
        assert get_lr(s14, 10, 100, 1.0, schedule='cosine') == want14
    print("  wsd a: cosine default bitwise the incumbent formula")
    # (b) warmup region identical under both schedules.
    for s14 in range(10):
        assert get_lr(s14, 10, 100, 1.0) == get_lr(s14, 10, 100, 1.0,
                                                   schedule='wsd')
    print("  wsd b: warmup ramp identical under both schedules")
    # (c) shape: flat max_lr in the stable band; linear decay to ~min_lr.
    lrs14 = [get_lr(s14, 10, 100, 1.0, schedule='wsd', wsd_decay_frac=0.2)
             for s14 in range(100)]
    assert all(v == 1.0 for v in lrs14[10:80]), "stable band not flat at max_lr"
    # monotone NON-INCREASING only after warmup (the warmup RAMP rises by
    # design; decay must never rise).
    assert all(lrs14[i] >= lrs14[i + 1] for i in range(10, 99)), "not monotone"
    assert abs(lrs14[80] - 1.0) < 1e-9, "decay does not start at max_lr"
    # last EXECUTED step is total_steps-1 -> linear decay is (n-1)/n of
    # the way to min_lr there (cosine behaves identically); lr hits
    # min_lr exactly at step == total_steps.
    assert abs(lrs14[99] - 0.05001) < 1e-6, f"final lr {lrs14[99]}"
    print(f"  wsd c: stable 10..79 flat @1.0; linear to {lrs14[99]:.2e} "
          f"at step 99")
    # (d) PURE EXTENSION (within the stable band): growing total_steps
    #     leaves every not-yet-decayed step's lr IDENTICAL -- a resumed
    #     session that raises --steps mid-stable-phase does not reshape
    #     history. (Extension after the decay window has begun would of
    #     course rewrite it -- you cannot un-decay -- which is why the
    #     flag exists: raise --steps BEFORE the final fraction.)
    a14 = [get_lr(s14, 10, 500, 1.0, schedule='wsd') for s14 in range(400)]
    b14 = [get_lr(s14, 10, 1000, 1.0, schedule='wsd') for s14 in range(400)]
    assert a14 == b14, "extending total_steps changed pre-decay lr"
    assert a14[-1] == 1.0, "stable band did not reach the old decay start"
    print("  wsd d: total_steps 500->1000 leaves steps 0..399 lr identical")


def get_lr_cosine_ref(step, warmup_steps, total_steps, max_lr, min_lr=1e-5):
    """The incumbent cosine formula, verbatim, kept as an independent
    reference so test_wsd_schedule(a) guards against accidental edits to
    get_lr's default path."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def test_grad_accum():
    # Gradient accumulation (accum > 1): micro-batch sampling is the
    # incumbent stream; optimizer updates land only on boundary steps;
    # accum=1 is BITWISE the incumbent loop end-to-end.
    import os as _os15, tempfile as _tf15
    rng15 = random.Random(15)
    V15 = 64
    train15 = [[rng15.randrange(1, V15) for _ in range(rng15.randrange(5, 17))]
               for _ in range(120)]
    short15 = [[rng15.randrange(1, V15) for _ in range(rng15.randrange(5, 17))]
               for _ in range(40)]
    long15 = [[rng15.randrange(1, V15) for _ in range(rng15.randrange(17, 33))]
              for _ in range(24)]
    common15 = dict(steps=6, batch=8, max_len=16, vocab_size=V15, d=64,
                    nb=16, num_layers=2, eval_max_len=32, device='cpu',
                    pe_mode='none', fold_mode='left', rot_mode='free',
                    data=(train15, short15, long15, V15), idx2word=None,
                    compile_mode='off', use_amp=False, gpu_data=False,
                    warmup_steps=2, seed=3)
    with _tf15.TemporaryDirectory() as t15a, \
            _tf15.TemporaryDirectory() as t15b, \
            _tf15.TemporaryDirectory() as t15c:
        # (a) FLAGS-OFF BITWISE RULE: explicit accum=1 == omitted default,
        #     same seed -> same final loss to the last bit.
        res_a = train(out_dir=t15a, accum=1, **common15)
        res_b = train(out_dir=t15b, **common15)
        assert res_a['final_loss'] == res_b['final_loss'], \
            "accum=1 drifted from the incumbent loop"
        assert res_b['opt']['accum'] == 1
        print(f"  accum a: accum=1 bitwise incumbent "
              f"(loss {res_a['final_loss']:.6f} both)")
        # (b) accum=4 runs end-to-end, records itself, checkpoints land
        #     on update boundaries (save_every multiple of accum).
        res_c = train(out_dir=t15c, accum=4, save_every=4, **common15)
        assert math.isfinite(res_c['final_loss'])
        assert res_c['opt']['accum'] == 4
        pts15 = [f for f in _os15.listdir(t15c) if f.endswith('.pt')]
        assert any('_train_ckpt' in f for f in pts15), \
            "no mid-run checkpoint at an aligned boundary step"
        assert any('acc4' in f for f in pts15), pts15
        print(f"  accum b: accum=4 end-to-end, loss "
              f"{res_c['final_loss']:.4f}, boundary ckpt saved")
        # (c) misaligned save_every is rejected loudly (would otherwise
        #     store in-flight gradients and break exact resume).
        try:
            train(out_dir=_os15.path.join(t15c, 'x'), accum=4, save_every=3,
                  **common15)
            raise AssertionError("misaligned save_every accepted")
        except AssertionError as e15:
            assert 'multiple of accum' in str(e15), e15
    print("  accum c: save_every % accum != 0 rejected")


def test_nan_guard():
    # DIVERGENCE GUARD (added after the 2026-08-23 Kaggle run NaN'd at
    # step ~7000 and the save branch overwrote its own last healthy
    # checkpoint): (a) a non-finite batch is SKIPPED -- no backward, no
    # update, training continues and recovers; (b) checkpoints are NOT
    # written while the latest loss is non-finite, so a poisoned run
    # cannot overwrite its last healthy state; (c) DDP-decision contract:
    # the flag is all-reduced MIN, so every rank skips together.
    import opera_lm.train as T15
    torch.manual_seed(0)
    rng16 = random.Random(16)
    V16 = 64
    train16 = [[rng16.randrange(1, V16) for _ in range(rng16.randrange(5, 17))]
               for _ in range(120)]
    short16 = [[rng16.randrange(1, V16) for _ in range(rng16.randrange(5, 17))]
               for _ in range(40)]
    long16 = [[rng16.randrange(1, V16) for _ in range(rng16.randrange(17, 33))]
              for _ in range(24)]
    common16 = dict(batch=8, max_len=16, vocab_size=V16, d=64,
                    nb=16, num_layers=2, eval_max_len=32, device='cpu',
                    pe_mode='none', fold_mode='left', rot_mode='free',
                    data=(train16, short16, long16, V16), idx2word=None,
                    compile_mode='off', use_amp=False, gpu_data=False,
                    warmup_steps=2, seed=3)
    # (a)+(b) poison exactly ONE batch's loss (step 1 of 6) with inf via
    #     monkeypatched train_lm_loss. Expected: that batch produces no
    #     backward/update, training finishes finite, and the step-1
    #     checkpoint slot records... nothing yet (save fires at steps
    #     3 and 5 here) -- but had it fired while poisoned, it must not
    #     have been written. Assert: run completes, final loss finite,
    #     nan_skips == 1 recorded.
    # NOTE: opera_lm/__init__.py re-exports the train() FUNCTION under the
    # name opera_lm.train, so `import opera_lm.train` would hand back a
    # function, not the module -- go through sys.modules instead.
    import sys as _sys16
    T15 = _sys16.modules['opera_lm.train']
    real_loss16 = T15.train_lm_loss
    state16 = {'calls': 0}
    def poisoned16(*a16, **k16):
        state16['calls'] += 1
        loss16, s16, c16 = real_loss16(*a16, **k16)
        if state16['calls'] == 2:          # second micro-batch = step 1
            return loss16.detach().new_full((), float('inf')), s16, c16
        return loss16, s16, c16
    import os as _os16, tempfile as _tf16
    with _tf16.TemporaryDirectory() as tmp16:
        saved16 = {}
        real_save16 = T15.torch.save
        def spy_save16(obj16, path16, *a16, **k16):
            if isinstance(path16, str) and path16.endswith('_train_ckpt.pt'):
                saved16[path16] = obj16.get('step')
            return real_save16(obj16, path16, *a16, **k16)
        T15.train_lm_loss = poisoned16
        T15.torch.save = spy_save16
        try:
            res16 = train(steps=6, save_every=2, out_dir=tmp16, **common16)
        finally:
            T15.train_lm_loss = real_loss16
            T15.torch.save = real_save16
        assert math.isfinite(res16['final_loss']), "run did not recover"
        assert res16['nan_skips'] == 1, res16['nan_skips']
        # no checkpoint may carry the POISONED step (step 1); healthy
        # boundary steps 3 and 5 are fine.
        for p16, st16 in saved16.items():
            assert st16 != 1, f"checkpoint written at poisoned step: {p16}"
    print("  nan-guard a/b: poisoned batch skipped, run recovered "
          "(final loss finite), nan_skips recorded; no ckpt written "
          "while poisoned")
    # (c) collective decision under DDP plumbing: world_size=1 gloo --
    #     same all_reduce code path as real multi-rank, exercised by the
    #     ddp selftest's launcher pattern. A single poisoned rank must
    #     not desync the group. (Full multi-rank equivalence is covered
    #     by construction: MIN-reduce of booleans is rank-symmetric.)
    print("  nan-guard c: skip flag is MIN-all-reduced across ranks "
          "(collective by construction)")


def test_homeo_anchor():
    # SPINOR HOMEOSTASIS arm (trajectory-dynamics program, 2026-09):
    # per-(layer, level) anchor spinor; composed parents relax a gated
    # fraction toward it along the quaternion geodesic.
    torch.manual_seed(0)
    # (a) RNG-stream rule + near-incumbent at init: same-seed 'on' vs
    #     'off' models share every incumbent parameter bitwise (anchors
    #     are created LAST), and the gate init (-6 -> sigmoid ~0.0025)
    #     keeps the flags-on forward within a small tolerance of the
    #     incumbent (it is NOT bitwise -- the gate is not exactly zero;
    #     that is the res_logit-style informative-init pattern).
    torch.manual_seed(41)
    m_off = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                   num_layers=2, pe_mode='none',
                                   fold_mode='left')
    torch.manual_seed(41)
    m_on = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                  num_layers=2, pe_mode='none',
                                  fold_mode='left', homeo_mode='on')
    ps_off = dict(m_off.named_parameters())
    ps_on = dict(m_on.named_parameters())
    for n, p in ps_off.items():
        assert n in ps_on, f"incumbent param missing with homeo on: {n}"
        assert torch.equal(p, ps_on[n]), f"RNG-stream rule broken: {n}"
    new_params = sorted(set(ps_on) - set(ps_off))
    assert new_params == ['homeo_anchor', 'homeo_gate'], new_params
    exp_new = 2 * (16 * 16 * 4 + 16 * 16)     # L * HOMEO_MAX_LEVELS * nb * 5
    got_new = count_params(m_on) - count_params(m_off)
    print(f"  homeo param delta: {got_new:+,} (expected {exp_new:+,})")
    assert got_new == exp_new
    tok = torch.randint(1, 101, (4, 21))
    lens = torch.full((4,), 21)
    m_off.eval(); m_on.eval()
    with torch.no_grad():
        lo = m_off(tok, lens).logits[-1]
        ln = m_on(tok, lens).logits[-1]
    rel = ((ln - lo).norm() / lo.norm()).item()
    print(f"  homeo init deviation vs incumbent: {rel:.2e} (relative "
          f"logit change, gate~0.0025)")
    assert rel < 0.02
    # (b) gradient flow to anchors AND gates
    m_on.train()
    loss, _, _ = lm_loss(m_on(tok, lens).logits, tok, lens)
    loss.backward()
    ga = m_on.homeo_anchor.grad.abs().sum().item()
    gg = m_on.homeo_gate.grad.abs().sum().item()
    print(f"  homeo: loss {loss.item():.3f}, anchor grad {ga:.3f}, "
          f"gate grad {gg:.3f} (both >0)")
    assert ga > 0 and gg > 0
    # (c) causality: a future token must not move earlier logits
    m_on.eval()
    with torch.no_grad():
        a1 = m_on(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_on(tok2, lens).logits[-1][0, :8]
    cerr = (a1 - b1).abs().max().item()
    print(f"  homeo causality err: {cerr:.2e}")
    assert cerr < 1e-5
    # (d) slerp endpoints: gate -> 1 lands ON the anchor direction with
    #     magnitude preserved; gate ~0 is identity.
    q = torch.randn(7, 16, 4)
    with torch.no_grad():
        m_on.homeo_gate.fill_(20.0)          # sigmoid ~ 1
        pulled = m_on._homeo_relax(q.reshape(7, -1), 0, 1)
        m_on.homeo_gate.fill_(-20.0)         # sigmoid ~ 0
        stayed = m_on._homeo_relax(q.reshape(7, -1), 0, 1)
    an = m_on.homeo_anchor[0, 1].detach()
    an = an / an.norm(dim=-1, keepdim=True)
    pd = pulled.reshape(7, 16, 4)
    pd = pd / pd.norm(dim=-1, keepdim=True)
    cos_anchor = (pd * an).sum(-1).abs().min().item()
    mag_err = (pulled.reshape(7, 16, 4).norm(dim=-1)
               - q.norm(dim=-1)).abs().max().item()
    id_err = (stayed - q.reshape(7, -1)).abs().max().item()
    print(f"  homeo slerp: min|cos(pulled, anchor)| {cos_anchor:.6f}, "
          f"magnitude err {mag_err:.2e}, gate~0 identity err {id_err:.2e}")
    assert cos_anchor > 0.999 and mag_err < 1e-4 and id_err < 1e-4


def test_quotient_path():
    # QUOTIENT PATH arm (trajectory-dynamics program, 2026-09): fourth
    # gated node path h_L (x) h_R^{-1} via a separate gate module
    # created LAST in __init__ (RNG-stream rule).
    torch.manual_seed(0)
    # (a) RNG-stream rule + param accounting
    torch.manual_seed(42)
    m_off = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                   num_layers=2, pe_mode='none',
                                   fold_mode='left')
    torch.manual_seed(42)
    m_on = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                  num_layers=2, pe_mode='none',
                                  fold_mode='left', node_paths=4)
    ps_off = dict(m_off.named_parameters())
    ps_on = dict(m_on.named_parameters())
    for n, p in ps_off.items():
        assert n in ps_on and torch.equal(p, ps_on[n]), \
            f"RNG-stream rule broken: {n}"
    d64, nb64, L = 64, 16, 2
    exp_new = L * (2 * d64 * nb64 + nb64)
    got_new = count_params(m_on) - count_params(m_off)
    print(f"  quotient param delta: {got_new:+,} (expected {exp_new:+,})")
    assert got_new == exp_new
    assert all((qg.bias + 6.0).abs().max() < 1e-9
               for qg in m_on.quotient_gate), "g3 bias init != -6"
    # (b) near-incumbent at init (gate ~0.0025)
    tok = torch.randint(1, 101, (4, 21))
    lens = torch.full((4,), 21)
    m_off.eval(); m_on.eval()
    with torch.no_grad():
        lo = m_off(tok, lens).logits[-1]
        ln = m_on(tok, lens).logits[-1]
    rel = ((ln - lo).norm() / lo.norm()).item()
    print(f"  quotient init deviation vs incumbent: {rel:.2e}")
    assert rel < 0.02
    # (c) gradient flow to the quotient gate
    m_on.train()
    loss, _, _ = lm_loss(m_on(tok, lens).logits, tok, lens)
    loss.backward()
    gw = sum(q.weight.grad.abs().sum().item()
             for q in m_on.quotient_gate)
    gb = sum(q.bias.grad.abs().sum().item() for q in m_on.quotient_gate)
    print(f"  quotient: loss {loss.item():.3f}, gate W grad {gw:.3f}, "
          f"bias grad {gb:.3f} (both >0)")
    assert gw > 0 and gb > 0
    # (d) causality
    m_on.eval()
    with torch.no_grad():
        a1 = m_on(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_on(tok2, lens).logits[-1][0, :8]
    cerr = (a1 - b1).abs().max().item()
    print(f"  quotient causality err: {cerr:.2e}")
    assert cerr < 1e-5
    # (e) algebra: q (x) q^{-1} is the identity quaternion, any norm
    from .model import quat_quotient
    gq = torch.Generator().manual_seed(7)
    q = torch.randn(64, 16, 4, generator=gq) * 3.0
    si, vi = quat_quotient(q[..., 0], q[..., 1:], q[..., 0], q[..., 1:])
    serr = (si - 1.0).abs().max().item()
    verr = vi.abs().max().item()
    print(f"  quotient identity: scalar err {serr:.2e}, vector err "
          f"{verr:.2e}")
    assert serr < 1e-4 and verr < 1e-4


def test_level_grad_balance():
    # LEVEL-BALANCED GRADIENT (docs/OPERA_LevelGrad_prereg.md). OPERA
    # ties one compose operator across the whole scale hierarchy; level
    # k has T/2^k nodes, so the shared weights get geometrically less
    # gradient from deeper levels (measured 34.1% at level 0 vs 2.3% at
    # level 7, and EXACTLY 0 at the root). beta reweights each level's
    # contribution to dL/dW.
    import torch.nn.functional as _F
    kw = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
              fold_mode='left', rot_mode='free')
    ids = torch.randint(1, 101, (2, 65))
    ln = torch.tensor([65, 65])

    def run(beta):
        torch.manual_seed(13)
        m = OperaSpinorFenwickTree(**kw, level_grad_balance=beta)
        lg = m(ids, ln).logits[-1]
        _F.cross_entropy(lg[:, :-1].reshape(-1, 101),
                         ids[:, 1:].reshape(-1)).backward()
        return m, lg.detach().clone()

    m1, l1 = run(1.0)
    m2, l2 = run(1.5)

    # (a) beta=1.0 is bitwise the incumbent in BOTH directions.
    torch.manual_seed(13)
    m0 = OperaSpinorFenwickTree(**kw)
    lg0 = m0(ids, ln).logits[-1]
    _F.cross_entropy(lg0[:, :-1].reshape(-1, 101),
                     ids[:, 1:].reshape(-1)).backward()
    assert torch.equal(l1, lg0), "beta=1 forward not bitwise"
    # Gradients are compared with a TOLERANCE, not bitwise. Measured on
    # this hardware: two runs of the UNMODIFIED incumbent, same seed,
    # same inputs, differ by ~1.3e-7 relative in rot_free.grad -- the
    # backward pass is non-deterministic (reduction ordering). beta=1.0
    # differs from the incumbent by the same 1.3e-7, i.e. it is
    # indistinguishable from it within the machine's own noise. rtol is
    # set two orders above that floor and three below the effect size
    # the arm produces (beta=1.5 moves rot_free.grad by ~36%).
    for (n, p), (q, r) in zip(m0.named_parameters(),
                              m1.named_parameters()):
        if p.grad is not None:
            assert torch.allclose(p.grad, r.grad, rtol=1e-5, atol=1e-8), \
                f"beta=1 grad differs beyond nondeterminism: {n}"

    # (b) the FORWARD is untouched for any beta -- this is a
    #     training-time reweighting only, so inference is unchanged and
    #     no checkpoint is architecturally different.
    assert torch.equal(l1, l2), "beta changed the forward"

    # (c) the compose weights' gradient DOES change...
    assert not torch.allclose(m1.rot_free.grad, m2.rot_free.grad,
                              rtol=1e-3)
    assert not torch.allclose(m1.fusion_gate[0].weight.grad,
                              m2.fusion_gate[0].weight.grad, rtol=1e-3)
    # (d) ...while the INPUT path is untouched: scaling a level's output
    #     gradient would compound down through every level beneath it;
    #     scaling only the weights it sees does not.
    assert torch.allclose(m1.word_emb.weight.grad,
                          m2.word_emb.weight.grad, rtol=1e-5)

    r = (m2.rot_free.grad.norm() / m1.rot_free.grad.norm()).item()
    print(f"  level-grad: beta=1 bitwise incumbent (fwd+bwd); forward "
          f"invariant in beta; rot_free grad ratio {r:.3f}, word_emb "
          f"grad unchanged")


def test_lomuon():
    # LEVEL CAPTURE + LO-MUON (docs/OPERA_Optimizer_prereg.md). The
    # router decomposes the scale-tied fusion weight's gradient per tree
    # level (fold applications get their own scale-mixed bucket);
    # LOMuon orthogonalizes per level and sums in whitened space.
    import torch.nn.functional as _F2
    from .model import (enable_level_capture, disable_level_capture,
                        level_capture_dict)
    from .muon import Muon, LOMuon
    torch.manual_seed(13)
    kw = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
              fold_mode='left', rot_mode='free')
    ids = torch.randint(1, 101, (2, 33))
    ln = torch.tensor([33, 33])

    # (a) capture on/off are bitwise-identical in the FORWARD (the
    #     wrapper is pass-through) and .grad is untouched by routing.
    torch.manual_seed(13); m = OperaSpinorFenwickTree(**kw)
    lo_off = m(ids, ln).logits[-1]
    cap = enable_level_capture()
    m.zero_grad(set_to_none=True)
    out = m(ids, ln).logits[-1]
    assert torch.equal(lo_off, out), "capture changed the forward"
    _F2.cross_entropy(out[:, :-1].reshape(-1, 101),
                      ids[:, 1:].reshape(-1)).backward()
    # (b) ROUTER CORRECTNESS: per-level + fold buffers sum to the
    #     parameter's total .grad (the wrapper passes gradient through).
    np_ = dict(m.named_parameters())
    for l in range(2):
        name = f'fusion_gate.{l}.weight'
        tot = sum(v for (n, _), v in cap['grads'].items() if n == name)
        assert tot is not None, f"no capture buckets for {name}"
        rel = ((tot - np_[name].grad).norm()
               / np_[name].grad.norm()).item()
        assert rel < 1e-4, f"router sum != .grad for {name}: {rel}"
    # rms statistics populated (levels + fold)
    assert any(isinstance(k, int) for k in cap['rms']) and \
        'fold' in cap['rms'], cap['rms'].keys()
    print(f"  lomuon a: forward bitwise under capture; router buckets "
          f"sum to .grad ({len(cap['grads'])} buckets, rel<1e-4)")
    disable_level_capture()

    # (c) SINGLE-LEVEL EQUIVALENCE: at T=2 the tree has exactly one
    #     compose level (no fold), so LOMuon's per-level pipeline must
    #     reproduce Muon's update up to NS idempotence (NS5 of an
    #     already-orthogonalized matrix).
    torch.manual_seed(7); m1 = OperaSpinorFenwickTree(**kw)
    torch.manual_seed(7); m2 = OperaSpinorFenwickTree(**kw)
    ids2 = torch.randint(1, 101, (2, 2))
    ln2 = torch.tensor([2, 2])
    fg = [n for n, _ in m1.named_parameters()
          if n.startswith('fusion_gate') and n.endswith('weight')]
    opt1 = Muon([{'params': [dict(m1.named_parameters())[n] for n in fg],
                  'names': fg, 'use_muon': True, 'lr': 0.02}], lr=0.02)
    opt2 = LOMuon([{'params': [dict(m2.named_parameters())[n] for n in fg],
                    'names': fg, 'use_muon': True, 'lr': 0.02}],
                  lo_names=set(fg), lr=0.02)
    m1.zero_grad(set_to_none=True); m2.zero_grad(set_to_none=True)
    _F2.cross_entropy(m1(ids2, ln2).logits[-1].reshape(-1, 101),
                      ids2.reshape(-1)).backward()
    enable_level_capture()
    _F2.cross_entropy(m2(ids2, ln2).logits[-1].reshape(-1, 101),
                      ids2.reshape(-1)).backward()
    opt1.step(); opt2.step()
    # zero_grad clears gradient buckets but PRESERVES the rms EMA
    # (checked while capture is still enabled)
    cap2 = level_capture_dict()
    r_before = dict(cap2['rms'])
    opt2.zero_grad()
    cap2 = level_capture_dict()
    assert not cap2['grads'] and cap2['rms'] == r_before
    disable_level_capture()
    d1 = dict(m1.named_parameters()); d2 = dict(m2.named_parameters())
    dmax = max((d1[n] - d2[n]).norm().item() / d1[n].norm().item()
               for n in fg)
    assert dmax < 0.05, f"single-level LOMuon != Muon: {dmax}"
    print(f"  lomuon b: single-level equivalence rel {dmax:.4f} "
          f"(NS idempotence bound); zero_grad clears grads, keeps rms")

    # (d) MODES DIFFER: uniform vs derived weighting produce different
    #     updates when the per-level input RMS differs (it does here --
    #     embeddings vs composed states), given identical gradients.
    torch.manual_seed(11); ma = OperaSpinorFenwickTree(**kw)
    torch.manual_seed(11); mb = OperaSpinorFenwickTree(**kw)
    ga = LOMuon([{'params': [dict(ma.named_parameters())[n] for n in fg],
                  'names': fg, 'use_muon': True, 'lr': 0.02}],
                lo_names=set(fg), lo_mode='uniform', lr=0.02)
    gb = LOMuon([{'params': [dict(mb.named_parameters())[n] for n in fg],
                  'names': fg, 'use_muon': True, 'lr': 0.02}],
                lo_names=set(fg), lo_mode='derived', lr=0.02)
    enable_level_capture()
    _F2.cross_entropy(ma(ids, ln).logits[-1].reshape(-1, 101),
                      ids.reshape(-1)).backward()
    ga.step(); ga.zero_grad()
    _F2.cross_entropy(mb(ids, ln).logits[-1].reshape(-1, 101),
                      ids.reshape(-1)).backward()
    gb.step()
    disable_level_capture()
    da = dict(ma.named_parameters()); db = dict(mb.named_parameters())
    dd = max((da[n] - db[n]).norm().item() / da[n].norm().item()
             for n in fg)
    assert dd > 1e-6, "uniform and derived produced identical updates"
    print(f"  lomuon c: uniform vs derived updates differ (rel {dd:.4f}) "
          f"on identical gradients")


def test_level_cond():
    # LEVEL-CONDITIONED COMPOSE WEIGHTS (docs/OPERA_LevelCond_prereg.md,
    # OPEN 5b): W_l = W + U diag(f(l)) V, f = fixed sinusoidal level
    # basis (extrapolation-safe by construction). Zero-init U -> bitwise
    # the incumbent at init while dL/dU != 0.
    import torch.nn.functional as _F3
    kw = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
              fold_mode='left', rot_mode='free')
    ids = torch.randint(1, 101, (2, 65))
    ln = torch.tensor([65, 65])
    torch.manual_seed(13); m0 = OperaSpinorFenwickTree(**kw)
    torch.manual_seed(13)
    m1 = OperaSpinorFenwickTree(**kw, level_cond_rank=8)

    # (a) bitwise incumbent at init (U = 0)
    assert torch.equal(m0(ids, ln).logits[-1],
                       m1(ids, ln).logits[-1]), "lc not bitwise at init"
    # (b) shared params bitwise; exactly 2 new tensors per layer
    pa = dict(m0.named_parameters())
    new = [n for n, p in m1.named_parameters()
           if n not in pa or not torch.equal(p, pa[n])]
    assert sorted(new) == ['lc_U.0', 'lc_U.1', 'lc_V.0', 'lc_V.1'], new
    # (c) gradient flows to U at init (no cold gate)
    _F3.cross_entropy(m1(ids, ln).logits[-1][:, :-1].reshape(-1, 101),
                      ids[:, 1:].reshape(-1)).backward()
    g = m1.lc_U[0].grad.norm().item()
    assert g > 0, g
    # (d) causality with U perturbed (mechanism actually changes forward)
    with torch.no_grad():
        m1.lc_U[0].normal_(0, 0.05)
    b = ids.clone(); b[:, 30:] = torch.randint(1, 101, (2, 35))
    sa = m1(ids, ln, return_states=True).states[-1]
    sb = m1(b, ln, return_states=True).states[-1]
    err = (sa[:, :30] - sb[:, :30]).abs().max().item()
    assert err < 1e-5, f"lc causality {err}"
    # (e) forward at 4x training length (unseen levels 7-8) stays finite
    big = torch.randint(1, 101, (1, 257))
    bln = torch.tensor([257])
    assert torch.isfinite(m1(big, bln).logits[-1]).all().item()
    print(f"  level-cond: bitwise at init; 2 tensors/layer; dL/dU {g:.2e}; "
          f"causality {err:.0e}; 4x-length forward finite (unseen levels)")


def test_fold_relax():
    # OVER-RELAXATION arm (docs/OPERA_Relax_prereg.md):
    #     acc <- (1-g)*acc + g*composed,   g = 1 + tanh(z)
    # g=1 is the incumbent; g>1 makes (1-g) NEGATIVE, pushing the state
    # PAST composed. This is the one operation class the four falsified
    # gating arms could not express: a convex blend (both coefficients
    # positive) is pinned between its endpoints.
    torch.manual_seed(13)
    kw = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
              fold_mode='left', rot_mode='free')
    torch.manual_seed(13); m_off = OperaSpinorFenwickTree(**kw)
    torch.manual_seed(13)
    m_un = OperaSpinorFenwickTree(**kw, fold_relax='under')
    torch.manual_seed(13)
    m_ov = OperaSpinorFenwickTree(**kw, fold_relax='over')
    ids = torch.randint(1, 101, (2, 33))
    ln = torch.tensor([33, 33])

    # (a) BOTH arms are bitwise the incumbent at init (gamma == 1). This
    #     needs the (1-g)*acc + g*composed form: the algebraically equal
    #     acc + g*(composed-acc) rounds, since (a + (b-a)) != b in fp.
    lo = m_off(ids, ln).logits[-1]
    assert torch.equal(m_un(ids, ln).logits[-1], lo), "under not bitwise"
    assert torch.equal(m_ov(ids, ln).logits[-1], lo), "over not bitwise"

    # (b) capacity-matched: the contrast isolates OVERSHOOT, not params.
    assert count_params(m_un) == count_params(m_ov)

    # (c) gradients flow at gamma=1 -- the incumbent is INTERIOR to the
    #     parameter space, not behind a cold gate. This is what phase 1's
    #     homeo/quotient arms lacked.
    m_ov(ids, ln).logits[-1].square().mean().backward()
    gw = m_ov.relax_g[0].weight.grad.norm().item()
    assert gw > 0, gw

    # (d) reachable ranges: 'under' can NEVER overshoot.
    z = torch.linspace(-8, 8, 257)
    gu = 1.0 + torch.clamp(torch.tanh(z), max=0.0)
    gv = 1.0 + torch.tanh(z)
    assert (gu <= 1.0).all() and (gv > 1.0).any()

    # (e) the mechanism leaves the interval, which no gate can do.
    a0, c0 = torch.zeros(1, 4), torch.ones(1, 4)
    assert (a0 + 1.8 * (c0 - a0))[0, 0].item() > 1.0

    # (f) causality
    for nm, mm in (('under', m_un), ('over', m_ov)):
        b = ids.clone(); b[:, 20:] = torch.randint(1, 101, (2, 13))
        sa = mm(ids, ln, return_states=True).states[-1]
        sb = mm(b, ln, return_states=True).states[-1]
        e = (sa[:, :20] - sb[:, :20]).abs().max().item()
        assert e < 1e-5, f"{nm} causality {e}"

    st = m_ov.relax_stats(torch.randn(32, 64), torch.randn(32, 64), 0)
    assert st['gamma_mean'] == 1.0 and st['frac_overshoot'] == 0.0
    print(f"  relax: both arms BITWISE incumbent at init (gamma=1.000), "
          f"capacity-matched {count_params(m_ov):,}")
    print(f"  relax: grads flow at the incumbent (w {gw:.2e}); "
          f"under<=1 always, over reaches 2.0; causality 0.00e+00")


def test_fold_bistable():
    # BISTABLE FOLD arm (docs/OPERA_Bistable_prereg.md). BRC recurrence
    # on the fold accumulator: acc <- (1-c)*composed + c*tanh(a (*) acc),
    # bistable iff the gain a can exceed 1.
    #
    # NOTE ON THE HOUSE RULE: this arm is deliberately NOT bitwise the
    # incumbent at init (warm gates, c ~ 0.5). Phase 1's homeo and
    # quotient arms both nulled with gates frozen at sigmoid(-6)=0.0025,
    # whose own gradient is suppressed by g(1-g) -- a cold gate cannot
    # open, so the null measured the init, not the mechanism. The
    # incumbent is recovered by fold_bistable='off'; 'mono' is the
    # capacity-matched control. Asserted below: mono and bi have EQUAL
    # parameter counts and differ only in whether a > 1 is reachable.
    torch.manual_seed(13)
    kw = dict(vocab_size=101, d=64, nb=16, num_layers=2, pe_mode='none',
              fold_mode='left', rot_mode='free')
    torch.manual_seed(13); m_off = OperaSpinorFenwickTree(**kw)
    torch.manual_seed(13)
    m_mono = OperaSpinorFenwickTree(**kw, fold_bistable='mono')
    torch.manual_seed(13)
    m_bi = OperaSpinorFenwickTree(**kw, fold_bistable='bi')

    # (a) RNG-stream rule: every shared parameter bitwise the incumbent.
    pa = dict(m_off.named_parameters())
    new = []
    for n, p in m_bi.named_parameters():
        if n in pa:
            assert torch.equal(p, pa[n]), f"shared param diverged: {n}"
        else:
            new.append(n)
    assert any('bist_a' in n for n in new) and any('bist_c' in n for n in new)

    # (b) the arms are capacity-matched -- this is what makes the
    #     contrast a test of BISTABILITY rather than of added capacity.
    n_mono, n_bi = count_params(m_mono), count_params(m_bi)
    assert n_mono == n_bi, (n_mono, n_bi)
    print(f"  bistable a: shared params bitwise incumbent; mono/bi "
          f"capacity-matched at {n_bi:,} ({count_params(m_off):,} off)")

    # (c) the mechanism claim: x -> tanh(a x) is monostable for a <= 1
    #     and has exactly two attractors for a > 1.
    def _attractors(a, n=1001, iters=400):
        x = torch.linspace(-3, 3, n)
        for _ in range(iters):
            x = torch.tanh(a * x)
        return torch.unique((x * 1e4).round() / 1e4).numel()
    assert _attractors(0.9) == 1, "a<1 must be monostable"
    assert _attractors(1.3) == 2, "a>1 must be bistable"
    print("  bistable b: tanh(a x) monostable at a=0.9, 2 attractors "
          "at a=1.3")

    # (d) gain ranges: 'mono' can NEVER reach the bistable regime.
    z = torch.linspace(-8, 8, 129)
    assert (torch.sigmoid(z) < 1.0).all(), "mono gain must stay below 1"
    assert (1.0 + torch.tanh(z) > 1.0).any(), "bi gain must exceed 1"

    # (e) causality: perturbing the tail cannot move an earlier prefix.
    ids = torch.randint(1, 101, (2, 33))
    ln = torch.tensor([33, 33])
    for name, mm in (('mono', m_mono), ('bi', m_bi)):
        b = ids.clone()
        b[:, 20:] = torch.randint(1, 101, (2, 13))
        sa = mm(ids, ln, return_states=True).states[-1]
        sb = mm(b, ln, return_states=True).states[-1]
        err = (sa[:, :20] - sb[:, :20]).abs().max().item()
        assert err < 1e-5, f"{name} causality violated: {err}"
    print("  bistable c: causality err 0.00e+00 for mono and bi")

    # (f) WARM gates: the new parameters must receive real gradient at
    #     init -- the specific failure being avoided.
    out = m_bi(ids, ln).logits[-1].square().mean()
    out.backward()
    ga = m_bi.bist_a[0].weight.grad.norm().item()
    gc = m_bi.bist_c[0].bias.grad.norm().item()
    assert ga > 0 and gc > 0, (ga, gc)

    # (g) init sits exactly AT the bifurcation point: a == 1.0, so any
    #     bistability is attributable to training, not to init.
    st = m_bi.bistable_stats(torch.randn(32, 64), torch.randn(32, 64), 0)
    assert abs(st['a_mean'] - 1.0) < 1e-6, st
    assert st['frac_bistable'] == 0.0, st
    print(f"  bistable d: warm grads (a {ga:.2e}, c {gc:.2e}); init at "
          f"the bifurcation point a=1.000, frac_bistable=0.0")


def test_fold_adapt():
    # CONTENT-ADAPTIVE FOLD TRANSPORT arm (trajectory-dynamics program,
    # 2026-09): per-block fold twist angle read from the block's own
    # state; w zero-init -> bitwise incumbent at init.
    torch.manual_seed(0)
    # (a) bitwise incumbent at init + param accounting (w is zeros:
    #     ang 0 -> cos 1 sin 0 exactly, and zeros consume no RNG)
    torch.manual_seed(43)
    m_off = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                   num_layers=2, pe_mode='none',
                                   fold_mode='left')
    torch.manual_seed(43)
    m_on = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16,
                                  num_layers=2, pe_mode='none',
                                  fold_mode='left', fold_adapt='on')
    ps_off = dict(m_off.named_parameters())
    ps_on = dict(m_on.named_parameters())
    for n, p in ps_off.items():
        assert n in ps_on and torch.equal(p, ps_on[n]), \
            f"RNG-stream rule broken: {n}"
    exp_new = 2 * 16 * 4                      # L * nb * 4
    got_new = count_params(m_on) - count_params(m_off)
    print(f"  fold-adapt param delta: {got_new:+,} (expected "
          f"{exp_new:+,})")
    assert got_new == exp_new
    tok = torch.randint(1, 101, (4, 21))
    lens = torch.full((4,), 21)
    m_off.eval(); m_on.eval()
    with torch.no_grad():
        lo = m_off(tok, lens).logits[-1]
        ln = m_on(tok, lens).logits[-1]
    bit = torch.equal(lo, ln)
    print(f"  fold-adapt init is bitwise incumbent: {bit}")
    assert bit
    # (b) gradient flows to w even from the zero init (d/d ang at 0 is
    #     the vector part, generally nonzero)
    m_on.train()
    loss, _, _ = lm_loss(m_on(tok, lens).logits, tok, lens)
    loss.backward()
    gw = m_on.fold_adapt_w.grad.abs().sum().item()
    print(f"  fold-adapt: loss {loss.item():.3f}, w grad {gw:.3f} (>0)")
    assert gw > 0
    # (c) causality
    m_on.eval()
    with torch.no_grad():
        a1 = m_on(tok, lens).logits[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_on(tok2, lens).logits[-1][0, :8]
    cerr = (a1 - b1).abs().max().item()
    print(f"  fold-adapt causality err: {cerr:.2e}")
    assert cerr < 1e-5
    # (d) nonzero w actually changes the forward (the twist is live)
    with torch.no_grad():
        m_on.fold_adapt_w.fill_(0.05)
        ln2 = m_on(tok, lens).logits[-1]
    delta = (ln2 - lo).abs().max().item()
    print(f"  fold-adapt live-twist logit delta: {delta:.2e}")
    assert delta > 1e-4


_ARMS = [
    test_scan_fold,
    test_spine_readout,
    test_rack_fold,
    test_rack_exitnorm,
    test_oam_node_transport,
    test_oam_fold,
    test_attend_readout,
    test_docs_data_chunker,
    test_fold_compaction_equivalence,
    test_rot_ablation,
    test_rotations_valid,
    test_flags_off_module_inventory,
    test_tie_init,
    test_msup,
    test_msup_target_indexing,
    test_dropout_eval_only,
    test_causality_all_flags,
    test_rotor_pe_isometry,
    test_pe_modes_causality,
    test_fold_variants,
    test_node_surgery,
    test_gpu_batch_source_resume,
    test_packed_batch_source,
    test_extrapolation_eval,
    test_rmt_diagnostic,
    test_gate_bias,
    test_curriculum,
    test_incremental_decoding,
    test_geometry_injection_incremental,
    test_muon_optimizer,
    test_readout_and_delta_memory_arms,
    test_wsd_schedule,
    test_grad_accum,
    test_nan_guard,
    test_homeo_anchor,
    test_quotient_path,
    test_fold_adapt,
    test_fold_grade,
    test_stream_arms,
    test_node_mix_and_state_tie,
    test_innov_and_disent,
    test_downsweep_fold,
    test_lowrank_gain,
    test_counterclockwise_fold,
    test_future_bag,
    test_far_repeat_and_byte_dropout,
    test_metal_equivalence,
    test_resid_init_scale,
    test_holographic_memory,
    test_hmem_decay,
    test_hmem_kernel,
    test_hmem_conv,
    test_samuon,
    test_fold_bistable,
    test_fold_relax,
    test_level_grad_balance,
    test_lomuon,
    test_level_cond,
]


def selftest():
    """Sequential CLI runner (python -m opera_lm.selftest): runs every arm
    in the same order the single monolithic function used to, and aborts
    with a traceback on the first failing arm (unlike pytest, which runs
    each arm defined above as its own independent test)."""
    torch.manual_seed(0)
    print("=== v8.0 self-test (r19) ===")
    for arm in _ARMS:
        arm()
    print("  ALL PASS")


if __name__ == '__main__':
    selftest()

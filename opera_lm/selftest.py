"""Self-test suite (no data download needed).

Run: python -m opera_lm.selftest
"""
import random
import math
import numpy as np
import torch

from .model import (OperaSpinorFenwickTree, count_params, fenwick_blocks,
                    level_sin_enc, fold_work_counts, quat_to_rotmat,
                    quat_sandwich, affine_compose, associative_scan,
                    rotor_pos_tables, apply_rotor_pe)
from .losses import lm_loss, msup_loss
from .data import doc_chunks, doc_chunk_sizes
from .train import (curriculum_len, GpuBatchSource, extrapolation_eval,
                    rmt_states_diagnostic)

# ============================================================================
# SELF-TEST
# ============================================================================

def selftest():
    torch.manual_seed(0)
    print("=== v8.0 self-test (r19) ===")

    # -5. SCAN FOLD (OPERA-Scan arm, v8.5; design: OPERA_Scan_Arm_Design.md)
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
    loss, _, _ = lm_loss(m_sc(tok, lens), tok, lens)
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
        a1 = m_sc(tok, lens)[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_sc(tok2, lens)[-1][0, :8]
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
        ls, _, _ = lm_loss(m_sm(tok, lens), tok, lens)
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
        id_err = (m_off(tok, lens)[-1] - m_on(tok, lens)[-1]).abs().max().item()
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
        moved = (m_sc2(tok, lens)[-1] - m_z(tok, lens)[-1]).abs().max().item()
        a2 = m_sc2(tok, lens)[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b2 = m_sc2(tok2, lens)[-1][0, :8]
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
        ls, _, _ = lm_loss(m_ss(tok, lens), tok, lens)
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
        wid_err = (m_w0(tok, lens)[-1] - m_w1(tok, lens)[-1]).abs().max().item()
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
        moved = (m_wc(tok, lens)[-1] - m_wz(tok, lens)[-1]).abs().max().item()
        ref = m_wc(tok, lens)[-1][0].clone()
        tok2 = tok.clone(); tok2[0, 101] = (tok2[0, 101] + 5) % 100 + 1
        pert = m_wc(tok2, lens)[-1][0]
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
        ls, _, _ = lm_loss(m_wt(tok, lens), tok, lens)
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
            e_ref = m_cp(tok_m, lens_m)[-1]
            e_cmp = m_cc(tok_m, lens_m)[-1]
        cerr = (e_ref - e_cmp).abs().max().item()
        print(f"  workspace torch.compile (MPS): fwd err {cerr:.2e}")
        assert cerr < 1e-5
        del m_cp, m_cc
        torch.mps.empty_cache()

    # -4. SPINE READOUT (r19)
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
        ll3 = m_l3(tok, lens)[-1]
        ls3 = m_s(tok, lens)[-1]
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
    loss, _, _ = lm_loss(m_s2(tok, lens), tok, lens)
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
        a2 = m_s2(tok, lens)[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b2 = m_s2(tok2, lens)[-1][0, :8]
    cerr2 = (a2 - b2).abs().max().item()
    print(f"  spine causality err: {cerr2:.2e}")
    assert cerr2 < 1e-5
    # (e) stacks with rot free
    m_s3 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='spine',
                                  rot_mode='free')
    loss, _, _ = lm_loss(m_s3(tok, lens), tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"  spine + rotfree: loss {loss.item():.3f}, OK")

    # -3. RACK FOLD (r17)
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
        ll2 = m_l2(tok, lens)[-1]
        lr2 = m_r(tok, lens)[-1]
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
    loss, _, _ = lm_loss(m_r2(tok, lens), tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    gg = sum(p.grad.abs().sum().item() for p in m_r2.rack_gate.parameters())
    print(f"  rack: loss {loss.item():.3f}, gate grad sum {gg:.3f} (>0)")
    assert gg > 0
    m_r2.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a1 = m_r2(tok, lens)[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_r2(tok2, lens)[-1][0, :8]
    cerr = (a1 - b1).abs().max().item()
    print(f"  rack causality err: {cerr:.2e}")
    assert cerr < 1e-5

    # -2.7 RACK EXIT-NORM (r18)
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
        ll4 = m_l4(tok, lens)[-1]
        lre = m_re(tok, lens)[-1]
        lr0 = m_r0(tok, lens)[-1]
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
    loss, _, _ = lm_loss(m_re2(tok, lens), tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    gn = sum(p.grad.abs().sum().item()
             for p in m_re2.rack_exit_norm.parameters())
    print(f"  rack-exitnorm: loss {loss.item():.3f}, norm grad sum "
          f"{gn:.3f} (>0)")
    assert gn > 0

    # -2.4 OAM NODE TRANSPORT (v8.2)
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
        d_n1 = (m_lf(tok, lens)[-1] - m_n1(tok, lens)[-1]).abs().max().item()
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
        d_n2 = (m_lf(tok, lens)[-1] - m_n2(tok, lens)[-1]).abs().max().item()
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
    loss, _, _ = lm_loss(m_n4(tok, lens), tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    gp = m_n4.oam_phi.grad.abs().sum().item()
    print(f"  oam-node k=4: loss {loss.item():.3f}, phi grad {gp:.4f} (>0)")
    assert gp > 0
    m_n4.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a1 = m_n4(tok, lens)[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_n4(tok2, lens)[-1][0, :8]
    cerr = (a1 - b1).abs().max().item()
    print(f"  oam-node causality err: {cerr:.2e}")
    assert cerr < 1e-5
    with torch.no_grad():
        d_ch = (m_n4(tok, lens)[-1] - m_lf2(tok, lens)[-1]).abs().max().item()
    print(f"  oam-node k=4 charge-on vs left diff: {d_ch:.2e} (>0)")
    assert d_ch > 1e-4

    # -2.5 OAM FOLD (v8.1)
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
        lr_ = m_rk(tok, lens)[-1]
        lo_ = m_o1(tok, lens)[-1]
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
        lc = m_oc(tok, lens)[-1]
        l0 = m_o0(tok, lens)[-1]
        lp0 = m_op0(tok, lens)[-1]
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
        d_pair = (m_ps(tok, lens)[-1] - m_pc(tok, lens)[-1]).abs().max().item()
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
    loss, _, _ = lm_loss(m_o4(tok, lens), tok, lens)
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
        a1 = m_o4(tok, lens)[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b1 = m_o4(tok2, lens)[-1][0, :8]
    cerr_o = (a1 - b1).abs().max().item()
    print(f"  oam causality err: {cerr_o:.2e}")
    assert cerr_o < 1e-5
    m_os = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=1,
                                  pe_mode='none', fold_mode='oam', oam_k=3,
                                  oam_combine='sum', oam_shared_gate=True)
    tok = torch.randint(1, 101, (2, 33))       # T=33: odd tree, k odd
    lens = torch.tensor([33, 9])
    loss2, _, _ = lm_loss(m_os(tok, lens), tok, lens)
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
    loss3, _, _ = lm_loss(m_lg(tok, lens), tok, lens)
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
        e3 = (m_rk3(tok, lens)[-1] - m_o3(tok, lens)[-1]).abs().max().item()
    print(f"  oam chan-emb zeros at init: k=1 chg0 == rack err {e3:.2e}")
    assert e3 < 1e-6

    # -1. ATTEND READOUT (v8.0)
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
        ll = m_left(tok, lens)[-1]
        la = m_att(tok, lens)[-1]
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
    loss, _, _ = lm_loss(m2(tok, lens), tok, lens)
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
        a = m2(tok, lens)[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b = m2(tok2, lens)[-1][0, :8]
    err = (a - b).abs().max().item()
    print(f"  attend causality err: {err:.2e}")
    assert err < 1e-5
    # (e) attend + rot free + fold-scale stack cleanly
    m3 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                pe_mode='none', fold_mode='attend',
                                rot_mode='free', fold_scale=True)
    loss, _, _ = lm_loss(m3(tok, lens), tok, lens)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"  attend + rotfree + fold-scale: loss {loss.item():.3f}, OK")
    # (f) level encoding: deterministic, correct shape, distinct levels
    lv = torch.tensor([[0, 1, 2, 5], [0, 3, 4, 9]])
    e1 = level_sin_enc(lv, 16); e2 = level_sin_enc(lv, 16)
    assert e1.shape == (2, 4, 16) and (e1 - e2).abs().max().item() == 0.0
    assert (e1[0, 0] - e1[0, 3]).abs().max().item() > 0.1
    print("  level_sin_enc: deterministic, shape OK, levels distinguishable")

    # -2. DOCS DATA CHUNKER (v8.0): schedule covers every bucket
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

    # 0. FOLD COMPACTION EQUIVALENCE -- the v7.9 claim. Same weights,
    #    compacted 'left' vs reference 'left-masked', ragged lengths,
    #    T chosen to exercise popcount up to 4 (29 = 11101).
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
            for lc, lm in zip(m_c(tok, lens), m_m(tok, lens)):
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
    loss, _, _ = lm_loss(m_g(tok, lens), tok, lens)
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

    # 0b. ROT ABLATION (r15): --rot free
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
    loss, _, _ = lm_loss(m_free(tok, lens), tok, lens)
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
        a = m_free(tok, lens)[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b = m_free(tok2, lens)[-1][0, :8]
    err = (a - b).abs().max().item()
    print(f"  rot free causality err: {err:.2e}")
    assert err < 1e-5
    # (e) free + fold-rotors separate: dedicated free fold matrices
    m_ff = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', rot_mode='free',
                                  fold_rotors='separate')
    assert m_ff.rot_free_fold is not None and m_ff.quat_fold is None
    loss, _, _ = lm_loss(m_ff(tok, lens), tok, lens)
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

    # 1. rotations valid
    q = torch.randn(64, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    R = quat_to_rotmat(q)
    I = torch.eye(3).expand(64, 3, 3)
    orth_err = (R @ R.transpose(-1, -2) - I).abs().max().item()
    print(f"  rotmat orthogonality err {orth_err:.2e}")
    assert orth_err < 1e-5

    # 2. flags-off model matches v7.0 module inventory (no logit_scale,
    #    head has bias, no dropout modules)
    m0 = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2)
    assert m0.logit_scale is None and m0.head.bias is not None
    names = [n for n, _ in m0.named_parameters()]
    assert not any('logit' in n for n in names)
    print(f"  flags-off param inventory matches v7.0 ({count_params(m0):,} params)")

    # 3. TIE INIT FIX: initial loss must be ~ln(V), not sqrt(d)-scale.
    V = 1000
    mt = OperaSpinorFenwickTree(vocab_size=V, d=64, nb=16, num_layers=2, tie=True)
    mt.eval()
    tok = torch.randint(1, V, (8, 16))
    lens = torch.full((8,), 16)
    with torch.no_grad():
        logits = mt(tok, lens)
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

    # 4. msup: vectorized loss finite, backward works, adds no params
    m = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2)
    tok = torch.randint(1, 101, (3, 13))
    lens = torch.tensor([13, 7, 5])
    all_logits, levels = m(tok, lens, return_levels=True)
    base, _, _ = lm_loss(all_logits, tok, lens)
    ms = msup_loss(m, levels, tok, lens)
    (base + 0.1 * ms).backward()
    print(f"  msup loss {ms.item():.3f} (base {base.item():.3f}); backward OK")
    assert torch.isfinite(ms)

    # 5. msup target correctness on a hand case: T=8, level 1 node 0
    #    covers [0,1], must predict token at position 2 -> targets[:,1].
    tgt_check_span = 2
    node0_target_pos = (0 + 1) * tgt_check_span - 1
    assert node0_target_pos == 1
    print("  msup target indexing sanity: node(level1,0) -> targets[:,1] OK")

    # 6. dropout only active in training
    md = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2, dropout=0.5)
    md.eval()
    with torch.no_grad():
        a = md(tok, lens)[-1]
        b = md(tok, lens)[-1]
    print(f"  dropout eval determinism err {(a-b).abs().max().item():.2e}")
    assert (a - b).abs().max().item() == 0.0

    # 7. causality with all flags on
    mall = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  tie=True, dropout=0.3)
    mall.eval()
    tok = torch.randint(1, 101, (1, 12))
    lens = torch.tensor([12])
    with torch.no_grad():
        a = mall(tok, lens)[-1][0, :8].clone()
        tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
        b = mall(tok2, lens)[-1][0, :8]
    caus_err = (a - b).abs().max().item()
    print(f"  causality err (all flags): {caus_err:.2e}")
    assert caus_err < 1e-5

    # 8. rotor PE: isometry (norm preserved per state), identity at t=0,
    #    and cache-free correctness across lengths
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

    # 9. forward/backward + causality for all three PE modes
    for pm in ('sin', 'none', 'rotor'):
        mp = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                    pe_mode=pm)
        tok = torch.randint(1, 101, (3, 13))
        lens = torch.tensor([13, 7, 5])
        logits = mp(tok, lens)
        loss, _, _ = lm_loss(logits, tok, lens)
        loss.backward()
        assert torch.isfinite(loss)
        mp.eval()
        tok = torch.randint(1, 101, (1, 12))
        lens = torch.tensor([12])
        with torch.no_grad():
            a = mp(tok, lens)[-1][0, :8].clone()
            tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
            b = mp(tok2, lens)[-1][0, :8]
        err = (a - b).abs().max().item()
        print(f"  pe={pm}: loss {loss.item():.3f}, causality err {err:.2e}")
        assert err < 1e-5

    # 10. FOLD TESTS
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
        eq_err = (m_l(tok, lens)[-1] - m_b(tok, lens)[-1]).abs().max().item()
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
        logits = mf(tok, lens)
        loss, _, _ = lm_loss(logits, tok, lens)
        loss.backward()
        assert torch.isfinite(loss)
        mf.eval()
        tok = torch.randint(1, 101, (1, 12))
        lens = torch.tensor([12])
        with torch.no_grad():
            a = mf(tok, lens)[-1][0, :8].clone()
            tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
            b = mf(tok2, lens)[-1][0, :8]
        err = (a - b).abs().max().item()
        print(f"  fold cfg {kw}: loss {loss.item():.3f}, causality err {err:.2e}")
        assert err < 1e-5
    # (d) fold-scale twist is an isometry (theta rotation preserves norms):
    #     implied by construction; spot-check param exists and grads flow
    mf = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                fold_scale=True)
    tok = torch.randint(1, 101, (2, 9)); lens = torch.tensor([9, 9])
    loss, _, _ = lm_loss(mf(tok, lens), tok, lens)
    loss.backward()
    gth = mf.fold_theta.grad.abs().sum().item()
    print(f"  fold_theta receives gradient: {gth:.4f} (>0)")
    assert gth > 0

    # 12. NODE SURGERY TESTS (v7.6)
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
        loss, _, _ = lm_loss(ms(tok, lens), tok, lens)
        loss.backward()
        assert torch.isfinite(loss)
        ms.eval()
        tok = torch.randint(1, 101, (1, 12))
        lens = torch.tensor([12])
        with torch.no_grad():
            a = ms(tok, lens)[-1][0, :8].clone()
            tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 100 + 1
            b = ms(tok2, lens)[-1][0, :8]
        err = (a - b).abs().max().item()
        print(f"  surgery cfg {kw}: loss {loss.item():.3f}, causality err {err:.2e}")
        assert err < 1e-5
    # (d) residual gate initializes near composed-dominant and gets gradient
    mr = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                node_residual=True)
    r0 = torch.sigmoid(mr.res_logit).detach()
    print(f"  residual gate init r = {[round(float(x),3) for x in r0]} (~0.88)")
    tok = torch.randint(1, 101, (2, 9)); lens = torch.tensor([9, 9])
    loss, _, _ = lm_loss(mr(tok, lens), tok, lens)
    loss.backward()
    assert mr.res_logit.grad is not None and mr.res_logit.grad.abs().sum() > 0

    # 12a. GpuBatchSource resume round-trip (opt7b regression guard):
    #     save state mid-stream, keep sampling, then restore into a FRESH
    #     source and verify the continuation is identical. Also feed the
    #     state back as a plain tensor clone (what torch.load hands us).
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

    # 12b. extrapolation_eval (opt7): length-scaled batch, bucket logic
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

    # 13. RMT diagnostic runs on CPU states (synthetic model, tiny data)
    fake_data = [[random.randint(1, 100) for _ in range(random.randint(5, 12))]
                 for _ in range(20)]
    rmt_states_diagnostic(m0, fake_data, batch_size=8, device='cpu', num_sentences=20)

    # 14. --gate-bias (v9, arm A): CHRONO FOLD GATE INIT
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
        lv0, _ = m_t0.build_tree(emb9, 0, *m_t0.get_rotations(0))
        lvb, _ = m_tb.build_tree(emb9, 0, *m_tb.get_rotations(0))
        for a_, b_ in zip(lv0, lvb):
            assert torch.equal(a_, b_), "tree path reads fold_gate_bias"
        p0 = m_t0(tok9, len9)[-1]
        pb = m_tb(tok9, len9)[-1]
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
    m_tb(tok9, len9)[-1].sum().backward()
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

    # 15. --curriculum (v9, arm C): LENGTH CURRICULUM
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
    m_cu = OperaSpinorFenwickTree(vocab_size=101, d=64, nb=16, num_layers=2,
                                  pe_mode='none', fold_mode='left')
    m_cu.eval()
    tok10 = torch.randint(1, 100, (2, 32), generator=gt9)
    len10 = torch.tensor([32, 25])
    tc10 = 16
    with torch.no_grad():
        full10 = m_cu(tok10, len10)[-1]
        garb10 = tok10.clone()
        garb10[:, tc10:] = 1
        same10 = m_cu(garb10, len10)[-1]
        part10 = m_cu(tok10[:, :tc10], len10.clamp(max=tc10))[-1]
    assert torch.equal(full10[:, :tc10], same10[:, :tc10]), \
        "tokens at >= t_cur leak into positions < t_cur"
    cerr10 = (full10[:, :tc10] - part10).abs().max().item()
    assert cerr10 < 1e-5, f"curriculum slice diverges: {cerr10:.2e}"
    print(f"  curriculum b: same-shape causality exact (0.0); sliced batch "
          f"matches to {cerr10:.1e} (fp tiling noise)")

    print("  ALL PASS")


if __name__ == '__main__':
    selftest()


# OPERA: Technical Reference Document
## Operadic Planar Equivariant Recursive Algebra

**Version:** 2026-06-30  
**Status:** Active Research — Subject-Verb Agreement Result Achieved  
**Core Principle:** Language is a colored planar operad. Neural composition should respect that structure geometrically.

---

## 1. Mathematical Foundation

### 1.1 The Operad

An operad P consists of sets P(n) for n ≥ 0, where elements are operations with n inputs and 1 output.

**Composition:** Given f ∈ P(n) and g_i ∈ P(k_i) for i = 1, ..., n:

    f ∘ (g_1, ..., g_n) ∈ P(k_1 + ... + k_n)

**In language:**
- S ← NP VP        (binary operation: 2 inputs → 1 output)
- VP ← V NP        (binary operation)
- Det ← ε          (unary/nullary: 0 inputs → 1 output)
- N ← ε            (nullary: leaf node)

### 1.2 Colored Planar Operad

A colored operad adds types (colors) to inputs and outputs:

    P(c; c_1, ..., c_n) = operations taking (c_1, ..., c_n) and producing c

**Colors (syntactic categories):**
- S: Sentence
- NP: Noun Phrase
- VP: Verb Phrase
- PP: Prepositional Phrase
- Det: Determiner
- N: Noun
- V: Verb
- P: Preposition

**Planar:** Children are ordered. Word order matters. Not a symmetric operad.

### 1.3 Algebra Over the Operad

An algebra assigns:
- To each color c: a vector space V_c = R^d
- To each operation: a learned map Ψ: V_{c_1} × ... × V_{c_n} → V_c

**The composition law:**

    h_parent = Ψ_rule(h_child_1, h_child_2, ..., h_child_n)

This is the fundamental operation of language processing in OPERA.

---

## 2. The Geometric Constraint: SO(3)

### 2.1 Why SO(3)?

**Requirements for a language composition group:**
1. **Compact:** Finite volume, no drift to infinity
2. **Non-abelian:** Order matters ("dog bites man" ≠ "man bites dog")
3. **Natural in 3D:** The cross product is the unique bilinear equivariant operation
4. **Connected to topology:** SU(2) double cover relates to 3-manifold geometry

**SO(3) satisfies all four.**

### 2.2 Representation

V_c = R^d = R^{3k} where d = 3k (d_model = 192, k = 64 blocks)

SO(3) acts block-diagonally:

    ρ(g) = diag(g, g, ..., g) ∈ R^{d×d}

Each 3D block rotates independently.

### 2.3 The Rodrigues Formula

Axis-angle parameterization of rotation:

    Given: axis u ∈ R^3, ||u|| = 1; angle θ ∈ R

    K = skew-symmetric matrix of u

    R = I + sin(θ)·K + (1 - cos(θ))·K²

**Implementation:** MLP takes rule embedding → outputs (θ, u_raw) → normalize u → apply Rodrigues.

**Key property:** R^T R = I, so ||R·x|| = ||x|| for all x.

---

## 3. The Composition Layer (Schur's Lemma)

### 3.1 Problem: SO(3)-Equivariant Binary Fusion

Given: x_L, x_R ∈ R^{3k}
Goal: h ∈ R^{3k} such that h(R·x_L, R·x_R) = R·h(x_L, x_R) for all R ∈ SO(3)

### 3.2 Schur's Lemma Answer

For R^3 with standard SO(3) action, the space of equivariant bilinear maps 
R^3 × R^3 → R^3 is spanned by:

1. **Projection:** α·x_L + β·x_R
2. **Cross product:** γ·(x_L × x_R)

**That's it.** No other bilinear equivariant operations exist in 3D.

### 3.3 The Fusion Formula (Per Block)

For each block b ∈ {1, ..., k}:

**Step 1: Rotate children**

    x̃_L^{(b)} = R_L^{(b)} · x_L^{(b)}
    x̃_R^{(b)} = R_R^{(b)} · x_R^{(b)}

**Step 2: Fuse (Schur's lemma)**

    h^{(b)} = α_r^{(b)} · x̃_L^{(b)} + β_r^{(b)} · x̃_R^{(b)} + γ_r^{(b)} · (x̃_L^{(b)} × x̃_R^{(b)})

**Step 3: Gated residual**

    s_r = sigmoid(MLP_gate(rule_embedding))  [scalar per rule]

    x_parent^{(b)} = s_r · h^{(b)} + (1 - s_r) · (x̃_L^{(b)} + x̃_R^{(b)})

**Step 4: Output rotation**

    x_parent^{(b)} = R_out^{(b)} · x_parent^{(b)}

**Step 5: Concatenate blocks**

    x_parent = [x_parent^{(1)}; ...; x_parent^{(k)}] ∈ R^d

### 3.4 The Per-Rule MLP (Honest Architecture)

Pure equivariant fusion is linear. For nonlinear boolean operations (AND, OR, NOT), 
add a small per-rule MLP after the equivariant composition:

    x_parent = MLP_r(x_parent^{equivariant})

**This is the honest architecture:**
- SO(3) structure provides **inductive bias for depth stability**
- MLP provides **expressivity for nonlinear operations**
- Both are needed

---

## 4. The Parser Architectures

### 4.1 Gold Tree (Oracle)

Tree is given. OPERA composes bottom-up. Upper bound.

### 4.2 Fixed Heuristic Trees

**Right-branching:** Always split at leftmost boundary.
- Mimics head-final structure
- Achieves 97% of gold tree performance
- No learning required

**Left-branching:** Always split at rightmost boundary.
- Mimics head-initial structure  
- Achieves 75% at PP=8 (wrong for English, still beats LSTM)

**Finding:** Any tree beats no tree. Even wrong structure helps.

### 4.3 Learned Soft Bracketing (The Breakthrough)

For each span (i, j), learn a softmax over split points k:

**Score:**

    s_{i,j,k} = MLP([e_i; e_j; p_k])

where e_i, e_j are boundary embeddings and p_k is split position embedding.

**Attention:**

    α_{i,j,k} = exp(s_{i,j,k}) / Σ_{k'} exp(s_{i,j,k'})

**Composition:**

    h_{i,j} = Σ_k α_{i,j,k} · Ψ(h_{i,k}, h_{k+1,j})

**No categories. No grammar rules. Just: where do I split?**

**Result:** 100% agreement accuracy, matching gold trees, with 3,243 parameters.

### 4.4 Full Soft CKY (The Next Step)

Add category prediction:

    s_{i,j,k}^{A→BC} = score(A→BC) + MLP([h_{i,k,B}; h_{k+1,j,C}])

Marginalize over all rules, splits, and categories. The real language model.

**Status:** Not yet implemented. De-risked by learned bracketing result.

---

## 5. The Empirical Results

### 5.1 Boolean Formula Depth Extrapolation

| Model | Params | Depth 4 | Depth 8 | Depth 16 | Depth 32 |
|-------|--------|---------|---------|----------|----------|
| **OPERA (full)** | 6,577 | 100% | 100% | 100% | **100%** |
| OPERA-no-cross (γ=0) | 6,577 | 92% | 97% | 91% | 93% |
| Tree-MLP (no SO(3)) | 11,473 | 60% | 62% | 50% | 57% |
| Tree-LSTM | 6,889 | 69% | 68% | 56% | 50% |

**Key finding:** SO(3) constraint is the mechanism. Tree-MLP has more parameters (75% more) and still fails.

**Cross product:** Helps but not essential. Full OPERA 3/3 seeds at 100%. No-cross 2/3 seeds.

### 5.2 Boolean-in-AST with Variable Arity

| Task | Arity | Depth 16 | Depth 32 | Notes |
|------|-------|----------|----------|-------|
| Binary only (and, or, not) | 1-2 | 100% | 100% | Core competence |
| With if_bool (conditional) | 1-3 | 100% | 81% | Routing is learnable |
| With and3 (pure ternary) | 1-3 | 98% | 74% | Ternary fusion has ceiling |

**Finding:** Arity 3 is harder than arity 2. More drift surface. But still far above baselines.

### 5.3 Subject-Verb Agreement (The NLP Result)

**Task:** Predict verb number (singular/plural) from prefix. Distractor NPs with opposite number prepended 30% of time.

| Model | Params | PP=0 | PP=1 | PP=2 | PP=4 | PP=8 |
|-------|--------|------|------|------|------|------|
| **OPERA (gold tree)** | 12,578 | 100% | 100% | 100% | 100% | **100%** |
| **OPERA (learned bracket)** | **3,243** | **100%** | **100%** | **100%** | **100%** | **100%** |
| OPERA (right-branch fixed) | 2,210 | 100% | 100% | 98% | 96% | 97% |
| OPERA (left-branch fixed) | 2,210 | 100% | 100% | 98% | 88% | 75% |
| **LSTM (no tree)** | 11,930 | 100% | 100% | **53%** | **57%** | **54%** |

**PP = number of intervening prepositional phrases.**

**Key findings:**
1. **Any tree beats no tree.** Even left-branching (wrong structure) beats LSTM by +20% at PP=8.
2. **Learned parser matches oracle.** 3,243 params, no gold tree given, 100% accuracy.
3. **LSTM collapses to chance.** ~55% at PP=2+ where structural tracking is required.
4. **Learned > heuristic.** Learned bracket (100%) beats right-branch (97%).

---

## 6. The Mechanism: Why It Works

### 6.1 Norm Preservation (Depth Stability)

SO(3) rotations satisfy R^T R = I. Therefore:

    ||R · x|| = ||x||

The state vector cannot grow or shrink. It can only rotate. At depth 32, the state is on the same sphere as the leaf embeddings. No explosion. No vanishing.

### 6.2 Angular Drift (The Ceiling)

Norm is preserved, but **direction** can drift. At arity 3 with 6 fusion weights, the model has more degrees of freedom to rotate into a "wrong" subspace. This is why ternary drops to 74% at depth 32 while binary stays at 100%.

**The SO(3) constraint is necessary but not sufficient for arbitrary computation at arbitrary depth.** It prevents norm drift, not all forms of representational degradation.

### 6.3 The Cross Product (Natural Nonlinearity)

The cross product a × b is:
- **Bilinear:** No learned parameters
- **Perpendicular:** Introduces information orthogonal to both inputs
- **Anticommutative:** a × b = -(b × a)

This is the **only** natural nonlinearity in 3D that respects rotation. It enables:
- **Cancellation:** Negation (antipodal embeddings flip the cross product sign)
- **Interaction:** AND (parallel vectors → zero cross product; antipodal → nonzero)

### 6.4 The MLP (Honest Trade-off)

Pure equivariant fusion cannot represent all boolean functions. The per-rule MLP adds learned nonlinearity after the geometric constraint.

**Architecture principle:** Structure provides stability. Expressivity provides capacity. Both are needed.

---

## 7. The Implementation

### 7.1 Core Components

| Component | File | Purpose |
|-----------|------|---------|
| SO(3) Rotation Layer | `opera.py` | Rodrigues formula, block rotations |
| Equivariant Fusion | `opera.py` | Schur's lemma composition |
| Soft Bracketing Parser | `opera_soft_bracket.py` | Learned split-point attention |
| Tree Batcher | `train_opera.py` | JIT-optimized padding to FIXED_N |
| MiniLang Generator | `mini_lang_agreement.py` | Synthetic grammar with distractors |
| Training Loop | `train_opera.py` | PyTorch, batched by length |

### 7.2 Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 192 | 64 blocks × 3D |
| num_blocks | 64 | SO(3) block count |
| FIXED_N | 80 | Max nodes per tree (padded) |
| MAX_ARITY | 3 | Ternary if_bool |
| Learning rate | 1e-3 | With gradient clipping |
| Batch size | 16 | Grouped by length |
| Training steps | 1000 | Boolean; 800-1200 for agreement |

### 7.3 JIT Optimization

**Critical:** All trees padded to exactly FIXED_N before stacking. JAX/PyTorch sees identical shapes across all batches. Compiles once. ~1.5-3 ms/step.

**Before fix:** 1.2s/step (recompilation per batch). **After fix:** 400× speedup.

---

## 8. References

| Work | Connection |
|------|------------|
| Spivak (2016) "Operadic Approach to Compositionality" | Mathematical foundation |
| de Felice (2022) "Categorical Tools for NLP" | Closest theoretical framework; no geometric constraint |
| Maillard et al. (2017) "Latent Tree Learning" | Soft CKY parser; standard LSTM composition |
| CYKNN (2026) | Neural CYK; no geometric constraint |
| Socher et al. (2011), Tai et al. (2015) | Tree-LSTM/Tree-RNN; no norm preservation |
| Linzen et al. (2016) | Subject-verb agreement as LSTM test |
| Montague (1970) | Formal semantics as typed lambda calculus |
| Lambek (1958) | Categorial grammar |
| Bridson (1996) "Formal Language Theory and 3-Manifolds" | 3-manifold / formal language connection |
| Epstein et al. "Word Processing in Groups" | Automatic groups; geometric group theory |

---

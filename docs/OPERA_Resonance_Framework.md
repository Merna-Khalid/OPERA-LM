# OPERA Resonance: Mathematical Framework
## Phase-Locking Tree OPERA — Grounded Research Document

**Version:** 2026-06-30  
**Core Principle:** Language is a system of coupled SO(3) rotators. The parse tree is the ground state of phase-locking dynamics.

---

## 1. The Mathematical Foundation

### 1.1 SO(3) as a Riemannian Manifold

SO(3) is the Lie group of 3D rotations. It is a compact, connected 3-manifold with a natural bi-invariant Riemannian metric induced by the negative trace form on its Lie algebra so(3):

$$g_{I}(X, Y) = -\text{tr}(XY) \quad \text{for } X, Y \in \mathfrak{so}(3)$$

This metric makes SO(3) a complete metric space (Hopf-Rinow theorem applies). Geodesics are one-parameter subgroups: $\gamma(t) = \exp(t\Xi)$ for $\Xi \in \mathfrak{so}(3)$ with $\text{tr}\Xi^2 = -1$.

**Key property:** The geodesic distance between two rotations $R_A, R_B \in SO(3)$ is the rotation angle of their relative rotation:

$$d_{\text{geodesic}}(R_A, R_B) = \|\log(R_A^T R_B)^\vee\| = \arccos\left(\frac{\text{tr}(R_A^T R_B) - 1}{2}\right)$$

This is the **angle metric**, the most natural metric on SO(3). It is identical to the geodesic metric induced by the Riemannian structure.

**Sources:** Hartley et al. (2013) "Rotation Averaging" (IJCV); Álvarez-Tuñón et al. (2023) arXiv:2401.05396; Berkeley Rotation Group notes on geodesics of SO(3).

### 1.2 The Chordal Distance (Practical Alternative)

The chordal distance embeds SO(3) in $\mathbb{R}^{3\times 3} \cong \mathbb{R}^9$ and uses the Frobenius norm:

$$d_{\text{chord}}(R_A, R_B) = \|R_A - R_B\|_F = 2\sqrt{2}\sin(\theta/2)$$

where $\theta = d_{\text{geodesic}}(R_A, R_B)$.

**Trade-off:** The chordal distance is not Riemannian but is more numerically stable (no $\arccos$ near degeneracy). For small angles, $d_{\text{chord}} \approx \sqrt{2} \cdot \theta$, so it approximates the geodesic metric.

**Recommendation for OPERA:** Use the **squared chordal distance** for training stability:

$$d_{\text{chord}}^2(R_A, R_B) = 8\sin^2(\theta/2) = 4(1 - \cos(\theta))$$

This is smooth, bounded in $[0, 8]$, and avoids the logarithmic map's numerical instability.

**Sources:** Hartley et al. (2013) "Rotation Averaging" Table 2; Geist et al. (2024) "Learning with 3D Rotations: A Hitchhiker's Guide to SO(3)" (ICML); Álvarez-Tuñón et al. (2023).

### 1.3 The 6D Rotation Representation

Zhou et al. (2019) proved that mapping two 3D vectors $v_1, v_2 \in \mathbb{R}^3$ to a rotation via Gram-Schmidt orthonormalization gives a continuous, single-basin parameterization of SO(3) with full coverage (including 180°).

**Algorithm:**
1. $a_1 = v_1 / \|v_1\|$
2. $b = v_2 - (a_1 \cdot v_2) a_1$
3. $a_2 = b / \|b\|$
4. $a_3 = a_1 \times a_2$ (ensures right-handedness)
5. $R = [a_1, a_2, a_3]$ (columns)

**Why this matters for OPERA:** The axis-angle (Rodrigues) parameterization has wraparound minima at $2\pi$ — the optimizer scatters across equivalent representations. The 6D representation avoids this entirely. Your experiment record confirmed: 6D solved S4 on 4/4 seeds, axis-angle on 0/4.

**Source:** Zhou et al. (2019) "On the Continuity of Rotation Representations in Neural Networks" (CVPR); your own experiment record Section 3.4.

---

## 2. Phase-Locking on SO(3)

### 2.1 From Kuramoto on the Circle to Kuramoto on Lie Groups

The classical Kuramoto model couples oscillators on the circle $S^1$ (SO(2)):

$$\dot{\theta}_i = \omega_i + \frac{K}{N} \sum_j \sin(\theta_j - \theta_i)$$

The order parameter $r = \left|\frac{1}{N}\sum_j e^{i\theta_j}\right|$ measures synchronization. When $r \to 1$, all oscillators are phase-locked.

**Extension to SO(3):** The Kuramoto model has been generalized to matrix Lie groups including SO(3). The dynamics are written as a control system on the Lie group with closed-form discrete-time solutions. For SO(3), the "phase difference" between two rotations is an element of the Lie algebra so(3) — a skew-symmetric matrix, which is exactly the cross product in vector form.

**Source:** University of Waterloo thesis (2025) on Kuramoto oscillators on Lie groups; "Local Synchronization on Matrix Lie Groups."

### 2.2 The Locking Strength as Cross-Product Magnitude

For two token states $h_i, h_j \in \mathbb{R}^{3k}$ (k blocks of 3D), the natural coupling on each block is:

$$\text{lock}_{ij}^{(b)} = \|h_i^{(b)} \times h_j^{(b)}\|$$

This is maximal when the two vectors are orthogonal ($\pi/2$ phase difference) and vanishes when parallel or antiparallel. This is **not** the standard Kuramoto $\sin(\Delta\theta)$ coupling — it is the **SO(3)-natural coupling** derived from Schur's lemma.

**Why this is the right coupling:** Schur's lemma states that for $\mathbb{R}^3$ with the standard SO(3) action, the space of equivariant bilinear maps $\mathbb{R}^3 \times \mathbb{R}^3 \to \mathbb{R}^3$ is spanned by:
1. Projection: $\alpha h_i + \beta h_j$
2. Cross product: $\gamma (h_i \times h_j)$

The cross product is the **only** natural nonlinearity. It is bilinear (no learned parameters), perpendicular (introduces orthogonal information), and anticommutative ($a \times b = -(b \times a)$).

**Source:** Your OPERA Technical Reference Section 3.2; standard representation theory (Schur's lemma).

### 2.3 The Affinity Matrix from Locking

Define the pairwise affinity between tokens i and j:

$$A_{ij} = \exp\left(-\frac{\sum_b \text{lock}_{ij}^{(b)}}{\tau \cdot k}\right) = \exp\left(-\frac{\sum_b \|h_i^{(b)} \times h_j^{(b)}\|}{\tau \cdot k}\right)$$

where $\tau$ is a temperature parameter. This is:
- **High** when vectors are orthogonal (strong syntactic binding)
- **Low** when vectors are parallel (weak binding)
- **Symmetric:** $A_{ij} = A_{ji}$
- **Non-negative:** $A_{ij} \geq 0$

The graph Laplacian is:

$$L = D - A, \quad D_{ii} = \sum_j A_{ij}$$

**Source:** Standard spectral clustering (von Luxburg, 2007); ISCT (2025) for hierarchical spectral clustering.

---

## 3. The Tree as Spectral Decomposition

### 3.1 The Fiedler Vector and Binary Splitting

The Fiedler vector $v_2$ is the eigenvector of the normalized Laplacian $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$ corresponding to the second-smallest eigenvalue $\lambda_2$.

**Key property:** The sign of the Fiedler vector gives a bipartition of the graph that minimizes the normalized cut. For a sequence graph, this gives a natural binary split.

**Recursive application:** Split → compute Laplacian on each part → split again → ... This yields a binary tree. The depth is $O(\log n)$.

**Source:** Fiedler (1973); von Luxburg (2007) "A Tutorial on Spectral Clustering."

### 3.2 Differentiable Power Iteration

Full eigendecomposition is $O(n^3)$. For the Fiedler vector, we use power iteration:

```
Initialize: v_0 ~ random, ||v_0|| = 1
For k = 1 to K:
    v_k = (I - L_sym) @ v_{k-1}  # since we want 2nd eigenvector of L_sym
    v_k = v_k - (v_k · v_1) v_1  # orthogonalize against constant vector v_1
    v_k = v_k / ||v_k||
Return v_K
```

**Differentiability:** Each step is a matrix-vector multiplication and normalization — fully differentiable via autograd. K = 10-20 iterations suffice for convergence.

**Source:** Yang et al. (2022) "Neural Networks Based on Power Method and Inverse Power Method for Solving Linear Eigenvalue Problems" (arXiv:2209.11134); Mumladze (2021) TUM Master's thesis on neural eigensolvers.

### 3.3 Soft Binary Tree from Fiedler Sign

Instead of hard thresholding at 0, use a soft split:

$$\text{split}_i = \sigma(v_{2,i} / \epsilon)$$

where $\sigma$ is the sigmoid and $\epsilon$ is a temperature. As $\epsilon \to 0$, this becomes a hard split. During training, $\epsilon$ is annealed from warm (soft) to cold (hard).

**Tree construction:**
1. Compute Fiedler vector $v_2$ of full sequence
2. Soft-split into left/right halves
3. Recursively compute Fiedler vector on each half
4. Continue until singletons

This gives a **continuous relaxation of a binary tree** where each node is a weighted combination of its children, with weights from the soft split.

---

## 4. The Heat Kernel Multi-Scale Tree

### 4.1 Diffusion on the Graph

The heat kernel on the graph is:

$$T(t) = \exp(-t L_{\text{sym}})$$

At different timescales $t$, this captures different levels of structure:
- **Small t:** Local neighborhoods (words → phrases)
- **Medium t:** Phrase clusters (phrases → clauses)
- **Large t:** Global structure (clauses → sentence)

**Tree levels = diffusion scales:** $t = 1, 2, 4, 8, ...$

At each scale, $T(t)_{ij}$ gives the "flow" from node i to node j. The tree parent of a node at scale t is the weighted combination of all nodes it diffuses to.

**Source:** Coifman & Lafon (2006) "Diffusion Maps"; spectral clustering literature.

### 4.2 OPERA Composition at Each Scale

For scale $t$ and node $i$:

$$h_i^{(t)} = \sum_j \frac{T(t)_{ij}}{\sum_k T(t)_{ik}} \cdot \Psi(h_i^{(t/2)}, h_j^{(t/2)})$$

where $\Psi$ is the OPERA composition function (6D rotation + Schur fusion).

This is a **coarsening operation:** fine-scale representations compose into coarse-scale representations via the diffusion weights.

---

## 5. The Full Resonance OPERA Architecture

### 5.1 Forward Pass

```
Input: token_ids [B, T]
Step 1: Embed -> h_0 [B, T, d]
Step 2: For each layer:
    a. Compute pairwise affinity A [B, T, T] from cross-product locking
    b. Build Laplacian L [B, T, T]
    c. Compute Fiedler vector v_2 [B, T] via differentiable power iteration
    d. Soft-split -> left/right weights [B, T] each
    e. Recursively split until tree is built
    f. Bottom-up OPERA composition with 6D rotations
    g. Multi-scale prefix predictions from each tree level
    h. Cross-block MLP + forget gate -> next layer input
Step 3: Output head -> logits [B, num_positions, vocab]
```

### 5.2 Complexity Analysis

| Component | Complexity | Notes |
|---|---|---|
| Affinity matrix | $O(T^2 \cdot d)$ | Pairwise cross-products |
| Laplacian | $O(T^2)$ | Sparse if thresholded |
| Power iteration (K steps) | $O(K \cdot T^2)$ | K=10-20, parallelizable |
| Tree construction | $O(T^2 \log T)$ | Recursive Fiedler on subproblems |
| OPERA composition | $O(T \cdot d)$ | Bottom-up tree |
| **Total per layer** | **$O(T^2 \log T)$** | For T=128, this is ~20K ops vs 2M for O(T^3) CKY |

**Comparison:**
- v3 soft CKY: $O(T^3)$ = 2M for T=128
- v4 Gumbel merge: $O(T \log T)$ but non-differentiable
- **Resonance OPERA: $O(T^2 \log T)$ = ~20K, fully differentiable**

### 5.3 Training Objectives

1. **Language modeling loss:** CE on next-token predictions at all prefix positions
2. **Tree regularization (optional):** Entropy of soft splits should be low (committed parses)
3. **Rotation smoothness:** $\sum_{ij} d_{\text{chord}}^2(R_i, R_j) \cdot A_{ij}$ — nearby tokens should have similar rotations

---

## 6. Risk Assessment and Mitigations

| Risk | Evidence | Mitigation |
|---|---|---|
| **Phase unwrapping** | Rodrigues failed in your experiments (0/4 seeds) | Use 6D Gram-Schmidt, not axis-angle |
| **Non-convex energy landscape** | Kuramoto has many local minima; synchronization only guaranteed if agents start close | Warm start: initialize rotations near identity; anneal temperature |
| **O(T^2) is still expensive** | For T=512, T^2=262K; for T=1024, T^2=1M | Use low-rank affinity: $A = UU^T$ with $U \in \mathbb{R}^{T \times k}$, $k \ll T$ |
| **Fiedler vector instability** | Eigenvectors are sensitive to perturbations when eigenvalues are close | Use shifted inverse power iteration for better separation |
| **"DC root" problem** | If everything locks at top level, tree collapses to star | Regularization: penalize all-to-one affinity; enforce locality bias $A_{ij} \propto \exp(-|i-j|)$ |

---

## 7. References

1. **Hartley, R., Trumpf, J., Dai, Y., & Li, H.** (2013). Rotation Averaging. *International Journal of Computer Vision*, 103(3), 267-305. — Geodesic and chordal distances on SO(3).

2. **Álvarez-Tuñón, O., et al.** (2023). Loss it right: Euclidean and Riemannian Metrics in ... arXiv:2401.05396. — Practical comparison of SO(3) metrics.

3. **Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.** (2019). On the Continuity of Rotation Representations in Neural Networks. *CVPR*. — 6D representation, full coverage.

4. **Geist, M., et al.** (2024). Learning with 3D Rotations: A Hitchhiker's Guide to SO(3). *ICML*. — Comprehensive guide to rotation representations and losses.

5. **Yang, Q., et al.** (2022). Neural Networks Based on Power Method and Inverse Power Method for Solving Linear Eigenvalue Problems. arXiv:2209.11134. — Differentiable power iteration.

6. **Mumladze, N.** (2021). Solving Eigenproblems with Neural Networks. *TUM Master's Thesis*. — Neural eigensolvers, SpectralNet, heat diffusion.

7. **von Luxburg, U.** (2007). A Tutorial on Spectral Clustering. *Statistics and Computing*, 17(4), 395-416. — Fiedler vector, normalized cuts.

8. **Coifman, R. R., & Lafon, S.** (2006). Diffusion Maps. *Applied and Computational Harmonic Analysis*, 21(1), 5-30. — Heat kernel, multi-scale structure.

9. **Berkeley Rotation Group.** Geodesics of the rotation group SO(3). rotations.berkeley.edu. — Geodesic dynamics, rigid body analogy.

10. **University of Waterloo.** (2025). Kuramoto Oscillators on Lie Groups. — SO(3) extension of Kuramoto model.

11. **Your Experiment Record** (2026). Rotation-Structured State Space Models. — 6D vs Cayley vs axis-angle bake-off; SSM length extrapolation.

12. **Your OPERA Technical Reference** (2026). Operadic Planar Equivariant Recursive Algebra. — Schur's lemma, SO(3) equivariant fusion, cross product as natural nonlinearity.


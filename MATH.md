# Multivariable Calculus in the Gesture Classifier

This document reads the hand gesture recognition system as a concrete instance of multivariable calculus. The classifier is a differentiable function; its training is gradient descent on a scalar loss; backpropagation is the multivariable chain rule applied to a composition of layers; and the weight matrix at each layer is, literally, a Jacobian. Every claim here is anchored to a specific file in this repository or an artefact under `runs/`.

---

## 1. The Input Space `R^279`

Each training sample is a single point in `R^279`. The 279 coordinates are deterministic functions of MediaPipe hand landmarks, not raw pixels, and have physical meaning:

```
x  =  [ right_hand_features (138)   |   left_hand_features (138)   |   right_present   |   left_present   |   inter_hand_distance ]
```

Per hand (138 coordinates, from `src/preprocessor.py`):

| Sub-block | Count | What it encodes |
|---|---|---|
| Normalized landmarks `(x, y, z)` | 63 | 21 joints in 3D after translating to the wrist and scaling by palm size |
| Bone vectors | 60 | 20 parent→child joint deltas (3 components each) |
| Finger extension ratios | 5 | straight-line MCP→tip distance divided by chain-sum-of-bones; in `[0, 1]` |
| Pairwise fingertip distances | 10 | `C(5, 2)` Euclidean distances between fingertips |

Two-hand assembly (`src/feature_assembler.py`) concatenates the right- and left-hand 138-vectors, then appends a right-hand presence flag, a left-hand presence flag, and one inter-hand wrist distance (zero when only one hand is present). The total is `138 + 138 + 1 + 1 + 1 = 279`.

The features are *not* mutually independent and do not form an orthonormal basis — they are deterministic non-linear functions of the original 21 landmarks. What matters is that the map `landmarks → x` is the same at training and inference time, so the model sees one consistent distribution.

The test set projected onto its first two principal components (PC1, PC2) is in `runs/evaluation/pca_input_space.png`. Distinct clusters in that scatter are visible class boundaries in `R^279` — each sample is one of 49,678 dots in that plot, and each dot is one point of the 279-dimensional space.

---

## 2. The Network as a Function Composition

The classifier is a single composed function

```
f  :  R^279  ->  R^28,           f  =  f_4  o  f_3  o  f_2  o  f_1
```

with intermediate dimensions

```
R^279  --f_1-->  R^256  --f_2-->  R^128  --f_3-->  R^64  --f_4-->  R^28
```

Each `f_l` (`l = 1, 2, 3`) is an affine map followed by BatchNorm and ReLU:

```
f_l(a)  =  ReLU( BN( W^l a + b^l ) )
```

The output layer is affine only:

```
f_4(a)  =  W^4 a + b^4         (logits; softmax is applied inside the cross-entropy loss)
```

This is exactly what `src/models/mlp.py:GestureMLP.forward` does. The matching constants in that file are:

```
INPUT_DIM = 279
HIDDEN_DIMS = (256, 128, 64)
DROPOUTS    = (0.3, 0.3, 0.2)
NUM_CLASSES = 28
```

Three caveats worth stating:

1. **ReLU is not differentiable at 0**, but the set `{ z : z = 0 }` has measure zero. PyTorch defines the subgradient at zero as `0`, which is what `verify_chain_rule_single_sample` uses.
2. **Dropout** is a stochastic layer. The chain-rule analysis below is for a single forward pass with the dropout mask frozen — which is what happens whenever `model.eval()` is set.
3. **BatchNorm** has two distinct modes. In `train()` it uses batch statistics; in `eval()` it uses the persisted running mean/variance. The chain-rule worked example below is run in `eval()` mode, so `γ_i` and the per-feature scale `1 / √(σ²_i + ε)` are constants for that forward pass.

---

## 3. The Loss Surface `L : R^P -> R`

Stack every weight, bias, and BatchNorm parameter into a single vector `θ ∈ R^P`. For the trained checkpoint at `runs/mlp_best.pt`,

```
P = (279·256 + 256) + 2·256                      # Linear 1 + BN 1
  + (256·128 + 128) + 2·128                      # Linear 2 + BN 2
  + (128· 64 +  64) + 2· 64                      # Linear 3 + BN 3
  + ( 64· 28 +  28)                              # Linear 4
  ≈ 1.07 · 10^5  parameters.
```

Cross-entropy applied to the softmax of the logits gives a scalar loss

```
L(θ)  =  - (1/N) Σ_n Σ_c y_{n,c}  log softmax( f_θ(x_n) )_c .
```

Training minimises `L` by gradient descent. The actual optimiser is Adam (`src/train.py`), which keeps exponential moving averages of the first and second moments of the gradient. Adam ≠ vanilla gradient descent — but the object being minimised is still `L`, and the gradient being computed at every step is still `∇_θ L(θ)` produced by backpropagation. The pure-gradient picture

```
θ_{t+1}  =  θ_t  -  α  ∇_θ L(θ_t)               (α = 1e-3 at start, per src/train.py)
```

captures what is geometrically happening even when Adam reshapes the step.

The convergence of `||∇_θ L||` toward zero is recorded epoch-by-epoch in `runs/training_log.csv` (column `grad_norm`) and plotted on a log y-axis in `runs/evaluation/grad_norm_vs_epoch.png`. On the current checkpoint the gradient norm starts near 1.65 at epoch 0 and reaches its smallest value (~0.40) near epoch 41 — the optimiser approaches a near-critical point on the loss surface.

---

## 4. Backpropagation Is the Multivariable Chain Rule

For a single sample `x`, the forward pass through layer `l` is

```
z^l   =  W^l a^{l-1}  +  b^l                     (pre-BN)
u^l   =  γ^l ⊙ ( z^l - μ^l ) / sqrt(σ²^l + ε)  +  β^l    (BN)
a^l   =  ReLU( u^l )                              (activation)
```

with `a^0 = x`. Cross-entropy loss `L` sits at the end of the chain.

**Backprop is the chain rule applied recursively.** The downstream "delta" at layer `l` is

```
δ^l   =  ∂L / ∂a^l ,
```

and the recursion is

```
δ^{l-1}  =  ( W^l )^T  δ_pre^l ,
δ_pre^l  =  δ^l  ⊙  1[u^l > 0]  ⊙  γ^l / sqrt(σ²^l + ε)
```

where `⊙` is elementwise. The first factor is the upstream gradient; the second is the ReLU gate; the third is BatchNorm's scale-by-`γ/√(var+ε)` factor at this position in the network.

For a single entry of the layer-1 weight matrix, the chain rule reads

```
∂L / ∂W^{(1)}_{i, j}     =    x_j     ·     γ^{(1)}_i / sqrt(σ²^{(1)}_i + ε)     ·     1[ u^{(1)}_i > 0 ]     ·     ∂L / ∂a^{(1)}_i .
                              ─────         ───────────────────────────────         ────────────────────         ────────────────────
                              input              BN scale                              ReLU gate                upstream gradient
```

This is the formula evaluated inside `src/mv_visualization.py:verify_chain_rule_single_sample`. The rest of this section walks the formula numerically.

### Worked example

We pick a sample whose chain has every factor non-trivially nonzero and whose ReLU gate is active, so the example is not vacuous. From `runs/evaluation/chain_rule_verification.csv`:

| Field | Value |
|---|---|
| `sample_index` | 13443 |
| `label_name` (true) | `count_1` (id 10) |
| `predicted_label_id` | 10 |
| `neuron_idx` (layer-1 output index `i`) | 29 |
| `input_idx` (layer-0 input index `j`) | 112 |
| `gate_active` | True |
| `x_j` | -0.2757134735584259 |
| `bn_scale` = `γ_i / √(σ²_i + ε)` | 0.13842201105956145 |
| `upstream_grad_a1_i` = `∂L / ∂a^{(1)}_i` | -0.026127735152840614 |

(Layer-1 neuron 0 is *dead* on this checkpoint — its pre-activation is below zero for every test sample, so its ReLU gate is always 0 and `∂L / ∂W^{(1)}_{0, j}` would be identically zero. We use a neuron that actually fires; see `src/mv_visualization.py:select_active_first_layer_neuron`.)

Plug the four numbers into the chain-rule formula:

```
∂L / ∂W^{(1)}_{29, 112}
   =   x_{112}              ·   γ^{(1)}_{29}/√(σ²+ε)     ·   1[u^{(1)}_{29} > 0]   ·   ∂L / ∂a^{(1)}_{29}
   =   (-0.2757134735584259) ·  0.13842201105956145      ·   1                     ·   (-0.026127735152840614)
   =   9.97160138924328 · 10^{-4} .
```

PyTorch's autograd returns

```
model.linears[0].weight.grad[29, 112]  =  9.97160212136805 · 10^{-4} .
```

The absolute error is

```
| manual − autograd |  =  7.32 · 10^{-11} ,
```

i.e. they agree to ~10 significant digits — the gap is at the float32/float64 conversion floor, not a derivation error. Across the 12 selected test samples in `chain_rule_verification.csv`, **12 / 12 pass** at tolerance `1e-5`, with a maximum absolute error of ≈ `7.3e-11`.

The takeaway is that the chain rule is the operational definition of what `loss.backward()` computes. Autograd is not a separate object that "approximates" the derivative; it *is* the chain rule executed mechanically over the same computation graph.

---

## 5. Jacobians at Each Layer

The pre-activation of layer `l` is `z^l = W^l a^{l-1} + b^l`. Differentiating with respect to `a^{l-1}`,

```
∂ z^l / ∂ a^{l-1}   =   W^l .
```

The weight matrix `W^l` *is* the Jacobian of the affine part of layer `l`. It is not a metaphor — the matrix entries that the optimiser updates each step are the Jacobian entries of `z^l` with respect to `a^{l-1}`.

The four Jacobian shapes in this network:

| Layer `l` | `W^l` shape | Maps |
|---|---|---|
| 1 | 256 × 279 | `R^279 → R^256` |
| 2 | 128 × 256 | `R^256 → R^128` |
| 3 | 64 × 128 | `R^128 → R^64` |
| 4 | 28 × 64 | `R^64 → R^28` |

(These come straight from `HIDDEN_DIMS = (256, 128, 64)` and `NUM_CLASSES = 28` in `src/models/mlp.py`.)

The full Jacobian of `f_l` (including BatchNorm and ReLU) is

```
∂ a^l / ∂ a^{l-1}   =   D^l  ·  W^l ,
```

where `D^l` is the `n_l × n_l` diagonal matrix whose `i`-th entry is the product of the BN scale `γ^l_i / √(σ²^l_i + ε)` and the ReLU gate `1[u^l_i > 0]`. So `W^l` alone is the Jacobian of one slice of the layer (the affine part), and `D^l · W^l` is the Jacobian of the whole layer. The recursive delta formula in §4 follows directly from chaining these per-layer Jacobians.

---

## 6. Loss Surface Geometry

Geometric facts of the loss surface visualised in `runs/evaluation/loss_surface_slice.png` and `runs/evaluation/grad_norm_vs_epoch.png`:

- **Level sets.** For a constant `c`, the set `{ θ : L(θ) = c }` is a level surface in `R^P`. Two points on the same level set have the same training loss.
- **Gradient perpendicularity.** `∇L(θ)` is everywhere perpendicular to the level set of `L` through `θ`. This is the standard MV-calculus result that the gradient points in the direction of steepest increase, and the orthogonal complement is the tangent space to the level surface.
- **Steepest descent.** `−∇L(θ)` is the direction of steepest decrease, which is what gradient descent and (geometrically) Adam follow.
- **Critical points.** At a local minimum, `∇L(θ) = 0`. Training does not in general reach a true zero but it does drive `||∇L||` down. On this checkpoint `||∇L||` falls from ~1.65 at epoch 0 to ~0.40 near epoch 41 (visible on `grad_norm_vs_epoch.png` and recorded in `runs/training_log.csv:grad_norm`).

The loss-surface slice in `loss_surface_slice.png` is a 2D slice of an `R^P`-dimensional surface. It fixes every parameter at the trained value except two scalar entries of `linears[0].weight` (chosen by `src/mv_visualization.py:select_active_first_layer_neuron` to avoid the dead-neuron caveat from §4) and sweeps those two over a 50 × 50 grid. The contour lines are level sets restricted to that slice. The local geometry around the trained minimum — concentric closed contours, gradient arrows pointing toward the centre — is exactly what the textbook level-set picture predicts. The true surface lives in `R^P` with `P ≈ 10^5`; this is one readable slice, not the whole surface.

---

## Summary

The classifier is a single function `f : R^279 → R^28`. Training minimises a scalar function `L : R^P → R`. The gradient of `L` with respect to each parameter is obtained by the multivariable chain rule applied along the computation graph; we have verified by direct numerical comparison that the manual chain-rule formula matches PyTorch's `loss.backward()` to ~10 significant digits on real data. The weight matrices are the Jacobians of the affine maps. The training trajectory is a discrete descent on a level-set-shaped loss surface in `R^P`. Multivariable calculus is not a metaphor for what this network does; it is the algorithm.

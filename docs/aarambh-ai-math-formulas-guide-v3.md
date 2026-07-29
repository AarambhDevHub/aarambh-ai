# Aarambh-AI v3: The Complete Math Formula Guide

### Every new formula from Phases 29-40, explained from scratch

This document covers the new math introduced in v3.0.0 — building on top of the 14 formulas from the v2 guide. If you haven't read that one yet, start there: the v3 formulas assume you're comfortable with dot products, softmax, cross-entropy, attention, and LoRA.

Every formula below follows the same pattern as the v2 guide:
- **What it's called**
- **How to read the symbols**
- **Why we use it**
- **The formula itself**
- **2 fully solved examples with real numbers**

---

## How V3's Formulas Fit Together

V3 adds three major mathematical innovations on top of V2:

| Formula | What it does | Used in Phase |
|---------|-------------|---------------|
| 15. Gated DeltaNet Recurrence | Compress past context into a linear-time recurrent state | 29 |
| 16. Top-k Block Indexer + Sparse Attention | Pick only the most relevant blocks to attend to | 30 |
| 17. Fine-Grained MoE Router + Shared Expert | Route tokens to many small experts plus a shared baseline | 31 |
| 18. Multi-Token Prediction Loss | Train multiple auxiliary future-token heads | 32 |
| 19. On-Policy Distillation KL | Student learns from teacher's distribution on its own outputs | 33 |
| 20. Fake Quantization with STE | Simulate low-precision rounding during training | 34 |
| 21. Forgetting Delta | Measure how much capability changes across checkpoints | 38 |

---

## Formula 15: Gated DeltaNet Recurrence (Linear Attention)

**Definition:** Instead of attending to all past tokens (O(n²)), Gated DeltaNet maintains a single recurrent state vector that compresses the entire past context, updating it with a gated delta rule at each step (O(n)).

**How to read it:**
```
s_t = (1 - α_t) ⊙ s_{t-1} + α_t ⊙ k_t ⊙ v_t
o_t = q_t^T · s_t
```
- `s_t` = the recurrent state at step t (a vector summarizing the past)
- `α_t` = a learned gate between 0 and 1, controlling how much new info to blend in
- `k_t` = a "key" vector derived from the current token
- `v_t` = a "value" vector derived from the current token
- `q_t` = a "query" vector that reads from the state
- `⊙` = element-wise multiplication (multiply matching positions)
- `o_t` = the output at step t

**Why we use it:** Standard attention costs O(n²) — doubling sequence length quadruples compute. The gated delta recurrence is O(n) — cost grows linearly with length. This is what makes long context (100K+ tokens) practical.

**Formula:**
```
s_t = (1 - α_t) ⊙ s_{t-1} + α_t ⊙ k_t ⊙ v_t
o_t = q_t^T · s_t
```

Where the gate is computed as: `α_t = σ(W_α · x_t)` (sigmoid of a learned projection of the input), so it's always between 0 and 1.

**Example 1 (2-dimensional vectors, step 1 with no prior state):**
```
Initial state: s_0 = [0, 0]  (nothing seen yet)

At step 1, input token produces:
  k_1 = [1.0, 0.5]
  v_1 = [2.0, 1.0]
  α_1 = [0.8, 0.6]  (gate — blend in 80% of first dim, 60% of second)

Step 1: update state
  s_1 = (1 - α_1) ⊙ s_0 + α_1 ⊙ k_1 ⊙ v_1
  First dimension:
    (1 - 0.8) × 0 + 0.8 × (1.0 × 2.0) = 0 + 0.8 × 2.0 = 1.6
  Second dimension:
    (1 - 0.6) × 0 + 0.6 × (0.5 × 1.0) = 0 + 0.6 × 0.5 = 0.3

  s_1 = [1.6, 0.3]

Now a query q_1 = [0.5, 2.0] reads from the state:
  o_1 = q_1^T · s_1 = (0.5 × 1.6) + (2.0 × 0.3) = 0.8 + 0.6 = 1.4
```

**Example 2 (continuing to step 2, showing the blending behavior):**
```
Now at step 2, new input:
  k_2 = [0.2, 1.0]
  v_2 = [3.0, 0.5]
  α_2 = [0.3, 0.9]  (gate — only blend 30% of first dim, but 90% of second)

Current state: s_1 = [1.6, 0.3]

Step 2: update state
  s_2 = (1 - α_2) ⊙ s_1 + α_2 ⊙ k_2 ⊙ v_2
  First dimension:
    (1 - 0.3) × 1.6 + 0.3 × (0.2 × 3.0) = 0.7 × 1.6 + 0.3 × 0.6
    = 1.12 + 0.18 = 1.30
  Second dimension:
    (1 - 0.9) × 0.3 + 0.9 × (1.0 × 0.5) = 0.1 × 0.3 + 0.9 × 0.5
    = 0.03 + 0.45 = 0.48

  s_2 = [1.30, 0.48]

Key observation: the old value 1.6 in the first dimension mostly persisted
(only 30% replaced by new info), while the old value 0.3 in the second
dimension was mostly replaced (90% overwritten by new info). The gate
controls this blending per dimension.

Query q_2 = [1.0, 1.0]:
  o_2 = (1.0 × 1.30) + (1.0 × 0.48) = 1.30 + 0.48 = 1.78
```

**Beginner question:** *How does this compare to regular attention?* Regular attention lets every token look at every other token directly — powerful but expensive. The delta recurrence compresses history into a single state vector, trading off some "perfect recall" for dramatically cheaper computation. The gating mechanism lets the model decide what to keep and what to overwrite.

---

## Formula 16: Top-k Block Indexer + Sparse Attention

**Definition:** For the remaining full-attention layers, a learned indexer computes a score for each block of past tokens, selects only the top-k blocks, and attention is computed only over those selected tokens — making the full-attention layers themselves much cheaper.

**How to read it:**
```
score(b) = mean-pool(Q) · W_index · mean-pool(K_block_b)
selected = top_k({score(1), ..., score(B)}, k)
output = Attention(Q, K_selected, V_selected)
```

**Why we use it:** Even with only a few full-attention layers, O(n²) at very long sequences is expensive. Block-sparse attention makes each layer O(n × k × block_size) instead of O(n²), where k ≪ n.

**Formula:**
```
For each block b:
  score_b = mean(Q_seq) · W_index · mean(K_block_b)

block_indices = argsort([score_1, ..., score_B])[:k]

output = softmax(Q · K_selected^T / √d) · V_selected
```

**Example 1 (4 blocks, select top 1):**
```
Sequence has 8 positions divided into 4 blocks of 2 tokens each:
  Block A: tokens [0, 1]  → mean(K) = [1.0, 0.0]
  Block B: tokens [2, 3]  → mean(K) = [0.0, 1.0]
  Block C: tokens [4, 5]  → mean(K) = [0.5, 0.5]
  Block D: tokens [6, 7]  → mean(K) = [1.0, 1.0]

Query at position 7: mean(Q) = [0.8, 0.2]
Indexer weight W_index = identity matrix (simplified)

Scores:
  Block A: 0.8×1.0 + 0.2×0.0 = 0.8
  Block B: 0.8×0.0 + 0.2×1.0 = 0.2
  Block C: 0.8×0.5 + 0.2×0.5 = 0.5
  Block D: 0.8×1.0 + 0.2×1.0 = 1.0

Top 1: Block D (score=1.0)
→ Attention is computed only over block D's 2 tokens
→ 4× cheaper than attending to all 8 tokens
```

**Example 2 (same setup, select top 2):**
```
Same scores as above: A=0.8, B=0.2, C=0.5, D=1.0

Top 2: Block D (1.0) and Block A (0.8)
→ Attention computed over blocks A and D = 4 tokens total
→ 2× cheaper than attending to all 8 tokens, but with more
  context than only picking 1 block
```

**Beginner question:** *How does the indexer learn good scores?* The indexer weights `W_index` are trained end-to-end — the model learns which blocks are useful for each query position, just like the attention mechanism itself learns which tokens are relevant.

---

## Formula 17: Fine-Grained MoE Router + Shared Expert

**Definition:** The router now selects from many small experts (e.g., 64) plus one or more always-active shared experts, providing finer-grained specialization and a guaranteed baseline of processing.

**How to read it:**
```
g = softmax(TopK(x · W_router, k))      (router weights for selected experts)
output_shared = SharedExpert(x)          (always active)
output_experts = Σ g_i × Expert_i(x)    (gated expert outputs)
output = output_shared + output_experts  (combined)
```

**Why we use it:** More, smaller experts provide richer combinations of specialized knowledge per token. The shared expert ensures every token gets basic processing even if the router's specialized picks vary widely across tokens.

**Formula:**
```
g = softmax(TopK(x · W_router, k))
output = SharedExpert(x) + Σ_i g_i × Expert_i(x)
```

**Example 1 (4 experts, select top 2, 1 shared expert):**
```
Input token embedding: x = [0.5, 1.0, -0.2]

Router weights W_router is a 3×4 matrix (3 input dims, 4 experts):
  Assume router logits after multiplication: [2.0, 0.5, 1.0, 0.1]

Top 2 experts: Expert 0 (2.0) and Expert 2 (1.0)

Softmax over the top 2 logits:
  e^2.0 = 7.389
  e^1.0 = 2.718
  sum = 10.107
  g_0 = 7.389/10.107 = 0.731
  g_2 = 2.718/10.107 = 0.269

Expert outputs (assuming each expert is a tiny network):
  Expert_0(x) = [1.0, 0.5]
  Expert_2(x) = [0.2, 0.8]

Gated expert output:
  0.731 × [1.0, 0.5] + 0.269 × [0.2, 0.8]
  = [0.731, 0.366] + [0.054, 0.215]
  = [0.785, 0.581]

Shared expert output (always active):
  SharedExpert(x) = [0.3, 0.4]

Final output = [0.785, 0.581] + [0.3, 0.4] = [1.085, 0.981]
```

**Example 2 (showing how shared expert provides a baseline):**
```
Same router, different input x = [1.0, 0.0, 0.5]

Router logits: [0.3, 1.5, 0.8, 2.0]
Top 2: Expert 3 (2.0) and Expert 1 (1.5)

Softmax:
  e^2.0 = 7.389
  e^1.5 = 4.482
  sum = 11.871
  g_3 = 0.622, g_1 = 0.378

Experts output:
  Expert_3(x) = [0.5, 0.1]
  Expert_1(x) = [0.3, 0.9]

Gated: 0.622×[0.5,0.1] + 0.378×[0.3,0.9] = [0.311,0.062] + [0.113,0.340] = [0.424, 0.402]

Shared expert: [0.3, 0.4] (same as Example 1 — it's always active)

Final: [0.424, 0.402] + [0.3, 0.4] = [0.724, 0.802]

→ Even though different experts were activated, the shared expert
  provides consistent baseline processing to every token.
```

**Beginner question:** *Why not just make the shared expert part of the router selection?* The shared expert is deliberately outside the router because it guarantees that basic, universally useful computation happens for every token — the router can't accidentally "forget" to include it.

---

## Formula 18: Multi-Token Prediction Loss

**Definition:** At each training position, the model predicts not just the next token but also 2-4 future tokens using lightweight auxiliary heads, and the total loss is a weighted sum of all prediction losses.

**How to read it:**
```
L_mtp = Σ_i w_i × CrossEntropy(Head_i(h_t), token_{t+i+1})
L_total = L_main + L_mtp
```
- `h_t` = the base model's hidden state at position t
- `Head_i` = a small neural network that predicts the token i+1 positions ahead
- `w_i` = a weight for how much each future-token prediction should matter

**Why we use it:** Predicting multiple future tokens gives the model more learning signal per training step without extra passes through the expensive base layers. The MTP heads also serve as a free draft model for speculative decoding.

**Formula:**
```
For each future offset i = 1, 2, ..., n:
  L_i = CrossEntropy(Head_i(h_t), target_{t+i})

L_mtp = w_1 × L_1 + w_2 × L_2 + ... + w_n × L_n
L_total = L_main + L_mtp
```

**Example 1 (predicting 2 future tokens, equal weights):**
```
Training example: "The cat sat on the mat"
Position t: "on"  (the model's hidden state is at "on")

Ground truth targets:
  next token (t+1): "the"
  2 tokens ahead (t+2): "mat"

Main loss (standard next-token prediction):
  predict "the" from h_t → loss_main = 0.5

MTP Head 1 (predicts 1 ahead — same as main, different head):
  Head_1(h_t) outputs logits → predicts "the"
  L_1 = CrossEntropy = 0.6

MTP Head 2 (predicts 2 ahead):
  Head_2(h_t) outputs logits → predicts "mat"
  L_2 = CrossEntropy = 1.2

MTP loss (w_1 = 0.3, w_2 = 0.3):
  L_mtp = 0.3 × 0.6 + 0.3 × 1.2 = 0.18 + 0.36 = 0.54

Total loss:
  L_total = 0.5 + 0.54 = 1.04
```

**Example 2 (showing prediction getting harder with distance):**
```
Same setup, different position.
Position t: "cat" → targets: "sat" (t+1), "on" (t+2)

Main loss: predict "sat" → loss_main = 0.3 (pretty confident)

Head 1: predict "sat" → L_1 = 0.4
Head 2: predict "on" → L_2 = 2.1 (harder to predict farther ahead)

MTP loss (w_1 = 0.3, w_2 = 0.3):
  L_mtp = 0.3 × 0.4 + 0.3 × 2.1 = 0.12 + 0.63 = 0.75

Total: 0.3 + 0.75 = 1.05

→ Notice: L_2 (2.1) is much larger than L_1 (0.4) because predicting
  "on" from "cat" is genuinely harder than predicting "sat".
  The farther ahead we predict, the less certain the model is.
```

**Beginner question:** *Does the model use MTP heads during inference?* Not by default — they're only for training. But they can optionally serve as a draft model for speculative decoding, since each head has already learned to predict future tokens from the base model's hidden states.

---

## Formula 19: On-Policy Distillation KL

**Definition:** During distillation, the student model generates its own output sequence, then the teacher model scores it. The distillation loss minimizes the KL divergence between the student's and teacher's output distributions on the student's own generated tokens.

**How to read it:**
```
L_distill = KL( P_student(t) ‖ P_teacher(t) )   for each generated token t
```
Where both distributions are computed over the student's *own* generated sequence (on-policy), not a fixed dataset (off-policy).

**Why we use it:** On-policy distillation corrects the distribution mismatch problem: if the student trains on teacher-written answers, it never learns to improve its own generation style. Scoring the student's own outputs bridges this gap.

**Formula:**
```
L_distill = (1/N) × Σ_i KL( P_student(token_i) ‖ P_teacher(token_i) )

Where KL(P‖Q) = Σ_j P(j) × log( P(j) / Q(j) )
```

(Note: KL divergence is Formula 13 from the v2 guide — but now applied to on-policy student-generated tokens rather than static dataset examples.)

**Example 1 (student and teacher mostly agree):**
```
Student generates token "Paris" at position i.
Student's probability distribution over vocabulary:
  P_student("Paris") = 0.85,  P_student("London") = 0.10,  P_student("Berlin") = 0.05

Teacher's probability distribution at same position:
  P_teacher("Paris") = 0.90,  P_teacher("London") = 0.08,  P_teacher("Berlin") = 0.02

KL(P_student ‖ P_teacher):
  "Paris": 0.85 × log(0.85/0.90) = 0.85 × log(0.944) = 0.85 × (-0.058) = -0.049
  "London": 0.10 × log(0.10/0.08) = 0.10 × log(1.25) = 0.10 × 0.223 = 0.022
  "Berlin": 0.05 × log(0.05/0.02) = 0.05 × log(2.5) = 0.05 × 0.916 = 0.046

  KL = (-0.049) + 0.022 + 0.046 = 0.019
  → Small KL (close distributions), student is already matching teacher well
```

**Example 2 (student is less confident than teacher):**
```
Student generates token "Eiffel" at a different position.
Student distribution:
  P_student("Eiffel") = 0.40,  P_student("Tower") = 0.35,  P_student("Building") = 0.25

Teacher distribution:
  P_teacher("Eiffel") = 0.85,  P_teacher("Tower") = 0.10,  P_teacher("Building") = 0.05

KL(P_student ‖ P_teacher):
  "Eiffel":  0.40 × log(0.40/0.85) = 0.40 × log(0.471) = 0.40 × (-0.753) = -0.301
  "Tower":   0.35 × log(0.35/0.10) = 0.35 × log(3.50) = 0.35 × 1.253 = 0.439
  "Building": 0.25 × log(0.25/0.05) = 0.25 × log(5.00) = 0.25 × 1.609 = 0.402

  KL = (-0.301) + 0.439 + 0.402 = 0.540
  → Much larger KL. The student is much less confident about "Eiffel"
    than the teacher. The distillation loss will push the student's
    distribution closer to the teacher's.
```

**Beginner question:** *Why KL instead of simple cross-entropy against teacher labels?* KL divergence captures the full distributional difference — it tells the student not just "teacher thinks the correct token is X" but "teacher's entire confidence profile looks like Y — try to match it." This richer signal produces better student models.

---

## Formula 20: Fake Quantization with Straight-Through Estimator

**Definition:** During QAT training, weights are "fake quantized" — rounded as if they were INT4 or INT8 — but gradients pass through the rounding operation unchanged (straight-through estimator), so the model learns to produce weights that work well at low precision.

**How to read it:**
```
Forward:  q = round(x / scale)     x̂ = q × scale
Backward: ∂L/∂x ≈ ∂L/∂x̂            (gradient ignores the rounding)
```
The key idea: during the forward pass, we simulate the damage quantization will do. During the backward pass, we pretend the rounding never happened, so gradients flow normally and the model adapts.

**Why we use it:** Post-training quantization always loses some accuracy because the weights were optimized for high-precision math. QAT makes the weights robust to quantization *during* training, so when you actually quantize later, much less quality is lost.

**Formula:**
```
Forward pass (fake quantization of weight W):
  scale = (max(W) - min(W)) / (2^bits - 1)
  q = clamp(round(W / scale), 0, 2^bits - 1)
  W_q = q × scale
  y = W_q · x    (forward pass uses quantized weights)

Backward pass (STE):
  ∂L/∂W = ∂L/∂W_q    (gradient flows through as if W = W_q)
```

**Example 1 (3-bit quantization of a single weight, bits=3):**
```
Original weight: W = 0.732 (32-bit float)

Forward (fake quant to 3 bits, 2^3 = 8 levels):
  Range of all weights in this layer: min = 0.0, max = 1.0
  scale = (1.0 - 0.0) / (8 - 1) = 1.0 / 7 = 0.143

  q = clamp(round(0.732 / 0.143), 0, 7)
    = clamp(round(5.12), 0, 7)
    = clamp(5, 0, 7) = 5

  W_q = 5 × 0.143 = 0.715

  → Forward pass uses 0.715 instead of 0.732 (small rounding error)
  → The model trains with this slight error, learning to compensate

  Backward (STE):
  ∂L/∂W = ∂L/∂W_q    (the gradient skips the rounding, flowing
                       as if the forward used 0.715 directly)
```

**Example 2 (showing how the model learns to avoid quantization loss):**
```
Same setup, but after QAT training the weights have shifted:

Original weight AFTER QAT training: W = 0.715

Forward:
  q = clamp(round(0.715 / 0.143), 0, 7)
    = clamp(round(5.0), 0, 7) = 5

  W_q = 5 × 0.143 = 0.715

  → The weight landed exactly on a quantized level!
  → No rounding error at all — the model learned to produce
    weights that quantize cleanly without loss.

This is the goal of QAT: not just to tolerate quantization
error, but to learn weights that minimize it.
```

**Beginner question:** *What's a "straight-through estimator"?* Normally, rounding (which has zero gradient almost everywhere) would kill all learning — the model couldn't adapt because the gradient through round() is zero. STE replaces that zero gradient with the identity function (1), so the gradient from loss → quantized weight passes straight through to the original weight, allowing normal training.

---

## Formula 21: Forgetting Delta (Capability Regression Measurement)

**Definition:** The forgetting delta measures how much a model's capability in a specific area (math, code, reasoning, etc.) changes between two checkpoints, producing a signed score that indicates improvement or regression.

**How to read it:**
```
Δ(c, s₁, s₂) = score_c(s₂) - score_c(s₁)
```
- `c` = a capability area (e.g., "math", "code", "reasoning")
- `s₁` = reference checkpoint (usually the one before an update)
- `s₂` = current checkpoint (after an update)
- `Δ` = positive means improvement, negative means forgetting

**Why we use it:** Without measurement, you can't tell if a new training run is silently destroying old capabilities. Forgetting delta quantifies this, enabling data-driven decisions.

**Formula:**
```
For each capability c:
  Δ_c = score at current checkpoint - score at reference checkpoint

Forgetting curve: plot Δ_c over all checkpoints for each capability
```

**Example 1 (fine-tuning improved math but regressed code):**
```
Before fine-tuning: checkpoint_500
After fine-tuning:  checkpoint_1000

Math probe:
  score_500 = 78 / 100
  score_1000 = 92 / 100
  Δ_math = 92 - 78 = +14   ✓ improvement

Code probe:
  score_500 = 81 / 100
  score_1000 = 73 / 100
  Δ_code = 73 - 81 = -8    ⚠ regression (forgetting)
```

**Example 2 (tracking over multiple checkpoints):**
```
Checkpoint series: step_1000 → step_2000 → step_3000 → step_4000

Factual knowledge probe:
  step_1000: 85/100  (baseline)
  step_2000: 87/100  Δ = +2  (slight improvement)
  step_3000: 82/100  Δ = -3  (starting to forget)
  step_4000: 79/100  Δ = -6  (continued regression)

→ The forgetting curve shows a clear downward trend.
  Action: mix more factual data into the next training batch.
```

**Beginner question:** *What counts as a "capability probe"?* Each probe is a fixed set of test examples for a capability area — 50 math problems, 50 code tasks, etc. — that stays identical forever, so score changes are purely due to model changes, not test differences.

---

# Quick Reference: V3 Formulas in One Table

| # | Formula | One-line meaning | Used in Phase |
|---|---------|-------------------|---------------|
| 15 | Gated DeltaNet Recurrence | Compress past context into a linear-time recurrent state with gated blending | 29 |
| 16 | Top-k Block Indexer + Sparse Attention | Score token blocks, attend only to the best ones | 30 |
| 17 | Fine-Grained MoE Router + Shared Expert | Route tokens to many small experts + always-active shared expert | 31 |
| 18 | Multi-Token Prediction Loss | Train multiple auxiliary heads predicting future tokens | 32 |
| 19 | On-Policy Distillation KL | Student matches teacher's distribution on its own generated outputs | 33 |
| 20 | Fake Quantization with STE | Simulate low-precision rounding in forward pass, ignore it in backward | 34 |
| 21 | Forgetting Delta | Score difference between checkpoints per capability | 38 |

---

# V3 Formulas in Context

```
 V2 foundation (Formulas 1-14):
 Dot Product, Matrix Multiply, Softmax, Attention, LayerNorm, GELU,
 Cross-Entropy, Gradient Descent, Adam, RoPE, LoRA, Quantization,
 KL Divergence, Perplexity

 V3 additions (Formulas 15-21):
 Gated DeltaNet        → replaces most attention layers with linear recurrence
 Sparse Attention      → makes remaining full-attention layers block-sparse
 Fine-Grained MoE      → more, smaller experts + shared expert
 MTP Loss              → extra training signal from future-token heads
 Distillation KL       → on-policy student-teacher distribution matching
 Fake Quant + STE      → quantization-robust weight training
 Forgetting Delta      → measurable capability regression tracking
```

---

*This guide covers the new mathematical formulas introduced in Aarambh-AI v3.0.0 (Phases 29-40), building on the v2 formula collection. Every formula is explained from first principles with two fully worked examples.*

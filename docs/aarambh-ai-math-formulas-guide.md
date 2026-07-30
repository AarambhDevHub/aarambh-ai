# Aarambh Studio: The Complete Math Formula Guide

### Every formula we use, explained like you've never seen math notation before

This document is for someone coming from a **non-math background** — maybe you know how to code, but formulas with Greek letters and weird symbols look scary. That's fine. Every formula below is broken down piece-by-piece, like reading a sentence word-by-word, before we ever touch a real number.

For every formula, you'll get:
- **What it's called** (definition)
- **How to read the symbols** (a beginner "translation" of the notation)
- **Why we use it in Aarambh Studio** (which phase it belongs to and what job it does)
- **The formula itself**
- **2 fully solved examples**, step by step, with real numbers

---

## How to Read Any Formula (read this first!)

Before we start, here's a "decoder ring" for symbols that show up again and again. Keep coming back to this section whenever a symbol confuses you.

| Symbol | Say it as | Meaning |
|---|---|---|
| `Σ` (sigma) | "sum of" | Add up a bunch of things |
| `x` | "x" | Usually the input |
| `y` | "y" | Usually the output or target/correct answer |
| `ŷ` (y-hat) | "y hat" | The model's *predicted* value (vs the real `y`) |
| `W` | "W" (weight matrix) | A grid of numbers the model learns during training |
| `b` | "bias" | An extra learned number added on, like a starting offset |
| `θ` (theta) | "theta" | A general stand-in for "all the model's learnable numbers" |
| `∂` (partial d) | "partial derivative" | "How much does this change if I nudge that a tiny bit" |
| `exp(x)` or `eˣ` | "e to the x" | A specific way of making numbers grow fast (~2.718 raised to power x) |
| `log(x)` | "log of x" | The opposite of exp — "undoes" exponential growth, used to measure surprise/error |
| `‖x‖` | "norm of x" | Roughly: "how big is this whole vector, as one single size number" |
| `argmax` | "arg-max" | "Which option has the biggest value" |
| `·` (dot) | "dot product" | Multiply matching pairs of numbers, then add them all up |

You don't need to memorize this table — just refer back to it whenever a symbol below looks unfamiliar.

---

## 1. Dot Product (the atom of everything)

**Definition:** The dot product takes two lists of numbers (vectors) of the same length, multiplies each matching pair together, and adds up all the results into one single number.

**How to read it:**
```
a · b = a1×b1 + a2×b2 + a3×b3 + ... + an×bn
```
Say it as: "a dot b equals: multiply each matching pair, then add them all up."

**Why we use it:** This is the single most-used operation in the *entire* model. Matrix multiplication (Phase 3, 4), attention (Phase 4, 15), and embeddings are all just dot products repeated thousands of times. If you understand this one formula, you understand 80% of what a neural network is "doing" mathematically.

**Formula:**
```
a · b = Σ (ai × bi)   for i = 1 to n
```

**Example 1 (simple):**
```
a = [2, 3, 4]
b = [1, 0, 5]

Step 1: multiply matching pairs
  2×1 = 2
  3×0 = 0
  4×5 = 20

Step 2: add them up
  2 + 0 + 20 = 22

Answer: a · b = 22
```

**Example 2 (with negative numbers, like real model weights):**
```
a = [0.5, -1.2, 3.0]
b = [2.0, 0.5, -0.5]

Step 1: multiply matching pairs
  0.5 × 2.0  = 1.0
  -1.2 × 0.5 = -0.6
  3.0 × -0.5 = -1.5

Step 2: add them up
  1.0 + (-0.6) + (-1.5) = -1.1

Answer: a · b = -1.1
```

**Beginner question:** *Why does the model care about this number at all?* Because a dot product measures "how aligned" two vectors are — a big positive number means they point in a similar direction (similar meaning, in the case of word embeddings), a negative number means they point in opposite directions.

---

## 2. Matrix Multiplication

**Definition:** Matrix multiplication is what happens when you take an entire grid of numbers (a matrix) and combine it with another grid, by computing a dot product for every row-column pairing.

**How to read it:**
```
C = A × B
```
"C equals A times B" — where every single entry in C is really just one dot product between a row of A and a column of B.

**Why we use it:** Every layer of the neural network (Phase 3, 4) is fundamentally a matrix multiplication. Your input (a row of numbers representing a token) gets multiplied by a weight matrix (the model's learned knowledge) to produce the next layer's numbers. This happens *thousands* of times per single response.

**Formula:**
```
C[i][j] = Σ ( A[i][k] × B[k][j] )   for all k
```
Translation: "the value at row i, column j of the result = dot product of row i of A with column j of B."

**Example 1 (2x2 matrices):**
```
A = [1 2]      B = [5 6]
    [3 4]          [7 8]

C[0][0] = (1×5) + (2×7) = 5 + 14 = 19
C[0][1] = (1×6) + (2×8) = 6 + 16 = 22
C[1][0] = (3×5) + (4×7) = 15 + 28 = 43
C[1][1] = (3×6) + (4×8) = 18 + 32 = 50

Result C = [19 22]
           [43 50]
```

**Example 2 (a 1×3 "token vector" times a 3×2 "weight matrix" — this is literally what happens inside the model):**
```
Token vector A = [1, 2, 3]      (a 1x3 row)

Weight matrix B = [1 0]
                   [0 1]
                   [1 1]        (a 3x2 grid)

Result[0][0] = (1×1)+(2×0)+(3×1) = 1+0+3 = 4
Result[0][1] = (1×0)+(2×1)+(3×1) = 0+2+3 = 5

Result = [4, 5]   (a new 1x2 vector — the "next layer's" representation)
```

**Beginner question:** *Why grids instead of single numbers?* Because a single token isn't described by one number — it's described by hundreds or thousands of numbers (its embedding). Matrices let you transform *all* those numbers together, in one structured operation, instead of writing separate formulas for each one.

---

## 3. Softmax Function

**Definition:** Softmax takes a list of raw, arbitrary numbers (which can be positive, negative, huge, or tiny) and converts them into a list of probabilities that all add up to exactly 1 (100%).

**How to read it:**
```
softmax(xi) = exp(xi) / Σ exp(xj)
```
Say it as: "take e raised to the power of this number, then divide by the sum of e-raised-to-the-power of every number in the list."

**Why we use it:** After the forward pass (Phase 4), the model produces a raw score for every possible next token — these scores are meaningless on their own (could be -50, could be 1000). Softmax turns them into clean probabilities like "42% chance the next word is 'chess'" — this is literally how the model picks its next word.

**Formula:**
```
softmax(x_i) = e^(x_i)  /  ( e^(x_1) + e^(x_2) + ... + e^(x_n) )
```

**Example 1 (raw scores for 3 possible next words):**
```
Raw scores: "cat" = 2.0, "dog" = 1.0, "car" = 0.1

Step 1: exponentiate each
  e^2.0 = 7.389
  e^1.0 = 2.718
  e^0.1 = 1.105

Step 2: sum them up
  7.389 + 2.718 + 1.105 = 11.212

Step 3: divide each by the sum
  "cat" = 7.389 / 11.212 = 0.659  → 65.9%
  "dog" = 2.718 / 11.212 = 0.242  → 24.2%
  "car" = 1.105 / 11.212 = 0.099  → 9.9%

Check: 65.9% + 24.2% + 9.9% = 100% ✓
```

**Example 2 (with a negative number included):**
```
Raw scores: A = 3.0, B = -1.0, C = 0.5

Step 1: exponentiate each
  e^3.0  = 20.086
  e^-1.0 = 0.368
  e^0.5  = 1.649

Step 2: sum them up
  20.086 + 0.368 + 1.649 = 22.103

Step 3: divide each by the sum
  A = 20.086 / 22.103 = 0.909 → 90.9%
  B = 0.368 / 22.103  = 0.017 → 1.7%
  C = 1.649 / 22.103  = 0.075 → 7.5%

Check: 90.9 + 1.7 + 7.5 = 100.1% (rounding) ✓
```

**Beginner question:** *Why exponentiate instead of just dividing the raw numbers directly?* Because raw scores can be negative, and you can't have a "negative probability." Exponentiating (e^x) always produces a positive number, no matter how negative x was — which is exactly what you need before turning things into clean percentages.

---

## 4. Scaled Dot-Product Attention

**Definition:** This is THE core formula of a transformer. It lets each token figure out which other tokens in the sequence are most relevant to it, and blend information from them accordingly.

**How to read it:**
```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √dk ) × V
```
- `Q` (Query) = "what am I looking for?" (a vector for the current token)
- `K` (Key) = "what do I contain?" (a vector for every token, including itself)
- `V` (Value) = "what information do I actually offer?" (a vector for every token)
- `Kᵀ` = the K matrix "flipped" (transposed) so the dot product works dimensionally
- `√dk` = square root of the dimension size, used to keep numbers from growing too large
- `softmax(...)` = turn the resulting scores into clean attention-weight percentages (Formula 3 above!)

**Why we use it:** This is what lets the model understand relationships between words — like knowing "it" refers to "the dog" three words earlier. Every transformer block (Phase 4) contains this exact formula, and Flash Attention (Phase 15) is just a faster way of computing this same math.

**Formula:**
```
Attention(Q, K, V) = softmax( QKᵀ / √dk ) V
```

**Example 1 (tiny simplified version, 1 query vs 2 keys/values, dk = 2):**
```
Query Q = [1, 0]
Keys:  K1 = [1, 0],  K2 = [0, 1]
Values: V1 = [10, 0], V2 = [0, 20]

Step 1: dot product Q with each key
  Q·K1 = (1×1)+(0×0) = 1
  Q·K2 = (1×0)+(0×1) = 0

Step 2: scale by √dk (dk = 2, so √2 ≈ 1.414)
  1 / 1.414 = 0.707
  0 / 1.414 = 0

Step 3: softmax these two scaled scores
  e^0.707 = 2.028
  e^0     = 1.0
  sum = 3.028
  weight1 = 2.028/3.028 = 0.670 → 67%
  weight2 = 1.0/3.028   = 0.330 → 33%

Step 4: weighted sum of the Values
  Output = 0.670×V1 + 0.330×V2
         = 0.670×[10,0] + 0.330×[0,20]
         = [6.70, 0] + [0, 6.60]
         = [6.70, 6.60]

Final attention output: [6.70, 6.60]
```

**Example 2 (query more aligned with the second key):**
```
Query Q = [0, 2]
Keys:  K1 = [1, 0],  K2 = [0, 1]
Values: V1 = [5, 5],  V2 = [1, 9]

Step 1: dot product
  Q·K1 = (0×1)+(2×0) = 0
  Q·K2 = (0×0)+(2×1) = 2

Step 2: scale by √2 ≈ 1.414
  0 / 1.414 = 0
  2 / 1.414 = 1.414

Step 3: softmax
  e^0     = 1.0
  e^1.414 = 4.113
  sum = 5.113
  weight1 = 1.0/5.113   = 0.196 → 19.6%
  weight2 = 4.113/5.113 = 0.804 → 80.4%

Step 4: weighted sum of Values
  Output = 0.196×[5,5] + 0.804×[1,9]
         = [0.98, 0.98] + [0.804, 7.236]
         = [1.784, 8.216]

Final attention output: [1.78, 8.22]
```
Notice: because Q was more aligned with K2, the output leans much more heavily toward V2 — this IS the model "paying more attention" to the second token.

**Beginner question:** *Why divide by √dk at all?* Without scaling, dot products can get very large as vector size grows, which makes softmax produce extremely lopsided (almost all-or-nothing) results — dividing by √dk keeps the scores in a reasonable, well-behaved range so training stays stable.

---

## 5. Layer Normalization

**Definition:** Layer normalization rescales a set of numbers so they have a mean (average) of 0 and a standard deviation (spread) of 1, then applies a small learned adjustment.

**How to read it:**
```
LayerNorm(x) = γ × ( (x - μ) / √(σ² + ε) ) + β
```
- `μ` (mu) = the average of all the numbers in x
- `σ²` (sigma squared) = the variance (how spread out the numbers are)
- `ε` (epsilon) = a tiny number added just to avoid dividing by zero
- `γ` (gamma) and `β` (beta) = small learned "rescale" and "shift" numbers

**Why we use it:** As numbers flow through many stacked layers (Phase 3, 4), they can grow huge or shrink to near-zero, making training unstable or impossible. Layer normalization resets things back to a well-behaved range after every layer, so training stays stable even in very deep networks.

**Formula:**
```
mean (μ) = (x1 + x2 + ... + xn) / n
variance (σ²) = ( (x1-μ)² + (x2-μ)² + ... + (xn-μ)² ) / n
normalized xi = (xi - μ) / √(σ² + ε)
output = γ × normalized_xi + β
```

**Example 1 (simple 4 numbers, assume γ=1, β=0 for simplicity):**
```
x = [2, 4, 4, 8]

Step 1: mean
  μ = (2+4+4+8)/4 = 18/4 = 4.5

Step 2: variance
  (2-4.5)² = 6.25
  (4-4.5)² = 0.25
  (4-4.5)² = 0.25
  (8-4.5)² = 12.25
  sum = 19.0
  σ² = 19.0/4 = 4.75

Step 3: normalize each (ε ≈ 0, ignoring for simplicity)
  √4.75 ≈ 2.179
  x1_norm = (2-4.5)/2.179   = -1.147
  x2_norm = (4-4.5)/2.179   = -0.230
  x3_norm = (4-4.5)/2.179   = -0.230
  x4_norm = (8-4.5)/2.179   = 1.606

Output (γ=1, β=0): [-1.147, -0.230, -0.230, 1.606]
```
Notice: the original numbers ranged from 2 to 8; after normalization they're centered around 0 with a controlled spread.

**Example 2 (with γ=2, β=1, showing the learned rescale/shift):**
```
x = [10, 20, 30]

Step 1: mean
  μ = (10+20+30)/3 = 20

Step 2: variance
  (10-20)² = 100
  (20-20)² = 0
  (30-20)² = 100
  sum = 200
  σ² = 200/3 = 66.67

Step 3: normalize
  √66.67 ≈ 8.165
  x1_norm = (10-20)/8.165 = -1.225
  x2_norm = (20-20)/8.165 = 0
  x3_norm = (30-20)/8.165 = 1.225

Step 4: apply γ=2, β=1
  output1 = 2×(-1.225)+1 = -1.45
  output2 = 2×(0)+1      = 1.0
  output3 = 2×(1.225)+1  = 3.45

Final output: [-1.45, 1.0, 3.45]
```

**Beginner question:** *Why bother re-adding γ and β after normalizing everything to a clean 0-1 range?* Because sometimes forcing everything to a strict standard range removes useful information — γ and β are learned during training, letting the model "undo" some of the normalization if that turns out to help.

---

## 6. GELU Activation Function

**Definition:** GELU (Gaussian Error Linear Unit) is a smooth, curved activation function that decides how much of a number "passes through" a layer, based on the number's own value.

**How to read it:**
```
GELU(x) ≈ 0.5 × x × ( 1 + tanh( √(2/π) × (x + 0.044715×x³) ) )
```
This looks intimidating, but conceptually: for very negative x, output ≈ 0 (blocked). For very positive x, output ≈ x (passed through almost unchanged). Near zero, it's a smooth, gentle curve rather than a sharp on/off switch.

**Why we use it:** Without activation functions (Phase 3), stacking many layers would mathematically collapse into being equivalent to just one layer — activations are what let the network learn complex, non-straight-line patterns. GELU specifically tends to train more smoothly than simpler alternatives like ReLU.

**Formula (simplified approximation used in practice):**
```
GELU(x) = 0.5 × x × (1 + tanh(0.7979 × (x + 0.044715 × x³)))
```

**Example 1 (x = 2, a clearly positive number):**
```
x = 2

Step 1: x³ = 8
Step 2: 0.044715 × 8 = 0.3577
Step 3: x + 0.3577 = 2 + 0.3577 = 2.3577
Step 4: 0.7979 × 2.3577 = 1.881
Step 5: tanh(1.881) ≈ 0.954
Step 6: 1 + 0.954 = 1.954
Step 7: 0.5 × 2 × 1.954 = 1.954

GELU(2) ≈ 1.954   (close to x itself — mostly "passed through")
```

**Example 2 (x = -1, a negative number):**
```
x = -1

Step 1: x³ = -1
Step 2: 0.044715 × -1 = -0.0447
Step 3: x + (-0.0447) = -1 - 0.0447 = -1.0447
Step 4: 0.7979 × -1.0447 = -0.8336
Step 5: tanh(-0.8336) ≈ -0.6822
Step 6: 1 + (-0.6822) = 0.3178
Step 7: 0.5 × (-1) × 0.3178 = -0.1589

GELU(-1) ≈ -0.159   (mostly blocked/shrunk toward 0, but not fully zeroed like ReLU would)
```

**Beginner question:** *How is this different from the simpler ReLU?* ReLU sharply cuts off anything negative to exactly 0, like a light switch. GELU is a smooth dimmer switch — negative numbers get shrunk toward zero gradually rather than being harshly clipped, which tends to help the model learn more smoothly.

---

## 7. Cross-Entropy Loss

**Definition:** Cross-entropy loss measures how "surprised" the model was by the correct answer — a low number means the model was confident and correct, a high number means it was confidently *wrong*.

**How to read it:**
```
Loss = - log( P(correct_token) )
```
Say it as: "negative log of the probability the model assigned to the actual correct token."

**Why we use it:** This is the exact number the training loop (Phase 6) tries to minimize. It directly measures the gap between what the model predicted (its softmax probabilities) and what the correct answer actually was.

**Formula:**
```
Loss = - Σ ( y_i × log(ŷ_i) )
```
Where `y_i` is 1 for the correct token and 0 for all others (this simplifies to just `-log(ŷ_correct)`).

**Example 1 (model was confident AND correct):**
```
Model's predicted probability for the correct token: 0.9 (90%)

Loss = -log(0.9)
     = -(-0.105)
     = 0.105

Loss ≈ 0.105   (small loss — the model did well)
```

**Example 2 (model was confident but WRONG — assigned only 5% to the correct token):**
```
Model's predicted probability for the correct token: 0.05 (5%)

Loss = -log(0.05)
     = -(-3.0)
     = 3.0

Loss ≈ 3.0   (large loss — the model is heavily penalized for being confidently wrong)
```

**Beginner question:** *Why use `log` instead of just `1 - probability`?* Because log punishes confident wrongness *much* more harshly than mild wrongness (compare: loss of 0.105 vs 3.0 above, for a shift from 90% to just 5% confidence) — this steep penalty pushes the model to be not just "roughly right" but genuinely well-calibrated in its confidence.

---

## 8. Gradient Descent (Weight Update Rule)

**Definition:** Gradient descent is the rule for how much to nudge each of the model's internal numbers (weights) after seeing how wrong a prediction was.

**How to read it:**
```
w_new = w_old - (learning_rate × gradient)
```
- `w_old` = the current value of a weight
- `gradient` = which direction (and how strongly) changing this weight would affect the loss
- `learning_rate` = how big a step to take (a small number you choose, like 0.001)
- `w_new` = the updated weight after this training step

**Why we use it:** This formula is applied to literally every single weight in the model, every single training step (Phase 6) — it's the actual mechanism of "learning." Without it, weights would just stay at their random starting values forever.

**Formula:**
```
w_new = w_old - η × (∂Loss / ∂w)
```
(η is "eta," a common symbol for learning rate; ∂Loss/∂w is the gradient — "how loss changes as this weight changes.")

**Example 1 (weight needs to decrease):**
```
Current weight: w_old = 0.50
Learning rate:  η = 0.1
Gradient:       ∂Loss/∂w = 2.0   (positive gradient means: increasing w increases loss)

w_new = 0.50 - (0.1 × 2.0)
      = 0.50 - 0.2
      = 0.30

The weight decreased from 0.50 to 0.30, because increasing it was hurting performance.
```

**Example 2 (weight needs to increase):**
```
Current weight: w_old = -0.20
Learning rate:  η = 0.05
Gradient:       ∂Loss/∂w = -4.0   (negative gradient means: increasing w decreases loss)

w_new = -0.20 - (0.05 × -4.0)
      = -0.20 - (-0.2)
      = -0.20 + 0.2
      = 0.0

The weight increased from -0.20 to 0.0, because increasing it was actually helping.
```

**Beginner question:** *Why subtract the gradient instead of adding it?* Because the gradient points in the direction that *increases* loss — and since we want to *decrease* loss, we always step in the opposite direction, which is why there's a minus sign.

---

## 9. Adam Optimizer Update

**Definition:** Adam is a smarter version of gradient descent that keeps a running memory of recent gradients (momentum) and automatically adjusts the step size for each individual weight.

**How to read it:**
```
m = β1×m + (1-β1)×gradient          (momentum: smoothed average of recent gradients)
v = β2×v + (1-β2)×gradient²          (adaptive scaling: smoothed average of squared gradients)
w_new = w_old - η × m / (√v + ε)
```
- `m` = "momentum" — remembers the general recent direction, like a rolling ball that doesn't stop instantly
- `v` = "velocity/variance tracker" — remembers how large recent gradients have been, to automatically slow down in bumpy areas
- `β1, β2` = "how much to remember the past" (commonly 0.9 and 0.999)
- `ε` = tiny number to avoid dividing by zero

**Why we use it:** Plain gradient descent (Formula 8) can be slow or unstable, especially on huge models like this one. Adam is the actual optimizer used in the training loop (Phase 6) and fine-tuning phases (10, 18) because it adapts automatically and converges faster in practice.

**Example 1 (first training step, starting from zero momentum):**
```
Previous m = 0, previous v = 0
β1 = 0.9, β2 = 0.999, η = 0.01, ε = 1e-8
gradient = 4.0

Step 1: update momentum
  m = 0.9×0 + 0.1×4.0 = 0.4

Step 2: update variance tracker
  v = 0.999×0 + 0.001×(4.0²) = 0.001×16 = 0.016

Step 3: update weight (w_old = 1.0)
  w_new = 1.0 - 0.01 × (0.4 / (√0.016 + 1e-8))
        = 1.0 - 0.01 × (0.4 / 0.1265)
        = 1.0 - 0.01 × 3.16
        = 1.0 - 0.0316
        = 0.9684
```

**Example 2 (a later step, with existing momentum already built up):**
```
Previous m = 0.4, previous v = 0.016
gradient = 3.0

Step 1: update momentum
  m = 0.9×0.4 + 0.1×3.0 = 0.36 + 0.3 = 0.66

Step 2: update variance tracker
  v = 0.999×0.016 + 0.001×(3.0²) = 0.016 + 0.009 = 0.025

Step 3: update weight (w_old = 0.9684)
  w_new = 0.9684 - 0.01 × (0.66 / (√0.025 + 1e-8))
        = 0.9684 - 0.01 × (0.66 / 0.1581)
        = 0.9684 - 0.01 × 4.175
        = 0.9684 - 0.04175
        = 0.9267
```

**Beginner question:** *Why not just use plain gradient descent everywhere?* Because Adam's "memory" of recent gradients smooths out noisy, jumpy updates, and its per-weight adaptive scaling means some weights can take bigger steps while others take smaller ones — this usually trains large models faster and more reliably than plain gradient descent.

---

## 10. RoPE (Rotary Position Embedding)

**Definition:** RoPE encodes a token's position in the sequence by literally *rotating* its vector by an angle that depends on its position — nearby tokens end up with similar rotation, far-apart tokens end up rotated very differently.

**How to read it:**
```
For a 2D pair of numbers (x, y) at position "pos":
  x_rotated = x × cos(pos × θ) - y × sin(pos × θ)
  y_rotated = x × sin(pos × θ) + y × cos(pos × θ)
```
This is literally the standard 2D rotation formula from geometry — `θ` (theta) controls how fast the rotation angle grows with position.

**Why we use it:** The model needs to know word *order* (Phase 16) — "dog bites man" vs. "man bites dog." RoPE bakes position information directly into the vector's geometry via rotation, and it naturally supports extending to longer sequences (RoPE scaling) since rotation math works at any angle/position.

**Formula:**
```
x' = x·cos(pos·θ) - y·sin(pos·θ)
y' = x·sin(pos·θ) + y·cos(pos·θ)
```

**Example 1 (position = 1, θ = 0.5 radians, vector = [1, 0]):**
```
x = 1, y = 0, pos = 1, θ = 0.5
angle = pos × θ = 1 × 0.5 = 0.5 radians

cos(0.5) ≈ 0.8776
sin(0.5) ≈ 0.4794

x' = 1×0.8776 - 0×0.4794 = 0.8776
y' = 1×0.4794 + 0×0.8776 = 0.4794

Rotated vector: [0.8776, 0.4794]
```

**Example 2 (position = 2, same θ, same original vector [1, 0]):**
```
x = 1, y = 0, pos = 2, θ = 0.5
angle = pos × θ = 2 × 0.5 = 1.0 radians

cos(1.0) ≈ 0.5403
sin(1.0) ≈ 0.8415

x' = 1×0.5403 - 0×0.8415 = 0.5403
y' = 1×0.8415 + 0×0.5403 = 0.8415

Rotated vector: [0.5403, 0.8415]
```
Notice: the same original vector [1,0] gets rotated to a *different* angle depending purely on its position (1 vs 2) — this rotation difference is how the model senses relative distance between tokens.

**Beginner question:** *Why rotation specifically, instead of just adding a "position number" to the vector?* Because rotation preserves the vector's length/magnitude while changing its direction — and critically, the *angle difference* between two rotated vectors depends only on their *relative* positions, which is exactly the kind of "how far apart are these two tokens" information attention needs.

---

## 11. LoRA Low-Rank Decomposition

**Definition:** LoRA represents a weight update as the product of two much smaller matrices instead of one huge matrix, dramatically cutting the number of trainable parameters during fine-tuning.

**How to read it:**
```
ΔW = A × B
```
- `ΔW` (delta W) = the full-size update you'd normally need to add to the original weight matrix
- `A` = a "tall-thin" matrix (learned)
- `B` = a "short-wide" matrix (learned)
- Multiplying these two small matrices together produces something the same shape as the full ΔW, but using far fewer numbers to represent it

**Why we use it:** Full fine-tuning (Phase 10) would mean training every single number in a huge weight matrix. LoRA instead trains just A and B — two small matrices — whose product approximates the needed change, cutting trainable parameters by orders of magnitude.

**Formula:**
```
Original weight: W (size m×n, huge)
LoRA update:     ΔW = A × B
                 A is (m × r), B is (r × n), where r is small (e.g. r=4 or r=8)
New weight:      W_new = W + ΔW = W + (A × B)
```

**Example 1 (tiny illustration: W is 2x2, but pretend it's actually huge; r=1):**
```
A (2×1) = [2]
          [3]

B (1×2) = [1, 4]

ΔW = A × B:
  ΔW[0][0] = 2×1 = 2
  ΔW[0][1] = 2×4 = 8
  ΔW[1][0] = 3×1 = 3
  ΔW[1][1] = 3×4 = 12

ΔW = [2  8]
     [3  12]

Only 2+2 = 4 numbers were trained (A and B combined) to produce
a full 2x2 = 4-number update. At small scale this looks similar,
but for a REAL weight matrix (say 4096×4096 = ~16.7 million numbers),
using r=8 means A and B together have only about 65,000 numbers —
a massive reduction.
```

**Example 2 (adding the LoRA update to an original weight):**
```
Original weight W = [10  20]
                     [30  40]

LoRA update ΔW (from Example 1) = [2  8]
                                    [3  12]

New weight W_new = W + ΔW:
  W_new[0][0] = 10+2  = 12
  W_new[0][1] = 20+8  = 28
  W_new[1][0] = 30+3  = 33
  W_new[1][1] = 40+12 = 52

W_new = [12  28]
        [33  52]
```

**Beginner question:** *How can two small matrices really capture a useful update to a huge matrix?* Because most useful behavior changes during fine-tuning don't need "full-rank" complexity — a low-rank approximation (a much simpler, compressed pattern) is often good enough to capture the specific behavior shift you're fine-tuning for, similar to how a rough sketch can still capture the essence of a detailed photo.

---

## 12. Quantization (Scale + Zero-Point)

**Definition:** Quantization maps a wide range of high-precision numbers (like 32-bit floats) down to a small set of low-precision integers (like 4-bit or 8-bit), using a scale factor and zero-point offset to convert back and forth.

**How to read it:**
```
q = round( x / scale ) + zero_point
x_approx = (q - zero_point) × scale
```
- `x` = the original high-precision number
- `scale` = how big a "step" each integer represents
- `zero_point` = an offset so negative numbers can still be represented in unsigned integers
- `q` = the quantized (compressed) integer
- `x_approx` = the number you get back when "decompressing" — close to original x, but not always exact

**Why we use it:** This is exactly the math behind the quantization stack (Phase 9) — it's how a huge model shrinks from many gigabytes down to a fraction of the size, letting it run on modest hardware like an i3 laptop.

**Formula:**
```
scale = (max_val - min_val) / (2^bits - 1)
q = round( (x - min_val) / scale )
x_approx = q × scale + min_val
```

**Example 1 (quantizing to 4-bit integers, range -1.0 to 1.0):**
```
min_val = -1.0, max_val = 1.0, bits = 4 (so 2^4 - 1 = 15 possible integer values)

scale = (1.0 - (-1.0)) / 15 = 2.0/15 = 0.1333

Quantize x = 0.42:
  q = round( (0.42 - (-1.0)) / 0.1333 )
    = round( 1.42 / 0.1333 )
    = round( 10.65 )
    = 11

Dequantize back:
  x_approx = 11 × 0.1333 + (-1.0)
           = 1.4667 - 1.0
           = 0.4667

Original: 0.42 → Recovered: 0.4667  (small precision loss, as expected)
```

**Example 2 (quantizing a different value in the same range):**
```
Same setup: min_val=-1.0, max_val=1.0, scale=0.1333

Quantize x = -0.75:
  q = round( (-0.75 - (-1.0)) / 0.1333 )
    = round( 0.25 / 0.1333 )
    = round( 1.876 )
    = 2

Dequantize back:
  x_approx = 2 × 0.1333 + (-1.0)
           = 0.2667 - 1.0
           = -0.7333

Original: -0.75 → Recovered: -0.7333  (again, a small, acceptable rounding error)
```

**Beginner question:** *Why not just always use the smallest possible bit size for maximum compression?* Because fewer bits means a bigger "scale" step size, which means more rounding error per number — push too far (like using only 2 bits) and the accumulated error across billions of weights starts noticeably hurting the model's output quality.

---

## 13. KL Divergence and the DPO Reference Ratio

**Definition:** KL (Kullback-Leibler) divergence measures how different one probability distribution is from another — used to make sure a fine-tuned model doesn't drift *too* far from its original behavior while still learning new preferences.

**How to read it:**
```
KL(P ‖ Q) = Σ P(x) × log( P(x) / Q(x) )
```
Say it as: "for every possible outcome x, take the probability under P, multiply by the log of (P's probability divided by Q's probability), and add these all up."
- `P` = the new (updated) model's probability distribution
- `Q` = the original (reference) model's probability distribution
- Result = 0 if P and Q are identical; grows larger the more they differ

**Why we use it:** Preference tuning should move toward preferred answers without discarding the starting model's behavior. GRPO includes an explicit KL term. Standard DPO does not add this full-distribution KL formula as a separate loss term; instead, its pairwise objective compares policy log-probabilities against frozen-reference log-probabilities.

```text
DPO loss = -log sigmoid(beta * [
    (log policy(chosen) - log reference(chosen))
  - (log policy(rejected) - log reference(rejected))
])
```

The reference ratios provide DPO's connection to KL-regularized reward optimization. Aarambh Studio precomputes those frozen-reference sequence scores once. Its explicit `--reference-free` mode sets the reference log-ratio to zero; that is a lower-memory variant and is not generally identical to standard DPO.

**Formula:**
```
KL(P ‖ Q) = Σ ( P(xi) × log( P(xi) / Q(xi) ) )   for all outcomes i
```

**Example 1 (P and Q are very similar):**
```
Outcome A: P=0.5, Q=0.5
Outcome B: P=0.5, Q=0.5

Term A: 0.5 × log(0.5/0.5) = 0.5 × log(1) = 0.5 × 0 = 0
Term B: 0.5 × log(0.5/0.5) = 0.5 × log(1) = 0.5 × 0 = 0

KL(P‖Q) = 0 + 0 = 0   (identical distributions → zero divergence)
```

**Example 2 (P and Q are quite different):**
```
Outcome A: P=0.9, Q=0.5
Outcome B: P=0.1, Q=0.5

Term A: 0.9 × log(0.9/0.5) = 0.9 × log(1.8) = 0.9 × 0.588 = 0.529
Term B: 0.1 × log(0.1/0.5) = 0.1 × log(0.2) = 0.1 × (-1.609) = -0.161

KL(P‖Q) = 0.529 + (-0.161) = 0.368

KL ≈ 0.368   (a clearly positive number, showing meaningful divergence
              between the new model's preferences and the original model's)
```

**Beginner question:** *Why is KL divergence never negative?* Mathematically, it's proven to always be ≥ 0 — it equals exactly 0 only when the two distributions are identical, and grows the more they differ; this makes it a reliable "distance-like" measure for how much a model's behavior has shifted.

---

## 14. Perplexity (model quality metric)

**Definition:** Perplexity is a single number summarizing how "confused" a model is on average when predicting a sequence of text — lower perplexity means the model finds the text more predictable (i.e., is a better language model for that data).

**How to read it:**
```
Perplexity = exp( average cross-entropy loss )
```
Say it as: "take the average loss (Formula 7) across a whole dataset, then exponentiate it (undo the log)."

**Why we use it:** This shows up in the evaluation harness (Phase 17) as one of the standard ways to score how good a language model is — lower perplexity on held-out text generally means better overall language modeling ability.

**Formula:**
```
Perplexity = e^( (1/N) × Σ Loss_i )    for N tokens
```

**Example 1 (a model that's quite confident/correct on average):**
```
Losses across 4 tokens: 0.1, 0.2, 0.15, 0.05

Average loss = (0.1+0.2+0.15+0.05)/4 = 0.5/4 = 0.125

Perplexity = e^0.125 ≈ 1.133

Perplexity ≈ 1.13   (very low — the model is barely "confused" at all)
```

**Example 2 (a model that's much less confident/accurate):**
```
Losses across 4 tokens: 2.0, 1.5, 2.5, 3.0

Average loss = (2.0+1.5+2.5+3.0)/4 = 9.0/4 = 2.25

Perplexity = e^2.25 ≈ 9.49

Perplexity ≈ 9.49   (much higher — on average, the model is roughly
                     as "confused" as if it were guessing among ~9-10
                     equally likely options)
```

**Beginner question:** *What's a "good" perplexity number?* It depends heavily on the dataset and vocabulary size, but generally: lower is always better, and comparing perplexity *between two versions of the same model on the same test data* is far more meaningful than looking at the raw number in isolation.

---

# Quick Reference: Every Formula in One Table

| # | Formula | One-line meaning | Used in Phase |
|---|---------|-------------------|----------------|
| 1 | Dot Product | Multiply matching pairs, add them up | 3, 4 (everywhere) |
| 2 | Matrix Multiplication | Combine two grids of numbers via repeated dot products | 3, 4 |
| 3 | Softmax | Turn raw scores into clean probabilities that sum to 1 | 4, 7 (output layer) |
| 4 | Scaled Dot-Product Attention | Let tokens decide which other tokens matter most | 4, 15 |
| 5 | Layer Normalization | Keep numbers in a stable, well-behaved range | 3, 4 |
| 6 | GELU Activation | Smoothly decide how much of a signal passes through | 3 |
| 7 | Cross-Entropy Loss | Measure how "surprised" the model was by the right answer | 6 |
| 8 | Gradient Descent | The basic rule for nudging weights to reduce error | 6 |
| 9 | Adam Optimizer | A smarter, adaptive version of gradient descent | 6, 10, 18 |
| 10 | RoPE | Encode token position via rotation | 16 |
| 11 | LoRA Decomposition | Approximate a big weight update with 2 small matrices | 10 |
| 12 | Quantization (scale/zero-point) | Compress numbers to fewer bits | 9 |
| 13 | KL Divergence | Measure how far a fine-tuned model has drifted | 24 |
| 14 | Perplexity | Single score summarizing model quality on text | 17 |

---

# Frequently Asked "Big Picture" Questions

**Q: Do I need to be good at math to understand these?**
No — every formula above is really just "multiply some numbers, add them up, maybe compare to another number." The scary-looking symbols (Σ, ∂, θ) are just shorthand for these simple repeated operations. Once you've worked through the solved examples by hand once, the notation stops looking scary.

**Q: Which formula is the single most important one to understand first?**
The **Dot Product** (Formula 1). Nearly everything else — matrix multiplication, attention, even parts of quantization — is built directly on top of this one simple idea: "multiply matching pairs of numbers, then add them up."

**Q: Why does the model need so much math instead of just "understanding" language directly?**
Because computers can only manipulate numbers — every one of these formulas is a specific, precise way of turning "understanding language" into something expressible as arithmetic on numbers. The *combination* of all these formulas, repeated billions of times, is what produces behavior that looks like understanding.

**Q: If I only remember one thing from each formula, what should it be?**
- Dot Product → "multiply and sum"
- Matrix Multiply → "many dot products at once"
- Softmax → "turn scores into percentages"
- Attention → "decide what to pay attention to"
- LayerNorm → "keep numbers stable"
- GELU → "smooth on/off switch"
- Cross-Entropy → "how surprised was the model"
- Gradient Descent → "nudge weights to do better next time"
- Adam → "smarter nudging with memory"
- RoPE → "encode position via rotation"
- LoRA → "cheap way to approximate a big update"
- Quantization → "compress numbers, lose a little precision"
- KL Divergence → "how far has the model drifted"
- Perplexity → "how confused is the model, on average"

---

*This guide covers the core mathematical formulas powering Aarambh Studio, from the basic building block (dot product) all the way to advanced fine-tuning and evaluation math — explained for a complete beginner, with 2 fully worked examples per formula.*

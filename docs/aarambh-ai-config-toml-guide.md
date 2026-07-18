# Aarambh-AI: The Complete Config (.toml) Guide

### Every field in every config file, explained like you've never opened a `.toml` file before

This guide walks through the checked-in config families used to train, tune,
and test Aarambh-AI. They cover Tiny through Large, CUDA and long-context
smoke runs, MoE, MTP, two-GPU training, vision projection/VQA, and vision-aware
self-learning.

Same format as the other 3 guides — for every field:
- **Definition**
- **Beginner explanation**
- **Why it matters**
- **Example (with real numbers from these exact configs)**
- **Diagram** (where useful)
- **Common beginner questions**

---

## What Is a TOML File, and Why Do We Use It?

**Definition:** TOML (Tom's Obvious, Minimal Language) is a simple text file format for storing settings/configuration, organized into `key = value` pairs and `[section]` headers.

**Beginner explanation:**
Instead of hardcoding numbers like "learning rate = 0.001" directly inside your Rust code (which means recompiling the program every time you want to test a different number), you put all these adjustable settings into a separate `.toml` file. The program reads this file at startup and configures itself accordingly.

**Why we use it:**
Training a model involves dozens of settings (model size, learning rate, batch size, etc.) that you want to change frequently between experiments — without a config file, you'd have to edit and recompile Rust code every single time. TOML files let you spin up a brand new training run just by writing a new `.toml` file and pointing the program at it.

**Example (a tiny slice of real TOML from `tiny_shakespeare_smoke.toml`):**
```toml
dataset_path = "data/tiny_shakespeare.txt"
vocab_size = 8000

[model]
hidden_dim = 128
n_layers = 2
```
Reading this: `dataset_path` and `vocab_size` are top-level settings (not inside any section). Then `[model]` starts a new section, and everything below it (`hidden_dim`, `n_layers`) belongs to that section — until the next `[section]` header appears.

**Diagram:**
```
   dataset_path = "..."      ← top-level setting
   vocab_size = 8000         ← top-level setting

   [model]                   ← section header
     hidden_dim = 128        ← belongs to [model]
     n_layers = 2             ← belongs to [model]

   [train]                   ← a new section starts
     lr = 0.001               ← belongs to [train]
```

**Common beginner questions:**
- *Q: Why TOML instead of JSON or YAML?* → TOML is designed specifically to be human-readable and hard to mess up (unlike YAML's tricky whitespace rules), while still being simple to parse in Rust — a very common choice for Rust config files.
- *Q: Do I need to write these by hand every time?* → No — usually you copy an existing config close to what you want and adjust just the fields you need to change, which is exactly why the checked-in configs exist as ready-made starting points.

---

# PART 1 — Top-Level Settings (Data & Run Setup)

These sit outside any `[section]` — they control how data is loaded and what hardware/precision the run uses.

## 1. `dataset_path`

**Definition:** The file path to the raw text data used for training.

**Beginner explanation:** This tells the program "here's the actual text file to learn from." Every training config points to a text file (or in vision configs, a JSONL of captions) somewhere on disk.

**Why it matters:** Without this, the program has no idea what data to train on — this is the literal doorway into Phase 2 (Data Pipeline).

**Example 1 (from `tiny_shakespeare.toml`):**
```toml
dataset_path = "data/tiny_shakespeare.txt"
```
This points to a small, classic dataset — the collected works of Shakespeare — commonly used for quick experiments because it's small and well-understood.

**Example 2 (from `wikitext103_large.toml`):**
```toml
dataset_path = "data/wikitext-103-raw/wiki.train.raw"
```
This points to WikiText-103, a much larger, real Wikipedia-derived dataset used for serious training runs, not just quick tests.

**Common beginner questions:**
- *Q: Why do some configs use Shakespeare and others use Wikipedia text?* → Shakespeare is small and fast — perfect for a quick smoke test to confirm the code works. WikiText-103 is large and diverse — used for actually training a capable model.

---

## 2. `tokenizer_path` / `tokenizer_save_path`

**Definition:** `tokenizer_save_path` is where a newly-trained tokenizer gets saved; `tokenizer_path` is where an already-trained tokenizer gets loaded from.

**Beginner explanation:** Recall from the Phases guide (Phase 1) — the tokenizer turns text into numbers. If you're training a brand-new tokenizer from your dataset, you use `tokenizer_save_path` to say where to save it. If a tokenizer already exists and you just want to reuse it (like for a vision or fine-tuning run built on an existing base model), you use `tokenizer_path` to load it.

**Why it matters:** Every model needs a tokenizer, and it needs to be the *same* tokenizer that was used originally — mixing up tokenizers between training and inference produces garbage output.

**Example 1 (training a new tokenizer, from `wikitext103_tiny.toml`):**
```toml
tokenizer_save_path = "checkpoints/wikitext103_tiny/tokenizer.json"
```
A brand new tokenizer gets built from this dataset and saved to this path.

**Example 2 (reusing an existing tokenizer, from `vision_vqa_instruct.toml`):**
```toml
tokenizer_path = "checkpoints/tiny_shakespeare/tokenizer.json"
```
This vision fine-tuning run reuses the tokenizer already trained during the `tiny_shakespeare` run — because it's building a vision capability *on top of* that already-trained base model, which must keep the exact same vocabulary.

**Diagram:**
```
  New base model training:
     dataset → [Train Tokenizer] → tokenizer_save_path (new file)

  Fine-tuning / vision on existing model:
     tokenizer_path (existing file) → [Load Tokenizer] → reuse as-is
```

**Common beginner questions:**
- *Q: What happens if I use the wrong tokenizer for a model?* → The token IDs won't line up with what the model actually learned — it would be like handing someone a book written in a code they were never taught, resulting in nonsense output.

---

## 3. `vocab_size`

**Definition:** The total number of unique tokens the tokenizer can produce — i.e., the size of its "vocabulary."

**Beginner explanation:** Recall from the Tokenizer phase — text gets split into pieces and each piece gets an ID number. `vocab_size` sets the ceiling on how many *different* pieces the tokenizer is allowed to recognize. It appears both at the top level (controls tokenizer training) and inside `[model]` (must exactly match, since the model's output layer needs one "slot" per possible token).

**Why it matters:** A bigger vocabulary means fewer tokens are needed to represent the same text (more of language is captured in single tokens), but it also makes the model's embedding and output layers larger (more memory/compute).

**Example 1 (smoke test configs, e.g. `tiny_shakespeare_smoke.toml`):**
```toml
vocab_size = 8000
```
A small vocabulary — fine for quick tests on a small, simple dataset like Shakespeare's text.

**Example 2 (full-scale configs, e.g. `wikitext103_medium.toml`):**
```toml
vocab_size = 32000
```
A much larger vocabulary — needed to properly cover the far more diverse language found in a large real-world Wikipedia dataset.

**Common beginner questions:**
- *Q: Why not always use the biggest vocab_size possible?* → Bigger vocab size increases the size (and training cost) of the embedding table and output layer — you want just enough vocabulary to represent your data well, not more.
- *Q: Do `vocab_size` at the top level and inside `[model]` ever mismatch on purpose?* → No — they must always match exactly, since this number determines how many "slots" the model's embedding table and output layer need.

---

## 4. `validation_split`

**Definition:** The fraction of the dataset held back from training, used only to check the model's performance on unseen data.

**Beginner explanation:** This connects directly to Formula/Phase concept of train/validation/test splitting (covered in your AI/ML/DL guide) — a small percentage of the data is set aside so you can honestly measure whether the model is actually learning general patterns, not just memorizing.

**Why it matters:** Without this, you'd have no way to catch overfitting (the model memorizing training data instead of learning general language patterns) during training.

**Example 1 (from `tiny_shakespeare.toml`):**
```toml
validation_split = 0.05
```
5% of the Shakespeare text is held back for validation — reasonable for a smaller dataset where you still want a meaningful validation sample.

**Example 2 (from `wikitext103_large.toml`):**
```toml
validation_split = 0.01
```
Just 1% held back — because WikiText-103 is so much larger, even 1% still gives you plenty of validation data, while maximizing how much data is available for actual training.

**Common beginner questions:**
- *Q: Why is `validation_split = 0.0` in some smoke test configs (e.g. `vision_vqa_smoke.toml`)?* → Smoke tests exist purely to confirm the code runs without crashing — they're not trying to actually measure model quality, so validation is skipped entirely to keep the test fast and simple.

---

## 5. `shuffle`

**Definition:** Whether to randomize the order of training examples before each pass through the data.

**Beginner explanation:** If the model always sees data in the exact same order every time, it can accidentally learn spurious patterns based on that ordering (e.g., "chapter 1 always comes before chapter 2, so I'll bias toward chapter-1-style text early in training"). Shuffling breaks this up.

**Why it matters:** Shuffling generally improves training stability and helps the model generalize better, since it prevents the model from learning order-dependent quirks of the raw dataset.

**Example 1 (smoke test, `vision_vqa_smoke.toml`):**
```toml
shuffle = false
```
Smoke tests often disable shuffling — since the goal is just to verify the code runs deterministically and quickly, not to train a good model.

**Example 2 (real training run, `wikitext103_tiny.toml`):**
```toml
shuffle = true
```
Real training runs almost always shuffle, to get the generalization benefits described above.

**Common beginner questions:**
- *Q: Does shuffling reorder the text within a single document, or just the order of documents/chunks?* → Just the order of chunks/documents being fed in — the text within each individual training chunk stays in its original, correct reading order.

---

## 6. `resume`

**Definition:** Whether to continue training from a previously saved checkpoint, instead of starting from a freshly-initialized (random) model.

**Beginner explanation:** Training large models can take hours or days. If your laptop needs to sleep, or a Kaggle session times out, you don't want to lose all that progress — `resume = true` tells the program to pick up exactly where it left off, using the last saved checkpoint.

**Why it matters:** This is essential for any long-running training job on hardware that isn't guaranteed to run uninterrupted for the full duration (like free-tier Kaggle GPU sessions with time limits).

**Example 1 (starting fresh, `wikitext103_cuda_smoke.toml`):**
```toml
resume = false
```
A quick one-off smoke test — no need to resume anything since it's meant to run start-to-finish in one short burst.

**Example 2 (long real training run, `medium_16k.toml`):**
```toml
resume = true
```
A large, long training run — `resume = true` means if this run gets interrupted (session timeout, crash, manual stop), restarting the program will pick back up from the last checkpoint instead of losing all progress.

**Diagram:**
```
   resume = false:  [Random Init] ──► Train from step 0

   resume = true:   [Load Last Checkpoint] ──► Continue from
                     saved step number, saved weights, saved
                     optimizer state (momentum, etc.)
```

**Common beginner questions:**
- *Q: What exactly gets restored when resuming?* → Typically the model weights, the optimizer's internal state (like Adam's momentum values from your math guide), and the current step count — so training truly continues as if it never stopped.

---

## 7. `device`

**Definition:** Which physical hardware the training computation should run on — CPU or a specific GPU.

**Beginner explanation:** This one setting decides whether all the matrix multiplications happen on your laptop's CPU cores, or on a GPU (`cuda:0` meaning "the first available NVIDIA GPU").

**Why it matters:** This is the single biggest lever for training speed. GPUs are built to do the exact kind of massively-parallel matrix math a neural network needs, and are typically 10-100x faster than a CPU for this workload.

**Example 1 (CPU, from `tiny_shakespeare_smoke.toml`):**
```toml
device = "cpu"
```
Used for smoke tests specifically designed to run on modest hardware (like your i3 laptop) without needing a GPU at all.

**Example 2 (GPU, from `wikitext103_medium.toml`):**
```toml
device = "cuda:0"
```
Targets the first CUDA-capable GPU (like the Kaggle T4 you've used) — necessary for training anything beyond tiny smoke-test scale in a reasonable amount of time.

**Common beginner questions:**
- *Q: What does the `:0` in `cuda:0` mean?* → It's the GPU index — if a machine has multiple GPUs, `cuda:0` is the first one, `cuda:1` the second, and so on; this matters for Multi-GPU Training (Phase 23).
- *Q: Can I just always set `device = "cuda:0"` even if I don't have a GPU?* → No — the program will fail to find a GPU and error out; you must match this setting to hardware you actually have available.

---

## 8. `dtype`

**Definition:** The numeric precision format used to store and compute the model's numbers during this run.

**Beginner explanation:** This connects to the Quantization concept from your Phases/Math guides, but applied at training time rather than just for final compression. Common values here:
- `f32` — standard 32-bit floating point (most precise, most memory-hungry)
- `f16` — 16-bit floating point (half the memory, faster on GPU, slightly less precise)
- `bf16` — "bfloat16," a 16-bit format with a wider representable range than plain `f16`, popular for large-scale training stability

**Why it matters:** Choosing a lower-precision dtype can dramatically speed up training and cut memory use — but going too low-precision without care can cause training instability (like NaN losses).

**Example 1 (CPU smoke test, `tiny_shakespeare_smoke.toml`):**
No `dtype` set at top level (defaults typically apply on CPU) — since CPU training often just uses standard precision without needing the GPU-specific speed tricks.

**Example 2 (GPU training, `wikitext103_large.toml`):**
```toml
dtype = "bf16"
```
`bf16` is used for large-scale GPU training runs — it's become close to an industry standard for training big models because it balances speed/memory savings with training stability better than plain `f16`.

**Diagram:**
```
   f32:  [sign][8 exponent bits][23 mantissa bits]   → most precise, most memory
   f16:  [sign][5 exponent bits][10 mantissa bits]   → less range, faster/smaller
   bf16: [sign][8 exponent bits][7 mantissa bits]    → same range as f32, less precision
```

**Common beginner questions:**
- *Q: Why is `bf16` preferred over `f16` for large training runs?* → `bf16` keeps the same *range* of representable numbers as full `f32` (same 8 exponent bits), which makes it much less prone to numerical overflow/underflow problems during training, even though it has fewer precision digits.
- *Q: Does using `f16`/`bf16` hurt final model quality?* → Generally only a little, if at all, for training — this is a well-established, widely-used trade-off in modern large model training.

---

# PART 2 — The `[model]` Section (Architecture)

This section defines the actual shape and size of the neural network itself.

## 9. `hidden_dim`

**Definition:** The size of the vector used to represent each token as it flows through the model — the "width" of the network.

**Beginner explanation:** Every token gets represented internally as a list of numbers (an embedding vector). `hidden_dim` decides how long that list is. Bigger `hidden_dim` = each token carries a richer, more detailed representation, but at the cost of more computation and memory.

**Why it matters:** This is one of the two biggest levers (along with `n_layers`) determining total model size and capability.

**Example 1 (small, from `wikitext103_tiny.toml`):**
```toml
hidden_dim = 384
```

**Example 2 (large, from `wikitext103_large.toml` / `large_16k.toml`):**
```toml
hidden_dim = 2048
```
More than 5x wider — a substantially more capable (and more compute-hungry) model.

**Diagram:**
```
   Small hidden_dim (384):    [●●●●●●●●] (short vector per token)
   Large hidden_dim (2048):   [●●●●●●●●●●●●●●●●●●●●●●●●●●●●] (long vector per token)
```

**Common beginner questions:**
- *Q: Does bigger `hidden_dim` always mean a better model?* → Generally yes for capability, but it needs to be matched with enough training data and compute — a huge `hidden_dim` trained on too little data won't reach its potential and just wastes resources.

---

## 10. `ffn_dim`

**Definition:** The size of the internal expansion inside each transformer block's Feed-Forward Network (FFN) sub-layer.

**Beginner explanation:** Inside each transformer block, after attention does its job (deciding what to pay attention to), a separate feed-forward network processes each token's representation further. This FFN typically expands the vector to a much larger size internally, then projects it back down — `ffn_dim` sets how large that internal expansion is.

**Why it matters:** This FFN sub-layer does a large share of the "thinking"/pattern-processing work in a transformer — its size heavily affects both model capability and compute cost.

**Example 1 (from `wikitext103_tiny.toml`):**
```toml
hidden_dim = 384
ffn_dim = 1024
```
Notice: `ffn_dim` (1024) is roughly 2.7x the `hidden_dim` (384) — a common ratio in transformer designs.

**Example 2 (from `wikitext103_medium.toml`):**
```toml
hidden_dim = 1024
ffn_dim = 3392
```
Again roughly 3.3x — this ratio (usually somewhere between ~2.7x and ~4x) is a standard transformer design choice, though exact values vary by architecture family.

**Common beginner questions:**
- *Q: Why not just make `ffn_dim` equal to `hidden_dim`?* → Research has consistently shown that expanding the FFN internally to several times the hidden size, then compressing back down, gives the network much more representational capacity than keeping it the same size throughout.

---

## 11. `n_layers`

**Definition:** How many transformer blocks are stacked on top of each other — the "depth" of the network.

**Beginner explanation:** Directly ties to the Deep Learning concept from your AI/ML/DL guide — this is literally the "how many layers deep" number. Each layer refines the token representations a bit further, building up increasingly abstract understanding.

**Why it matters:** Along with `hidden_dim`, this is the other primary lever for total model size/capability.

**Example 1 (small, from `wikitext103_tiny.toml`):**
```toml
n_layers = 8
```

**Example 2 (large, from `wikitext103_large.toml`):**
```toml
n_layers = 24
```
Three times deeper — significantly more capacity for learning complex, layered patterns.

**Diagram:**
```
   n_layers = 8:   [L1][L2][L3][L4][L5][L6][L7][L8]

   n_layers = 24:  [L1][L2][L3]...................[L24]
                   (three times as many stacked transformer blocks)
```

**Common beginner questions:**
- *Q: Is it better to increase `n_layers` or `hidden_dim` for more capability?* → Both matter, and the ideal balance is usually determined empirically — modern research generally suggests scaling both together roughly proportionally rather than maxing out just one.

---

## 12. `n_heads` and `n_kv_heads`

**Definition:** `n_heads` is the number of parallel "attention heads" in each transformer block's attention mechanism. `n_kv_heads` is the number of heads specifically used for the Key and Value projections in a technique called Grouped-Query Attention (GQA), which can be fewer than `n_heads`.

**Beginner explanation:** Recall the Scaled Dot-Product Attention formula — instead of computing attention just once per block, transformers typically split it into several parallel "heads," each independently learning to attend to different kinds of relationships (one head might focus on grammar, another on long-range topic relevance, etc.). `n_kv_heads` being smaller than `n_heads` means several Query heads *share* the same Key/Value heads — this saves memory and compute with only a small quality trade-off.

**Why it matters:** Multiple heads let the model capture several different *kinds* of relationships between tokens simultaneously, rather than being limited to one single "view" of attention. The `n_kv_heads` trick (GQA) is specifically why modern models can serve long-context inference more cheaply.

**Example 1 (from `wikitext103_tiny.toml`):**
```toml
n_heads = 6
n_kv_heads = 2
```
6 Query heads, but only 2 Key/Value head groups — so every 3 Query heads share one Key/Value head, cutting KV-cache memory usage substantially compared to using 6 separate KV heads.

**Example 2 (from `wikitext103_large.toml`):**
```toml
n_heads = 32
n_kv_heads = 8
```
32 Query heads sharing just 8 KV head groups (4 Query heads per KV group) — the same ratio-based memory-saving trick, scaled up for a much bigger model.

**Diagram:**
```
   Standard Multi-Head Attention (n_heads = n_kv_heads):
   Q-head1 ── K/V-head1
   Q-head2 ── K/V-head2
   Q-head3 ── K/V-head3
   (every query head has its OWN key/value head — more memory)

   Grouped-Query Attention (n_kv_heads < n_heads):
   Q-head1 ┐
   Q-head2 ├── shared K/V-head1
   Q-head3 ┘
   (multiple query heads SHARE one key/value head — less memory)
```

**Common beginner questions:**
- *Q: Why not just always use the same number of KV heads as Query heads?* → You could, but GQA (fewer KV heads) significantly reduces memory needed to cache Keys/Values during long-context inference, with only a small, usually acceptable, quality cost — a very popular trade-off in modern models.
- *Q: Does `n_heads` need to divide evenly into `hidden_dim`?* → Yes — each head gets an equal slice of the hidden dimension, so `hidden_dim` must be evenly divisible by `n_heads` (e.g. 384 / 6 = 64 per head).

---

## 13. `max_seq_len`

**Definition:** The maximum number of tokens the model can process in a single sequence (context window).

**Beginner explanation:** This directly connects to the Long Context / RoPE Scaling phase from your Phases guide — it's the "how much can it read at once" ceiling. A `max_seq_len` of 512 means the model can consider at most 512 tokens of context (roughly a few paragraphs) when generating its next prediction.

**Why it matters:** Larger `max_seq_len` lets the model handle longer documents/conversations, but increases memory and compute cost substantially — attention cost grows faster than the sequence length itself, which is exactly why Flash Attention (Phase 15) matters so much for long sequences.

**Example 1 (short context, `wikitext103_tiny.toml`):**
```toml
max_seq_len = 512
```

**Example 2 (long context, `medium_16k.toml` / `large_16k.toml`):**
```toml
max_seq_len = 16384
```
32x longer context than the tiny config — this is exactly the kind of setting that requires RoPE scaling (see below) to work correctly, since it goes far beyond typical short-context training ranges.

**Common beginner questions:**
- *Q: Can I just set `max_seq_len` to a huge number for every config?* → Longer max sequence length dramatically increases memory/compute requirements — smoke tests and small experiments deliberately use short `max_seq_len` to stay fast and cheap, while long-context configs are specifically designed (with RoPE scaling) to handle much longer sequences.

---

## 14. `rope_theta`

**Definition:** The base frequency value used in RoPE (Rotary Position Embedding) calculations — it controls how quickly the rotation angle changes as token position increases.

**Beginner explanation:** Recall the RoPE formula from your Math Formulas guide — position is encoded by rotating a token's vector by an angle based on its position. `rope_theta` is the constant that determines how "fast" that rotation angle grows per position step. A larger `rope_theta` makes the rotation change more slowly across positions, which helps the model represent much longer sequences without the rotation "wrapping around" too quickly.

**Why it matters:** This single number is a key ingredient in how far a model's context length can stretch before position information becomes ambiguous.

**Example 1 (shorter-context configs, `wikitext103_tiny.toml`):**
```toml
rope_theta = 10000.0
```
The standard, widely-used default for shorter context lengths.

**Example 2 (long-context configs, `wikitext103_medium.toml`, `medium_16k.toml`):**
```toml
rope_theta = 500000.0
```
A much larger base value — deliberately chosen to support the longer `max_seq_len` these configs are built for, spreading the rotation out over a much longer range of positions.

**Common beginner questions:**
- *Q: Why do long-context configs need a bigger `rope_theta`?* → With a small `rope_theta`, rotation angles start repeating (aliasing) at long distances, making the model confuse far-apart positions — a bigger `rope_theta` keeps rotation angles distinct across a much longer range of positions.

---

## 15. `norm_eps`

**Definition:** The tiny constant (`ε`, epsilon) added inside Layer Normalization's formula purely to avoid dividing by zero.

**Beginner explanation:** Straight from your Layer Normalization formula in the Math Formulas guide — recall the denominator `√(σ² + ε)`. If, for some rare batch, variance (`σ²`) happened to be exactly zero, dividing by zero would crash training. `norm_eps` is a tiny safety buffer added to prevent that.

**Why it matters:** Without this small safeguard, training can occasionally hit a divide-by-zero error and crash, especially early in training when weights are still random and unstable.

**Example (identical across virtually every config, e.g. `tiny_shakespeare_smoke.toml`):**
```toml
norm_eps = 0.00001
```
This value (1e-5) is a very standard, widely-used default across transformer implementations — it's small enough to have essentially zero effect on normal training, but large enough to prevent divide-by-zero crashes.

**Common beginner questions:**
- *Q: Why is this the same in almost every config?* → Because it's purely a numerical-stability safeguard, not a model-capability lever — there's rarely a good reason to change it, so it stays fixed across experiments.

---

## 16. `tie_embeddings`

**Definition:** Whether the input token-embedding table and the final output prediction layer share the exact same weights, instead of being two separate learned matrices.

**Beginner explanation:** A transformer needs a table that converts token IDs → vectors at the input (the embedding layer), and a separate step at the output that converts the model's final vector back into probabilities over the vocabulary (recall the Softmax formula). "Tying" these means literally reusing the same weight matrix for both jobs instead of learning two independent ones.

**Why it matters:** Tying embeddings cuts the number of trainable parameters significantly (since the vocabulary-sized weight matrix is often one of the largest single components in a smaller model), with minimal to no quality loss — especially beneficial for smaller models.

**Example (from every config in this set, e.g. `wikitext103_tiny.toml`):**
```toml
tie_embeddings = true
```
The language-model configs tie embeddings by default — a deliberate,
consistent design choice across this model family to save parameters.

**Diagram:**
```
   Untied:  Input Embedding Table (separate) ──► Model ──► Output Layer (separate)
                    ~12M params                              ~12M params

   Tied:    Shared Embedding Table ──► Model ──► (same table, reused)
                    ~12M params total (not doubled)
```

**Common beginner questions:**
- *Q: Does tying embeddings ever hurt quality?* → For smaller models it's usually a clear win (fewer parameters, similar quality); for very large models the effect is more debated, but this project consistently ties embeddings across all its configs.

---

## 16A. Multi-Token Prediction: `[model.mtp]`

This optional section adds future-token auxiliary heads during base training:

```toml
[model.mtp]
num_future_tokens = 2
aux_loss_weight = 0.3
```

`num_future_tokens` includes the normal main head. A value of `2` means the
main head predicts `t+1` and one auxiliary head predicts `t+2`; a value of `3`
adds another head for `t+3`. It must be at least 2 and cannot exceed
`max_seq_len`.

`aux_loss_weight` scales the mean of all auxiliary-head losses before adding it
to the normal next-token loss. It must be finite and in `(0, 1]`. The default
`0.3` keeps MTP as supporting supervision rather than replacing the main
language-model objective.

Omitting `[model.mtp]` creates no auxiliary tensors and preserves ordinary
training and inference. MTP checkpoints can use one-checkpoint speculative
decoding with `infer --speculative`; `--draft-tokens` cannot exceed the trained
`num_future_tokens` horizon.

## 16B. Native QAT: `[model.qat]`

This optional section enables weight-only fake quantization in the base
training loop:

```toml
[model.qat]
bits = "int4"
granularity = "export_aligned"
targets = ["attention", "ffn"]
```

`bits` is `int4` or `int8`. `export_aligned` is the recommended granularity
because its forward values match the existing Q4_K_M or Q8 GGUF exporter.
`per_tensor` and `per_output_channel` are available for controlled experiments.

`targets` may contain `attention`, `ffn`, `moe_router`, `delta_net`,
`dsa_indexer`, `mtp`, and `lm_head`. Embeddings, RMSNorm weights, DeltaNet
convolution weights, and scalar recurrent parameters are never selected. In
export-aligned INT4 mode, DSA indexers intentionally simulate INT8 to match the
exporter's precision policy.

Omitting `[model.qat]` keeps the existing full-precision training path. QAT is
activated only by the training constructor, so loading the same config for
inference does not fake-quantize weights. For continuation, set
`retrofit_from` to an exact SafeTensors checkpoint; names and shapes must match.
Resume also requires the saved and configured QAT policies to be identical.

Use `configs/qat_smoke.toml` for the two-step CPU check and
`configs/qat_tiny.toml` for an exact Tiny continuation. QAT checkpoints store
floating-point master weights and use the normal GGUF exporter afterward.

# PART 3 — RoPE Scaling: `[model.rope_scaling]`

This section only appears in the long-context configs (`wikitext103_long_smoke.toml`, `medium_16k.toml`, `large_16k.toml`) — it's the YaRN technique for stretching a model's context length beyond what it was originally trained for.

## 17. `method`

**Definition:** Which specific RoPE-scaling algorithm to use.

**Beginner explanation:** There are several published techniques for extending RoPE to longer contexts; this project uses **YaRN** (Yet another RoPE extensioN method), a well-regarded approach that adjusts rotation frequencies smartly across different "wavelength" ranges rather than uniformly.

**Example (from `medium_16k.toml`):**
```toml
[model.rope_scaling]
method = "yarn"
```

**Common beginner questions:**
- *Q: Are there other methods besides YaRN?* → Yes (like simple "linear scaling" or "NTK-aware scaling"), but YaRN is generally considered one of the most effective for preserving quality at longer extended lengths, which is why it's used here.

---

## 18. `factor`

**Definition:** How many times longer the new (extended) context length is, compared to the model's original training context length.

**Beginner explanation:** If a model was originally trained comfortably at 2048 tokens, and you want it to handle 16384 tokens, that's an 8x extension — `factor = 8.0`.

**Why it matters:** This is the core "how far are we stretching" number that all the other YaRN math is built around.

**Example 1 (from `medium_16k.toml`):**
```toml
original_max_seq_len = 2048
factor = 8.0
```
Check: 2048 × 8.0 = 16384 — matches this config's `max_seq_len = 16384`.

**Example 2 (from `large_16k.toml`):**
```toml
original_max_seq_len = 4096
factor = 4.0
```
Check: 4096 × 4.0 = 16384 — same final target length (16384), but starting from a longer original base, so a smaller stretching factor is needed.

**Common beginner questions:**
- *Q: Why do `medium_16k.toml` and `large_16k.toml` reach the same final 16384 length via different factors?* → Because they start from different `original_max_seq_len` values — the larger model was apparently already trained/designed with a longer native context (4096) than the medium one (2048), so it needs less stretching to reach the same final target.

---

## 19. `original_max_seq_len`

**Definition:** The context length the model's RoPE embeddings were originally designed/trained for, before any scaling is applied.

**Beginner explanation:** This is the "known good" baseline length — YaRN scaling math needs to know this starting point to correctly compute how to stretch things out.

**Example 1 (from `wikitext103_long_smoke.toml`):**
```toml
original_max_seq_len = 128
```
A tiny baseline, appropriate for a quick smoke test.

**Example 2 (from `large_16k.toml`):**
```toml
original_max_seq_len = 4096
```
A much larger, realistic baseline for an actual production-scale long-context model.

**Common beginner questions:**
- *Q: Where does this number come from?* → It should match whatever `max_seq_len` the base model was actually trained with before you started applying context-extension techniques.

---

## 20. `beta_fast` and `beta_slow`

**Definition:** Two tuning parameters in the YaRN formula that control the boundary between which rotation "frequencies" get scaled aggressively versus left mostly alone.

**Beginner explanation:** Not all parts of the RoPE rotation behave the same way at long distances — some frequencies (roughly, "fast-changing" ones) need more correction than others ("slow-changing" ones). `beta_fast` and `beta_slow` set the boundary ramp between these two behaviors, based on the original YaRN research paper's recommended defaults.

**Example (identical across all RoPE-scaling configs, e.g. `medium_16k.toml`):**
```toml
beta_fast = 32.0
beta_slow = 1.0
```
These specific values are the standard defaults recommended in the YaRN paper, and this project keeps them consistent across all its long-context configs rather than tuning them per-run.

**Common beginner questions:**
- *Q: Do I need to understand the deep math behind these two numbers?* → Not for practical use — these are established default values from published YaRN research; you'd only tune them if you were doing dedicated research into RoPE scaling behavior itself.

---

## 21. `attn_factor`

**Definition:** A scaling correction applied to attention score magnitudes, to compensate for the RoPE stretching changing the "typical" scale of attention scores.

**Beginner explanation:** When you stretch RoPE's rotation behavior, it can subtly shift how large or small attention scores tend to be (recall the `√dk` scaling in your Attention formula) — `attn_factor` is a small correction multiplier to keep things well-calibrated after RoPE scaling is applied.

**Example 1 (from `wikitext103_long_smoke.toml`, factor=2.0):**
```toml
attn_factor = 1.069314718
```

**Example 2 (from `medium_16k.toml`, factor=8.0):**
```toml
attn_factor = 1.207944154
```
Notice: as the scaling `factor` increases (2.0 → 8.0), `attn_factor` also increases — this value is mathematically derived from the scaling factor itself (following the YaRN paper's formula: roughly `0.1 × ln(factor) + 1`), rather than being an independent free choice.

**Common beginner questions:**
- *Q: Do I need to calculate this by hand?* → No — this is typically computed automatically from the `factor` value by the YaRN implementation, but it's still explicitly written in the config for clarity and reproducibility.

---

# PART 4 — Context Schedule: `[[context_schedule]]`

This appears in the long-context configs as a list of stages — notice the *double* square brackets `[[...]]`, which in TOML means "this is an array of tables" (multiple repeated sections, not just one).

## 22. `[[context_schedule]]` (max_seq_len + until_step)

**Definition:** A staged schedule that gradually increases the training sequence length as training progresses, rather than training at the full long context length from step one.

**Beginner explanation:** Training directly at a huge sequence length (like 16384 tokens) from the very first step is expensive and can be less stable. Instead, these configs start training at a *shorter* sequence length, then progressively increase it in stages as training proceeds — this connects to the GPU Scale-Up idea (Phase 14) but applied to sequence length instead of model size.

**Why it matters:** This saves substantial compute early in training (short sequences are much cheaper to process) while still ending up with a model fully capable of handling the full long context by the end of training.

**Example (from `medium_16k.toml`, full 3-stage schedule):**
```toml
[[context_schedule]]
max_seq_len = 4096
until_step = 5000

[[context_schedule]]
max_seq_len = 8192
until_step = 15000

[[context_schedule]]
max_seq_len = 16384
until_step = 30000
```
Reading this: from step 0 to step 5000, train with sequences up to 4096 tokens. From step 5000 to 15000, step up to 8192 tokens. From step 15000 to the final step 30000, train at the full 16384 tokens.

**Second example (from `wikitext103_long_smoke.toml`, a tiny 2-stage version for quick testing):**
```toml
[[context_schedule]]
max_seq_len = 128
until_step = 1

[[context_schedule]]
max_seq_len = 256
until_step = 2
```
The exact same *concept*, just compressed down to a 2-step smoke test that finishes almost instantly — step 1 uses length 128, and by step 2, length 256.

**Diagram:**
```
   Step:        0 ────────── 5000 ────────── 15000 ────────── 30000
   Seq Length:  [4096 tokens] │ [8192 tokens]  │  [16384 tokens]
                (cheaper)      (medium cost)     (most expensive, but
                                                  only for the final,
                                                  shorter stretch)
```

**Common beginner questions:**
- *Q: Does the model "forget" the shorter-context training once it moves to longer stages?* → No — the earlier stages still contribute useful general language learning; the later long-context stages specifically teach the model to handle the extended range, building on top of what it already learned.
- *Q: Why not just always use this staged approach, even for short-context models?* → For models that never need long context in the first place (like the small `wikitext103_tiny.toml` config, `max_seq_len = 512`), there's no benefit — the staged schedule specifically exists to make long-context training economical.

---

# PART 5 — The `[vision]` Section

This section only appears in the four vision-related configs, connecting directly to Phases 19-21 (Vision Encoder, Vision-Language Training, Vision-Aware Self-Learning) from your Phases guide.

## 23. `mode`

**Definition:** Which vision-training stage this particular config run is performing.

**Beginner explanation:** There are two distinct vision modes seen here:
- `"projector_pretrain"` — training just the small projector layer that translates CLIP's image embeddings into the language model's format (Phase 19).
- `"vlm_instruction"` — a further fine-tuning stage where the combined vision+language model is trained on instruction-style visual question-answering data (closer to Phase 20's vision-language training, applied in an instruction-following style).

**Why it matters:** These are genuinely different training objectives requiring different data and (often) different parts of the model to be actively trained — the config needs to tell the program which one is happening.

**Example 1 (from `vision_projector_pretrain.toml`):**
```toml
[vision]
mode = "projector_pretrain"
```

**Example 2 (from `vision_vqa_instruct.toml`):**
```toml
[vision]
mode = "vlm_instruction"
```

**Diagram:**
```
   Stage 1: projector_pretrain
     [Frozen CLIP] → [Projector (being trained)] → [Frozen Language Model]
     (only the small projector learns anything new)

   Stage 2: vlm_instruction
     [Frozen CLIP] → [Trained Projector] → [Language Model (further tuned)]
     (building on stage 1's projector, now teaching instruction-following
      behavior with images)
```

**Common beginner questions:**
- *Q: Why train the projector separately first, instead of training everything together from the start?* → This mirrors why LoRA/fine-tuning phases exist generally — training just the small projector first is cheap and fast, and gives the language model a working "translation" of images before you invest more expensive compute into deeper instruction-tuning.

---

## 24. `base_model_path`

**Definition:** The file path to the already-trained (text-only) language model checkpoint that the vision capability is being built on top of.

**Beginner explanation:** Vision training doesn't start from a blank, randomly-initialized model — it starts from an existing, already-trained text model, and adds vision understanding on top.

**Example (identical across all 4 vision configs, e.g. `vision_projector_smoke.toml`):**
```toml
base_model_path = "checkpoints/tiny_shakespeare/step_000050/model.safetensors"
```
Every vision config here builds on top of the same base checkpoint — a specific saved step (step 50) from the `tiny_shakespeare` training run.

**Common beginner questions:**
- *Q: Why "step_000050" specifically?* → This is just whichever specific saved checkpoint step was chosen as the base for these vision experiments — likely an early, quick checkpoint suitable for smoke-testing the vision pipeline itself, rather than needing a fully-converged text model.

---

## 25. `clip_config_path` and `clip_weights_path`

**Definition:** File paths to the configuration and the actual trained weights of the CLIP vision encoder (Phase 19) used to process images.

**Beginner explanation:** Recall from your Phases guide — CLIP-B/32 is the frozen (untouched) vision encoder that converts raw images into embeddings. These two paths point to that encoder's config and weight files, which get loaded but not further trained.

**Example 1 (smoke test, tiny CLIP, from `vision_projector_smoke.toml`):**
```toml
clip_config_path = "data/vision_smoke/clip_tiny_config.json"
clip_weights_path = "data/vision_smoke/clip_tiny.safetensors"
```
A deliberately tiny, fast CLIP model — used purely to verify the vision pipeline code works correctly, without needing the full CLIP-B/32 model.

**Example 2 (real training, full CLIP-B/32, from `vision_projector_pretrain.toml`):**
```toml
clip_config_path = "data/vision/clip_b32_config.json"
clip_weights_path = "data/vision/clip_b32.safetensors"
```
The actual full-scale CLIP-B/32 model referenced in your v2 roadmap.

**Common beginner questions:**
- *Q: Why have a separate tiny CLIP model just for testing?* → Loading and running the full CLIP-B/32 model is slow and memory-heavy — for a quick smoke test that just needs to confirm "does the code run without crashing," a tiny stand-in model gets the job done in a fraction of the time.

---

## 26. `caption_jsonl`

**Definition:** The path to a JSONL file containing image-caption pairs, used to train the projector to connect images with matching text descriptions.

**Beginner explanation:** This directly connects to the JSONL formatting concept from your Dataset Creation guide — each line is a JSON record pairing an image with its caption, used exactly as described in Phase 20 (Vision-Language Training).

**Example 1 (smoke test, from `vision_projector_smoke.toml`):**
```toml
caption_jsonl = "data/vision_smoke/train_smoke_4.jsonl"
```
Notice the filename literally says "smoke_4" — likely just 4 tiny example image-caption pairs, enough to confirm the training loop runs correctly.

**Example 2 (real training, from `vision_projector_pretrain.toml`):**
```toml
caption_jsonl = "data/coco_captions/train.jsonl"
```
This points to (a version of) the COCO Captions dataset — a well-known, large public dataset of real photos paired with human-written captions, exactly the kind of public dataset described in your Dataset Creation guide.

**Common beginner questions:**
- *Q: Why doesn't the `vlm_instruction` mode config use `caption_jsonl`?* → Looking at `vision_vqa_smoke.toml` and `vision_vqa_instruct.toml`, they don't include this field — instruction-style VQA (Visual Question Answering) training likely uses a differently-structured dataset (question+image+answer triples) rather than simple image+caption pairs, loaded through a different mechanism.

---

## 27. `image_root`

**Definition:** The folder path where the actual image files themselves are stored (referenced by the `caption_jsonl` file).

**Beginner explanation:** The JSONL file typically stores just a filename or relative image path per record — `image_root` tells the program which folder to look inside to actually find those image files on disk.

**Example 1 (smoke test, from `vision_projector_smoke.toml`):**
```toml
image_root = "data/vision_smoke/images"
```

**Example 2 (real training, from `vision_projector_pretrain.toml`):**
```toml
image_root = "data/coco_captions/images/val2017"
```
Points specifically to COCO's `val2017` image folder — a standard, well-known subset of the COCO dataset.

**Common beginner questions:**
- *Q: Why keep captions and images in separate files/folders instead of embedding images directly into the JSONL?* → Images are binary data and would bloat a JSONL file enormously (and JSONL is meant for lightweight, line-by-line text records) — keeping images as separate files and just referencing their filenames/paths is far more practical and efficient.

---

## 28. `projector_hidden_mult`

**Definition:** A multiplier controlling the internal size of the projector's hidden layer, relative to some base dimension.

**Beginner explanation:** Recall Phase 19 — the projector is the small translator network between CLIP's image embeddings and the language model's expected input format. `projector_hidden_mult` controls how much bigger the projector's internal working layer is, relative to its base size — a bigger multiplier gives the projector more capacity to learn a more sophisticated translation, at the cost of more parameters.

**Example 1 (smoke test, from `vision_projector_smoke.toml`):**
```toml
projector_hidden_mult = 1
```
Minimal size — enough to verify the code runs, no need for real translation capacity.

**Example 2 (real training, from `vision_projector_pretrain.toml`):**
```toml
projector_hidden_mult = 4
```
4x larger internal capacity — appropriate for a projector that actually needs to learn a genuinely useful translation from real CLIP-B/32 embeddings into real language-model-compatible representations.

**Common beginner questions:**
- *Q: Is there a "right" value for this?* → It's a capacity/cost trade-off tuned experimentally — bigger generally helps quality up to a point, then gives diminishing returns while still costing more compute/memory.

---

## 29. `max_caption_tokens`

**Definition:** The maximum number of tokens allowed for a single image's caption/description during training.

**Beginner explanation:** Just like `max_seq_len` caps how much text the whole model can process, this specifically caps how long a single image caption is allowed to be before it gets cut off.

**Example 1 (smoke test, from `vision_projector_smoke.toml`):**
```toml
max_caption_tokens = 32
```

**Example 2 (real training, from `vision_projector_pretrain.toml`):**
```toml
max_caption_tokens = 128
```
4x longer — real captions (especially in instruction-style VQA data) can be considerably more detailed/descriptive than the short placeholder captions used in a smoke test.

**Common beginner questions:**
- *Q: What happens to a caption longer than this limit?* → It gets truncated (cut off) at the limit — so setting this too short risks losing important information from longer, more detailed captions.

---

## 30. `max_samples`

**Definition:** A hard cap on the total number of training examples used, regardless of how many are actually available in the dataset.

**Beginner explanation:** This field only appears in the smoke test vision configs — it forces the training run to use just a handful of examples, no matter how large the underlying dataset folder might actually be.

**Example (from `vision_projector_smoke.toml` and `vision_vqa_smoke.toml`):**
```toml
max_samples = 4
```
Just 4 training examples total — purely to let the whole training loop run start-to-finish in seconds, confirming the code path works correctly before committing to a real, full-scale run.

**Common beginner questions:**
- *Q: Why is this field completely absent from the real training configs (`vision_projector_pretrain.toml`, `vision_vqa_instruct.toml`)?* → Because real training runs are meant to use the *entire* available dataset (or as much as practical) — capping sample count would defeat the purpose of a genuine training run.

---

# PART 6 — The `[train]` Section (Training Hyperparameters)

These fields map directly onto the Gradient Descent and Adam Optimizer formulas from your Math Formulas guide.

## 31. `lr` (Learning Rate)

**Definition:** The step size used in the Gradient Descent / Adam weight-update formula — how big a nudge to make to the weights on each training step.

**Beginner explanation:** Directly the `η` (eta) symbol from your Gradient Descent and Adam formulas. Too high, and training can become unstable or diverge; too low, and training takes forever to make meaningful progress.

**Example 1 (smoke test, higher lr, from `wikitext103_cuda_smoke.toml`):**
```toml
lr = 0.001
```
Smoke tests often use a relatively high learning rate since they only run for a couple of steps and just need to see the loss number change at all, confirming the mechanism works.

**Example 2 (real large-model training, lower lr, from `wikitext103_large.toml`):**
```toml
lr = 0.00015
```
A much smaller learning rate — larger models generally need smaller learning rates to train stably, since bigger weight matrices can accumulate error more easily with large steps.

**Common beginner questions:**
- *Q: Why do bigger models in this config set consistently use smaller learning rates?* → Comparing `wikitext103_tiny.toml` (`lr = 0.001`) through `wikitext103_large.toml` (`lr = 0.00015`) — this is a very standard pattern: larger models are more sensitive to overly large weight updates, so learning rate is scaled down as model size scales up.

---

## 32. `batch_size` and `grad_accum_steps`

**Definition:** `batch_size` is how many training examples are processed together in one forward/backward pass. `grad_accum_steps` is how many such batches get processed and their gradients added together *before* actually updating the weights once.

**Beginner explanation:** Sometimes you want a bigger "effective" batch size than your GPU memory can physically hold at once. Gradient accumulation solves this: process several smaller batches one after another, add up (accumulate) their gradients, and only apply one weight update after several batches' worth of gradients have been collected — mathematically equivalent to having used one giant batch.

**Why it matters:** This is how you get the training-stability benefits of large batch sizes even on hardware with limited memory (like your i3 laptop, or a free-tier Kaggle GPU).

**Example 1 (from `medium_16k.toml`):**
```toml
batch_size = 1
grad_accum_steps = 32
```
Effective batch size = 1 × 32 = **32** — achieved without ever needing to fit 32 examples' worth of activations in memory simultaneously (crucial since each example here can be up to 16384 tokens long!).

**Example 2 (from `wikitext103_small.toml`):**
```toml
batch_size = 16
grad_accum_steps = 2
```
Effective batch size = 16 × 2 = **32** — the *same* effective batch size as Example 1, but achieved differently, because this model uses much shorter sequences (`max_seq_len = 1024`) and can therefore afford a bigger physical `batch_size` per step.

**Diagram:**
```
   Without grad accumulation:
     [Batch of 32] ──► Forward+Backward ──► Update weights ONCE
     (needs memory for all 32 examples at once)

   With grad accumulation (batch_size=1, accum=32):
     [Batch of 1] ──► Forward+Backward ──► accumulate gradient
     [Batch of 1] ──► Forward+Backward ──► accumulate gradient
     ... (32 times total) ...
     [Batch of 1] ──► Forward+Backward ──► accumulate gradient
                                          ──► Update weights ONCE
     (only ever needs memory for 1 example at a time)
```

**Common beginner questions:**
- *Q: Is gradient accumulation exactly mathematically identical to one big batch?* → Very close to identical for the weight update math itself — the main practical difference is things like Batch/Layer Normalization statistics being computed per small batch instead of across the full effective batch, though Layer Norm (used here) is less sensitive to this than Batch Norm would be.

---

## 33. `max_epochs` and `max_steps`

**Definition:** `max_epochs` is how many full passes through the entire dataset to make. `max_steps` is a hard cap on the total number of individual training steps (weight updates) regardless of epoch count.

**Beginner explanation:** Whichever limit is hit *first* stops training — `max_steps` often acts as a practical safety cap, especially useful for large datasets where even a fraction of one epoch could be an enormous number of steps.

**Example 1 (from `tiny_shakespeare.toml`):**
```toml
max_epochs = 20
max_steps = 5000
```
With a very small dataset, 20 full passes through it might be needed to learn meaningful patterns — but `max_steps = 5000` provides a hard stopping point regardless.

**Example 2 (from `wikitext103_large.toml`):**
```toml
max_epochs = 1
max_steps = 150000
```
With a much larger dataset, even a single full epoch is a massive amount of training — `max_epochs = 1` here effectively means "just do as much of one full pass as fits within 150,000 steps."

**Common beginner questions:**
- *Q: Why would you ever need more than 1 epoch?* → For small datasets (like Shakespeare's relatively short text), the model needs to see the same data multiple times to properly learn from it — for huge datasets (like all of WikiText-103), often even a fraction of one pass provides plenty of unique training signal.

---

## 34. `warmup_steps`

**Definition:** The number of initial training steps during which the learning rate gradually ramps up from a very small value to its full target value, instead of starting at full strength immediately.

**Beginner explanation:** Right at the very start of training, the model's weights are still randomly initialized and quite "fragile" — taking full-sized gradient steps immediately can cause instability. Warmup gently eases the learning rate up over the first several steps before letting it reach full strength.

**Why it matters:** This is a well-established technique for improving training stability, especially important for larger models and longer training runs.

**Example 1 (short warmup, from `tiny_shakespeare_smoke.toml`):**
```toml
warmup_steps = 4
```
A tiny warmup for a tiny smoke test — enough to demonstrate the mechanism works without needing a lengthy ramp-up.

**Example 2 (longer warmup, from `wikitext103_large.toml`):**
```toml
warmup_steps = 3000
```
A much longer, more gradual ramp-up appropriate for a serious, large-scale training run where stability matters a great deal.

**Diagram:**
```
   Learning Rate
       │
   Full lr ┤              ┌─────────────────────────
       │                 ╱
       │                ╱
       │               ╱ (gradual ramp-up)
       │              ╱
     0 ┤─────────────┘
       └──────────────────────────────────────► Steps
             warmup_steps         (then full lr, later decaying)
```

**Common beginner questions:**
- *Q: What happens right after warmup ends?* → Typically the learning rate then follows a decay schedule (often gradually decreasing) for the rest of training, working alongside `min_lr_ratio` below.

---

## 35. `min_lr_ratio`

**Definition:** The lowest point the learning rate is allowed to decay to by the end of training, expressed as a fraction (ratio) of the original full learning rate.

**Beginner explanation:** Learning rate doesn't just warm up and then stay flat forever — it typically decays gradually over the course of training. `min_lr_ratio` sets a floor: the learning rate will decay down to (at most) this fraction of its peak value, never all the way to absolute zero.

**Example 1 (from `wikitext103_cuda_smoke.toml`):**
```toml
min_lr_ratio = 1.0
```
A ratio of 1.0 means the learning rate never decays at all during this ultra-short smoke test — it just stays flat, since there isn't enough time in 2 steps for a decay schedule to matter.

**Example 2 (from `wikitext103_large.toml`):**
```toml
min_lr_ratio = 0.1
```
The learning rate will decay down to 10% of its peak value by the end of training — a very standard choice, allowing the model to make large adjustments early and much smaller, fine-grained adjustments later.

**Common beginner questions:**
- *Q: Why not let the learning rate decay all the way to exactly zero?* → Keeping a small floor lets the model continue making tiny, useful adjustments right up through the final training steps, rather than effectively "freezing" learning too early.

---

## 36. `weight_decay`

**Definition:** A small penalty added during training that gently pulls all weights toward zero, in addition to the normal gradient-based updates.

**Beginner explanation:** Without any counterbalance, weights can sometimes grow unnecessarily large during training, which can hurt the model's ability to generalize to new data. Weight decay adds a small constant "pull toward zero" force, which acts as a regularizer, keeping weights from growing excessively and helping prevent overfitting.

**Example 1 (from `wikitext103_long_smoke.toml`):**
```toml
weight_decay = 0.0
```
No weight decay at all for this ultra-short smoke test — there's no meaningful risk of overfitting in just 2 training steps.

**Example 2 (from `wikitext103_large.toml`):**
```toml
weight_decay = 0.1
```
A small but meaningful regularization pull, standard for real, longer training runs where overfitting becomes a genuine concern.

**Common beginner questions:**
- *Q: Is weight decay the same thing as the KL Divergence "leash" from DPO in the math guide?* → No — they're different regularization concepts. Weight decay pulls weights toward zero as a general anti-overfitting measure; KL divergence in DPO/PPO specifically measures and limits how far the model's *output behavior* drifts from a reference model.

---

## 37. `beta1`, `beta2`, `epsilon`

**Definition:** These are the exact `β1`, `β2`, and `ε` parameters from the Adam Optimizer formula in your Math Formulas guide.

**Beginner explanation:**
- `beta1` controls how much "momentum" (memory of recent gradient direction) is kept.
- `beta2` controls how much memory of recent squared-gradient magnitude (used for adaptive scaling) is kept.
- `epsilon` is the tiny safety constant preventing division by zero in Adam's update formula.

**Example (essentially identical across every config in this set, e.g. `tiny_shakespeare.toml`):**
```toml
beta1 = 0.9
beta2 = 0.95
epsilon = 0.00000001
```
These are extremely standard, widely-used Adam defaults (note: `beta2 = 0.95` here is a slightly more aggressive/common variant than the original paper's `0.999`, often preferred for language model training specifically).

**Common beginner questions:**
- *Q: Why are these values kept identical across nearly every config, unlike `lr` which changes a lot?* → These Adam hyperparameters are generally quite robust defaults that work well across a wide range of model sizes and datasets — unlike learning rate, which needs careful tuning per model size, these rarely need adjustment.

---

## 38. `clip_grad_norm`

**Definition:** A ceiling on how large the overall gradient (across all weights combined) is allowed to be during any single training step — if it exceeds this value, it gets scaled back down.

**Beginner explanation:** Occasionally, a particular training batch can produce an unusually large gradient (an "exploding gradient"), which if applied directly could badly destabilize training in one bad step. Gradient clipping catches this and rescales the gradient down to a maximum allowed size before applying it.

**Example (identical across every config in this set, e.g. `wikitext103_medium.toml`):**
```toml
clip_grad_norm = 1.0
```
A very standard, widely-used value — capping the total gradient "size" at 1.0 across every config in this project, regardless of model size.

**Common beginner questions:**
- *Q: How often does clipping actually kick in during normal training?* → Usually only occasionally, on unusually difficult batches — most of the time, gradients naturally stay below this ceiling and clipping does nothing; it's a safety net for the occasional bad step, not a constant limiter.

---

## 39. `save_every_n_steps`

**Definition:** How frequently (in training steps) the model's current weights get saved to disk as a checkpoint.

**Beginner explanation:** This directly connects to the `resume` field discussed earlier — checkpoints are what `resume = true` actually loads from. More frequent saving means less lost progress if something interrupts training, but costs disk space and a bit of time each save.

**Example 1 (smoke test, no saving, from `wikitext103_long_smoke.toml`):**
```toml
save_every_n_steps = 0
```
A value of 0 means checkpointing is disabled entirely — reasonable for a 2-step smoke test that finishes almost instantly, with nothing worth saving.

**Example 2 (real training, from `wikitext103_large.toml`):**
```toml
save_every_n_steps = 5000
```
A checkpoint gets saved every 5,000 steps — frequent enough to avoid losing too much progress if the run gets interrupted, but not so frequent that saving itself becomes a significant time/disk overhead.

**Common beginner questions:**
- *Q: Does a smaller `save_every_n_steps` slow down training?* → Slightly — writing a checkpoint to disk takes some time, so there's a genuine trade-off between safety (frequent saves) and raw training speed (fewer interruptions to save).

---

## 40. `log_every_n_steps`

**Definition:** How often (in steps) training progress metrics (like the current loss value) get printed/logged.

**Beginner explanation:** This connects to the Cross-Entropy Loss formula and Evaluation Harness concept from your other guides — this is simply how often you get a visible readout of "here's the loss right now," so you can watch training progress in something like a terminal or log file.

**Example (from `wikitext103_tiny.toml`):**
```toml
log_every_n_steps = 10
```
A loss value gets printed every 10 steps — frequent enough to closely monitor progress, without flooding the console with output every single step.

**Common beginner questions:**
- *Q: Why not log every single step?* → For a run with tens of thousands of steps, printing every single one would flood your terminal/log file with far more detail than useful for spotting overall trends — periodic logging strikes a practical balance.

---

## 41. `eval_steps`

**Definition:** How often (in steps) the model gets evaluated on the held-out validation set (from `validation_split`), during training.

**Beginner explanation:** This connects directly to the Evaluation Harness concept and the train/validation/test splitting idea from your other two guides — periodically, training pauses briefly to check performance on data the model *hasn't* directly trained on, to catch overfitting early.

**Example 1 (smoke test, no eval, from `vision_vqa_smoke.toml`):**
```toml
eval_steps = 0
```
Disabled entirely — smoke tests aren't trying to genuinely measure quality.

**Example 2 (real training, from `wikitext103_medium.toml`):**
```toml
eval_steps = 1000
```
Validation performance gets checked every 1,000 steps — frequent enough to catch problems developing over the course of a long run.

**Common beginner questions:**
- *Q: Does running evaluation slow down training?* → Yes, briefly — each evaluation pass takes some extra time, which is exactly why it's done periodically (every 1000 steps) rather than continuously.

---

## 42. `seed`

**Definition:** A fixed starting number used to initialize all of training's random number generation (weight initialization, data shuffling order, etc.), making the entire run reproducible.

**Beginner explanation:** Computers don't generate *truly* random numbers — they use algorithms that produce a long, seemingly-random sequence starting from a "seed" number. Using the exact same seed means every "random" choice made during training (how weights start out, what order data is shuffled into) will be identical between two runs.

**Why it matters:** Reproducibility is crucial for debugging and fair comparison — if you change one hyperparameter and want to know if the *change* caused a difference (not just random chance), keeping the seed identical between runs isolates that variable properly.

**Example (identical across literally every single config in this set):**
```toml
seed = 42
```
Training configs use deterministic seeds so differences between comparable
runs come from configuration changes rather than uncontrolled random luck.

**Common beginner questions:**
- *Q: Why specifically 42?* → No deep technical reason — it's a widely-used, almost joking convention in programming/ML culture (a reference to "The Hitchhiker's Guide to the Galaxy," where 42 is "the answer to life, the universe, and everything"), that stuck around simply because it needs to be *some* fixed number, and 42 is a common default choice.

---

## 43. `checkpoint_dir`

**Definition:** The folder path where this specific training run's checkpoints (and other saved artifacts) get stored.

**Beginner explanation:** Every config points to its own distinctly-named folder, keeping each experiment's saved checkpoints neatly separated from every other experiment's.

**Example 1 (from `wikitext103_tiny.toml`):**
```toml
checkpoint_dir = "checkpoints/wikitext103_tiny"
```

**Example 2 (from `vision_vqa_instruct.toml`):**
```toml
checkpoint_dir = "adapters/vision_vqa_dora"
```
Notice this one is inside an `adapters/` folder rather than `checkpoints/` — a meaningful naming distinction, since this run is doing DoRA fine-tuning (Phase 18), which produces small adapter weights layered on top of a base model, rather than a complete standalone model checkpoint.

**Common beginner questions:**
- *Q: Why does the folder naming convention change between `checkpoints/` and `adapters/`?* → It reflects what's actually being saved — full model checkpoints (`checkpoints/`) versus lightweight fine-tuning adapters (`adapters/`) that only make sense layered on top of an existing base model — a helpful organizational convention for keeping the project's saved artifacts clearly understandable.

---

# Quick Reference: Every Config File at a Glance

| Config File | Purpose | Model Size (hidden_dim / layers) | Context | Special Feature |
|---|---|---|---|---|
| `tiny_shakespeare_smoke.toml` | Tiniest possible smoke test | 128 / 2 | 64 | Fastest possible sanity check |
| `tiny_shakespeare.toml` | Real small training run | 384 / 8 | 256 | CPU-friendly full run |
| `wikitext103_cuda_smoke.toml` | GPU smoke test | 384 / 2 | 128 | Confirms CUDA path works |
| `wikitext103_tiny.toml` | Small real training run | 384 / 8 | 512 | — |
| `wikitext103_small.toml` | Medium-small training run | 768 / 12 | 1024 | — |
| `wikitext103_medium.toml` | Medium training run | 1024 / 24 | 2048 | Larger rope_theta |
| `wikitext103_large.toml` | Large training run | 2048 / 24 | 4096 | GQA (n_kv_heads=8) |
| `wikitext103_long_smoke.toml` | Long-context smoke test | 384 / 2 | 256 | YaRN + context schedule |
| `medium_16k.toml` | Medium, long-context (16K) | 1024 / 24 | 16384 | YaRN + 3-stage schedule |
| `large_16k.toml` | Large, long-context (16K) | 2048 / 24 | 16384 | YaRN + 3-stage schedule |
| `vision_projector_smoke.toml` | Vision projector smoke test | 384 / 8 | 256 | Tiny CLIP, 4 samples |
| `vision_projector_pretrain.toml` | Real vision projector training | 384 / 8 | 512 | Full CLIP-B/32, COCO data |
| `vision_vqa_smoke.toml` | VQA instruction smoke test | 384 / 8 | 256 | Tiny CLIP, 4 samples |
| `vision_vqa_instruct.toml` | Real VQA instruction fine-tune | 384 / 8 | 512 | Full CLIP-B/32, LLaVA-style data |

---

# Frequently Asked "Big Picture" Questions

**Q: What's the actual pattern behind having a "smoke test" version of almost every config?**
Every major training scenario (base training, long-context, vision) has a matching tiny/fast "smoke" config specifically designed to run in seconds and confirm the code path works correctly — before committing real time/compute to the full-scale version. This is genuinely good engineering practice: catch bugs cheap, before they cost you hours of real training time.

**Q: Why do model sizes scale up in such a specific way (tiny → small → medium → large)?**
This mirrors the GPU Scale-Up phase from your Phases guide — each size tier is a deliberately chosen checkpoint along a scaling ladder, letting you validate that training works correctly at each size before committing to the next, more expensive tier.

**Q: How do all these fields connect back to the Math Formulas guide?**
Almost every `[train]` field maps directly onto a specific symbol in the Gradient Descent/Adam formulas (`lr`→η, `beta1`→β1, `beta2`→β2, `epsilon`→ε, `weight_decay`→the regularization term). The `[model.rope_scaling]` fields map directly onto the RoPE formula's variables. This is the practical, "turn the dial" layer sitting right on top of that math.

**Q: If I wanted to create my own new config for a custom training run, what's the safest approach?**
Copy the smoke-test version closest to what you want first, get it running successfully end-to-end (confirming paths, dataset, and basic mechanics work), then scale it up field-by-field toward your real target config — exactly the pattern this whole set of 14 files already demonstrates.

---

*This guide covers every field across all 14 `.toml` configuration files used to train Aarambh-AI — from the smallest CPU-only smoke test through full long-context and vision-language training runs.*

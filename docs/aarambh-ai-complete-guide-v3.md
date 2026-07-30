# Aarambh Studio v3.0.0: The Complete Guide

### Everything we built in v3, in plain human language

This document covers **v3.0.0** — Phases 29 through 40 — building on the v1.0.0 (Phases 1-13) and v2.0.0 (Phases 14-28) foundations. Each phase assumes the model from v2 is already working; v3 adds hybrid linear attention, sparse attention, fine-grained MoE, multi-token prediction, distillation, QAT, video, document, agents, forgetting diagnostics, and max thinking.

---

## The V3 Big Picture

v2 gave you a working multimodal LLM with vision, MoE, DPO, speculative decoding, and a server. v3 makes it **faster, more efficient, multimodal across more modalities, and self-improving without forgetting**:

```
 V2 Model (dense + MoE, full attention, text + image)
      │
      ▼
 Hybrid Attention (Gated DeltaNet + Sparse) → cheaper long context
 Fine-Grained MoE + Shared Expert → better compute-per-parameter
 Multi-Token Prediction → denser training signal + free draft head
 Distillation → smaller model learns from larger teacher
 QAT → quantization folded into training, less accuracy loss
 Video + Document → vision extended beyond static images
 Agent Chains → multi-step tool use
 Forgetting Diagnostics → measure what's being lost during updates
 Max Thinking → 16K-token reasoning budget
```

---

## Phase 29: Gated DeltaNet (Hybrid Linear Attention)

**Definition:** A hybrid transformer where most attention layers are replaced with Gated DeltaNet — a linear-time recurrence that processes long contexts far more cheaply than full attention — while a minority of full-attention layers are retained for tasks that need precise token-to-token lookup.

**Beginner explanation:**
Full attention (Phase 4) gets expensive very quickly as text gets longer — it's O(n²), meaning doubling the sequence length quadruples the compute. Gated DeltaNet is a "linear attention" variant that compresses past context into a single recurrent state, so cost grows linearly with length instead. A few full-attention layers are kept because some tasks genuinely need to look back at specific earlier tokens rather than a compressed summary.

**Why we need it:**
Without this, scaling to very long documents or conversations is computationally brutal on limited hardware. Hybrid attention gives near-full-attention quality at a fraction of the cost for long sequences.

**Example:**
```
Full attention (all 24 layers):
  Reading 100K tokens: O(n²) = 10 billion operations

Hybrid (20 linear + 4 full):
  20 layers do O(n) = 2 million operations each
  4 layers do O(n²) but n is manageable with the linear layers handling most of the work
  → roughly 10-40× cheaper at long context
```

**Diagram:**
```
Traditional: [Full][Full][Full][Full][Full][Full]...  all layers O(n²)

Hybrid:      [Linear][Linear][Full][Linear][Linear][Full]...
               O(n)     O(n)   O(n²)   O(n)    O(n)   O(n²)
                                    ↑
                    Full-attention layers kept for precise lookup
```

**Common beginner questions:**
- *Q: Does linear attention work as well as full attention?* → On most tasks, almost as well for a fraction of the cost. Tasks requiring precise token recall (like finding a specific sentence in a long document) benefit from the retained full-attention layers.
- *Q: What is the "DeltaNet" part?* → A specific form of linear attention using a delta rule — each new token updates the recurrent state by adding only the *difference* (delta) between what the state predicted and what the token actually is.

---

## Phase 30: DeepSeek Sparse Attention (DSA)

**Definition:** Instead of every query attending to every past token, DSA selects only the most relevant blocks of past tokens using a learned indexer — making even the retained full-attention layers cheaper.

**Beginner explanation:**
The full-attention layers kept from Phase 29 still have the O(n²) problem. DSA fixes this by having each query pick only a small number of "blocks" (groups of tokens) to look at, based on a learned scoring system. It's like having a smart bookmark system that says "this query probably only needs to look at these 3 sections of the document" rather than re-reading everything.

**Why we need it:**
Even with only a few full-attention layers, O(n²) cost at very long sequences adds up. DSA makes those layers much cheaper by attending to a sparse subset of tokens.

**Example:**
```
Regular attention at layer 4:
  Query at position 5,000 attends to all 5,000 past positions

DSA at layer 4 (block_size=64, top_k=4):
  Query at position 5,000: indexer scores all 5,000 positions
  Only attends to the top 4 blocks (256 tokens total)
  → 20× cheaper at this layer
```

**Diagram:**
```
   Positions: [0.....1024....2048....3072....4096....5120]
                                                  ↑ Query
   Regular: attends to ALL 5120 positions
   DSA:     indexer picks blocks [512] [2048] [3072] [4096]
            attends to only 256 positions total
```

**Common beginner questions:**
- *Q: Doesn't the model miss important information by being sparse?* → The indexer is trained to pick the most relevant blocks for each query — it learns where to look, similar to how you'd skim a document for relevant sections.
- *Q: Is this from DeepSeek?* → The technique is inspired by DeepSeek's sparse attention work, adapted to this codebase's architecture.

---

## Phase 31: Fine-Grained MoE + Shared Expert

**Definition:** An upgrade to v2's MoE (Phase 22) that splits experts into many smaller, finer-grained units and adds always-active shared experts that every token uses.

**Beginner explanation:**
v2 had a small number of large experts. v3 increases the number of experts dramatically but makes each one much smaller. A shared expert (always active for every token) learns universal patterns, while the fine-grained experts specialize in narrower domains. Think of it like: instead of 8 general-purpose doctors, you have 64 narrow specialists plus 2 general practitioners that everyone sees.

**Why we need it:**
Fine-grained experts activate more combinations of specialists per input, improving model quality. The shared expert ensures basic processing isn't lost when the router picks different specialists for different tokens.

**Example:**
```
v2 MoE: 8 experts, 2 activated per token
  Expert 1: general math, Expert 2: general language
  → Each expert must cover a broad domain

v3 Fine-Grained MoE: 64 experts + 2 shared, 8 activated per token
  Expert 1: addition, Expert 2: multiplication, Expert 3: grammar, ...
  + Shared: always-active base knowledge
  → Sharper specialization, better quality
```

**Diagram:**
```
v2 (coarse):              v3 (fine-grained + shared):
  ┌──────┐ ┌──────┐        ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐
  │Exp 1│ │Exp 2│  ...     │E1││E2││E3││E4││E5││E6││E7││E8│... 64 experts
  └──────┘ └──────┘        └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘
  2 active per token        ┌──┐┌──┐
                              │Shr││Shr│  ← always active
                              └──┘└──┘
                            8 + 2 active per token → more nuanced
```

**Common beginner questions:**
- *Q: Won't 64 experts make the model huge?* → Each expert is much smaller than v2's experts, so total parameter count stays similar while offering more combinatorial variety per forward pass.
- *Q: What does the shared expert learn?* → Patterns useful for all tokens — basic syntax, common knowledge — so every token gets a baseline of useful processing regardless of which specialized experts are picked.

---

## Phase 32: Multi-Token Prediction (MTP)

**Definition:** During training, the model predicts multiple future tokens at each position (not just the next one) using lightweight auxiliary heads, providing a denser learning signal per training example.

**Beginner explanation:**
Normally, the model predicts only "what is the very next word?" at each position. MTP adds small extra prediction heads that try to predict "what is the word after next? And the one after that?" All these predictions share the same base model, so the model gets more feedback per training step without extra forward passes through the big network.

**Why we need it:**
More learning signal per token = faster, more sample-efficient training. The MTP heads also serve as a free draft model for speculative decoding (Phase 25) — no separate checkpoint needed.

**Example:**
```
Position: "The cat sat on ___"
  Standard training: predict only "the" (next token)
  MTP (2 extra heads):
    Head 1: predict "the"  (next token)
    Head 2: predict "mat"  (token after next)
    Head 3: predict "."    (2 tokens ahead)
  → 3x more training signal from the same input!
```

**Diagram:**
```
Input tokens: [The, cat, sat, on, ___]
                     │
                     ▼
              Base model layers
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   Head 0         Head 1         Head 2
   (next)        (2 ahead)      (3 ahead)
   predicts       predicts       predicts
   "the"          "mat"          "."
```

**Common beginner questions:**
- *Q: Does MTP make inference slower?* → No — the auxiliary heads are used only during training. During inference, only the main head is used (or optionally, the MTP heads can serve as a draft for speculative decoding).
- *Q: How many future tokens does MTP predict?* → Configurable — typically 2-4 future tokens, trading off between extra signal and additional compute during training.

---

## Phase 33: On-Policy Distillation

**Definition:** A process where a smaller "student" model is trained on its own generated outputs (on-policy) that have been scored by a larger "teacher" model, making the student learn from the teacher's preferences.

**Beginner explanation:**
Distillation is like a student learning from a teacher's example. But there's a subtlety: if the student trains on the teacher's *static* example answers, the student's own generation style doesn't get corrected. On-policy distillation fixes this: the student generates an answer first, then the teacher scores it. This way, the student learns to improve its own style of output rather than just imitating a fixed set of teacher answers.

**Why we need it:**
A distilled student can be much smaller and faster than the teacher while retaining most of the teacher's quality. On-policy training (vs off-policy) reduces the distribution mismatch between training and inference.

**Example:**
```
Off-policy distillation:
  Teacher writes: "The capital of France is Paris."
  Student learns to write: "The capital of France is Paris."
  → But what if the student naturally writes "Paris is France's capital"?

On-policy distillation:
  Student generates: "Paris is France's capital."
  Teacher scores it: 95/100 (basically correct)
  Student learns: "My natural style is fine, keep refining it"
  → Student improves its ACTUAL output style, not a fixed imitation
```

**Diagram:**
```
         Prompt
           │
           ▼
  ┌────────────────┐
  │ Student model   │── generates answer
  └────────────────┘
           │
           ▼
  ┌────────────────┐
  │ Teacher model   │── scores the student's answer
  └────────────────┘
           │
           ▼
  ┌────────────────┐
  │ Student update  │── learns from teacher's score
  └────────────────┘
           │
           ▼
  Improved student (on its own generation style)
```

**Common beginner questions:**
- *Q: Do we need both models loaded at once?* → Yes, during distillation training both teacher and student are in memory, which is why it typically runs on GPU.
- *Q: Does distillation replace fine-tuning?* → It's a complement — you'd pre-train or fine-tune first, then distill to get a smaller, faster model with similar quality.

---

## Phase 34: Native QAT (Quantization-Aware Training)

**Definition:** QAT (Quantization-Aware Training) simulates the effects of quantization during training itself, so the model learns to produce weights that remain accurate even after being compressed to low-precision formats.

**Beginner explanation:**
v2's quantization (Phase 9) was applied *after* training — you train in full precision, then compress. The problem: quantization introduces small errors, and the model was never trained to handle them. QAT fixes this by inserting "fake quantization" operations during training that mimic what real quantization will do. The model learns to work around these errors, so when you actually quantize it later, much less quality is lost.

**Why we need it:**
Post-training quantization always loses some quality. QAT typically recovers most or all of that loss, giving you the same model quality in a much smaller, faster package.

**Example:**
```
Post-training quantization (v2):
  Train in FP32 → quantize to INT4 → quality drops 2-3%

QAT (v3):
  Train with fake INT4 noise → real INT4 quantization → quality drops ~0.5%
  → The model learned to cope with quantization during training
```

**Diagram:**
```
Training time:
  [Weights] → [FakeQuant] → [Forward pass] → [Backward pass]
                 ↑
         Simulates INT4 rounding
         (but gradients still flow for learning)

Export time:
  [Weights] → [Real INT4 Quantization]
                 ↑
         Much less quality lost because
         model was trained to handle it
```

**Common beginner questions:**
- *Q: Does QAT make training slower?* → Slightly, due to the extra quantization simulation operations, but the benefit of retaining quality at low precision usually outweighs the small training overhead.
- *Q: What's "fake quantization"?* → It rounds the weights as if they were INT4/INT8 during the forward pass, but keeps the underlying floating-point values for gradient computation — so the model learns that rounding to low precision is coming and adapts.

---

## Phase 35: Native Video Understanding

**Definition:** Extending the vision pipeline (Phases 19-21) from static images to video by decoding video files, sampling representative frames, encoding temporal position information, and training on video question-answering data.

**Beginner explanation:**
A video is just a sequence of images (frames) shown rapidly. The model takes a video, picks key frames (either evenly spaced or at scene changes), processes each frame through the same vision encoder used for static images, adds information about each frame's position in time, and fuses them all together so the language model can reason about the video's content moment by moment.

**Why we need it:**
Without video understanding, the model is blind to the vast amount of information available in videos — tutorials, demonstrations, news clips, surveillance, etc.

**Example:**
```
Input: Video of someone baking a cake (30 seconds, 900 frames)
       │
       ▼
Frame sampler picks 8 representative frames:
  Frame 1: mixing ingredients
  Frame 3: pouring batter
  Frame 5: cake in oven
  Frame 8: finished cake
       │
       ▼
Each frame → CLIP encoder → temporal position added → fused
       │
       ▼
"Describe the baking process shown in this video"
→ Model can describe each step using the temporal ordering
```

**Diagram:**
```
 Video file (.mp4)
      │
      ▼
 ┌────────────┐
 │ Decode      │ (OpenH264)
 └────────────┘
      │
      ▼
 ┌────────────┐
 │ Sample      │ (uniform or scene-aware)
 │ frames      │
 └────────────┘
      │
      ▼
 ┌────────────┐
 │ CLIP encode │ (same frozen ViT as images)
 │ each frame  │
 └────────────┘
      │
      ▼
 ┌────────────┐
 │ Add temporal│ (learned position per frame)
 │ position    │
 └────────────┘
      │
      ▼
 ┌────────────┐
 │ Fuse with   │
 │ text tokens │
 └────────────┘
      │
      ▼
 LLM reasons about video content
```

**Common beginner questions:**
- *Q: How many frames does the model process per video?* → Configurable, typically 4-16 frames depending on video length and available memory.
- *Q: Does it need a separate video-specific training?* → Yes — Phase 35 includes training on video QA datasets like NExT-QA so the model learns to connect frame sequences to answers.

---

## Phase 36: Native Document Understanding

**Definition:** Extending the vision pipeline to PDFs and scanned documents with layout-aware encoding, so the model can read and reason about document structure (columns, tables, headers).

**Beginner explanation:**
Documents aren't just images of text — they have layout: two columns, a header here, a table there, a footnote at the bottom. Phase 36 rasterizes each page into an image, processes it through the same vision encoder as images/video, but additionally encodes each page region's 2D position (row and column in the page layout) so the model understands document structure.

**Why we need it:**
Business, legal, academic, and scientific knowledge is largely stored in documents. A model that can't read documents can't access this knowledge.

**Example:**
```
Input: Research paper PDF (2 columns, 4 pages)
       │
       ▼
Page rasterizer converts each page to an image
Layout projector encodes 2D position of each visual token
       │
       ▼
"What is the main result of Figure 3?"
→ Model understands: "Figure 3 is in the right column of page 2,
  it shows the accuracy comparison between Method A and Method B"
```

**Diagram:**
```
 PDF Document
      │
      ▼
 ┌──────────────┐
 │ Rasterize     │  (each page → image)
 │ pages         │
 └──────────────┘
      │
      ▼
 ┌──────────────┐
 │ CLIP encode   │  (same frozen ViT)
 └──────────────┘
      │
      ▼
 ┌──────────────┐
 │ Layout        │  (adds 2D row/column position)
 │ Projector     │
 └──────────────┘
      │
      ▼
 ┌──────────────┐
 │ Fuse with     │
 │ text tokens   │
 └──────────────┘
      │
      ▼
 LLM reasons about document structure
```

**Common beginner questions:**
- *Q: Is this OCR (optical character recognition)?* → Not exactly. The model learns to "read" the visual patterns of text directly from the rasterized page images, without needing a separate OCR step.
- *Q: Does it handle handwriting?* → Only as well as the CLIP vision encoder can interpret it — naturally, printed text works much better than handwriting.

---

## Phase 37: Long-Horizon Tool-Use Chains

**Definition:** Upgrading v2's single-call tool use (Phase 26) into multi-step chains where the model can call a tool, receive a result, decide whether to continue, and make additional calls — all within one conversation turn.

**Beginner explanation:**
v2's tool use was "one and done" — model calls a calculator, gets an answer, done. Many real tasks need multiple steps: search for a fact, use the fact in a calculation, then use the result to answer a follow-up question. Phase 37 makes this possible by adding a tool chain orchestrator that lets the model decide "I should call another tool now" or "I have enough information to answer."

**Why we need it:**
Real-world problem solving is rarely a single step — you search, compute, verify, refine. Multi-step tool chains enable the model to solve complex, multi-stage problems autonomously.

**Example:**
```
User: "What was the population of France in 2020, and how does it compare to 1950?"

Step 1: Model calls search("France population 2020")
        → Result: 67.4 million

Step 2: Model calls search("France population 1950")
        → Result: 41.8 million

Step 3: Model calls calculator(67.4 - 41.8)
        → Result: 25.6 million

Step 4: Model synthesizes final answer:
        "France's population grew from 41.8M (1950) to 67.4M (2020),
         an increase of 25.6 million people."
```

**Diagram:**
```
 User Question
      │
      ▼
 ┌─────────────┐
 │ Tool Chain   │  (max 64 steps)
 │ Orchestrator │
 └─────────────┘
      │
      ▼
 ┌──────────┐   ┌──────────┐   ┌──────────┐
 │ Call     │ → │ Receive  │ → │ Decide   │
 │ Tool     │   │ Result   │   │ Continue?│
 └──────────┘   └──────────┘   └──────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
                   More tools needed?      Answer ready
                        │                       │
                        └──→ back to start      ▼
                                          Final answer
```

**Common beginner questions:**
- *Q: Can the model get stuck in an infinite tool loop?* → The chain has a configurable max step limit (1-64), so it can't loop forever.
- *Q: Does the model have access to all tools at once?* → It can choose from the tools available in the configuration, picking which to call based on the current sub-task.

---

## Phase 38: Forgetting Diagnostics

**Definition:** A diagnostic system that measures catastrophic forgetting — how much the model loses previously learned capabilities when it learns new things — using controlled, repeatable capability probes across checkpoints.

**Beginner explanation:**
When you fine-tune or self-learn new behaviors, the model might "forget" older skills — like getting better at math but worse at creative writing. Phase 38 adds a measurement toolkit: a set of fixed test probes across 8 capability areas (math, code, reasoning, factual, vision, video, document, tool-use). After every significant update, the model is retested on all probes, and the results are plotted as forgetting curves over time.

**Why we need it:**
Without measurement, you can't tell if a new training run is silently destroying old capabilities. Forgetting diagnostics make this visible, enabling data-driven decisions about when to stop training or mix in old data.

**Example:**
```
After fine-tuning on advanced math:
  Math probe score:      78% → 92%  (+14%)  ✓
  Code probe score:      81% → 73%  (-8%)   ⚠
  Reasoning probe score:  74% → 76%  (+2%)   ✓

→ We can see that math improved but code regressed.
  The forgetting curve visualizes this over all checkpoints.
```

**Diagram:**
```
 Checkpoint (step N)
      │
      ▼
 ┌─────────────────────────┐
 │ 8 Capability Probes     │
 │ • Math    • Code        │
 │ • Reasoning • Factual   │
 │ • Vision  • Video      │
 │ • Document • Tool-use   │
 └─────────────────────────┘
      │
      ▼
 ┌─────────────────────────┐
 │ ForgettingCurve         │
 │ (score per capability   │
 │  over steps/time)       │
 └─────────────────────────┘
      │
      ▼
 Decision: "Code is regressing — add code data to next replay batch"
```

**Common beginner questions:**
- *Q: Does this slow down training?* → Running all 8 probes takes some time, but it's configurable — you can run fewer probes or test less frequently.
- *Q: Is this only for the self-learning module?* → It integrates with both the self-learning loop and the main training loop via optional config `[forgetting]`.

---

## Phase 39: Max Thinking Mode

**Definition:** A fifth reasoning depth level above High — "Max" mode gives the model a 16,384-token thinking budget, enabling extended multi-step reasoning for very hard problems.

**Beginner explanation:**
v2 had four thinking modes: None (0 tokens), Low (256), Medium (1024), High (4096). Max gives 16,384 tokens of thinking space — that's about 12,000 words of internal reasoning before the model commits to an answer. This is for genuinely hard problems that benefit from exploring many reasoning paths.

**Why we need it:**
Some problems — complex math proofs, multi-hop reasoning, detailed code generation — need more room to think than even High provides. Max mode unlocks deeper reasoning for these cases.

**Example:**
```
Low mode: "42" (256 thinking tokens — basic reasoning)
Medium mode: "Let me calculate... 6×7 = 42" (1024 tokens — shows work)
High mode: "6×7 = 42, but let me verify: 7+7+7+7+7+7+7 = 14+14+14 = 42 ✓"
  (4096 tokens — thorough verification)
Max mode: explores multiple solution strategies, checks each one,
  compares approaches, selects the best, explains reasoning fully
  (16384 tokens — deep chains of thought)
```

**Diagram:**
```
  Thinking Budget:
  None:    0 tokens ─── immediate answer
  Low:    256 tokens ── quick reasoning
  Medium: 1024 tokens ── show work
  High:   4096 tokens ── thorough verification
  Max:   16384 tokens ── deep multi-path exploration
```

**Common beginner questions:**
- *Q: Won't Max mode be very slow?* → Yes — it generates up to 16K thinking tokens, so responses take longer. It should be used selectively for genuinely hard problems.
- *Q: Does Max always give better answers?* → Not always. Some problems don't benefit from more thinking — but for complex multi-step problems, the extra space usually helps significantly.

---

## Phase 40: v3.0.0 Source Release

**Definition:** The final hardening and source-release phase that freezes the v3.0.0 application version, updates all 19 workspace packages to version 3.0.0, and ships the release through GitHub source.

**Beginner explanation:**
Same structure as v2.0.0 release (Phase 28): version bumps, CHANGELOG, CI updates, release notes. No pretrained weights, adapters, or binaries are attached — just the reviewed source tree.

---

# Quick Reference: V3 Phases in One Table

| # | Phase | One-line meaning |
|---|-------|-------------------|
| 29 | Gated DeltaNet (Hybrid Linear Attention) | Replace most attention layers with cheap linear recurrence |
| 30 | DeepSeek Sparse Attention (DSA) | Make remaining full-attention layers attend sparsely |
| 31 | Fine-Grained MoE + Shared Expert | Many small experts + always-active shared experts |
| 32 | Multi-Token Prediction (MTP) | Predict multiple future tokens per position during training |
| 33 | On-Policy Distillation | Student learns from teacher's scores on its own outputs |
| 34 | Native QAT | Simulate quantization during training to retain accuracy |
| 35 | Native Video Understanding | Decode, sample, and reason about video frames |
| 36 | Native Document Understanding | Rasterize and layout-encode PDFs and scanned documents |
| 37 | Long-Horizon Tool-Use Chains | Multi-step tool orchestration within one conversation |
| 38 | Forgetting Diagnostics | Measure capability regression across 8 areas over time |
| 39 | Max Thinking Mode | 16,384-token thinking budget for hard problems |
| 40 | v3.0.0 Source Release | Version bump, CHANGELOG, CI, ship source |

---

# How V3 Fits Into the Full Stack

```
 v1 (Phases 1-13):  Core LLM from scratch
 v2 (Phases 14-28): GPU, vision, MoE, DPO, server, speculative decode
                     ↓
 v3 (Phases 29-40):  Hybrid attention → cheaper long context
                     Sparse attention → faster full-attention layers
                     Fine-grained MoE → better capacity utilization
                     MTP → denser training signal
                     Distillation → smaller models from teachers
                     QAT → quantization-robust weights
                     Video + Document → full multimodal coverage
                     Agent chains → multi-step tool use
                     Forgetting diagnostics → measurable safety
                     Max thinking → deeper reasoning
```

---

*This guide covers Aarambh Studio's completed v3.0.0 roadmap through Phase 40 — extending the from-scratch Rust/Candle LLM with hybrid attention, sparse attention, fine-grained MoE, multi-token prediction, distillation, QAT, video/document understanding, agent chains, forgetting diagnostics, and max thinking.*

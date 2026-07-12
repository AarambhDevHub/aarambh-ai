# Aarambh-AI — Docs

This folder holds the learning material behind Aarambh-AI — a from-scratch decoder-only LLM built in Rust using Candle. If you've ever looked at this repo and wondered *"okay but how does any of this actually work?"*, start here.

These docs aren't API references or code comments. They're written for someone coming in with **zero background in AI/ML** — a beginner who codes but has never touched a neural network before. The goal is that by the end of these guides, you understand not just *what* Aarambh-AI does, but *why* every piece exists and *how* the math underneath it actually works.

---

## What's in this folder

### 1. `aarambh-ai-complete-guide.md`
**The full project walkthrough — every phase, explained.**

This covers all 28 completed phases of Aarambh-AI, from v1.0.0 through the
production v2.0.0 source release:

- **v1 (Phases 1–13):** Tokenizer, Data Pipeline, Neural Network Primitives, Full Model Forward Pass, Custom Kernels (CPU SIMD + GPU prep), Training Loop, Inference Engine + CLI, Thinking Engine, Quantization Stack, Fine-Tuning (LoRA/QLoRA/SFT), GRPO Reinforcement Learning, Safety Layer, Self-Learning.
- **v2 (Phases 14–28):** GPU Scale-Up, Flash Attention CUDA Kernels, Long Context (RoPE scaling), Evaluation Harness, DoRA Fine-Tuning, Vision Encoder + Projector, Vision-Language Training, Vision-Aware Self-Learning, Mixture of Experts, Multi-GPU Training, DPO Preference Tuning, Speculative Decoding, Tool Use / Function Calling, Inference Server, and the v2.0.0 production source release.

Each phase includes a plain-English definition, a beginner explanation, why it's needed, a worked example, and a diagram. Read this first — it's the map of the whole project.

### 2. `aarambh-ai-math-formulas-guide.md`
**The math underneath every phase, explained from zero.**

Once you know *what* each phase does, this file explains the actual formulas doing the work — Dot Product, Matrix Multiplication, Softmax, Scaled Dot-Product Attention, Layer Normalization, GELU Activation, Cross-Entropy Loss, Gradient Descent, Adam Optimizer, RoPE, LoRA Decomposition, Quantization, KL Divergence, and Perplexity.

Every formula comes with a symbol-by-symbol translation (so `Σ`, `∂`, `θ` stop looking scary) and **two fully solved numeric examples** worked by hand, step by step. Read this after the phases guide, whenever you want to understand the actual arithmetic behind a specific phase.

### 3. `ai-ml-dl-dataset-creation-guide.md`
**The foundation underneath everything — terminology and where the training data comes from.**

Two parts:
- **Part 1** untangles the terminology soup: AI, Machine Learning, Deep Learning, Neural Networks, NLP, LLMs, Generative AI, and the three types of ML (supervised, unsupervised, reinforcement) — how they all nest inside each other.
- **Part 2** walks through the practical pipeline of building a real training dataset: web scraping, public dumps (Common Crawl, Wikipedia, Gutenberg), APIs, cleaning, deduplication, filtering, PII removal, formatting into JSONL, train/val/test splitting, and licensing/ethics.

Read this first if you're completely new to AI in general, or read it alongside the other two whenever data-related phases (like Phase 2, Data Pipeline) come up.

### 4. `aarambh-ai-config-toml-guide.md`
**Every field in every `.toml` config file, explained — the practical "turn the dial" layer.**

This walks through the checked-in training and inference configurations in
`configs/` — Tiny/Small/Medium/Large, CUDA, long-context, MoE, distributed,
vision, and smoke configurations — field by field:

- **Top-level settings:** `dataset_path`, `tokenizer_path`, `vocab_size`, `validation_split`, `shuffle`, `resume`, `device`, `dtype`.
- **`[model]` architecture:** `hidden_dim`, `ffn_dim`, `n_layers`, `n_heads`/`n_kv_heads` (Grouped-Query Attention), `max_seq_len`, `rope_theta`, `norm_eps`, `tie_embeddings`.
- **`[model.rope_scaling]` (YaRN):** `method`, `factor`, `original_max_seq_len`, `beta_fast`/`beta_slow`, `attn_factor` — how the long-context configs stretch to 16K tokens.
- **`[[context_schedule]]`:** the staged sequence-length ramp-up used during long-context training.
- **`[vision]`:** `mode`, CLIP config/weights paths, `caption_jsonl`, `image_root`, `projector_hidden_mult`, `max_caption_tokens`, `max_samples`.
- **`[train]` hyperparameters:** `lr`, `batch_size`/`grad_accum_steps`, `warmup_steps`, `min_lr_ratio`, `weight_decay`, Adam's `beta1`/`beta2`/`epsilon`, `clip_grad_norm`, checkpointing, and more — each one tied back to the exact formula it came from in the math guide.

Read this whenever you're about to write a new training config, or want to understand exactly what a specific field in an existing one actually does.

---

## Suggested reading order

If you're starting from zero:

```
ai-ml-dl-dataset-creation-guide.md   →  understand the terminology + where data comes from
            │
            ▼
aarambh-ai-complete-guide.md         →  understand what Aarambh-AI actually builds, phase by phase
            │
            ▼
aarambh-ai-math-formulas-guide.md    →  understand the exact math powering each phase
            │
            ▼
aarambh-ai-config-toml-guide.md      →  understand how to actually configure and run a training job
```

If you already know the basics and just want the project-specific details, jump straight to `aarambh-ai-complete-guide.md` and use the other three as reference whenever a term, formula, or config field is unfamiliar.

---

## Who this is for

- Anyone reading the Aarambh-AI codebase for the first time and wondering what a given module actually does.
- Contributors who want to understand a phase deeply enough to help extend it.
- Future-me, six months from now, who forgot why a formula was written a certain way.

No prior ML background is assumed anywhere in these three files. If something is still unclear after reading, that's a gap in the doc, not a gap in you — feel free to open an issue.

---

## Keeping these docs updated

The v2 roadmap is complete. Future changes must update the matching guide,
configuration reference, architecture section, and changelog in the same pull request.

---

## Support Aarambh-AI

If these docs or the project itself helped you, consider supporting the work:

- ☕ [Buy Me a Coffee](https://www.buymeacoffee.com/aarambhdevhub)
- 💖 [GitHub Sponsors](https://github.com/sponsors/aarambh-darshan)
- 🎓 [Topmate](https://topmate.io/darshan_vichhi) — 1-on-1 mentoring and paid sessions
- 🪙 [Razorpay](https://razorpay.me/@aarambhdevhub) — for India-based support

Every bit helps keep this project — and the free educational content around it — going.

---

*Part of the Aarambh Dev Hub ecosystem. Built with Rust, one phase at a time.*

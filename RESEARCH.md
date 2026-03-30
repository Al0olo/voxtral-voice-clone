# Research Findings

This document consolidates research on TTS LLM fine-tuning, codec-LLM architectures,
and Arabic TTS production systems. Generated from analysis of 30+ papers and
implementations across the field.

## Table of Contents

1. [Why LoRA Broke Our Model](#why-lora-broke-our-model)
2. [Production Fine-Tuning Strategy](#production-fine-tuning-strategy)
3. [Similar Architectures](#similar-architectures)
4. [Arabic TTS Production Pipeline](#arabic-tts-production-pipeline)
5. [Key Papers](#key-papers)

---

## Why LoRA Broke Our Model

The core issue is **autoregressive error cascading unique to TTS**. Unlike text LLMs
where a small logit perturbation causes a synonym substitution, in TTS codec-LLMs it
causes wrong semantic tokens which cascade into garbled phonemes within 10-20 tokens.

Three reinforcing mechanisms:

1. **Perturbation magnitude vs. voice distinction**: Preset voices differ by only 1-3%
   cosine similarity. LoRA rank 8 across 182 weight matrices introduces cumulative
   perturbations larger than inter-voice differences, causing voice identity collapse.

2. **Strong pretraining traps**: Research (arXiv:2602.02855) shows strong pretraining
   can slow LoRA convergence by trapping optimization in a "correlated search phase."
   Ministral-3B's compact distilled representations are especially fragile.

3. **Confirmed across the field**: Fish Speech (GitHub #1163, #1120), Qwen3-TTS
   (GitHub #39), all report garbled output from LoRA fine-tuning with similar failures.

---

## Production Fine-Tuning Strategy

Ordered by risk (lowest first):

### Tier 1: Codec Encoder (149M params) -- Current Work
Train the encoder on multilingual data including Arabic. Frozen LLM, frozen decoder.
This is our V3 training with speaker verification loss, gradient balancer, native 24kHz data.

### Tier 2: Audio Token Embedding (28M params, 0.8% of model)
Re-learn the codec-to-LLM mapping. Supported by SpeechMapper, LiSTEN, UniAudio 1.5
research. The LLM has already learned speech generation structure; we just adapt the
translation layer between discrete codes and continuous embeddings.

### Tier 3: Selective Layer Fine-Tuning (~8% of params)
Based on CSP-FT (arXiv:2501.14273): fine-tune only 2 layers (most and least
contributing). Matches full fine-tuning performance with 2x speedup and significantly
mitigates catastrophic forgetting. For Voxtral: layers 0-1 and 24-25.

### Tier 4: DPO Post-Training
Rejection sampling with WER + speaker similarity + UTMOS scoring. Arabic-specific DPO
with diacritization accuracy as additional criterion. Very low LR (8e-8), 1 epoch max.

### What NOT to Do
- LoRA rank > 4 on the full LLM -- proven to break the model
- Full fine-tuning of all 3.4B params -- catastrophic forgetting risk
- Skip the diacritization pipeline for Arabic -- 4.3x WER gap vs English

---

## Similar Architectures

| System | Backbone | Semantic Gen | Acoustic Gen | Frame Rate | Codec |
|--------|----------|-------------|-------------|------------|-------|
| **Voxtral** | Ministral-3B (3.4B) | AR (CE loss) | Flow-matching (390M) | 12.5 Hz | VQ-FSQ |
| **VALL-E 2** | Custom (~1B) | AR | NAR (grouped) | 75 Hz | EnCodec RVQ |
| **SoundStorm** | Conformer | N/A (input) | Parallel MaskGIT | 50 Hz | SoundStream |
| **CosyVoice 2** | Qwen2.5-0.5B | AR | Chunk-aware FM | 25 Hz | FSQ |
| **F5-TTS** | DiT | N/A | FM (continuous) | N/A | None |
| **MaskGCT** | Custom | Masked parallel | Masked parallel | Variable | Custom |
| **Bark** | 3x GPT (80M) | AR | AR + NAR | 75 Hz | EnCodec |

CosyVoice 2 is architecturally closest to Voxtral (both use LLM + FM + FSQ).

---

## Arabic TTS Production Pipeline

### Voxtral's Arabic Performance (from paper)

| Metric | Voxtral | ElevenLabs v3 | Gap |
|--------|---------|---------------|-----|
| WER | 2.68% | 3.67% | Voxtral wins |
| UTMOS | 3.07 | 2.50 | Voxtral wins |
| Speaker Sim | 0.746 | 0.546 | Voxtral wins |

Voxtral already outperforms ElevenLabs on Arabic. But English WER is 0.63% --
the 4.3x gap indicates substantial room for Arabic improvement.

### Recommended Production Architecture

```
Input Text
  -> Language Detection (Arabic/English segments)
  -> Arabic segments: Diacritization (Sadeed/CATT)
  -> English segments: pass-through
  -> Merge normalized text
  -> Voice Reference -> Codec Encoder -> Voice Tokens
  -> Voxtral LLM -> Semantic Tokens
  -> Flow-Matching Transformer -> Acoustic Tokens
  -> Codec Decoder -> 24kHz Audio
  -> Quality Gate: ASR re-transcribe (WER check) + Speaker Verify (ECAPA sim >= 0.70)
```

### Phased Roadmap

| Phase | Weeks | Focus | Expected Impact |
|-------|-------|-------|----------------|
| 1 | 1-4 | Diacritization + baseline | Arabic WER 2.7% -> 1.8% |
| 2 | 4-8 | Codec encoder + Arabic data | Speaker sim 0.746 -> 0.78+ |
| 3 | 8-14 | LLM LoRA + Arabic DPO | Arabic WER < 1.5%, UTMOS > 3.5 |
| 4 | 14-20 | Multi-dialect + code-switching | Production across MSA + 3-4 dialects |
| 5 | Ongoing | Production hardening | Edge deployment, compliance |

### MENA Compliance Notes

- Saudi PDPL: voice data is biometric, requires explicit consent, data localization
- UAE PDPL: less prescriptive but still requires consent
- Deploy on regional GPU clusters (AWS Bahrain, Azure UAE) for data sovereignty
- Build voice consent and watermarking from day one

### License Warning

Voxtral's CC BY-NC 4.0 license **prohibits commercial use** of the open weights.
For production at a $2B company, you need either:
1. Commercial license from Mistral (contact sales)
2. Use Mistral's La Plateforme API
3. Train your own model from scratch using the research findings

---

## Key Papers

| Paper | ArXiv | Key Contribution |
|-------|-------|-----------------|
| Voxtral TTS | 2603.25551 | Hybrid VQ-FSQ codec + AR semantic + FM acoustic |
| VALL-E 2 | 2406.05370 | Human parity zero-shot TTS |
| CosyVoice 2 | 2412.10117 | LLM + FSQ + chunk-aware flow-matching |
| MaskGCT | 2409.00750 | Masked generative codec transformer |
| CSP-FT | 2501.14273 | Partial layer fine-tuning (2 layers = full FT quality) |
| EnCodec | 2210.13438 | Gradient-norm loss balancer, RVQ codec |
| UtterTune | 2508.09767 | LoRA for pronunciation control |
| When Fine-Tuning Fails | 2603.10904 | Data diversity critical for TTS fine-tuning |
| SpeechMapper | 2601.20417 | Projection-only training for audio-LLM alignment |
| MR-FlowDPO | 2512.10264 | DPO adapted for flow-matching |
| Habibi-TTS | GitHub | 12+ Arabic dialect unified TTS |
| NileTTS | ACL 2026 | Egyptian Arabic TTS pipeline |
| SawtArabi | Interspeech 2025 | Arabic-English code-switch TTS benchmark |

# Research: Fine-Tuning TTS LLM Backbones

Production R&D reference for Voxtral-4B-TTS (Ministral-3B backbone) adaptation.
Compiled March 30, 2026.

---

## 1. Why LoRA Broke Our Model

### The Observed Failure

LoRA rank 8 applied to all 26 layers of the 3.4B Ministral backbone produced garbled
speech for ALL voices, including presets. Rank 2 partially worked but still damaged
preset voice quality. This section synthesizes the theoretical and empirical explanations.

### Root Cause: Subspace Interference in Autoregressive Token Generation

TTS codec-LLMs are fundamentally different from text LLMs. The model must maintain
**extremely precise probability distributions** over 131,072 text tokens + 9,088 audio
tokens, where even small perturbations to the logit space cause cascading errors in
autoregressive generation.

**Key mechanism**: LoRA adds a low-rank perturbation ΔW = BA to every targeted weight
matrix. In a text LLM, small logit perturbations cause minor word substitutions. In a
TTS codec-LLM, small logit perturbations cause:

1. **Wrong semantic tokens** → phoneme substitutions → garbled words
2. **Wrong acoustic tokens** → timbre/prosody corruption → robotic/noisy speech
3. **Autoregressive error accumulation** → each bad token conditions the next,
   creating a cascade that produces complete gibberish within 10-20 tokens

### Theoretical Support

**"When pre-training hurts LoRA fine-tuning"** (arXiv:2602.02855, Nwemadji et al., Feb 2026)
mathematically demonstrates that strong pre-training can *slow down* LoRA convergence.
The key insight: LoRA's low-rank update subspace must align with the task-relevant
gradient subspace. When the pretrained model is already very good (as Ministral-3B is
for speech), the gradient signal for adaptation is weak, and the LoRA matrices struggle
to find the right direction—potentially spending many steps in a "correlated search phase"
where they perturb the model without improving the target task.

**Forgetting geometry** (arXiv:2603.09684): Catastrophic forgetting in LoRA follows
F = α(1 − cos²θ_min) + β, where θ_min is the minimum principal angle between the
fine-tuning gradient subspace and the pretrained knowledge subspace. When they are
nearly orthogonal (as speech generation vs. new voice adaptation likely is), forgetting
is severe and *rank-independent*. Higher rank ≠ less forgetting; it just means
more parameters to misalign.

### Empirical Confirmation from Other TTS Systems

| System | Issue | Reference |
|--------|-------|-----------|
| **Fish Speech** | LoRA setup destroys pretrained weights due to init order bug; training loss starts at ~22 instead of ~8 | GitHub #1163 |
| **Fish Speech** | Post-GRPO models "almost unable to be finetuned with LoRA" | GitHub #1120 |
| **Qwen3-TTS 0.6B** | LoRA fine-tuned model outputs pure noise despite decreasing training loss; missing text projection step in fine-tuning code | GitHub #39 |
| **Qwen3-TTS** | LR 2e-5 causes noise; 2e-6 partially works; 0.6B model remains broken even with fixes | GitHub #39 |
| **Fish Speech** | After LoRA merge, noise in inference; proposed fix: selective target_modules instead of all layers | GitHub #1230 |

### Why Rank Matters (and Why Rank 8 Is Worse Than Rank 2)

- **Rank 2**: Adds a 2-dimensional perturbation subspace. The model's 3072-dim hidden
  space has ~3070 dimensions preserved exactly. The perturbation is small enough that
  most of the pretrained autoregressive generation behavior survives.

- **Rank 8**: Adds an 8-dimensional perturbation across all 26 layers. With attention
  (q/k/v/o) + FFN (gate/up/down) = 7 matrices per layer, that's 26×7 = 182 weight
  matrices each perturbed by rank-8. The cumulative perturbation is massive—it shifts
  the entire residual stream geometry enough to destroy the delicate token probability
  distributions the model learned for speech.

### The Preset Voice Problem

Voxtral's preset voices are defined by fixed embedding sequences hardcoded into the
model. LoRA modifies the weight matrices that *process* these embeddings, so even
unmodified preset embeddings produce different hidden states, attention patterns, and
output logits. The model loses the learned association between these specific embedding
patterns and their corresponding voices.

**Critical number**: Preset-to-preset cosine similarity is 0.97-0.99. The LLM
distinguishes voices based on 1-3% cosine differences in embedding space. LoRA rank 8
introduces perturbations that are *larger* than these inter-voice differences, causing
voice identity collapse.

### Practical Recommendations

1. **Do NOT apply LoRA to all layers for TTS codec-LLMs.** This is the single most
   important takeaway. Unlike text LLMs, TTS models have zero tolerance for
   distribution shift in the output space.

2. **If LoRA is necessary**, use:
   - Rank 1-2 maximum
   - Target only specific layers (CSP-FT approach: highest + lowest contributing layers)
   - Target only specific module types (e.g., only attention Q/V, not FFN)
   - Use OPLoRA (orthogonal projection) to constrain updates to the orthogonal
     complement of pretrained singular directions (arXiv, AAAI 2025)

3. **Prefer non-LoRA adaptation strategies** (see Section 5):
   - Embedding-only fine-tuning (28M params, 0.8% of model)
   - CSP-FT: ~8% of params, 2x faster training
   - DPO post-training (Voxtral's own approach)
   - Freeze everything, adapt only codec encoder

---

## 2. Papers on TTS LLM Fine-Tuning

### 2.1 VALL-E / VALL-E 2 (Microsoft)

**VALL-E** (arXiv:2301.02111, Jan 2023)
- First neural codec language model for TTS
- Treats TTS as conditional language modeling over EnCodec tokens
- AR model for first codebook, NAR model for remaining codebooks
- 3-second reference audio for zero-shot voice cloning
- No fine-tuning needed for new voices (pure in-context learning)

**VALL-E 2** (arXiv:2406.05370, Jun 2024)
- Achieves human parity in zero-shot TTS (first system to do so)
- **Repetition Aware Sampling**: Prevents infinite loops by tracking token repetition
- **Grouped Code Modeling**: Organizes codec codes into groups for shorter sequences
- Trained on 80K hours of LibriLight
- Key lesson: Scaling data + model size > fine-tuning for voice quality

**Fine-tuning approach**: VALL-E does NOT fine-tune for new voices. It relies entirely
on in-context learning (voice conditioning tokens). This is architecturally similar
to Voxtral's approach.

### 2.2 Bark (Suno AI)

**Architecture**: Three-stage pipeline:
1. Text → semantic tokens (GPT-based)
2. Semantic → coarse acoustic tokens (GPT-based)
3. Coarse → fine acoustic tokens (non-autoregressive)

**Fine-tuning**: Community-driven (not officially supported by Suno).
- Semantic token generation for voice clones discovered by community member gitmylo
- Fine-tuning involves training on extracted semantic tokens from target speakers
- No published paper; techniques are empirical/community-developed

### 2.3 Tortoise-TTS (James Betker)

**Architecture**: Autoregressive + diffusion decoder hybrid
- GPT-2 architecture generates MEL tokens
- Diffusion model converts to high-quality spectrograms
- Multi-voice conditioning via CLVP (CLIP-like Voice Pairing)

**Fine-tuning for new languages** (Martin Thissen, Medium 2023):
1. Train a new tokenizer for the target language
2. Fine-tune the AR model on target language data
3. Modify inference code for language-specific handling
4. Upload adapted weights to HuggingFace

**Key insight**: Tortoise's AR model can be fine-tuned for new languages, but requires
retraining the tokenizer—not just the model weights.

### 2.4 XTTS / XTTS-v2 (Coqui AI)

**Paper**: arXiv:2406.04904 (Jun 2024)
**Architecture**: Extension of Tortoise with multilingual modifications
- GPT-based AR model → VQ-VAE decoder
- 17 languages supported (including Arabic)
- 6-second reference audio for voice cloning
- 24kHz output

**Fine-tuning approach**:
- Full fine-tuning of the GPT component on target language data
- ~100+ hours of audio recommended for new language adaptation
- 8+ hours training on single A100-40GB
- HiFi-GAN vocoder typically frozen during adaptation
- Community demonstrated Arabic fine-tuning (NileTTS, Egyptian Arabic, 38h data)

**Relevance to Voxtral**: XTTS proved that GPT-based TTS models CAN be fine-tuned for
new languages without destroying existing capabilities, given sufficient data diversity.

### 2.5 Fish Speech (Fish Audio)

**Architecture**: LLAMA-based codec language model
- VQGAN codec for semantic token extraction
- LLAMA backbone for text → semantic token generation
- Supports 10+ languages including Arabic
- Trained on 1M+ hours

**LoRA fine-tuning** (official):
1. Extract semantic tokens via VQGAN
2. Pack dataset into protobuf format
3. LoRA fine-tune LLAMA with config `r_8_alpha_16`
4. Merge LoRA weights before inference

**Known issues** (critical for our project):
- LoRA setup order bug destroys pretrained weights (GitHub #1163)
- Selective target_modules proposed to reduce damage (GitHub #1230)
- Post-GRPO models cannot be LoRA fine-tuned effectively
- Full parameter fine-tuning causes quality degradation on longer utterances

### 2.6 Orpheus TTS (Canopy AI, 3B)

**Architecture**: LLaMA-3B backbone fine-tuned for speech token generation.

**LoRA configuration** (official, via Unsloth):
- Rank: 64 (much higher than typical, possibly because full model was trained
  from scratch on speech data rather than adapted from text LLM)
- Alpha: 64 (equal to rank)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- RSLoRA enabled for stability
- Optionally trains LM head and embedding tokens
- Trainable on consumer GPUs (RTX 4090, 24GB)

**Key insight**: Orpheus uses high rank (64) successfully because its backbone was
fully trained on speech from the start—unlike Voxtral where the backbone was pretrained
on text (Ministral-3B) and then adapted. The text-pretrained model has stronger
"text knowledge" that LoRA can more easily disrupt.

**Full fine-tuning**: Also supported at 8-bit precision (not bf16).

### 2.7 MaskGCT (Amphion)

**Paper**: arXiv:2409.00750 (Sep 2024), accepted ICLR 2025
**Architecture**: Fully non-autoregressive, mask-and-predict paradigm
- Stage 1: Text → semantic tokens (masked prediction from SSL embeddings)
- Stage 2: Semantic → acoustic tokens (masked prediction conditioned on semantic)
- No explicit text-speech alignment needed during training
- No phone-level duration prediction needed
- Parallel token generation at inference

**Fine-tuning**: Not the primary design goal. MaskGCT emphasizes zero-shot
in-context learning from 100K hours of in-the-wild speech data. The non-autoregressive
design avoids the cascading error problem that makes LoRA dangerous for AR models.

**Relevance**: MaskGCT's success with 100K hours suggests that scaling data is more
important than fine-tuning for zero-shot TTS quality.

### 2.8 CosyVoice / CosyVoice 2 / CosyVoice 3 (Alibaba)

**CosyVoice 2** (arXiv:2412.10117, Dec 2024):
- Text-Speech LM + Flow Matching + HiFi-GAN vocoder
- FSQ codec for improved codebook utilization
- Chunk-aware causal flow matching for streaming
- Pre-trained LLM backbone (can use any LLM)

**CosyVoice 2-EU** (2025, hi-paris): **Most relevant paper for our use case**
- Systematic component-level adaptation of CosyVoice 2 for French/German
- Controlled ablation across 6 data budgets (50-1500h per language)
- **Key findings**:
  - Text-Speech LM fine-tuning: 83-91% WER reduction with 250-500h data
  - Flow Matching: Secondary gains for prosody/rhythm at larger data scales
  - HiFi-GAN vocoder: Can remain frozen
  - Bilingual training (FR+DE joint) improves speaker similarity
  - LoRA adaptation supported (memory-efficient, backbone-agnostic)

**CosyVoice 3** (arXiv:2505.17589, May 2025):
- Scaled from 0.5B to 1.5B parameters
- 10K → 1M hours training data, 9 languages + 18 Chinese dialects
- Novel speech tokenizer via supervised multi-task training
- **Differentiable reward model for post-training** (DPO-like)

### 2.9 "When Fine-Tuning Fails" (arXiv:2603.10904, Mar 2026)

**Authors**: Anupam Purwar, Aditya Choudhary
**System**: Qwen-0.5B based TTS

**Key findings**:
- LoRA fine-tuning of the LLM backbone consistently outperforms frozen base:
  - DNS-MOS improvements up to +0.42 points
  - Increased voice similarity
  - SNR gains up to 34%
- **BUT**: Only for speakers with sufficiently diverse training data
- Speakers with low acoustic variability data get NO benefit from fine-tuning
- Fine-tuning success is strongly governed by training data characteristics:
  acoustic energy variability and perceptual quality diversity

**Practical implication**: For Arabic voice cloning, we need diverse acoustic data
per speaker (different emotions, speaking rates, recording conditions), not just
large amounts of monotonic studio recordings.

---

## 3. Arabic TTS Fine-Tuning

### 3.1 "More Data, Fewer Diacritics: Scaling Arabic TTS" (arXiv:2603.01622, Mar 2026)

**Key contribution**: Built a 4,000-hour Arabic TTS dataset using automated pipeline:
- Voice Activity Detection → Speech Recognition → Automatic Diacritization → Noise Filtering

**Findings on diacritics**:
- Diacritized data produces better models at all scales
- BUT increased training data substantially compensates for missing diacritics
- At 4,000 hours, the gap between diacritized and undiacritized training narrows significantly
- Plan to release a public Arabic TTS model that works WITHOUT diacritization

**Implication for Voxtral**: If we can gather 1,000+ hours of Arabic data, we may not
need perfect diacritization. Automatic diacritizers like Sadeed + large data volume
can compensate.

### 3.2 Sadeed: Arabic Diacritization SLM (arXiv:2504.21635, 2025)

- Fine-tuned decoder-only LM (Kuwain 1.5B) for diacritization
- Competitive with larger models using modest compute
- Introduces SadeedDiac-25 benchmark for fair evaluation across genres
- Directly applicable as a preprocessing step for Arabic TTS data

### 3.3 CATT-Whisper: Multimodal Diacritic Restoration

- Combines text + speech representations for dialectal Arabic diacritization
- Early fusion vs. cross-attention fusion strategies
- WER 0.55, CER 0.13 on test sets
- Useful for dialects where text-only diacritization fails

### 3.4 NileTTS: Egyptian Arabic TTS via LLM Synthetic Data (ACL 2026)

**Pipeline**: LLM-to-Speech (aclanthology.org/2026.abjadnlp-1.6)
- Uses LLMs to generate Egyptian Arabic text content
- Converts to speech via audio synthesis
- Transcribes and verifies for quality
- Fine-tunes **XTTS v2** on 38 hours of synthetic data
- First publicly available Egyptian Arabic TTS dataset + open-source model

**Relevance**: Demonstrates that XTTS-style models CAN be fine-tuned for Arabic dialects
with relatively small amounts of data (38h), using synthetic data augmentation.

### 3.5 Saudi Dialect LoRA Fine-tuning (arXiv:2508.13525, Aug 2025)

- ALLaM-7B-Instruct fine-tuned with LoRA for Saudi dialects (Hijazi, Najdi)
- 5,466 synthetic instruction-response pairs
- Saudi dialect rate: 47.97% → 84.21%
- MSA leakage: 32.63% → 6.21%
- Text-only LLM, but demonstrates dialect adaptation via LoRA is viable

### 3.6 Arabic Prosody and Oversmoothing

- FastPitch-based Arabic TTS with cepstral-domain analysis of mel oversmoothing
- Adversarial training to reduce oversmoothing artifacts
- Multi-speaker synthesis using XTTSv2-generated synthetic voices
- Arabic prosody modeling requires attention to:
  - Gemination (shadda) timing
  - Emphatic consonants and their coarticulation effects
  - Iʿrāb (case endings) for formal Arabic
  - Dialectal intonation patterns (MSA vs. Egyptian vs. Gulf vs. Levantine)

### 3.7 Practical Approach for Arabic Voxtral Adaptation

Given the research, the recommended pipeline for Arabic:

1. **Text preprocessing**: Arabic text → Sadeed diacritizer → diacritized input
2. **Data**: Target 500-1000h diverse Arabic audio (mixed MSA + target dialect)
3. **Adaptation strategy** (in priority order):
   a. Train codec encoder on Arabic speech data (codec-only, no LLM changes)
   b. Fine-tune audio_token_embedding (28M params) on Arabic voice data
   c. If needed: CSP-FT on 2 LLM layers (~8% of params)
   d. DPO post-training with Arabic preference data
4. **Avoid**: Full LoRA on all layers, high learning rates, small/monotonic datasets

---

## 4. The Ministral-3B Backbone

### 4.1 Ministral 3 Technical Report (arXiv:2601.08584, Jan 2026)

**Architecture**: Decoder-only transformer
- 3.4B language model + 0.4B vision encoder
- 32 attention heads, 8 KV heads, head dim 128
- Hidden dim 3072, FFN dim 9216 (SwiGLU)
- 256K context window
- RoPE with theta = 1,000,000

**Training method**: **Cascade Distillation** from Mistral Small 3.1 (24B)
- Iterative pruning + distillation from parent model
- Trained on 1-3 trillion tokens (vs. 15-36T for Llama 3, Qwen 3)
- Much less training data than competitors, relying on distilled knowledge

**Multilingual support**: 11 languages
- English, French, Spanish, German, Italian, Portuguese, Dutch
- Chinese, Japanese, Korean, **Arabic**

**Post-training**: SFT → Online DPO (ODPO)

**Key implications for Voxtral**:
1. Arabic is a first-class supported language in the backbone. The tokenizer has
   Arabic tokens and the model has seen Arabic text during pretraining.
2. The model was distilled (not trained from scratch), meaning its internal
   representations are compact and may be MORE fragile to perturbation than
   models trained on larger data (the distilled knowledge is "compressed").
3. The 256K context window means the model can handle long voice conditioning
   sequences without positional encoding issues.

### 4.2 Voxtral's Use of Ministral-3B

From the Voxtral paper (arXiv:2603.25551):
- Ministral-3B initialized the LLM backbone
- Text embeddings frozen during LLM training phase
- LLM was trained to predict semantic tokens autoregressively
- DPO post-training applied to the hybrid discrete-continuous setting:
  - Standard DPO for semantic token generation
  - Flow-based DPO for acoustic prediction
  - Rejection sampling for preference data (WER, speaker similarity, UTMOS-v2)
- Achieves 68.4% preference rate over ElevenLabs Flash v2.5 in voice cloning

---

## 5. Production-Grade TTS Fine-Tuning Strategies

### 5.1 Company Approaches (Published/Inferred)

| Company | Architecture | Voice Adaptation | Published? |
|---------|-------------|-----------------|------------|
| **Mistral (Voxtral)** | Ministral-3B + flow matching + VQ-FSQ codec | In-context (3s ref) + DPO post-training | Yes (arXiv:2603.25551) |
| **ElevenLabs** | Transformer/GAN hybrid (unpublished) | IVC: inference-time conditioning (30s); PVC: per-voice fine-tuning (30min audio) | No paper |
| **Cartesia** | SSM (State Space Model, Mamba-family) | API-based voice design + cloning | No paper (SSM architecture blog only) |
| **PlayHT** | Diffusion-based (PlayDiffusion) + AR | Voice cloning from 30s audio | Partial (PlayDiffusion on HuggingFace) |
| **WellSaid Labs** | LLM context layer + custom synthesis | Professional voice actors, per-voice training, 96kHz | No paper |
| **Alibaba (CosyVoice)** | LLM + Flow Matching + HiFi-GAN | Zero-shot + component-level fine-tuning | Yes (arXiv:2412.10117, 2505.17589) |

### 5.2 Proven Fine-Tuning Strategies (Ordered by Risk)

#### Strategy A: Codec Encoder Only (Risk: Lowest)
**What**: Train only the codec encoder to produce codes the frozen LLM accepts.
**Evidence**: Your own project (Voxtral voice clone) demonstrates this works.
**Params**: ~149M (4.4% of total model)
**Pro**: Zero risk to LLM, preserves all preset voices, codec is self-contained
**Con**: Cannot teach the LLM new phoneme sequences (e.g., Arabic-specific sounds)

#### Strategy B: Audio Token Embedding Only (Risk: Low)
**What**: Fine-tune mm_audio_embeddings (9088×3072) to better map codec tokens to LLM space.
**Evidence**: Analogous to tokenizer-aware cross-lingual adaptation (ACL 2026):
embedding relearning achieves up to 20% improvement across 96 languages while
training only the embedding layer, with minimal catastrophic forgetting.
**Params**: ~28M (0.8% of total model)
**Pro**: Adjusts the interface between codec and LLM without touching the LLM
**Con**: Cannot change the LLM's generation behavior, only how it receives audio tokens

#### Strategy C: Characteristic-Specific Partial Fine-Tuning (Risk: Medium-Low)
**What**: Fine-tune only 2 LLM layers (~8% of params) selected by contribution analysis.
**Paper**: arXiv:2501.14273 (CSP-FT)
**Method**: Analyze each layer's contribution to speaker/emotion characteristics,
fine-tune highest + lowest contributing layers.
**Pro**: 2x faster than full fine-tuning, matches or beats full FT quality
**Con**: Requires layer contribution analysis; layer selection may differ for
Voxtral's architecture

#### Strategy D: Selective LoRA (Risk: Medium)
**What**: LoRA on carefully selected modules only.
**Evidence**: UtterTune (arXiv:2508.09767) uses LoRA on Q/K/V/O projections only
in CosyVoice 2, improving accent correctness from 0.472 to 0.975.
**Config**: Rank 2-4, target only attention projections, NOT FFN
**Pro**: Memory-efficient, well-supported by frameworks
**Con**: Still risks perturbation damage; requires careful module selection

#### Strategy E: DPO Post-Training (Risk: Medium)
**What**: Preference optimization after initial model training.
**Evidence**: Voxtral itself uses this. MPO (arXiv:2509.00685) extends to
multidimensional preferences. ARDM-DPO for diffusion models.
**Pro**: Directly optimizes for human-perceived quality metrics
**Con**: Requires expensive preference data generation (rejection sampling)
**Papers**:
- MPO (arXiv:2509.00685): Multidimensional preference optimization for TTS
- ARDM-DPO (arXiv:2509.18928): DPO for autoregressive diffusion TTS
- LPO (arXiv:2508.14947): Linear PO, avoids DPO overfitting/collapse
- ADPO (arXiv:2602.09533): Autoregressive-aware DPO
- Preference Alignment for LM-TTS (arXiv:2409.12403): Empirical DPO evaluation

#### Strategy F: Component-Level Fine-Tuning (Risk: Medium-High)
**What**: Fine-tune Text-Speech LM (dominant) + optionally Flow Matching.
**Evidence**: CosyVoice 2-EU shows LM fine-tuning gives 83-91% WER reduction
with 250-500h data per language. Vocoder stays frozen.
**Pro**: Well-studied, systematic, with clear ablation evidence
**Con**: Requires 250+ hours of target language data; higher risk than embedding-only

#### Strategy G: Full Fine-Tuning (Risk: Highest)
**What**: Update all LLM parameters.
**Evidence**: Orpheus TTS does this at 8-bit precision. Works when backbone was
purpose-trained for speech. Risky when backbone was text-pretrained (like Ministral).
**Pro**: Maximum expressivity
**Con**: Catastrophic forgetting of text knowledge, requires large data, expensive

### 5.3 Recommended Production Pipeline for Arabic Voxtral

Based on all evidence, the recommended staged approach:

```
Phase 1: Codec Encoder (current, ~149M params)
├── Train on Arabic + multilingual 24kHz data
├── Losses: mel + speaker verification + ASR distillation + diversity
├── Validate: semantic code utilization, speaker similarity, mel loss
└── Goal: Arabic codes that the frozen LLM accepts

Phase 2: Audio Token Embedding (28M params, 0.8% of model)
├── Fine-tune mm_audio_embeddings on Arabic voice data
├── Use frozen LLM as evaluation: does generation quality improve?
├── Preserve preset embeddings (multi-task: Arabic + preset voice data)
└── Goal: Better mapping from Arabic codec codes to LLM embedding space

Phase 3: Optional - CSP-FT or Selective LoRA (~8% of params)
├── Only if Phase 1+2 insufficient for Arabic prosody/diacritics
├── CSP-FT: analyze layer contributions, fine-tune 2 layers
├── OR: LoRA rank 2 on attention-only (Q/V), NOT FFN
├── Mixed training: Arabic + preset voice data to prevent forgetting
└── Goal: Arabic-specific prosody modeling

Phase 4: DPO Post-Training
├── Generate N Arabic utterances per prompt
├── Rank by: WER (Arabic ASR) + speaker similarity + UTMOS
├── Train preference model on (preferred, rejected) pairs
├── Apply Voxtral-style hybrid DPO (semantic + flow-based)
└── Goal: Human-preference-aligned Arabic quality
```

### 5.4 Data Requirements (Derived from Literature)

| Data Volume | Expected Capability | Evidence |
|------------|-------------------|----------|
| 50-100h | Basic intelligibility, limited speaker diversity | CosyVoice 2-EU |
| 250-500h | Good WER reduction (83-91%), acceptable quality | CosyVoice 2-EU |
| 500-1000h | Robust quality, diacritics partially compensated | arXiv:2603.01622 |
| 1000-4000h | Near-native quality, diacritics largely compensated | arXiv:2603.01622 |
| 3s per voice | Zero-shot voice cloning (Voxtral native capability) | Voxtral paper |
| 30min per voice | High-quality voice cloning (ElevenLabs PVC-style) | ElevenLabs docs |

---

## 6. Additional Key Papers

### Parameter-Efficient Fine-Tuning Theory

| Paper | arXiv | Key Insight |
|-------|-------|-------------|
| LoRA Learns Less and Forgets Less | 2405.09673 | Low-rank LoRA better preserves OOD performance than full FT |
| OPLoRA (AAAI 2025) | — | Orthogonal projection constrains LoRA to preserve top-k singular triples |
| When pre-training hurts LoRA | 2602.02855 | Strong pretraining can slow LoRA convergence via prolonged search phase |
| On Catastrophic Forgetting in LoRA | 2603.09684 | Forgetting follows geometric law based on task subspace angles |
| Modular Fine-tuning for Translation | ACL 2026 | Per-language LoRA adapters, merged for multi-directional transfer |

### TTS-Specific

| Paper | arXiv | Key Insight |
|-------|-------|-------------|
| When Fine-Tuning Fails (TTS) | 2603.10904 | Data diversity per speaker determines fine-tuning success |
| CSP-FT | 2501.14273 | 2-layer fine-tuning (~8% params) matches full FT |
| UtterTune | 2508.09767 | LoRA on Q/K/V/O for pronunciation control, Japanese CosyVoice 2 |
| CosyVoice 2 | 2412.10117 | FSQ codec + streaming + pre-trained LLM backbone |
| CosyVoice 3 | 2505.17589 | 1.5B params, 1M hours, differentiable reward for post-training |
| MaskGCT | 2409.00750 | Non-AR mask-predict avoids cascading error problem |
| VALL-E 2 | 2406.05370 | Human parity zero-shot TTS via repetition-aware sampling |
| Voxtral TTS | 2603.25551 | Hybrid VQ-FSQ codec, DPO for discrete+continuous |
| MPO for TTS | 2509.00685 | Multidimensional DPO for intelligibility+similarity+prosody |
| Tokenizer-Aware Cross-Lingual | ACL 2026 | Embedding relearning for 96 languages, minimal forgetting |
| Scaling Arabic TTS | 2603.01622 | 4Kh Arabic data compensates for missing diacritics |

---

## 7. Key Takeaways

1. **LoRA on all layers is toxic for TTS codec-LLMs.** The autoregressive generation
   of discrete speech tokens has near-zero tolerance for distribution shift. This is
   confirmed across Fish Speech, Qwen3-TTS, and our own Voxtral experiments.

2. **The safest adaptation path is encoder → embedding → selective layers → DPO.**
   Each stage adds risk but also expressivity. Start at the periphery (codec, embeddings)
   and work inward (LLM layers) only as needed.

3. **Data diversity > data volume for fine-tuning success.** Per arXiv:2603.10904,
   acoustic variability in training data is the strongest predictor of fine-tuning
   success, not total hours.

4. **Arabic is already supported by the backbone.** Ministral-3B was trained on Arabic
   text. The challenge is Arabic *speech* tokens, not Arabic text understanding.

5. **The CosyVoice 2-EU ablation is the closest analogue to our task.** Their
   component-level approach (LM > Flow Matching > Vocoder) provides a clear roadmap
   for how to prioritize Voxtral components for Arabic adaptation.

6. **DPO post-training is the production quality lever.** Voxtral, CosyVoice 3, and
   multiple 2025-2026 papers show that preference optimization is the final step
   that bridges the gap to production quality. Build rejection sampling infrastructure
   early.

7. **For Arabic specifically**: Invest in a diacritization pipeline (Sadeed) and
   gather diverse dialectal data. The arXiv:2603.01622 result showing 4Kh compensates
   for missing diacritics is encouraging but not sufficient—diacritized data still
   produces better models at every scale.

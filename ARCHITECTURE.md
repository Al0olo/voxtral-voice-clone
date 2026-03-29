# Voxtral-4B-TTS Codec Architecture

Technical reference for the codec encoder, decoder, quantizer, and LLM backbone
of `mistralai/Voxtral-4B-TTS-2603`.

Paper: https://mistral.ai/static/research/voxtral-tts.pdf

## Model Overview

| Component | Params | Role |
|-----------|--------|------|
| LLM Backbone (Ministral-3B) | ~3.4B | Text + voice conditioning -> semantic tokens |
| Acoustic Transformer | ~390M | Semantic -> acoustic tokens (flow-matching) |
| Codec Encoder | ~149M | Audio -> discrete codes (**trained by this project**) |
| Codec Decoder | ~149M | Codes -> audio waveform |
| Quantizer (VQ-FSQ) | ~2M | Continuous -> discrete codes |
| Audio Token Embedding | ~28M | Codes -> 3072-dim LLM embeddings |

## Codec Encoder (149M params, 114 weight tensors)

4-stage conv-transformer encoder following the paper specification:

- Input: raw audio `[B, 1, T]` at 24kHz
- Patch size: 240 samples (10ms)
- Hidden dim: 1024, FFN dim: 4096, head dim: 128
- 8 attention heads, 8 KV heads, QK norm, ALiBi positional bias, causal
- Sliding window attention: 16 -> 8 -> 4 -> 2 (halved at each downsampling)
- LayerScale initialized at 0.01
- Output: `[B, 292, L]` latent (256 semantic + 36 acoustic dims)
- Downsampling: strides (2,2,2,1) -> 8x temporal compression
- Frame rate: 24000 / (240 x 8) = 12.5 Hz

### Encoder Structure

```
input_proj (CausalConv1d):         240 -> 1024, kernel=7, weight_norm
encoder_blocks.0 (Transformer):    2 layers, window=16
encoder_blocks.1 (CausalConv1d):   1024 -> 1024, kernel=4, stride=2
encoder_blocks.2 (Transformer):    2 layers, window=8
encoder_blocks.3 (CausalConv1d):   1024 -> 1024, kernel=4, stride=2
encoder_blocks.4 (Transformer):    2 layers, window=4
encoder_blocks.5 (CausalConv1d):   1024 -> 1024, kernel=4, stride=2
encoder_blocks.6 (Transformer):    2 layers, window=2
encoder_blocks.7 (CausalConv1d):   1024 -> 292, kernel=3, stride=1
```

## Quantizer (VQ-FSQ Hybrid)

The 292-dim latent is split and quantized independently:

- **Semantic** (dims 0-255): VQ codebook, 8192 entries x 256 dims
  - Euclidean distance nearest-neighbor lookup
  - During training: applied with 50% probability (50% passthrough)
  - ASR-distilled from Whisper for phoneme content

- **Acoustic** (dims 256-291): FSQ, 21 levels x 36 channels
  - `tanh` bounding -> uniform quantization to 21 levels
  - During training: 50% quantize, 25% dither, 25% passthrough

Total: 37 codes per frame (1 semantic + 36 acoustic)
Bitrate: 12.5 Hz x (log2(8192) + 36 x log2(21)) = 2.14 kbps

### Embedding Table Layout

The audio token embedding `[9088, 3072]` maps codes to LLM space:
- Codebook 0 (semantic): offset=0, size=8194 (8192 + 2 special tokens)
- Codebook 1-36 (acoustic): offset=8194+k*23, size=23 each (21 + 2 special)

Voice embedding per frame = **sum of 37 lookups** from this table.

## LLM Backbone (Ministral-3B, 26 layers)

- hidden_dim = 3072, n_heads = 32, n_kv_heads = 8, head_dim = 128
- FFN: SwiGLU, hidden = 9216
- RoPE: theta = 1,000,000
- vocab_size = 131,072

### Prompt Structure

```
[BOS=1] [BEGIN_AUDIO=25] [AUDIO=24 x N] [text tokens...] [END=25]
```

Voice reference tokens come first, then text. Audio token positions are
replaced by voice embeddings (sum of 37 codebook lookups per frame).

## Training Recipe (from paper)

### Codec Training
```
L = alpha * L_feature + beta * L_ASR + gamma^t * L_L1 + gamma^t * L_STFT + delta * L_commit
```
Where alpha=1.0, beta=1.0, gamma=0.9999^t (exponential decay), delta=0.1.

### Key Training Details
- Discriminator: 8 multi-resolution STFT discriminators
- Feature matching loss is the primary reconstruction signal
- L1/STFT losses decay as discriminator strengthens
- Text embeddings are frozen during LLM training
- LLM backbone initialized from Ministral-3B

## Research Findings

### What Was Missing from the Open-Weight Release
The codec encoder weights were deliberately stripped from the checkpoint.
The architecture is fully specified in the serving framework, and the decoder +
quantizer weights are included. Only the encoder weights need to be trained.

### Key Discoveries

**Binary code saturation**: Without the paper's stochastic quantization schedule
(50/25/25 for acoustic, 50% for semantic), the encoder produces only extreme
codes (0 and 20). The stochastic schedule teaches intermediate values naturally.

**Semantic codebook collapse**: Without ASR distillation from Whisper, all encoder
outputs map to the same semantic codebook entry. The Whisper cosine loss provides
the linguistic content supervision needed for semantic diversity.

**Voice identity is the LLM's job**: The codec encoder should focus purely on
reconstruction quality. Voice identity emerges during LLM training where it
processes (voice_reference, text) -> audio pairs. Attempts to add voice identity
losses to the encoder training are counterproductive.

**Preset embedding structure**: Voice embeddings are sums of 37 codebook lookups.
Preset-to-preset cosine similarity is very high (0.97-0.99). The LLM distinguishes
voices based on 2-3% cosine differences.

### Training Pipeline

1. **Phase 1 - Encoder**: Train codec encoder with reconstruction + ASR distillation + discriminator
2. **Phase 2 - LoRA**: Fine-tune the LLM with LoRA to interpret our encoder's code patterns
3. **Phase 3 (optional)**: DPO post-training for improved speaker similarity

## Checkpoint Structure

```
consolidated.safetensors
  layers.*                              (237 weights) - LLM backbone
  norm.weight                           (1 weight)    - final LLM norm
  acoustic_transformer.*                (33 weights)  - flow-matching transformer
  audio_tokenizer.decoder_blocks.*      (112 weights) - codec decoder
  audio_tokenizer.output_proj.*         (2 weights)   - codec decoder output
  audio_tokenizer.quantizer.*           (2 buffers)   - VQ-FSQ codebook
  audio_tokenizer.input_proj.*          (2 weights)   - codec encoder [TRAINED]
  audio_tokenizer.encoder_blocks.*      (112 weights) - codec encoder [TRAINED]
  mm_audio_embeddings.*                 (2 weights)   - token embeddings
```

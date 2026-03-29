# Voxtral Voice Clone

Training the missing codec encoder for Mistral's **Voxtral-4B-TTS**, enabling
zero-shot voice cloning on the open-weight model.

## What This Does

Mistral released [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
with an important gap: the codec encoder weights were not included. Without them,
the model is limited to 20 preset voices and cannot clone new voices from audio.

This project:

1. **Trains the codec encoder** from scratch, following the paper's training recipe
   (stochastic quantization, ASR distillation, multi-resolution discriminator)
2. **Fine-tunes the LLM** with LoRA so it interprets our encoder's output for
   voice identity transfer
3. **Provides tooling** to inject the trained weights and enable `ref_audio`
   voice cloning

## Architecture

The Voxtral codec is a VQ-FSQ hybrid that compresses audio to 2.14 kbps:
- 12.5 Hz frame rate (240-sample patch at 24kHz, 8x downsampling)
- 1 semantic code (VQ, 8192 entries) + 36 acoustic codes (FSQ, 21 levels)
- Voice embeddings = sum of 37 codebook lookups per frame -> `[N, 3072]`

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical breakdown,
weight mapping, and research findings.

## Quick Start

### Requirements

- 1x GPU with >= 80GB VRAM (A100/H100/GH200)
- Voxtral-4B-TTS-2603 weights downloaded
- Python 3.10+

```bash
pip install -r requirements.txt
```

### Phase 1: Train Codec Encoder

```bash
export VOXTRAL_CKPT=/path/to/Voxtral-4B-TTS-2603
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_encoder.py
```

The script auto-detects and combines datasets from:
- LibriSpeech (train-clean-360, train-other-500)
- Common Voice (English, Arabic)
- Generated preset clips

Training follows the paper's recipe:
- Stochastic quantization (50% quantize / 25% dither / 25% passthrough for acoustic)
- Whisper ASR distillation for semantic token diversity
- 8 multi-resolution STFT discriminators with feature matching
- Exponentially decaying reconstruction losses

### Phase 2: Full Pipeline LoRA

```bash
python train_full_pipeline.py
```

Distills the LLM to interpret our encoder's voice embeddings by matching hidden
states between preset embeddings (teacher) and our encoder's output (student)
across all 26 transformer layers.

### Inject Weights & Clone

```bash
python inject_encoder.py

# Enable custom voices in the tokenizer
export VOXTRAL_VOICE_DIR=/path/to/voice_embeddings
python patch_tokenizer.py
```

After injection, any serving framework that loads the checkpoint will have
`ref_audio` cloning enabled. Pass a reference audio clip and the model will
generate speech in that voice.

## Training Details

### Phase 1 - Codec Encoder

| Hyperparameter | Value |
|---|---|
| Batch size | 4 |
| Max audio length | 4 seconds |
| Learning rate | 3e-4 (cosine decay) |
| Optimizer | AdamW (betas 0.8, 0.99) |
| Discriminator warmup | 2000 steps |
| Epochs | 10-20 |

### Phase 2 - LoRA Distillation

| Hyperparameter | Value |
|---|---|
| LoRA rank | 8 |
| LoRA targets | wq, wk, wv, wo (all 26 layers) |
| Learning rate | 2e-5 |
| Loss | MSE on hidden states at audio positions |

### Dataset Recommendations

For research-quality results:
- **Minimum**: ~30k clips (~100h) from LibriSpeech train-clean-360
- **Recommended**: ~300k clips (~900h) mixing LibriSpeech + Common Voice
- **Production**: 1M+ clips across languages and speaker diversity

## Research Narrative

### The Problem

Voxtral's codec encoder is deliberately withheld from the open-weight release.
The decoder, quantizer, and all LLM weights are available, but the encoder --
which converts raw audio to the discrete code space -- is missing. This means
the model cannot process reference audio for voice cloning.

### Reverse Engineering the Architecture

The encoder architecture is fully specified in the model's serving code. We
discovered a 4-stage convolutional-transformer encoder with:
- 149M parameters across 114 weight tensors
- 8 causal transformer layers with ALiBi attention
- Sliding window attention halving at each downsampling stage
- VQ-FSQ hybrid quantization splitting 292 latent dims

### The Invisible Walls

**Wall 1 - Codebook Collapse**: Naive training produces `sem_util=1/8192`
(only 1 of 8192 semantic codes used). Solved by ASR distillation from Whisper.

**Wall 2 - Binary Code Saturation**: Without stochastic quantization, acoustic
codes collapse to extremes (0 and 20 only). The 50/25/25 schedule from the
paper teaches intermediate values.

**Wall 3 - Training-Inference Mismatch**: An encoder that reconstructs audio
perfectly can still produce voice embeddings the LLM rejects. Solved by Phase 2
LoRA distillation that adapts the LLM to our encoder's code patterns.

**Wall 4 - Embedding Sensitivity**: The LLM distinguishes 20 preset voices
using only 2-3% cosine similarity differences. Our embeddings must match not
just the statistical distribution but the per-dimension sparsity pattern of
genuine voice embeddings.

### Current Status

Phase 1 (codec encoder training) is producing promising results with
paper-aligned training. Phase 2 (LoRA distillation) follows.

## License

This project is licensed under [CC BY-NC 4.0](LICENSE).
The trained weights are derivative of Mistral's Voxtral-4B-TTS model and
subject to its license terms.

## Citation

```bibtex
@misc{voxtral-voice-clone,
  title={Training the Missing Voxtral Codec Encoder for Zero-Shot Voice Cloning},
  author={al0olo},
  year={2025},
  url={https://github.com/al0olo/voxtral-voice-clone}
}
```

## Acknowledgements

- [Mistral AI](https://mistral.ai) for the Voxtral-4B-TTS model and paper
- [OpenAI Whisper](https://github.com/openai/whisper) for ASR distillation
- The LibriSpeech and Common Voice communities for open audio datasets

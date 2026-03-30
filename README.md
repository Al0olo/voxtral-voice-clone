# Voxtral Voice Clone

Training the missing codec encoder for Mistral's **Voxtral-4B-TTS**, enabling
zero-shot voice cloning on the open-weight model.

**Status: The encoder produces intelligible speech from reference audio.
Voice identity preservation is actively improving with V3 training.**

## Results

We successfully trained a codec encoder that:
- Produces codes the LLM accepts without any fine-tuning (no LoRA needed)
- Generates clear, intelligible speech from reference audio clips
- Produces embeddings with norms matching preset voices (4.6 vs 3.7 target)
- Uses 50-200+ unique semantic codes per utterance (matching preset range of 52-152)

Voice identity is improving with V3 training (speaker verification loss + native 24kHz data).

## What This Does

Mistral released [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
with an important gap: the codec encoder weights were not included. Without them,
the model is limited to 20 preset voices and cannot clone new voices from audio.

This project trains the missing encoder from scratch using techniques from
the Voxtral paper, EnCodec, and speaker verification research.

## Architecture

The Voxtral codec is a VQ-FSQ hybrid that compresses audio to 2.14 kbps:
- 12.5 Hz frame rate (240-sample patch at 24kHz, 8x downsampling)
- 1 semantic code (VQ, 8192 entries) + 36 acoustic codes (FSQ, 21 levels)
- Voice embeddings = sum of 37 codebook lookups per frame -> `[N, 3072]`

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical breakdown,
weight mapping, and research findings.

## Quick Start

### Requirements

- 4x A100-80GB (or equivalent, ~320GB total VRAM for multi-GPU training)
- Voxtral-4B-TTS-2603 weights downloaded
- Python 3.10+

```bash
pip install -r requirements.txt
pip install speechbrain  # for ECAPA-TDNN speaker verification loss
```

### Train Codec Encoder

```bash
export MODEL_DIR=/path/to/Voxtral-4B-TTS-2603
export HF_CACHE=/path/to/data_cache
export OUTPUT_DIR=/path/to/encoder_output

torchrun --nproc_per_node=4 train_encoder.py
```

Training uses LibriTTS-R (native 24kHz, 585h, 2456 speakers) downloaded
automatically from HuggingFace.

### Inject Weights & Test

```bash
python inject_encoder.py
python patch_tokenizer.py

# Serve with vLLM
vllm serve /path/to/model --omni --gpu-memory-utilization 0.45
```

## Training Recipe (V3)

The current training combines techniques from multiple research papers:

| Component | Source | Purpose |
|-----------|--------|---------|
| Stochastic VQ (50/25/25) | Voxtral paper | Prevents code saturation |
| ASR distillation (Whisper) | Voxtral paper | Semantic token diversity |
| Codebook diversity loss | Our innovation | Breaks semantic collapse (1 -> 200+ codes) |
| Frozen speaker loss (ECAPA-TDNN) | SAC paper | Explicit speaker identity preservation |
| Gradient-norm balancer | EnCodec/AudioCraft | Auto-scales loss contributions |
| Acoustic distribution shaping | Preset analysis | Matches preset code statistics |
| Multi-resolution STFT disc (64ch, 8 scales) | Voxtral paper + EnCodec | Waveform quality |
| Native 24kHz data (LibriTTS-R) | Our discovery | Fixes 16kHz upsampling artifacts |
| Adam beta1=0.5 | EnCodec/HiFi-GAN | GAN training stability |

### V3 Training Metrics (current best)

| Metric | V1 (initial) | V2 (diversity fix) | V3 (current) |
|--------|-------------|-------------------|-------------|
| mel loss | 1.8 (plateau) | 1.5 (plateau) | **0.87** (dropping) |
| sem_util | 1/8192 | 70-100 | **228/8192** |
| speaker loss | N/A | N/A | **0.37** (dropping) |
| acoustic codes | all 18s | diverse 3-15 | diverse 3-18 |
| speech output | hums | intelligible, wrong voice | intelligible, identity improving |

## Research Narrative

### The Problem

Voxtral's codec encoder is deliberately withheld from the open-weight release.
The paper provides full architecture details, but training a compatible encoder
requires solving multiple interacting problems.

### The Invisible Walls (and how we broke them)

**Wall 1 - Codebook Collapse** (`sem_util=1/8192`): All encoder outputs map to
one codebook entry. ASR distillation alone is insufficient because the encoder
can produce diverse continuous features that all land in one Voronoi cell.
Solved with **entropy-based codebook diversity loss** (sem_util 1 -> 200+).

**Wall 2 - Acoustic Code Saturation**: Without stochastic quantization, codes
collapse to extremes (0 and 20). Solved with the paper's 50/25/25 schedule
plus **acoustic distribution shaping** targeting preset statistics (mean~10).

**Wall 3 - Speaker Identity Loss**: Mel reconstruction alone does NOT preserve
speaker identity at low bitrates (2.14 kbps). The codec can sound good while
erasing speaker-discriminative features. Solved with **frozen ECAPA-TDNN
speaker verification loss** (cosine similarity between original and reconstructed
speaker embeddings).

**Wall 4 - Sample Rate Mismatch**: Training on 16kHz LibriSpeech upsampled to
24kHz means the 8-12kHz band (where speaker timbre lives) contains interpolated
artifacts. Solved by switching to **LibriTTS-R** (native 24kHz).

**Wall 5 - Loss Balancing**: Manual loss weights (ASR=5, diversity=5) caused mel
to plateau at 1.5 for 2+ epochs. Solved with **EnCodec's gradient-norm balancer**
that auto-scales each loss based on gradient magnitude.

### Key Discovery: LoRA Is Not Needed

We initially assumed the LLM would reject our encoder's embeddings and require
LoRA fine-tuning. Testing showed the opposite: the original unmodified LLM
produces clear speech from our encoder's codes. LoRA at rank 8 destroyed the
base model; rank 2 partially worked but damaged preset voices. The encoder's
codes are "legal" tokens that the LLM accepts natively.

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
- [Meta AI](https://github.com/facebookresearch/audiocraft) for EnCodec's gradient balancer
- [SpeechBrain](https://speechbrain.github.io/) for the ECAPA-TDNN speaker verification model
- [OpenAI Whisper](https://github.com/openai/whisper) for ASR distillation
- The LibriTTS-R and Common Voice communities for open audio datasets

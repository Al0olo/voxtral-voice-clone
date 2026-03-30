# Datasets for Voxtral Codec Encoder Training

## Critical Requirement: Native 24kHz Audio

The Voxtral codec operates at 24kHz. Training on lower sample rate audio (e.g. 16kHz
LibriSpeech) upsampled to 24kHz causes the encoder to lose speaker identity information
in the 8-12kHz band where timbre, breathiness, and formant details live.

**Always prefer native 24kHz datasets. Downsampling from 48kHz is acceptable.**

---

## Currently Used

| Dataset | Language | Hours | Speakers | Sample Rate | Clips | Source |
|---------|----------|-------|----------|-------------|-------|--------|
| LibriTTS-R clean-360 | English | 245h | ~1,100 | **24kHz** | 116K | [HuggingFace](https://huggingface.co/datasets/mythicinfinity/libritts_r) |
| LibriTTS-R other-500 | English | 340h | ~1,300 | **24kHz** | 205K | [HuggingFace](https://huggingface.co/datasets/mythicinfinity/libritts_r) |
| **Total** | | **585h** | **~2,456** | | **321K** | |

---

## English Datasets

### LibriTTS-R (current primary)
- **Hours**: 585h
- **Speakers**: 2,456
- **Sample Rate**: 24kHz native
- **Source**: Restored LibriSpeech audiobooks with speech enhancement
- **License**: CC BY 4.0
- **Download**: `mythicinfinity/libritts_r` on HuggingFace or [openslr.org/141](https://openslr.org/141)
- **Notes**: Clean, well-segmented, sentence-level. Our primary training dataset.

### VCTK
- **Hours**: 44h
- **Speakers**: 110
- **Sample Rate**: 48kHz (downsample to 24kHz)
- **Source**: Professional studio recordings, newspaper sentences
- **License**: ODC-By v1.0
- **Download**: [datashare.ed.ac.uk](https://datashare.ed.ac.uk/handle/10283/3443)
- **Notes**: Excellent accent diversity (British + international English). Small but
  high quality. Good supplement for speaker diversity.

### Common Voice English (v24)
- **Hours**: 3,779h total / 2,750h validated
- **Speakers**: 99,289
- **Sample Rate**: 48kHz (downsample to 24kHz)
- **Source**: Crowd-sourced Mozilla recordings
- **License**: CC-0 (public domain)
- **Download**: [commonvoice.mozilla.org](https://commonvoice.mozilla.org/en/datasets) or `mozilla-foundation/common_voice_17_0` on HuggingFace
- **Notes**: Massive speaker diversity (40x more than LibriTTS-R). Variable quality.
  Recommend filtering: `up_votes > down_votes`, duration 2-15s, max 20 clips/speaker.

### Emilia (English subset)
- **Hours**: 46,000h+ English
- **Speakers**: Thousands
- **Sample Rate**: 24kHz (preprocessed)
- **Source**: YouTube, podcasts, interviews -- in-the-wild conversational speech
- **License**: Non-commercial research only
- **Download**: `amphion/Emilia` on HuggingFace (requires access request)
- **Notes**: Nuclear option. Massive scale, spontaneous speech, real conversational
  dynamics. Best for production-quality training but requires significant storage (~2TB+).

### GigaSpeech
- **Hours**: 10,000h (XL) / 250h (S)
- **Speakers**: Many thousands
- **Sample Rate**: 16kHz (NOT recommended -- upsampling artifacts)
- **License**: Apache 2.0
- **Notes**: Large but 16kHz native. Use only if 24kHz alternatives exhausted.

---

## Arabic Datasets

### Arabic-Diacritized-TTS
- **Hours**: ~15h
- **Speakers**: Multiple
- **Sample Rate**: Varies (check per-sample)
- **Source**: Fully diacritized MSA text + high-quality audio
- **License**: Open
- **Download**: `Nourhann/Arabic-Diacritized-TTS` on HuggingFace
- **Notes**: Key dataset for Arabic diacritics training. Fully vocalized text critical
  for correct Arabic TTS prosody.

### Arabic TTS WAV 24k
- **Hours**: 12.5h
- **Speakers**: 2 (1 male, 1 female)
- **Sample Rate**: **24kHz** native
- **Source**: Studio-quality recordings with IPA phonetic transcriptions
- **License**: Open
- **Download**: `NeoBoy/arabic-tts-wav-24k` on HuggingFace (2.15 GB)
- **Notes**: Small but native 24kHz. Good starting point for Arabic codec training.
  Includes phonetic IPA alongside Arabic text.

### ArVoice
- **Hours**: 83.5h
- **Speakers**: 7 human + synthetic voices
- **Sample Rate**: Varies
- **Source**: Multi-speaker MSA with fully diacritized transcriptions
- **License**: Open (check per-subset)
- **Notes**: Diverse demographics (age, gender). 10h human voices, rest synthetic.
  Good for codec encoder training (human subset) and LLM fine-tuning (full set).

### NileTTS (Egyptian Arabic)
- **Hours**: 38h
- **Speakers**: Multiple
- **Sample Rate**: Varies
- **Source**: Egyptian dialect, medical/sales/general conversation domains
- **License**: Apache 2.0
- **Download**: HuggingFace (see `KickItLikeShika/NileTTS` GitHub)
- **Notes**: Egyptian dialect specifically. Good for dialect diversity.

### Common Voice Arabic (v24)
- **Hours**: 157h total / 92h validated
- **Speakers**: 1,651
- **Sample Rate**: 48kHz (downsample to 24kHz)
- **Source**: Crowd-sourced Mozilla recordings
- **License**: CC-0
- **Download**: [commonvoice.mozilla.org](https://commonvoice.mozilla.org/ar/datasets)
- **Notes**: Good speaker diversity for Arabic. Filter for quality.

### Habibi-TTS (Multi-Dialect Arabic)
- **Hours**: TBD
- **Speakers**: Multiple per dialect
- **Sample Rate**: Varies
- **Source**: Unified-dialectal Arabic synthesis framework
- **License**: MIT
- **Download**: `SWivid/Habibi-TTS` on GitHub
- **Notes**: Covers 12+ Arabic dialects (MSA, Saudi, UAE, Egyptian, Moroccan, Iraqi,
  Algerian, Tunisian, Levantine, Sudanese, Libyan, Omani). Framework includes benchmark
  dataset. Outperforms ElevenLabs on dialectal Arabic.

### Sadeed Tashkeela (Diacritization Corpus)
- **Hours**: Text only (53M+ words)
- **Source**: Cleaned Tashkeela corpus with full diacritization
- **License**: Open
- **Download**: `Misraj/Sadeed_Tashkeela` on HuggingFace
- **Notes**: Text-only diacritization dataset. Not audio, but essential for generating
  properly diacritized prompts for Arabic TTS fine-tuning.

---

## Hindi Datasets

### IndicVoices-R Hindi
- **Hours**: ~100h+ (26,318 samples, 46.6 GB)
- **Speakers**: Multiple
- **Sample Rate**: Varies (some 24kHz)
- **Source**: Restored speech from IndicVoices project
- **License**: Open (CC BY 4.0)
- **Download**: `SPRINGLab/IndicVoices-R_Hindi` on HuggingFace
- **Notes**: Best Hindi dataset for codec training. Used to train F5-Hindi-24KHz.

### IndicTTS-Hindi
- **Hours**: 10.3h
- **Speakers**: 2 (1 male, 1 female)
- **Sample Rate**: 48kHz (downsample to 24kHz)
- **Source**: Studio-quality recordings from IIT Madras
- **License**: CC BY 4.0
- **Download**: `SPRINGLab/IndicTTS-Hindi` on HuggingFace
- **Notes**: Small but high quality. Professional studio recording.

### IndicVoices (Full Multilingual)
- **Hours**: 23,700h (22 Indian languages, 11,200h transcribed)
- **Speakers**: 51,000+
- **Sample Rate**: Varies
- **Source**: Read, extempore, and conversational speech from 400+ Indian districts
- **License**: Open
- **Download**: `ai4bharat/IndicVoices` on HuggingFace
- **Notes**: Massive multilingual Indian dataset. Hindi is a major subset. Includes
  conversational speech which is rare and valuable.

### Common Voice Hindi (v17)
- **Hours**: ~20h validated
- **Speakers**: ~1,000
- **Sample Rate**: 48kHz (downsample to 24kHz)
- **License**: CC-0
- **Download**: [commonvoice.mozilla.org](https://commonvoice.mozilla.org/hi/datasets)
- **Notes**: Small but diverse speakers.

---

## Multilingual / Large-Scale

### Emilia
- **Hours**: 101,000h (base), 216,000h (Emilia-Large)
- **Languages**: English, Chinese, German, French, Japanese, Korean
- **Sample Rate**: 24kHz (preprocessed)
- **License**: Non-commercial research
- **Download**: `amphion/Emilia` on HuggingFace
- **Notes**: Largest open speech generation dataset. In-the-wild conversational speech.
  Includes Emilia-Pipe preprocessing pipeline. Does NOT include Arabic or Hindi.

### CML-TTS
- **Hours**: 3,176h
- **Languages**: Dutch, French, German, Italian, Portuguese, Polish, Spanish
- **Sample Rate**: Varies
- **License**: Open
- **Notes**: Does NOT include English, Arabic, or Hindi. Useful for European language expansion.

### MLS (Multilingual LibriSpeech)
- **Hours**: 50,000h+
- **Languages**: 8 languages (English, German, Dutch, French, Spanish, Italian, Portuguese, Polish)
- **Sample Rate**: 16kHz (NOT recommended)
- **License**: CC BY 4.0
- **Notes**: Large but 16kHz. No Arabic or Hindi.

---

## Recommended Training Mixtures

### Phase 1: Codec Encoder (current)
| Dataset | Hours | Purpose |
|---------|-------|---------|
| LibriTTS-R (clean+other) | 585h | Primary English training |
| **Total** | **585h** | |

### Phase 2: Speaker Diversity
| Dataset | Hours | Purpose |
|---------|-------|---------|
| LibriTTS-R | 585h | English base |
| VCTK | 44h | Accent diversity |
| Common Voice English (curated) | ~500h | Massive speaker diversity |
| Arabic TTS 24k | 12.5h | Arabic native 24kHz |
| Common Voice Arabic | ~92h | Arabic speaker diversity |
| **Total** | **~1,234h** | |

### Phase 3: Production Multilingual
| Dataset | Hours | Purpose |
|---------|-------|---------|
| All Phase 2 | ~1,234h | Base |
| ArVoice (human subset) | ~10h | Arabic multi-speaker |
| Arabic-Diacritized-TTS | ~15h | Diacritized Arabic |
| IndicVoices-R Hindi | ~100h | Hindi |
| Emilia English (subset) | ~5,000h | Scale + conversational diversity |
| **Total** | **~6,359h** | |

### Arabic Fine-Tuning (LLM, after encoder)
| Dataset | Type | Purpose |
|---------|------|---------|
| Arabic-Diacritized-TTS | Audio + diacritized text | Diacritics + prosody |
| ArVoice | Audio + diacritized text | Multi-speaker MSA |
| Habibi-TTS | Audio + text | Multi-dialect coverage |
| NileTTS | Audio + text | Egyptian dialect |
| Sadeed Tashkeela | Text only (53M words) | Diacritization training data |
| Common Voice Arabic | Audio + text | Speaker diversity |

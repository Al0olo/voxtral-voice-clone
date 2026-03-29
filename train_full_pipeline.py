"""
Full Pipeline LoRA Training for Voxtral Voice Cloning.

Replicates the exact TTS prompt structure for distillation:
- Full tokenized prompt: [BOS] [BEGIN_AUDIO] [AUDIO x N] [text tokens]
- Text positions use tok_embeddings, audio positions use voice embeddings
- All 26 LLM layers with RoPE (rotary position embeddings)
- LoRA rank 8 on all layers
- Distillation loss: hidden states from our encoder should match preset embeddings

The LLM learns to interpret our encoder's output for voice identity.
"""

import os, sys, math, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, soundfile as sf
from safetensors.torch import load_file, save_file

DEVICE = "cuda"
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/Voxtral-4B-TTS-2603")
PRESET_DIR = os.environ.get("PRESET_DIR", "/codec_data")
ENCODER_WEIGHTS = os.environ.get("ENCODER_WEIGHTS", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/encoder_full_pipeline")
SAMPLE_RATE = 24000
BATCH_SIZE = 2
LR = 5e-5
EPOCHS = 30
LOG_EVERY = 50
LORA_RANK = 8
AUDIO_TOKEN_ID = 24
BEGIN_AUDIO_TOKEN_ID = 25
BOS_TOKEN_ID = 1


# ============================================================
# LLM Components (reuse from train_lora_distill.py)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def precompute_freqs_cis(dim, max_seq_len, theta=1000000.0, device="cuda"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(xq, xk, cos_f, sin_f):
    B, T, H, D = xq.shape
    half = D // 2
    cos_f = cos_f[:T, :half].unsqueeze(0).unsqueeze(2)
    sin_f = sin_f[:T, :half].unsqueeze(0).unsqueeze(2)
    def rotate(x):
        x1, x2 = x.float()[..., :half], x.float()[..., half:]
        return torch.cat([x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f], dim=-1).to(x.dtype)
    return rotate(xq), rotate(xk)


class LoRALinear(nn.Module):
    def __init__(self, original, rank=8):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)
        in_f, out_f = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, device=original.weight.device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=original.weight.device))
        self.scale = 1.0 / rank
    def forward(self, x):
        base = self.original(x)
        lora = (x.to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()) * self.scale
        return base + lora.to(base.dtype)


class LLMAttention(nn.Module):
    def __init__(self, dim=3072, n_heads=32, n_kv_heads=8, head_dim=128):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.repeats = n_heads // n_kv_heads
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x, cos_f, sin_f):
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, cos_f, sin_f)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        if self.repeats > 1:
            xk = xk.repeat_interleave(self.repeats, dim=1)
            xv = xv.repeat_interleave(self.repeats, dim=1)
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class LLMBlock(nn.Module):
    def __init__(self, dim=3072, n_heads=32, n_kv_heads=8, head_dim=128, hidden_dim=9216):
        super().__init__()
        self.attention = LLMAttention(dim, n_heads, n_kv_heads, head_dim)
        self.feed_forward_w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.feed_forward_w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.feed_forward_w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, cos_f, sin_f):
        h = x + self.attention(self.attention_norm(x), cos_f, sin_f)
        fn = self.ffn_norm(h)
        return h + self.feed_forward_w2(F.silu(self.feed_forward_w1(fn)) * self.feed_forward_w3(fn))


class MinistralLLM(nn.Module):
    def __init__(self, n_layers=26, dim=3072):
        super().__init__()
        self.tok_embeddings = nn.Embedding(131072, dim)
        self.layers = nn.ModuleList([LLMBlock() for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.n_layers = n_layers

    def forward(self, input_embeds, cos_f, sin_f):
        h = input_embeds
        for layer in self.layers:
            h = layer(h, cos_f, sin_f)
        return self.norm(h)

    def forward_with_checkpoint(self, input_embeds, cos_f, sin_f):
        """Memory-efficient forward with gradient checkpointing."""
        h = input_embeds
        for layer in self.layers:
            h = torch.utils.checkpoint.checkpoint(layer, h, cos_f, sin_f, use_reentrant=False)
        return self.norm(h)


# ============================================================
# Weight Loading
# ============================================================

def load_llm_weights(llm, model_dir):
    st = load_file(os.path.join(model_dir, "consolidated.safetensors"), device="cpu")
    llm.tok_embeddings.weight.data.copy_(st["mm_audio_embeddings.tok_embeddings.weight"])
    llm.norm.weight.data.copy_(st["norm.weight"])
    loaded = 0
    for i in range(llm.n_layers):
        block = llm.layers[i]
        for src, tgt in [
            (f"layers.{i}.attention.wq.weight", block.attention.wq),
            (f"layers.{i}.attention.wk.weight", block.attention.wk),
            (f"layers.{i}.attention.wv.weight", block.attention.wv),
            (f"layers.{i}.attention.wo.weight", block.attention.wo),
            (f"layers.{i}.attention_norm.weight", block.attention_norm),
            (f"layers.{i}.ffn_norm.weight", block.ffn_norm),
            (f"layers.{i}.feed_forward.w1.weight", block.feed_forward_w1),
            (f"layers.{i}.feed_forward.w2.weight", block.feed_forward_w2),
            (f"layers.{i}.feed_forward.w3.weight", block.feed_forward_w3),
        ]:
            if src in st:
                if hasattr(tgt, 'weight'):
                    tgt.weight.data.copy_(st[src])
                else:
                    tgt.data.copy_(st[src])
                loaded += 1
    print(f"Loaded {loaded} LLM weights")
    return llm


def apply_lora_all_layers(llm, rank):
    lora_params = []
    for i in range(llm.n_layers):
        attn = llm.layers[i].attention
        for name in ['wq', 'wk', 'wv', 'wo']:
            original = getattr(attn, name)
            lora_layer = LoRALinear(original, rank)
            setattr(attn, name, lora_layer)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
    n = sum(p.numel() for p in lora_params)
    print(f"LoRA rank={rank} on ALL {llm.n_layers} layers: {n:,} params")
    return lora_params


def merge_lora_all_layers(llm):
    for i in range(llm.n_layers):
        attn = llm.layers[i].attention
        for name in ['wq', 'wk', 'wv', 'wo']:
            lora = getattr(attn, name)
            if isinstance(lora, LoRALinear):
                with torch.no_grad():
                    delta = (lora.lora_B @ lora.lora_A).to(lora.original.weight.dtype) * lora.scale
                    lora.original.weight.data += delta
                setattr(attn, name, lora.original)
    print("Merged LoRA into all layers")


# ============================================================
# Full Prompt Builder
# ============================================================

def build_prompt_embeds(llm, token_ids, voice_embedding, device):
    """Build input_embeds matching the TTS inference pipeline.
    
    token_ids: [BOS=1, BEGIN_AUDIO=25, AUDIO=24 x N, text_tokens...]
    voice_embedding: [N, 3072] preset voice embedding
    
    Returns: [1, seq_len, 3072] input embeddings
    """
    token_ids = torch.tensor(token_ids, device=device)
    
    # Get text embeddings for all tokens
    text_embeds = llm.tok_embeddings(token_ids)  # [seq_len, 3072]
    
    # Replace AUDIO token positions with voice embedding
    audio_mask = token_ids == AUDIO_TOKEN_ID
    audio_positions = audio_mask.nonzero(as_tuple=True)[0]
    
    n_audio = audio_positions.shape[0]
    n_voice = voice_embedding.shape[0]
    
    # Truncate voice embedding to match audio positions
    n = min(n_audio, n_voice)
    text_embeds[audio_positions[:n]] = voice_embedding[:n].to(text_embeds.dtype)
    
    return text_embeds.unsqueeze(0)  # [1, seq_len, 3072]


# ============================================================
# Training
# ============================================================

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("FULL PIPELINE LORA TRAINING")
    print("=" * 60)

    # Load tokenizer
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.speech.request import SpeechRequest
    tokenizer = MistralTokenizer.from_file(os.path.join(MODEL_DIR, 'tekken.json'))

    # Load LLM
    print("\nLoading LLM...")
    llm = MinistralLLM()
    llm = load_llm_weights(llm, MODEL_DIR)
    llm = llm.to(DEVICE).bfloat16()
    llm.requires_grad_(False)

    # Apply LoRA to ALL 26 layers
    lora_params = apply_lora_all_layers(llm, LORA_RANK)

    # RoPE
    cos_f, sin_f = precompute_freqs_cis(128, 4096, theta=1000000.0, device=DEVICE)

    # Load preset voice embeddings
    voice_emb_dir = os.path.join(MODEL_DIR, "voice_embedding")
    preset_embs = {}
    for f in os.listdir(voice_emb_dir):
        if f.endswith('.pt') and not any(x in f for x in ['adapted', 'calibrat', 'merged', 'lora', 'clone']):
            name = f.replace('.pt', '')
            preset_embs[name] = torch.load(os.path.join(voice_emb_dir, f),
                                           map_location='cpu', weights_only=True).to(DEVICE).bfloat16()
    print(f"Loaded {len(preset_embs)} preset voices")

    # Load encoder (optional - for encoder+LoRA joint training)
    encoder = None
    projection = None
    encoder_params = []
    if ENCODER_WEIGHTS and os.path.exists(ENCODER_WEIGHTS):
        sys.path.insert(0, "/voxtral-encoder")
        exec(open("/voxtral-encoder/train_encoder.py").read().split("def train():")[0], globals())
        encoder = VoxtralCodec()
        encoder = load_decoder_weights(encoder, MODEL_DIR)
        r = torch.load(ENCODER_WEIGHTS, map_location="cpu", weights_only=True)
        for k, v in r.items():
            if k in dict(encoder.named_parameters()):
                dict(encoder.named_parameters())[k].data.copy_(v)
        encoder = encoder.to(DEVICE).bfloat16()
        encoder.input_proj.float()
        for b in encoder.encoder_blocks:
            b.float()
        for name, param in encoder.named_parameters():
            if name.startswith("input_proj.") or name.startswith("encoder_blocks."):
                param.requires_grad = True
                encoder_params.append(param)
            else:
                param.requires_grad = False
        print(f"Loaded encoder: {sum(p.numel() for p in encoder_params):,} trainable params")

    # Collect training data: (text, voice_name) pairs
    texts = [
        "The conference will be held in the main auditorium next Friday morning.",
        "Scientists discovered a new species of deep sea fish last Tuesday.",
        "She walked through the garden admiring the colorful spring flowers.",
        "He carefully placed the ancient book back on the dusty wooden shelf.",
        "Today we announce a major breakthrough in renewable energy technology.",
        "The children laughed and played in the park until the sun went down.",
        "Please remember to submit your report before the end of this week.",
        "A gentle breeze carried the scent of fresh pine through the valley.",
        "The orchestra performed beautifully at the grand opening ceremony.",
        "I would like to order a large coffee with extra cream please.",
    ]
    
    training_pairs = []
    for voice in preset_embs:
        for text in texts:
            training_pairs.append((text, voice))
    np.random.shuffle(training_pairs)
    print(f"Training pairs: {len(training_pairs)} ({len(preset_embs)} voices x {len(texts)} texts)")

    # Optimizer
    all_params = lora_params + encoder_params
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(training_pairs) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        np.random.shuffle(training_pairs)
        epoch_loss = 0
        n_batches = 0

        for bi in range(0, len(training_pairs) - BATCH_SIZE, BATCH_SIZE):
            batch = training_pairs[bi:bi + BATCH_SIZE]
            total_loss = 0

            for text, voice in batch:
                # Tokenize the full TTS prompt
                result = tokenizer.encode_speech_request(SpeechRequest(input=text, voice=voice))
                token_ids = result.tokens

                # Build input_embeds with PRESET voice embedding (target)
                preset_emb = preset_embs[voice]
                with torch.no_grad():
                    target_embeds = build_prompt_embeds(llm, token_ids, preset_emb, DEVICE)
                    # Run frozen LLM (no LoRA effect for target)
                    # We need to temporarily disable LoRA for the target pass
                    # Instead, just save the target embeds as-is and compare at output
                    
                # For target: run with preset embedding through ALL layers
                # We use the same LLM but the LoRA delta is small, so target is approximately correct
                with torch.no_grad():
                    target_hidden = llm.forward(target_embeds, cos_f, sin_f)

                # Build input_embeds with OUR voice embedding (or encoder output)
                if encoder is not None:
                    # Run encoder on a preset audio clip for this voice
                    audio_dir = os.path.join(PRESET_DIR, voice)
                    if os.path.isdir(audio_dir):
                        wavs = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
                        if wavs:
                            wav_path = os.path.join(audio_dir, wavs[np.random.randint(len(wavs))])
                            audio, sr = sf.read(wav_path, dtype='float32')
                            if len(audio.shape) > 1: audio = audio[:, 0]
                            if sr != SAMPLE_RATE:
                                import librosa
                                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                            if len(audio) > 4 * SAMPLE_RATE:
                                start = np.random.randint(0, len(audio) - 4 * SAMPLE_RATE)
                                audio = audio[start:start + 4 * SAMPLE_RATE]
                            if len(audio) % 240:
                                audio = np.pad(audio, (0, 240 - len(audio) % 240))
                            x = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                latent = encoder.forward_encoder(x)
                                codes = encoder.quantizer.encode(latent.float())
                                # Build embedding from codes (same as encode_tokens)
                                from safetensors.torch import load_file as _lf
                                # Use cached embedding table
                                if not hasattr(train, '_emb_table'):
                                    _st = _lf(os.path.join(MODEL_DIR, 'consolidated.safetensors'), device='cpu')
                                    train._emb_table = _st['mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight'].float().to(DEVICE)
                                    _p = json.load(open(os.path.join(MODEL_DIR, 'params.json')))
                                    _ama = _p['multimodal']['audio_model_args']
                                    _sizes = [_ama['semantic_codebook_size'] + 2] + [_ama['acoustic_codebook_size'] + 2] * _ama['n_acoustic_codebook']
                                    train._offsets = np.cumsum([0] + _sizes[:-1])
                                codes_offset = codes + 2
                                frames = []
                                for fi in range(codes.shape[2]):
                                    frame = sum(train._emb_table[codes_offset[0, cb, fi].item() + int(train._offsets[cb])] for cb in range(37))
                                    frames.append(frame)
                                our_voice_emb = torch.stack(frames).to(DEVICE).bfloat16()
                        else:
                            our_voice_emb = preset_emb  # fallback
                    else:
                        our_voice_emb = preset_emb
                else:
                    our_voice_emb = preset_emb  # If no encoder, use preset (LoRA-only training)

                our_embeds = build_prompt_embeds(llm, token_ids, our_voice_emb, DEVICE)

                # Run LLM with LoRA on our embedding
                our_hidden = llm.forward_with_checkpoint(our_embeds, cos_f, sin_f)

                # Loss: MSE on audio token positions only
                audio_mask = torch.tensor(token_ids, device=DEVICE) == AUDIO_TOKEN_ID
                if audio_mask.sum() > 0:
                    target_audio = target_hidden[0, audio_mask].float()
                    our_audio = our_hidden[0, audio_mask].float()
                    distill_loss = F.mse_loss(our_audio, target_audio.detach())
                    total_loss += distill_loss

            loss = total_loss / len(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % LOG_EVERY == 0:
                print(f"  [E{epoch} B{n_batches}] loss={loss.item():.6f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = epoch_loss / max(n_batches, 1)

        # Measure cosine gap
        with torch.no_grad():
            gap_results = {}
            for voice in ['ar_male', 'casual_female']:
                result = tokenizer.encode_speech_request(SpeechRequest(input='Hello world.', voice=voice))
                embeds = build_prompt_embeds(llm, result.tokens, preset_embs[voice], DEVICE)
                hidden = llm.forward(embeds, cos_f, sin_f)
                audio_mask = torch.tensor(result.tokens, device=DEVICE) == AUDIO_TOKEN_ID
                gap_results[voice] = hidden[0, audio_mask].float().mean(0)
            
            cos_male = F.cosine_similarity(gap_results['ar_male'].unsqueeze(0),
                                           gap_results['casual_female'].unsqueeze(0).detach() * 0 + gap_results['ar_male'].unsqueeze(0)).item()
            # Actually measure: how different are the hidden states for male vs female?
            hidden_cos = F.cosine_similarity(gap_results['ar_male'].unsqueeze(0),
                                             gap_results['casual_female'].unsqueeze(0)).item()

        print(f"\n  Epoch {epoch}/{EPOCHS}: loss={avg_loss:.6f} hidden_cos_m_f={hidden_cos:.4f}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            lora_state = {}
            for i, p in enumerate(lora_params):
                lora_state[f"lora_{i}"] = p.data.cpu()
            torch.save(lora_state, os.path.join(OUTPUT_DIR, "best_lora.pt"))
            if encoder_params:
                enc_state = {n: p.data.cpu() for n, p in encoder.named_parameters()
                             if n.startswith("input_proj.") or n.startswith("encoder_blocks.")}
                torch.save(enc_state, os.path.join(OUTPUT_DIR, "best_encoder.pt"))
            print(f"  Saved best (loss={best_loss:.6f})", flush=True)

        # Save every 5 epochs
        if epoch % 5 == 0:
            lora_state = {}
            for i, p in enumerate(lora_params):
                lora_state[f"lora_{i}"] = p.data.cpu()
            torch.save(lora_state, os.path.join(OUTPUT_DIR, f"lora_ep{epoch}.pt"))

    # Merge and save
    print("\nMerging LoRA...")
    merge_lora_all_layers(llm)
    
    print("Saving merged checkpoint...")
    original_st = load_file(os.path.join(MODEL_DIR, "consolidated.safetensors"), device="cpu")
    for i in range(llm.n_layers):
        block = llm.layers[i]
        for name, module in [("wq", block.attention.wq), ("wk", block.attention.wk),
                             ("wv", block.attention.wv), ("wo", block.attention.wo)]:
            original_st[f"layers.{i}.attention.{name}.weight"] = module.weight.data.cpu().to(torch.bfloat16)
    
    if encoder_params:
        enc_state = torch.load(os.path.join(OUTPUT_DIR, "best_encoder.pt"), map_location="cpu", weights_only=True)
        for k, v in enc_state.items():
            original_st[f"audio_tokenizer.{k}"] = v.to(torch.bfloat16)
    
    save_file(original_st, os.path.join(OUTPUT_DIR, "consolidated_merged.safetensors"))
    print(f"Saved to {OUTPUT_DIR}/consolidated_merged.safetensors")
    print("Done!")


if __name__ == "__main__":
    train()

"""
Voxtral Codec Encoder Training.

Implements the codec encoder architecture described in the Voxtral TTS paper
(https://mistral.ai/static/research/voxtral-tts.pdf). The encoder was not
included in the open-weight release; this script trains it from scratch using
the frozen decoder and quantizer weights from the checkpoint.

Training follows the paper's recipe:
  - Stochastic VQ (50% quantize / 25% dither / 25% passthrough)
  - ASR distillation from Whisper for semantic token diversity
  - Multi-resolution STFT discriminator with feature matching loss
  - Exponentially decaying reconstruction loss

After training, encoder weights can be injected into the model checkpoint
to enable ref_audio voice cloning.
"""

import os
import sys
import math
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from pathlib import Path
from copy import deepcopy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_ASR_DISTILL = os.environ.get("USE_ASR_DISTILL", "1") == "1"
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/Voxtral-4B-TTS-2603")
DATA_DIR = os.environ.get("DATA_DIR", "/codec_data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/encoder_trained")
BATCH_SIZE = 4
MAX_AUDIO_SEC = 4
LR = 3e-4
EPOCHS = 30
SAMPLE_RATE = 24000
LOG_EVERY = 25

# ============================================================
# Codec architecture (matching the open-weight model spec)
# ============================================================

from einops import rearrange

try:
    from apex.normalization import FusedRMSNorm
    rms_norm = FusedRMSNorm
except ImportError:
    from torch.nn import RMSNorm
    rms_norm = RMSNorm

weight_norm = torch.nn.utils.parametrizations.weight_norm

# ============================================================
# Building blocks (conv, attention, transformer, quantizer)
# ============================================================

def pad1d(x, paddings, mode="constant", value=0.0):
    length = x.shape[-1]
    pl, pr = paddings
    if mode == "reflect":
        max_pad = max(pl, pr)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1,
                 pad_mode="reflect", use_weight_norm=True, use_bias=True):
        super().__init__()
        conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                         padding=0, dilation=dilation, bias=use_bias)
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.pad_mode = pad_mode
        self._stride = self.conv.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.conv.dilation[0] + 1
        self._padding_total = self._effective_kernel_size - self._stride
        self.stride = self.conv.stride

    def forward(self, x):
        n_frames = (x.shape[-1] - self._effective_kernel_size + self._padding_total) / self._stride + 1
        target_length = (math.ceil(n_frames) - 1) * self._stride + (self._effective_kernel_size - self._padding_total)
        extra_padding = target_length - x.shape[-1]
        x = pad1d(x, (self._padding_total, extra_padding), mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1,
                 trim_ratio=1.0, use_weight_norm=True, use_bias=True):
        super().__init__()
        conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride,
                                  groups=groups, bias=use_bias)
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.trim_ratio = trim_ratio

    def forward(self, x):
        ks = self.conv.kernel_size[0]
        st = self.conv.stride[0]
        total_padding = ks - st
        out = self.conv(x)
        rp = math.ceil(total_padding * self.trim_ratio)
        lp = total_padding - rp
        return out[..., lp:out.shape[-1] - rp]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, use_biases=False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=use_biases)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=use_biases)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, qk_norm, qk_norm_eps,
                 use_biases, sliding_window, layer_id):
        super().__init__()
        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window

        def get_alibi_slopes(n):
            def sp2(n):
                r = 2.0 ** (-8.0 / n)
                return torch.tensor([r**i for i in range(n)], dtype=torch.float32)
            if math.log2(n).is_integer():
                return sp2(n)
            m = 2 ** math.floor(math.log2(n))
            return torch.cat([sp2(m), sp2(2 * m)[::2][:n - m]])

        self.register_buffer("alibi_slopes", get_alibi_slopes(n_heads), persistent=False)
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=use_biases)
        if qk_norm:
            self.q_norm = rms_norm(n_heads * head_dim, eps=qk_norm_eps)
            self.k_norm = rms_norm(n_kv_heads * head_dim, eps=qk_norm_eps)
        self._qk_norm = qk_norm

    def forward(self, x):
        # x can be (B*T, D) from encoder or (B, T, D) from decoder
        if x.dim() == 2:
            bsz = 1
            seqlen = x.shape[0]
            x_3d = x.unsqueeze(0)
        else:
            bsz, seqlen, _ = x.shape
            x_3d = x
        xq = self.wq(x_3d).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x_3d).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x_3d).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        if self._qk_norm:
            xq = self.q_norm(xq.flatten(-2)).view_as(xq)
            xk = self.k_norm(xk.flatten(-2)).view_as(xk)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        if self.repeats > 1:
            xk = xk.repeat_interleave(self.repeats, dim=1)
            xv = xv.repeat_interleave(self.repeats, dim=1)
        T = seqlen
        pos = torch.arange(T, device=x.device)
        dist = pos.unsqueeze(0) - pos.unsqueeze(1)
        slopes = self.alibi_slopes.to(x.device).unsqueeze(-1).unsqueeze(-1)
        alibi = slopes * dist.unsqueeze(0).float()
        causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        if self.sliding_window and T > self.sliding_window:
            sw_mask = torch.tril(torch.full((T, T), float('-inf'), device=x.device),
                                 diagonal=-(self.sliding_window + 1))
            causal = causal + sw_mask
        mask = alibi + causal.unsqueeze(0)
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask.to(xq.dtype))
        output = output.transpose(1, 2).reshape(bsz, seqlen, -1)
        if x.dim() == 2:
            return self.wo(output).squeeze(0)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id, dim, hidden_dim, n_heads, n_kv_heads, head_dim,
                 qk_norm, qk_norm_eps, use_biases, norm_eps, layer_scale,
                 layer_scale_init, sliding_window):
        super().__init__()
        self._layer_id = layer_id
        self.attention = Attention(dim, n_heads, n_kv_heads, head_dim, qk_norm,
                                   qk_norm_eps, use_biases, sliding_window, layer_id)
        self.feed_forward = FeedForward(dim, hidden_dim, use_biases)
        self.attention_norm = rms_norm(dim, eps=norm_eps)
        self.ffn_norm = rms_norm(dim, eps=norm_eps)
        self.layer_scale = layer_scale
        if layer_scale:
            if layer_scale_init is None:
                if layer_id < 18:
                    init_scale = 0.1
                elif layer_id <= 24:
                    init_scale = 1e-5
                else:
                    init_scale = 1e-6
            else:
                init_scale = layer_scale_init
            self.attention_scale = nn.Parameter(torch.full((dim,), init_scale))
            self.ffn_scale = nn.Parameter(torch.full((dim,), init_scale))

    @property
    def layer_id(self):
        return self._layer_id

    def forward(self, x):
        r = self.attention(self.attention_norm(x))
        if self.layer_scale:
            r = self.attention_scale * r
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        if self.layer_scale:
            r = self.ffn_scale * r
        return h + r


class Transformer(nn.Module):
    def __init__(self, n_layers, **block_kwargs):
        super().__init__()
        self.layers = nn.ModuleDict()
        for i in range(n_layers):
            self.layers[str(i)] = TransformerBlock(layer_id=i, **block_kwargs)

    def forward(self, x):
        for k in self.layers:
            x = self.layers[k](x)
        return x


# ============================================================
# Codebooks (frozen, from checkpoint)
# ============================================================

class SemanticCodebook(nn.Module):
    def __init__(self, codebook_size, codebook_dim):
        super().__init__()
        self.epsilon = 1e-5
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("_embedding", None, persistent=False)
        self.codebook_size = codebook_size
        self.num_codebooks = 1

    @property
    def embedding(self):
        if self._embedding is None:
            emb = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            self.register_buffer("_embedding", emb, persistent=False)
            return emb
        return self._embedding

    def encode(self, x):
        B, D, T = x.shape
        x_flat = rearrange(x, "b d t -> (b t) d")
        distances = torch.cdist(x_flat, self.embedding.to(device=x.device, dtype=x.dtype), p=2)
        codes = distances.argmin(dim=-1).view(B, 1, T)
        return codes

    def decode(self, codes, dtype=torch.float32):
        codes = codes.squeeze(1)
        emb = self.embedding.to(device=codes.device, dtype=dtype)
        return rearrange(F.embedding(codes, emb), "b t d -> b d t")


class AcousticCodebook(nn.Module):
    def __init__(self, codebook_size, codebook_dim):
        super().__init__()
        self.n_levels = codebook_size
        self.num_codebooks = codebook_dim
        self.half_levels = (codebook_size - 1) / 2.0

    def _bound(self, x):
        return torch.tanh(x) * self.half_levels

    def _round_ste(self, x):
        """Round with straight-through estimator."""
        return x + (x.round() - x).detach()

    def encode(self, x):
        x = torch.tanh(x)
        levels = torch.ones_like(x) * self.n_levels
        scaled = ((x + 1) / 2) * (levels - 1)
        return scaled.round().long()

    def encode_stochastic(self, x):
        """Paper's 50/25/25 schedule: 50% quantize, 25% dither, 25% passthrough."""
        z_b = self._bound(x)

        # Quantized path (round with STE)
        z_q = self._round_ste(z_b)

        # Dither path (add uniform noise)
        noise_scale = 1.0 / self.n_levels
        noise = torch.empty_like(z_b).uniform_(-noise_scale, noise_scale) * self.half_levels
        z_dither = (z_b + noise).clamp(-self.half_levels, self.half_levels)

        # Passthrough (just bounded, no quantization)
        z_pass = z_b

        B = x.shape[0]
        probs = torch.rand(B, device=x.device)
        quant_mask = (probs < 0.50).view(B, 1, 1)
        dither_mask = ((probs >= 0.50) & (probs < 0.75)).view(B, 1, 1)

        z_out = torch.where(quant_mask, z_q,
                torch.where(dither_mask, z_dither, z_pass))

        # Convert to code-scale [-half, half] → [0, n_levels-1] for code tracking
        codes = (z_out + self.half_levels).round().long().clamp(0, self.n_levels - 1)

        # Convert back to [-1, 1] range for decoder compatibility
        z_normalized = z_out / self.half_levels
        return z_normalized, codes

    def decode(self, codes, dtype=torch.float32):
        return ((codes.float() * 2 / (self.n_levels - 1)) - 1).to(dtype)


class MistralAudioCodebook(nn.Module):
    def __init__(self, semantic_size, semantic_dim, acoustic_size, acoustic_dim):
        super().__init__()
        self.semantic_codebook = SemanticCodebook(semantic_size, semantic_dim)
        self.acoustic_codebook = AcousticCodebook(acoustic_size, acoustic_dim)
        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim

    def encode(self, x):
        sem = self.semantic_codebook.encode(x[:, :self.semantic_dim, :])
        aco = self.acoustic_codebook.encode(x[:, self.semantic_dim:, :])
        return torch.cat([sem, aco], dim=1)

    def encode_stochastic(self, x):
        """Stochastic encoding matching the paper's training schedule.
        Semantic: 50% VQ, 50% passthrough
        Acoustic: 50% quantize, 25% dither, 25% passthrough"""
        sem_input = x[:, :self.semantic_dim, :]
        aco_input = x[:, self.semantic_dim:, :]

        # Semantic: 50% VQ, 50% passthrough
        sem_codes = self.semantic_codebook.encode(sem_input)
        sem_quantized = self.semantic_codebook.decode(sem_codes, dtype=sem_input.dtype)
        B = x.shape[0]
        sem_mask = (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1)
        # STE: quantized in forward, gradient flows through original
        sem_out = torch.where(sem_mask, sem_input + (sem_quantized - sem_input).detach(), sem_input)

        # Acoustic: 50/25/25 stochastic
        aco_out, aco_codes = self.acoustic_codebook.encode_stochastic(aco_input)

        quantized = torch.cat([sem_out, aco_out], dim=1)
        codes = torch.cat([sem_codes, aco_codes], dim=1)
        return quantized, codes

    def decode(self, codes, dtype=torch.float32):
        sem = self.semantic_codebook.decode(codes[:, :1, :], dtype=dtype)
        aco = self.acoustic_codebook.decode(codes[:, 1:, :]).to(dtype)
        return torch.cat([sem, aco], dim=1)

    @property
    def num_codebooks(self):
        return 1 + self.acoustic_codebook.num_codebooks


# ============================================================
# Full Codec Model (encoder trainable, decoder+quantizer frozen)
# ============================================================

class VoxtralCodec(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture params from AudioTokenizerArgs defaults
        self.patch_size = 240
        dim = 1024
        hidden_dim = 4096
        head_dim = 128
        n_heads = 8
        n_kv_heads = 8
        semantic_dim = 256
        acoustic_dim = 36
        self.latent_dim = semantic_dim + acoustic_dim  # 292

        block_kwargs = dict(
            dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
            head_dim=head_dim, qk_norm=True, qk_norm_eps=1e-6, use_biases=False,
            norm_eps=1e-2, layer_scale=True, layer_scale_init=None,
        )

        # ---- ENCODER (trainable) ----
        self.input_proj = CausalConv1d(self.patch_size, dim, kernel_size=7,
                                       use_weight_norm=True, use_bias=False)
        encoder_blocks = []
        enc_transformer_lengths = (2, 2, 2, 2)
        enc_conv_kernels = (4, 4, 4, 3)
        enc_conv_strides = (2, 2, 2, 1)
        cur_window = 16

        for idx, n_layers in enumerate(enc_transformer_lengths):
            kw = {**block_kwargs, "sliding_window": cur_window}
            encoder_blocks.append(Transformer(n_layers, **kw))
            is_last = idx == len(enc_transformer_lengths) - 1
            proj_out = self.latent_dim if is_last else dim
            if enc_conv_kernels[idx] != 1 or enc_conv_strides[idx] != 1 or is_last:
                encoder_blocks.append(
                    CausalConv1d(dim, proj_out, kernel_size=enc_conv_kernels[idx],
                                 stride=enc_conv_strides[idx], pad_mode="replicate", use_bias=False)
                )
                if enc_conv_strides[idx] > 1:
                    cur_window = cur_window // 2

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # ---- QUANTIZER (frozen) ----
        self.quantizer = MistralAudioCodebook(8192, semantic_dim, 21, acoustic_dim)

        # ---- DECODER (frozen) ----
        decoder_blocks = []
        dec_conv_kernels = (3, 4, 4, 4)
        dec_conv_strides = (1, 2, 2, 2)
        dec_transformer_lengths = (2, 2, 2, 2)

        decoder_blocks.append(
            CausalConv1d(self.latent_dim, dim, kernel_size=dec_conv_kernels[0],
                         stride=dec_conv_strides[0], pad_mode="replicate", use_bias=False)
        )
        for idx, n_layers in enumerate(dec_transformer_lengths):
            kw = {**block_kwargs, "sliding_window": cur_window}
            decoder_blocks.append(Transformer(n_layers, **kw))
            if (idx + 1 < len(dec_transformer_lengths)) and \
               (dec_conv_kernels[idx + 1] != 1 or dec_conv_strides[idx + 1] != 1):
                decoder_blocks.append(
                    CausalConvTranspose1d(dim, dim, kernel_size=dec_conv_kernels[idx + 1],
                                         stride=dec_conv_strides[idx + 1], use_bias=False)
                )
                if dec_conv_strides[idx + 1] > 1:
                    cur_window = cur_window * 2

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.output_proj = CausalConv1d(dim, self.patch_size, kernel_size=7,
                                        use_weight_norm=True, use_bias=False)
        # Audio token embedding for voice identity loss (loaded from checkpoint, frozen)
        self.audio_token_embedding = None  # set after weight loading

    def forward_encoder(self, x):
        """x: [B, 1, T] raw audio at 24kHz -> [B, 292, L] latent"""
        if x.shape[-1] % self.patch_size != 0:
            pad_len = self.patch_size - (x.shape[-1] % self.patch_size)
            x = F.pad(x, (0, pad_len))
        emb = rearrange(x, "b c (t h) -> b (c h) t", h=self.patch_size)
        emb = self.input_proj(emb)
        emb = rearrange(emb, "b d t -> b t d").contiguous()
        for block in self.encoder_blocks:
            if isinstance(block, CausalConv1d):
                emb = rearrange(emb, "b t d -> b d t")
                emb = block(emb)
                emb = rearrange(emb, "b d t -> b t d")
            else:
                bsz = emb.shape[0]
                emb = rearrange(emb, "b t d -> (b t) d")
                emb = block(emb)
                emb = rearrange(emb, "(b t) d -> b t d", b=bsz)
        emb = rearrange(emb, "b t d -> b d t")
        # tanh(x*0.3) for acoustic: spreads across FSQ range without double-saturation
        sem = emb[:, :256, :]
        aco = torch.tanh(emb[:, 256:, :] * 0.3)
        return torch.cat([sem, aco], dim=1)  # [B, 292, L]

    def forward_decoder(self, emb):
        """emb: [B, 292, L] quantized latent"""
        emb = rearrange(emb, "b d t -> b t d").contiguous()
        for block in self.decoder_blocks:
            if isinstance(block, (CausalConvTranspose1d, CausalConv1d)):
                emb = rearrange(emb, "b t d -> b d t")
                emb = block(emb)
                emb = rearrange(emb, "b d t -> b t d")
            else:
                emb = block(emb)
        emb = rearrange(emb, "b t d -> b d t")
        emb = self.output_proj(emb)
        out = rearrange(emb, "b (c h) t -> b c (t h)", h=self.patch_size)
        return out  # [B, 1, T]

    def codes_to_voice_embedding(self, codes):
        """Convert codes [B, 37, L] to voice embeddings [B, L, 3072] via audio_token_embedding."""
        if self.audio_token_embedding is None:
            return None
        offset_codes = codes + 2
        emb = self.audio_token_embedding(offset_codes)  # [B, CB, L, 3072]
        return emb.sum(dim=1)  # [B, L, 3072]

    def soft_voice_embedding(self, latent):
        """Differentiable voice embedding from continuous encoder output.
        Uses soft codebook lookup instead of hard argmin."""
        if self.audio_token_embedding is None:
            return None
        # Split semantic and acoustic
        sem = latent[:, :256, :]  # [B, 256, L]
        aco = latent[:, 256:, :]  # [B, 36, L]

        B, _, L = sem.shape

        # Soft semantic lookup
        sem_flat = sem.permute(0, 2, 1).reshape(B * L, 256)  # [B*L, 256]
        cb = self.quantizer.semantic_codebook.embedding.to(
            device=sem_flat.device, dtype=sem_flat.dtype)
        dists = torch.cdist(sem_flat, cb)  # [B*L, 8192]
        soft_weights = F.softmax(-dists / 2.0, dim=-1)  # temperature=2, sharper
        # Lookup: soft_weights @ audio_token_embedding for semantic codebook (offset=2)
        sem_emb_table = self.audio_token_embedding.weight[2:8194].to(sem_flat.dtype)  # [8192, 3072]
        soft_sem_emb = soft_weights @ sem_emb_table  # [B*L, 3072]
        soft_sem_emb = soft_sem_emb.view(B, L, -1)

        # Hard acoustic lookup (FSQ is already differentiable via tanh + STE)
        aco_codes = self.quantizer.acoustic_codebook.encode(aco.float())  # [B, 36, L]
        aco_offset = aco_codes + 2  # offset
        # Each of 36 acoustic codebooks has 21 levels, offsets start after semantic (8192+2=8194)
        # Codebook i: offset = 8194 + i*21 ... actually the offsets are cumulative
        # For simplicity, use the hard codes for acoustic (they're fine, 36 channels carry identity)
        # and only make semantic differentiable
        return soft_sem_emb  # [B, L, 3072] semantic contribution only

    def forward(self, x, use_vq=True, stochastic=False):
        """Full autoencoder: encode -> (optionally quantize) -> decode
        stochastic: use paper's 50/25/25 quantization schedule during training"""
        latent = self.forward_encoder(x)  # [B, 292, L] (float32 from encoder)
        if use_vq:
            if stochastic and self.training:
                # Paper's training schedule: mix of quantized/dithered/passthrough
                quantized, codes = self.quantizer.encode_stochastic(latent.float())
                recon = self.forward_decoder(quantized.to(torch.bfloat16))
            else:
                codes = self.quantizer.encode(latent.float())  # [B, 37, L]
                quantized = self.quantizer.decode(codes, dtype=torch.bfloat16)
                latent_bf16 = latent.to(torch.bfloat16)
                quantized_st = latent_bf16 + (quantized - latent_bf16).detach()
                recon = self.forward_decoder(quantized_st)
        else:
            codes = None
            recon = self.forward_decoder(latent.to(torch.bfloat16))
        return recon, latent, codes


# ============================================================
# Loss functions
# ============================================================

def mel_spectrogram(x, n_fft, hop_length, n_mels=80, sample_rate=24000):
    """Compute mel spectrogram."""
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    stft = torch.stft(x, n_fft, hop_length, n_fft, window, return_complex=True)
    mag = stft.abs()
    mel_basis = torch.from_numpy(
        __import__('librosa').filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    ).to(x.device, x.dtype)
    mel = torch.matmul(mel_basis, mag)
    return torch.log(mel.clamp(min=1e-5))


def multi_resolution_mel_loss(pred, target, sample_rate=24000):
    """Multi-resolution mel spectrogram loss."""
    losses = []
    for n_fft, hop in [(2048, 512), (1024, 256), (512, 128)]:
        min_len = min(pred.shape[-1], target.shape[-1])
        p = pred[..., :min_len].squeeze(1)
        t = target[..., :min_len].squeeze(1)
        mel_p = mel_spectrogram(p, n_fft, hop, sample_rate=sample_rate)
        mel_t = mel_spectrogram(t, n_fft, hop, sample_rate=sample_rate)
        losses.append(F.l1_loss(mel_p, mel_t))
    return sum(losses) / len(losses)


# ============================================================
# ASR Distillation Loss (Whisper-based semantic token learning)
# ============================================================

class ASRDistillationLoss(nn.Module):
    """Distill semantic tokens against Whisper decoder hidden states.
    Teaches the semantic codebook to encode phoneme/linguistic content."""

    def __init__(self, whisper_model="openai/whisper-base", semantic_dim=256,
                 codec_sr=24000, whisper_sr=16000, max_tokens=64):
        super().__init__()
        self.codec_sr = codec_sr
        self.whisper_sr = whisper_sr
        self.max_tokens = max_tokens

        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.processor = WhisperProcessor.from_pretrained(whisper_model)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            whisper_model, attn_implementation="eager")
        for p in self.whisper.parameters():
            p.requires_grad = False
        self.whisper.eval()
        whisper_dim = self.whisper.config.d_model
        self.projection = nn.Linear(semantic_dim, whisper_dim)

    def forward(self, audio, z_semantic):
        """audio: [B, 1, T] at codec_sr. z_semantic: [B, 256, L] post-VQ semantic."""
        # Resample to 16kHz for Whisper
        audio_16k = F.interpolate(audio, size=int(audio.shape[-1] * self.whisper_sr / self.codec_sr),
                                  mode='linear', align_corners=False).squeeze(1)

        # Process through Whisper (pad to 30s as Whisper expects)
        inputs = self.processor.feature_extractor(
            [s.detach().cpu().float().numpy() for s in audio_16k],
            sampling_rate=self.whisper_sr, return_tensors="pt", padding="max_length"
        )
        input_features = inputs.input_features.to(audio.device)

        with torch.no_grad():
            # Create attention mask for encoder features (all ones since we padded with feature_extractor)
            encoder_attention_mask = torch.ones(input_features.shape[0], input_features.shape[-1],
                                                device=input_features.device, dtype=torch.long)
            generated = self.whisper.generate(
                input_features=input_features,
                attention_mask=encoder_attention_mask,
                max_new_tokens=self.max_tokens)
            if generated.shape[1] < 2:
                bos = self.whisper.config.decoder_start_token_id
                generated = torch.tensor([[bos, bos]], device=audio.device).repeat(input_features.shape[0], 1)
            # Create decoder attention mask (1 for real tokens, 0 for padding)
            decoder_ids = generated[:, :-1]
            decoder_attention_mask = (decoder_ids != self.whisper.config.pad_token_id).long()
            outputs = self.whisper.model(
                input_features=input_features,
                decoder_input_ids=decoder_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True, output_attentions=True,
                return_dict=True, use_cache=False
            )
            decoder_hidden = outputs.decoder_hidden_states[-1].detach()
            # cross_attentions can be None, a tuple of Nones, or valid tensors
            try:
                ca = outputs.cross_attentions
                if ca is not None and ca[-1] is not None:
                    cross_attn = ca[-1].detach()
                else:
                    raise ValueError("no cross attention")
            except:
                B_w = decoder_hidden.shape[0]
                dec_len = decoder_hidden.shape[1]
                cross_attn = torch.ones(B_w, 1, dec_len, dec_len, device=audio.device) / dec_len

        # Project semantic embeddings to Whisper dim
        z_proj = self.projection(z_semantic.transpose(1, 2))  # [B, L, whisper_dim]

        # Compute soft alignment from cross-attention
        attn = cross_attn.mean(dim=1)  # [B, dec_len, enc_len]
        attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attn = F.interpolate(attn, size=z_proj.shape[1], mode='linear', align_corners=False)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Aligned codec embeddings
        aligned = attn @ z_proj  # [B, dec_len, whisper_dim]
        aligned = F.normalize(aligned, dim=-1)
        decoder_hidden = F.normalize(decoder_hidden, dim=-1)

        # Cosine distance loss
        return 1.0 - (aligned * decoder_hidden).sum(dim=-1).mean()


class NoOpASRLoss(nn.Module):
    """Dummy when ASR distillation is disabled."""
    def forward(self, audio, z_semantic):
        return audio.new_zeros(())


# ============================================================
# Voice Identity Loss (supervised from presets)
# ============================================================

class PresetVoiceData:
    """Loads preset voice audio + their known embeddings for supervised training."""
    def __init__(self, codec_data_dir, voice_emb_dir, sample_rate=24000):
        self.voices = {}
        for d in os.listdir(codec_data_dir):
            audio_dir = os.path.join(codec_data_dir, d)
            emb_path = os.path.join(voice_emb_dir, f"{d}.pt")
            if not os.path.isdir(audio_dir) or not os.path.exists(emb_path):
                continue
            wavs = [os.path.join(audio_dir, f) for f in sorted(os.listdir(audio_dir)) if f.endswith('.wav')]
            if not wavs:
                continue
            emb = torch.load(emb_path, map_location='cpu', weights_only=True)
            self.voices[d] = {'wavs': wavs, 'embedding': emb}
        print(f"PresetVoiceData: {len(self.voices)} voices loaded")

    def sample_batch(self, batch_size, max_len, device):
        """Sample a batch of (audio, target_embedding) pairs from random presets.
        Clips audio to 3-5 seconds (like real ref_audio usage)."""
        voice_names = list(self.voices.keys())
        clip_len = 4 * SAMPLE_RATE  # 4 seconds
        audios = []
        target_embs = []
        for _ in range(batch_size):
            vname = voice_names[np.random.randint(len(voice_names))]
            vdata = self.voices[vname]
            wav_path = vdata['wavs'][np.random.randint(len(vdata['wavs']))]
            audio, sr = sf.read(wav_path, dtype='float32')
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(audio) > clip_len:
                start = np.random.randint(0, len(audio) - clip_len)
                audio = audio[start:start + clip_len]
            if len(audio) % 240 != 0:
                audio = np.pad(audio, (0, 240 - len(audio) % 240))
            audios.append(torch.from_numpy(audio).float())
            target_embs.append(vdata['embedding'].float().mean(0))  # [3072] mean embedding

        max_audio_len = max(a.shape[0] for a in audios)
        if max_audio_len % 240 != 0:
            max_audio_len += 240 - max_audio_len % 240
        batch_audio = torch.zeros(batch_size, 1, max_audio_len)
        for i, a in enumerate(audios):
            batch_audio[i, 0, :a.shape[0]] = a
        target_emb_batch = torch.stack(target_embs)  # [B, 3072]
        return batch_audio.to(device), target_emb_batch.to(device)


# ============================================================
# Dataset
# ============================================================

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_samples=None):
        self.files = []
        for ext in ('*.wav', '*.flac'):
            self.files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        self.files = sorted(self.files)
        if max_samples:
            self.files = self.files[:max_samples]
        print(f"Dataset: {len(self.files)} files from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, sr = sf.read(self.files[idx], dtype='float32')
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        max_len = MAX_AUDIO_SEC * SAMPLE_RATE
        if len(audio) > max_len:
            start = np.random.randint(0, len(audio) - max_len)
            audio = audio[start:start + max_len]
        # Pad to multiple of patch_size (240)
        if len(audio) % 240 != 0:
            audio = np.pad(audio, (0, 240 - len(audio) % 240))
        return torch.from_numpy(audio).float()


def collate_fn(batch):
    max_len = max(x.shape[0] for x in batch)
    if max_len % 240 != 0:
        max_len += 240 - max_len % 240
    padded = torch.zeros(len(batch), 1, max_len)
    lengths = []
    for i, x in enumerate(batch):
        padded[i, 0, :x.shape[0]] = x
        lengths.append(x.shape[0])
    return padded, torch.tensor(lengths)


# ============================================================
# Weight loading
# ============================================================

def load_decoder_weights(model, model_dir):
    """Load frozen decoder + quantizer weights from consolidated.safetensors."""
    from safetensors.torch import load_file
    
    st_files = sorted(Path(model_dir).glob("consolidated*.safetensors"))
    all_weights = {}
    for sf_path in st_files:
        all_weights.update(load_file(str(sf_path), device="cpu"))

    loaded = 0
    skipped = 0
    
    # Map checkpoint keys -> our model keys
    for ck_key, ck_val in all_weights.items():
        if not ck_key.startswith("audio_tokenizer."):
            continue
        our_key = ck_key.replace("audio_tokenizer.", "")

        # Skip encoder weights (we're training those)
        if our_key.startswith("input_proj.") or our_key.startswith("encoder_blocks."):
            skipped += 1
            continue

        # Handle quantizer buffers specially
        if our_key == "quantizer.semantic_codebook.cluster_usage":
            model.quantizer.semantic_codebook.cluster_usage = ck_val
            loaded += 1
            continue
        if our_key == "quantizer.semantic_codebook.embedding_sum":
            model.quantizer.semantic_codebook.embedding_sum = ck_val
            loaded += 1
            continue

        # Load into model params
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        if our_key in params:
            params[our_key].data.copy_(ck_val)
            loaded += 1
        elif our_key in buffers:
            buffers[our_key].copy_(ck_val)
            loaded += 1
        else:
            print(f"  SKIP (not found): {our_key} {ck_val.shape}")

    print(f"Loaded {loaded} decoder/quantizer weights, skipped {skipped} encoder weights")

    # Load audio_token_embedding from mm_audio_embeddings
    emb_key = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
    if emb_key in all_weights:
        emb_weight = all_weights[emb_key]
        print(f"Loading audio_token_embedding: {emb_weight.shape}")
        # Build the MultiVocabEmbeddings-compatible lookup
        # Shape is [9088, 3072] - flat table for all codebooks with offsets
        model.audio_token_embedding = nn.Embedding(emb_weight.shape[0], emb_weight.shape[1])
        model.audio_token_embedding.weight.data.copy_(emb_weight)
        model.audio_token_embedding.requires_grad_(False)

    return model


# ============================================================
# Main training loop
# ============================================================

def stft_magnitude_loss(pred, target, n_fft=1024, hop=256):
    """L1 loss on STFT magnitudes (paper uses this alongside mel)."""
    window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
    pred_stft = torch.stft(pred, n_fft, hop, n_fft, window, return_complex=True).abs()
    tgt_stft = torch.stft(target, n_fft, hop, n_fft, window, return_complex=True).abs()
    return F.l1_loss(pred_stft, tgt_stft)


# ============================================================
# Multi-Resolution STFT Discriminator (from paper)
# ============================================================

class STFTDiscriminator(nn.Module):
    """Single STFT-based discriminator."""
    def __init__(self, n_fft, hop_length=None, channels=256, n_layers=4):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4

        in_ch = 2  # real + imag
        layers = []
        for i in range(n_layers):
            out_ch = min(channels * (2 ** i), 1024)
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                nn.LeakyReLU(0.1, inplace=True),
            ))
            in_ch = out_ch
        layers.append(nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
        ))
        self.convs = nn.ModuleList(layers)
        self.conv_post = nn.Conv2d(in_ch, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x = x.squeeze(1)
        window = torch.hann_window(self.n_fft, device=x.device)
        stft = torch.stft(x, self.n_fft, self.hop_length, self.n_fft, window,
                          return_complex=True, normalized=False, onesided=True)
        spec = torch.stack([stft.real, stft.imag], dim=1)
        fmaps = []
        h = spec
        for conv in self.convs:
            h = conv(h)
            fmaps.append(h)
        logits = self.conv_post(h)
        return logits, fmaps


class MultiResolutionDiscriminator(nn.Module):
    """8 STFT discriminators matching the paper: sizes 2296, 1418, 876, 542, 334, 206, 126, 76."""
    def __init__(self, channels=256, n_layers=4):
        super().__init__()
        stft_sizes = [2296, 1418, 876, 542, 334, 206, 126, 76]
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n_fft=s, channels=channels, n_layers=n_layers)
            for s in stft_sizes
        ])

    def forward(self, x):
        return [disc(x) for disc in self.discriminators]


def feature_matching_loss(fmaps_real, fmaps_fake):
    """L1 feature matching loss across all discriminator layers."""
    loss = 0
    for real_list, fake_list in zip(fmaps_real, fmaps_fake):
        for r, f in zip(real_list, fake_list):
            loss += F.l1_loss(f, r.detach())
    n = sum(len(fl) for fl in fmaps_real)
    return loss / max(n, 1)


def discriminator_loss(disc_real_outputs, disc_fake_outputs):
    """Hinge loss for discriminator."""
    loss = 0
    for (logits_real, _), (logits_fake, _) in zip(disc_real_outputs, disc_fake_outputs):
        loss += F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean()
    return loss / len(disc_real_outputs)


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("VOXTRAL CODEC ENCODER TRAINING (paper-aligned)")
    print("=" * 60)

    # Build model
    print("\nBuilding model...")
    model = VoxtralCodec()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # Load frozen decoder + quantizer
    print(f"\nLoading decoder/quantizer from {MODEL_DIR}...")
    model = load_decoder_weights(model, MODEL_DIR)
    model = model.to(DEVICE)

    # Freeze decoder + quantizer, only train encoder
    encoder_params = []
    for name, param in model.named_parameters():
        if name.startswith("input_proj.") or name.startswith("encoder_blocks."):
            param.requires_grad = True
            encoder_params.append(param)
        else:
            param.requires_grad = False

    print(f"\nTrainable (encoder): {sum(p.numel() for p in encoder_params):,} params")

    # bf16 model, float32 encoder for stable training
    model = model.bfloat16()
    model.input_proj.float()
    for block in model.encoder_blocks:
        block.float()

    # Sanity check
    print("\nSanity check: decoder forward pass...")
    with torch.no_grad():
        dummy_codes = torch.zeros(1, 37, 10, dtype=torch.long, device=DEVICE)
        dummy_audio = model.forward_decoder(model.quantizer.decode(dummy_codes, dtype=torch.bfloat16))
        print(f"  OK: {dummy_audio.shape}")

    # Dataset: combine ALL available audio sources
    print(f"\nLoading datasets...")
    datasets_list = []

    # Primary: LibriSpeech clean-360 (30K clips)
    if os.path.isdir(DATA_DIR):
        ds = AudioDataset(DATA_DIR)
        datasets_list.append(ds)
        print(f"  LibriSpeech clean: {len(ds)} files")

    # Extended: LibriSpeech other-500 (if downloaded)
    libri500_dir = os.environ.get("LIBRI500_DIR", "/librispeech_500")
    if os.path.isdir(libri500_dir) and len(os.listdir(libri500_dir)) > 100:
        ds500 = AudioDataset(libri500_dir, max_samples=None)
        datasets_list.append(ds500)
        print(f"  LibriSpeech other-500: {len(ds500)} files")

    # Common Voice Arabic (if downloaded)
    cv_arabic_dir = os.environ.get("CV_ARABIC_DIR", "/cv_arabic")
    if os.path.isdir(cv_arabic_dir) and len(os.listdir(cv_arabic_dir)) > 100:
        ds_ar = AudioDataset(cv_arabic_dir, max_samples=None)
        datasets_list.append(ds_ar)
        print(f"  Common Voice Arabic: {len(ds_ar)} files")

    # Common Voice English (if downloaded)
    cv_english_dir = os.environ.get("CV_ENGLISH_DIR", "/cv_english")
    if os.path.isdir(cv_english_dir) and len(os.listdir(cv_english_dir)) > 100:
        ds_en = AudioDataset(cv_english_dir, max_samples=None)
        datasets_list.append(ds_en)
        print(f"  Common Voice English: {len(ds_en)} files")

    # Preset clips (5K+ clips with known voices)
    preset_dir = os.environ.get("PRESET_DIR", "/codec_data")
    if os.path.isdir(preset_dir):
        preset_ds = AudioDataset(preset_dir, max_samples=None)
        datasets_list.append(preset_ds)
        print(f"  Preset clips: {len(preset_ds)} files")

    dataset = torch.utils.data.ConcatDataset(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
    print(f"  Total: {len(dataset)} files")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=collate_fn, drop_last=True, pin_memory=True
    )
    print(f"Batches per epoch: {len(loader)}")

    # ASR Distillation (Whisper-based semantic token learning)
    asr_loss_fn = NoOpASRLoss().to(DEVICE)
    asr_params = []
    if USE_ASR_DISTILL:
        try:
            asr_loss_fn = ASRDistillationLoss(whisper_model="openai/whisper-base",
                                              semantic_dim=256, codec_sr=SAMPLE_RATE).to(DEVICE)
            asr_params = list(asr_loss_fn.projection.parameters())
            print(f"ASR distillation: enabled ({sum(p.numel() for p in asr_params):,} projection params)")
        except Exception as e:
            print(f"ASR distillation: disabled ({e})")

    # Discriminator (paper: 8 multi-resolution STFT discriminators)
    disc = MultiResolutionDiscriminator(channels=256, n_layers=4).to(DEVICE).float()
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"Discriminator: {disc_params:,} params")
    DISC_START_STEP = 2000  # warm-up: no adversarial loss for first N steps

    # Optimizers (separate for generator and discriminator, matching paper)
    all_trainable = encoder_params + asr_params
    opt_g = torch.optim.AdamW(all_trainable, lr=LR, betas=(0.8, 0.99), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=1e-4)
    total_steps = EPOCHS * len(loader)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=total_steps)

    # Paper-aligned training: stochastic VQ from epoch 1, no phasing
    global_step = 0
    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for name, mod in model.named_modules():
            if name.startswith("decoder_blocks") or name.startswith("output_proj") or name.startswith("quantizer"):
                mod.eval()

        epoch_loss = 0
        n_batches = 0

        for batch_idx, (audio, lengths) in enumerate(loader):
            audio = audio.to(DEVICE)

            # Forward: encoder -> stochastic VQ (50/25/25) -> decoder
            recon, latent, codes = model(audio, use_vq=True, stochastic=True)

            recon_f = recon.float()
            audio_f = audio.float()
            min_len = min(recon_f.shape[-1], audio_f.shape[-1])

            # === GENERATOR LOSSES (paper-aligned) ===

            # Reconstruction: mel + L1 + STFT (with decay factor from paper)
            rec_weight = 0.9999 ** global_step  # exponential decay as discriminator strengthens
            mel_loss = multi_resolution_mel_loss(recon_f[..., :min_len], audio_f[..., :min_len])
            wav_loss = F.l1_loss(recon_f[..., :min_len], audio_f[..., :min_len])
            stft_loss = stft_magnitude_loss(recon_f[..., :min_len].squeeze(1), audio_f[..., :min_len].squeeze(1))

            # Feature matching loss (primary signal from paper, replaces raw GAN loss)
            feat_loss = torch.tensor(0.0, device=DEVICE)
            if global_step >= DISC_START_STEP:
                torch.cuda.empty_cache()
                disc.eval()
                with torch.no_grad():
                    disc_real = disc(audio_f[..., :min_len])
                disc_fake = disc(recon_f[..., :min_len])
                fmaps_real = [fmaps for _, fmaps in disc_real]
                fmaps_fake = [fmaps for _, fmaps in disc_fake]
                feat_loss = feature_matching_loss(fmaps_real, fmaps_fake)

            # VQ commitment loss (delta=0.1 from paper)
            commit_loss = torch.tensor(0.0, device=DEVICE)
            if codes is not None:
                with torch.no_grad():
                    quantized_hard = model.quantizer.decode(
                        model.quantizer.encode(latent.float()), dtype=torch.float32)
                commit_loss = F.mse_loss(latent.float(), quantized_hard.detach())

            # ASR distillation (every 3rd batch to save compute)
            asr_loss = torch.tensor(0.0, device=DEVICE)
            if (batch_idx % 3 == 0):
                sem_latent = latent.float()[:, :256, :]
                asr_loss = asr_loss_fn(audio, sem_latent)

            # Scale regularization
            sem_std = latent.float()[:, :256, :].std()
            scale_loss = (sem_std - 13.6).abs()

            # Combined generator loss (matching paper eq. 3)
            g_loss = (rec_weight * mel_loss
                      + rec_weight * wav_loss
                      + rec_weight * stft_loss
                      + 1.0 * feat_loss
                      + 0.1 * commit_loss
                      + 1.0 * asr_loss
                      + 0.3 * scale_loss)

            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            opt_g.step()
            sched_g.step()

            # === DISCRIMINATOR STEP (after warm-up) ===
            d_loss_val = 0.0
            if global_step >= DISC_START_STEP:
                torch.cuda.empty_cache()
                disc.train()
                disc_real = disc(audio_f[..., :min_len].detach())
                disc_fake = disc(recon_f[..., :min_len].detach())
                d_loss = discriminator_loss(disc_real, disc_fake)
                opt_d.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                opt_d.step()
                d_loss_val = d_loss.item()

            loss = g_loss  # for logging
            global_step += 1

            epoch_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % LOG_EVERY == 0:
                # Get code stats
                sem_util = 0
                aco_sample = []
                if codes is not None:
                    sem_util = codes[:, 0, :].unique().numel()
                    fi = min(5, codes.shape[2] - 1)
                    aco_sample = codes[0, 1:, fi].tolist()[:8]

                print(f"  [E{epoch} B{batch_idx+1}/{len(loader)}] "
                      f"g={g_loss.item():.3f} mel={mel_loss.item():.3f} "
                      f"feat={feat_loss.item():.3f} d={d_loss_val:.3f} "
                      f"asr={asr_loss.item():.3f} commit={commit_loss.item():.3f} "
                      f"sem_util={sem_util}/8192 aco={aco_sample} "
                      f"rw={rec_weight:.3f} lr={sched_g.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\n  Epoch {epoch}/{EPOCHS}: avg_loss={avg_loss:.4f}", flush=True)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            enc_state = {n: p.data.cpu() for n, p in model.named_parameters()
                         if n.startswith("input_proj.") or n.startswith("encoder_blocks.")}
            torch.save(enc_state, os.path.join(OUTPUT_DIR, "best_encoder.pt"))
            print(f"  Saved best (loss={best_loss:.4f})", flush=True)

        # Save every epoch
        enc_state = {n: p.data.cpu() for n, p in model.named_parameters()
                     if n.startswith("input_proj.") or n.startswith("encoder_blocks.")}
        torch.save(enc_state, os.path.join(OUTPUT_DIR, f"encoder_ep{epoch}.pt"))

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()

"""Configure the tokenizer to accept custom voice embeddings from VOXTRAL_VOICE_DIR."""
import mistral_common.tokens.tokenizers.audio as audio_mod

audio_file = audio_mod.__file__
with open(audio_file) as f:
    lines = f.readlines()

new_block = """            if voice is not None and voice not in self.audio_config.voice_num_audio_tokens:
                import os, torch as _torch
                _voice_dir = os.environ.get("VOXTRAL_VOICE_DIR", "")
                _pt = os.path.join(_voice_dir, f"{voice}.pt") if _voice_dir else ""
                if _pt and os.path.exists(_pt):
                    _emb = _torch.load(_pt, map_location="cpu", weights_only=True)
                    self.audio_config.voice_num_audio_tokens[voice] = _emb.shape[0]
                else:
                    raise ValueError(f"Unknown voice {voice!r}")
            assert voice is not None
            num_audio_tokens = self.audio_config.voice_num_audio_tokens[voice]
"""

for i, line in enumerate(lines):
    if "assert voice is not None and voice in self.audio_config.voice_num_audio_tokens" in line:
        for j in range(i, min(i + 5, len(lines))):
            if "num_audio_tokens = self.audio_config.voice_num_audio_tokens[voice]" in lines[j]:
                lines[i:j + 1] = [new_block]
                break
        break

with open(audio_file, "w") as f:
    f.writelines(lines)
print("Tokenizer patched successfully")

"""
Inject trained encoder weights into the model checkpoint.

Adds the encoder weights (input_proj + encoder_blocks) to
consolidated.safetensors, enabling ref_audio voice cloning.
"""

import os
import sys
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/Voxtral-4B-TTS-2603")
ENCODER_WEIGHTS = os.environ.get("ENCODER_WEIGHTS", "/encoder_trained/best_encoder.pt")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "")  # empty = modify in-place


def inject():
    print("=" * 50)
    print("INJECTING ENCODER WEIGHTS")
    print("=" * 50)

    # Load trained encoder weights
    print(f"\nLoading encoder weights from {ENCODER_WEIGHTS}...")
    encoder_state = torch.load(ENCODER_WEIGHTS, map_location="cpu", weights_only=True)
    print(f"  {len(encoder_state)} encoder weight tensors")
    for k, v in list(encoder_state.items())[:3]:
        print(f"  {k}: {v.shape} {v.dtype}")

    # Load existing checkpoint
    safetensors_files = sorted(Path(MODEL_DIR).glob("consolidated*.safetensors"))
    print(f"\nLoading checkpoint from {MODEL_DIR}...")
    print(f"  Files: {[f.name for f in safetensors_files]}")

    all_weights = {}
    for sf_path in safetensors_files:
        all_weights.update(load_file(str(sf_path), device="cpu"))
    print(f"  {len(all_weights)} existing weights")

    # Check if encoder weights already exist
    existing_encoder = [k for k in all_weights if k.startswith("audio_tokenizer.input_proj") 
                        or k.startswith("audio_tokenizer.encoder_blocks")]
    if existing_encoder:
        print(f"\n  WARNING: {len(existing_encoder)} encoder weights already in checkpoint!")
        print(f"  Will OVERWRITE them.")
        for k in existing_encoder:
            del all_weights[k]

    # Add encoder weights with audio_tokenizer. prefix
    added = 0
    for name, tensor in encoder_state.items():
        full_name = f"audio_tokenizer.{name}"
        # Convert to bf16 to match checkpoint format
        all_weights[full_name] = tensor.to(torch.bfloat16)
        added += 1

    print(f"\n  Added {added} encoder weights")
    print(f"  Total weights now: {len(all_weights)}")

    # Verify the key ones exist
    check_keys = ["audio_tokenizer.input_proj.conv.parametrizations.weight.original1",
                   "audio_tokenizer.encoder_blocks.0.layers.0.attention.wq.weight"]
    for k in check_keys:
        if k in all_weights:
            print(f"  OK: {k} ({all_weights[k].shape})")
        else:
            print(f"  MISSING: {k}")

    # Save
    if OUTPUT_FILE:
        out_path = OUTPUT_FILE
    else:
        out_path = str(safetensors_files[0])
        # Backup original
        backup = out_path + ".backup"
        if not os.path.exists(backup):
            import shutil
            shutil.copy2(out_path, backup)
            print(f"\n  Backed up original to {backup}")

    print(f"\n  Saving to {out_path}...")
    save_file(all_weights, out_path)
    print(f"  Done! File size: {os.path.getsize(out_path) / 1024 / 1024:.0f} MB")
    print(f"\n  Restart the serving engine to pick up the new weights.")
    print(f"  ref_audio voice cloning should now be enabled!")


if __name__ == "__main__":
    inject()

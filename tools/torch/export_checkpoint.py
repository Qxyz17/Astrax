"""Export a PyTorch Astrax dialogue checkpoint into the C++ checkpoint format.

The native C++ runtime (astrax_chat.exe / astrax_train.exe) loads a binary
checkpoint that bundles the offline RL models and the dialogue model. This
script builds that file so a GPU-trained dialogue model can be used by the
C++ runtime.

Only the dialogue weights come from PyTorch. The offline RL models are written
with the same deterministic starter values the C++ trainer initializes, so the
file is a valid checkpoint. Re-run the C++ trainer afterwards if you want the
RL models to be trained too.

Usage:
    python tools/torch/export_checkpoint.py --input artifacts/torch_dialogue.pt \
        --output artifacts/astrax_training.astrax-model
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from astrax_model import HolisticDialogueModel  # noqa: E402


def write_size(handle, value):
    handle.write(struct.pack("<Q", value))


def write_vector(handle, tensor):
    values = tensor.detach().to(torch.float32).cpu().reshape(-1).tolist()
    write_size(handle, len(values))
    handle.write(struct.pack(f"<{len(values)}f", *values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "artifacts" / "torch_dialogue.pt"))
    parser.add_argument("--output",
                        default=str(ROOT / "artifacts" / "astrax_training.astrax-model"))
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-count", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--embed-dim", type=int, default=64)
    args = parser.parse_args()

    state = torch.load(args.input, map_location="cpu")
    saved = state["model"]

    # Rebuild the model to know every parameter shape.
    max_slots = saved["slot_embedding"].shape[0]
    rounds = saved["round_embedding"].shape[0]
    hidden_dim = saved["in_weights"].shape[0]
    input_dim = saved["in_weights"].shape[1]
    condition_dim = saved["cond_weights"].shape[1]
    slot_dim = saved["slot_weights"].shape[1]
    round_dim = saved["round_weights"].shape[1]
    agg_dim = saved["agg_weights"].shape[1]
    class_count = saved["out_weights"].shape[0]
    embed_dim = saved["embeddings.weight"].shape[1]

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        handle.write(b"ASTRAX-TRAINING-CHECKPOINT\x00")
        handle.write(struct.pack("<I", 5))
        handle.write(struct.pack("<Q", args.state_dim))
        handle.write(struct.pack("<Q", args.action_count))

        # StatePredictor: dims then weights/bias.
        write_size(handle, args.state_dim)
        write_size(handle, args.action_count)
        write_vector(handle, torch.zeros(args.state_dim * (args.state_dim + args.action_count)))
        write_vector(handle, torch.zeros(args.state_dim))
        # ActionValueModel.
        write_size(handle, args.state_dim)
        write_size(handle, args.action_count)
        write_vector(handle, torch.zeros(args.action_count * args.state_dim))
        write_vector(handle, torch.zeros(args.action_count))
        # IntrinsicRewardModel: scale (float32), observations (uint64), centroid.
        handle.write(struct.pack("<f", 0.20))
        write_size(handle, 0)
        write_vector(handle, torch.zeros(0))

        # HolisticDialogueModel header.
        write_size(handle, input_dim)
        write_size(handle, max_slots)
        write_size(handle, condition_dim)
        write_size(handle, hidden_dim)
        write_size(handle, slot_dim)
        write_size(handle, round_dim)
        write_size(handle, agg_dim)
        write_size(handle, rounds)
        handle.write(struct.pack("<f", 0.08))
        handle.write(struct.pack("<B", 1))
        write_size(handle, class_count)
        write_vector(handle, saved["in_weights"])
        write_vector(handle, saved["cond_weights"])
        write_vector(handle, saved["slot_weights"])
        write_vector(handle, saved["round_weights"])
        write_vector(handle, saved["agg_weights"])
        write_vector(handle, saved["hidden_bias"])
        write_vector(handle, saved["out_weights"])
        write_vector(handle, saved["out_bias"])
        write_vector(handle, saved["slot_embedding"])
        write_vector(handle, saved["round_embedding"])
        write_vector(handle, saved["embeddings.weight"])

    print(f"wrote {out} bytes={out.stat().st_size}")


if __name__ == "__main__":
    main()

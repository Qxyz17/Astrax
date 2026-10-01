"""Train the Astrax holistic dialogue model on a GPU (or CPU) with PyTorch.

Reads the same Wikipedia document corpora and BPE subword table as the native
C++ trainer, and writes a checkpoint that the C++ runtime can load.

Usage:
    python tools/torch/train_astrax.py --document-epochs 200 --device cuda
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from astrax_model import HolisticDialogueModel, byte_loss  # noqa: E402

DATA = ROOT / "data"


def read_documents(paths, max_documents):
    documents = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                documents.append(line)
                if len(documents) >= max_documents:
                    return documents
    return documents


def load_merges(path):
    merges = []
    if not path.exists():
        raise RuntimeError(f"missing subword table: {path}")
    data = path.read_bytes()
    if data[:11] != b"ASTRAX-BPE-1":
        raise RuntimeError("subword table has an unexpected header")
    offset = 11
    (count,) = int.from_bytes(data[offset:offset + 4], "little"),
    offset += 4
    for _ in range(count):
        (left_len,) = int.from_bytes(data[offset:offset + 2], "little"),
        offset += 2
        left = data[offset:offset + left_len].decode("utf-8")
        offset += left_len
        (right_len,) = int.from_bytes(data[offset:offset + 2], "little"),
        offset += 2
        right = data[offset:offset + right_len].decode("utf-8")
        offset += right_len
        merges.append((left, right))
    return merges


class BpeEncoder:
    def __init__(self, merges, vocab_size):
        self.merges = merges
        self.base = {}
        self.next_id = 1
        self.vocab_size = vocab_size

    def token_id(self, token):
        if token not in self.base:
            if self.next_id >= self.vocab_size:
                return 0
            self.base[token] = self.next_id
            self.next_id += 1
        return self.base[token]

    def encode(self, text):
        ids = []
        for character in text:
            ids.append(self.token_id(character))
        for left, right in self.merges:
            left_id = self.token_id(left)
            right_id = self.token_id(right)
            result_id = self.token_id(left + right)
            index = 0
            while index + 1 < len(ids):
                if ids[index] == left_id and ids[index + 1] == right_id:
                    ids[index] = result_id
                    del ids[index + 1]
                else:
                    index += 1
        return ids


def make_batch(documents, encoder, max_slots, device):
    features_ids = []
    targets = []
    active = []
    for document in documents:
        data = document.encode("utf-8")
        ids = encoder.encode(document)
        features_ids.append(ids)
        byte_targets = list(data[:max_slots])
        byte_targets += [256] * (max_slots - len(byte_targets))
        mask = [1] * min(len(data), max_slots) + [0] * max(0, max_slots - len(data))
        targets.append(byte_targets)
        active.append(mask)
    max_ids = max(len(item) for item in features_ids)
    padded = [item + [0] * (max_ids - len(item)) for item in features_ids]
    id_mask = [[1] * len(item) + [0] * (max_ids - len(item)) for item in features_ids]
    return (
        torch.tensor(padded, device=device),
        torch.tensor(id_mask, device=device),
        torch.tensor(targets, device=device),
        torch.tensor(active, dtype=torch.float, device=device),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-documents", type=int, default=100000)
    parser.add_argument("--max-slots", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts" / "torch_dialogue.pt"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device={device}")

    documents = read_documents(
        [DATA / "wiki_zh.txt", DATA / "wiki_en.txt"], args.max_documents)
    if not documents:
        raise RuntimeError("no corpus found under data/")
    merges = load_merges(DATA / "astrax_subwords.bin")
    encoder = BpeEncoder(merges, vocab_size=30000)
    print(f"documents={len(documents)} merges={len(merges)}")

    model = HolisticDialogueModel(
        max_response_codepoints=args.max_slots,
        vocab_size=30000,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    if args.resume and pathlib.Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        print(f"resumed from {args.checkpoint} at epoch {start_epoch}")

    order = list(range(len(documents)))
    for epoch in range(start_epoch, args.document_epochs):
        model.train()
        # Deterministic shuffle per epoch.
        generator = torch.Generator().manual_seed(epoch)
        permutation = torch.randperm(len(order), generator=generator).tolist()
        total_loss = 0.0
        batches = 0
        started = time.time()
        for batch_start in range(0, len(order), args.batch_size):
            indices = [order[permutation[index]]
                       for index in range(batch_start,
                                          min(batch_start + args.batch_size, len(order)))]
            batch_docs = [documents[index] for index in indices]
            ids, id_mask, targets, active = make_batch(
                batch_docs, encoder, args.max_slots, device)
            features = model.encode_features(ids, id_mask)
            condition = torch.zeros(
                ids.shape[0], model.condition_dim, device=device)
            logits = model(features, condition)
            loss = byte_loss(logits, targets, active)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        print(f"epoch {epoch} loss={total_loss / max(1, batches):.4f} "
              f"seconds={time.time() - started:.1f}")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "epoch": epoch},
            args.checkpoint)
        print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()

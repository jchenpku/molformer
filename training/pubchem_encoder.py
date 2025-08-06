#!/usr/bin/env python3
# molformer/training/pubchem_encoder.py
#
# Fully self-contained version that always finds the vocabulary file,
# no matter from which directory the module is executed.

import os
import regex as re
import torch
import numpy as np         # noqa: F401  (kept for backwards compatibility)
import random              # noqa: F401
import collections

# --------------------------------------------------------------------
# Helper: resolve the vocabulary file relative to this source file
# --------------------------------------------------------------------
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_VOCAB_PATH = os.path.join(_THIS_DIR, "pubchem_canon_zinc_final_vocab_sorted.pth")
if not os.path.isfile(_VOCAB_PATH):
    raise FileNotFoundError(
        f"Cannot locate MolFormer vocabulary at {_VOCAB_PATH}.\n"
        "If you moved the file, update _VOCAB_PATH accordingly."
    )


class Encoder:
    """
    Character-level tokenizer / encoder used by MolFormer.

    The logic is unchanged from the original repository; the only
    modification is that `pubchem_canon_zinc_final_vocab_sorted.pth`
    is loaded with an **absolute** path so that `torch.load` succeeds
    regardless of the current working directory.
    """

    def __init__(self,
                 max_length: int = 500,
                 add_bos: bool = True,
                 add_eos: bool = True,
                 feature_size: int = 32):

        # ---- FIX: load vocabulary with absolute path ----------------
        self.vocab_encoder = torch.load(_VOCAB_PATH, map_location="cpu")
        # -------------------------------------------------------------

        self.max_length      = max_length
        self.min_length      = 1
        self.mod_length      = 42
        self.mlm_probability = 0.15
        self.avg_length      = 66
        self.tail            = 122

        # Caches & buckets used during data preparation
        self.b0_cache = collections.deque()
        self.b1_cache = collections.deque()
        self.b2_cache = collections.deque()
        self.b3_cache = collections.deque()
        self.bucket0  = collections.deque()
        self.bucket1  = collections.deque()
        self.bucket2  = collections.deque()
        self.bucket3  = collections.deque()

        if feature_size == 32:
            self.b0_max, self.b1_max, self.b2_max, self.b3_max = 1100, 700, 150, 50
        else:
            self.b0_max, self.b1_max, self.b2_max, self.b3_max = 1382, 871, 516, 311

        # ----------------------------------------------------------------
        # Build cutoffs and vocab dictionaries
        # ----------------------------------------------------------------
        values = list(self.vocab_encoder.values())
        num_top     = sum(c > 100_000 for c in values)
        middle_top  = sum(c > 50 for c in values) - num_top
        self.cutoffs = [num_top + 4, middle_top]

        self.char2id = {"<bos>": 0, "<eos>": 1, "<pad>": 2, "<mask>": 3}
        self.id2char = {0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<mask>"}

        pos = 0
        for key in self.vocab_encoder.keys():
            self.char2id[key]   = pos + 4
            self.id2char[pos+4] = key
            pos += 1

        self.char2id["<unk>"] = pos + 4
        self.id2char[pos+4]   = "<unk>"

        self.pad  = self.char2id["<pad>"]
        self.mask = self.char2id["<mask>"]
        self.eos  = self.char2id["<eos>"]
        self.bos  = self.char2id["<bos>"]

        # Pre-compiled SMILES regex
        self.pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
        self.regex   = re.compile(self.pattern)

        self.add_bos = add_bos
        self.add_eos = add_eos

    # ----------------------------------------------------------------
    # Encoding utilities
    # ----------------------------------------------------------------
    def encode(self, char_list):
        if self.add_bos:
            char_list = ["<bos>"] + char_list
        if self.add_eos:
            char_list = char_list + ["<eos>"]
        return torch.tensor([self.char2id[c] for c in char_list])

    def encoder(self, tokens):
        return [self.encode(mol) for mol in tokens]

    # ----------------------------------------------------------------
    # Data-processing pipeline used during pre-training
    # ----------------------------------------------------------------
    def process_text(self, text):
        mod_length = self.mod_length
        avg_length = self.avg_length

        for mol in text:
            raw_regex = self.regex.findall(mol["text"].strip("\n"))
            length = len(raw_regex)

            if self.min_length < length < mod_length:
                target_bucket, target_cache, max_size = self.bucket0, self.b0_cache, self.b0_max
            elif mod_length <= length < avg_length:
                target_bucket, target_cache, max_size = self.bucket1, self.b1_cache, self.b1_max
            elif avg_length <= length < self.tail:
                self.b2_cache.append(raw_regex); continue
            elif self.tail <= length < self.max_length:
                self.b3_cache.append(raw_regex); continue
            else:
                continue  # outside length limits

            if len(target_bucket) < max_size:
                target_bucket.append(raw_regex)
            else:
                target_cache.append(raw_regex)

        # Flush caches into buckets and return lists for training
        out0 = list(self.bucket0) + list(self._drain_cache(self.b0_cache, self.b0_max))
        out1 = list(self.bucket1) + list(self._drain_cache(self.b1_cache, self.b1_max))
        out2 = list(self._drain_cache(self.b2_cache, self.b2_max))
        out3 = list(self._drain_cache(self.b3_cache, self.b3_max))
        self.bucket0.clear(); self.bucket1.clear()
        return out0, out1, out2, out3

    def _drain_cache(self, cache, max_items):
        """Move up to `max_items` elements from the right side of a deque."""
        n = min(len(cache), max_items)
        return [cache.pop() for _ in range(n)]

    # ----------------------------------------------------------------
    # Masked-language-modeling helpers
    # ----------------------------------------------------------------
    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens/labels for MLM: 80 % mask, 10 % random, 10 % unchanged.
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.size(), self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for idx, seq in enumerate(inputs.tolist()):
                for j, token in enumerate(seq):
                    if token in (self.bos, self.eos, self.pad):
                        special_tokens_mask[idx, j] = True
        else:
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # compute loss only on masked tokens

        # 80 % → [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.size(), 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask

        # 10 % → random token
        indices_random = torch.bernoulli(torch.full(labels.size(), 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.char2id), labels.size(), dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    # ----------------------------------------------------------------
    # Public helpers used during training
    # ----------------------------------------------------------------
    def pack_tensors(self, tokens):
        array  = self.encoder(tokens)
        array  = torch.nn.utils.rnn.pad_sequence(array, batch_first=True, padding_value=self.pad)
        special_mask = [
            [1 if t in (self.bos, self.eos, self.pad) else 0 for t in seq] for seq in array.tolist()
        ]
        return self.mask_tokens(array, special_mask)

    def process(self, text):
        arrays, targets = [], []
        for bucket in self.process_text(text):
            if bucket:
                array, target = self.pack_tensors(bucket)
                arrays.append(array)
                targets.append(target)
        return arrays, targets


if __name__ == "__main__":
    enc = Encoder()
    print(f"Loaded vocabulary with {len(enc.vocab_encoder)} entries from {_VOCAB_PATH}")

import pickle
from dataclasses import dataclass
from pathlib import Path

# the plan:
# - bytes as initial "atoms"
# - merge frequently adjacent pairs
# - merge until the target size is reached


@dataclass(frozen=True)
class ControlToken:
    name: str


Token = ControlToken | bytes


@dataclass
class Tokenizer:
    vocab_size: int
    encoder: dict[Token, int]
    decoder: dict[int, Token]
    merges: dict[tuple[int, int], "_Merge"]

    def encode(self, text: str) -> list[int]:
        out = []
        piece_cache: dict[bytes, tuple[int, ...]] = {}
        for piece in _pretokenize(text):
            piece_bytes = piece.encode("utf-8")
            encoded = piece_cache.get(piece_bytes)
            if encoded is None:
                encoded = _apply_merges(piece_bytes, self.merges)
                piece_cache[piece_bytes] = encoded
            out.extend(encoded)
        return out

    def decode(self, tokens: list[int], errors: str = "strict") -> str:
        dec = (self.decoder[tok] for tok in tokens)
        dec = (
            f" {r.name} ".encode() if isinstance(r, ControlToken) else r for r in dec
        )
        dec = b"".join(dec)
        return dec.decode("utf-8", errors=errors)

    def get_control_token(self, name: str) -> int:
        return self.encoder[ControlToken(name)]


def save(tok: Tokenizer, file: Path):
    with open(file, "wb") as fp:
        pickle.dump(tok, fp)


def load(file: Path) -> Tokenizer:
    with open(file, "rb") as fp:
        tok = pickle.load(fp)
    assert isinstance(tok, Tokenizer), tok
    return tok


def build(
    texts: list[str],
    *,
    max_size: int,
    min_frequency: int,
    control_tokens: list[str] = ["<|EOS|>", "<|pad|>"],
) -> Tokenizer:
    min_size = 256 + len(control_tokens)
    if max_size < min_size:
        raise ValueError(
            f"max_size must be at least {min_size} for byte-level BPE "
            "with control tokens"
        )

    # init encoder
    encoder = {bytes([i]): i for i in range(256)}
    encoder |= {ControlToken(c): i + 256 for i, c in enumerate(control_tokens)}
    # init decoder
    decoder = {i: bytes([i]) for i in range(256)}
    decoder |= {i + 256: ControlToken(c) for i, c in enumerate(control_tokens)}
    # init merges (empty)
    merges = dict[tuple[int, int], _Merge]()

    sequences = _count_sequences(texts)
    while len(decoder) < max_size:
        counts = _count_pairs(sequences)
        if not counts:
            break
        # find the top vocab pair and combine
        (top_lhs, top_rhs), freq = max(counts.items(), key=lambda kv: kv[1])
        if freq < min_frequency:
            break
        lhs = decoder[top_lhs]
        rhs = decoder[top_rhs]
        assert isinstance(lhs, bytes)
        assert isinstance(rhs, bytes)
        new_tok = lhs + rhs
        new_vid = len(decoder)
        # symmetrically update vocab dictionaries
        encoder[new_tok] = new_vid
        decoder[new_vid] = new_tok
        # update the merge rank
        merges[(top_lhs, top_rhs)] = _Merge(len(merges), new_vid)
        # update the sequences
        sequences = _merge_sequences(sequences, (top_lhs, top_rhs), new_vid)
    return Tokenizer(len(decoder), encoder, decoder, merges)


def _apply_merges(
    seq: bytes | tuple[int, ...],
    merges: dict[tuple[int, int], "_Merge"],
) -> tuple[int, ...]:
    if len(seq) < 2:
        return tuple(seq)

    while True:
        # find all mergeable adjacent pairs in the current sequence
        # and pick the earliest-learned pair (lowest rank).
        best_pair: tuple[int, int] | None = None
        best_rank: int | None = None
        best_new_id = 0
        prev = seq[0]
        for cur in seq[1:]:
            pair = (prev, cur)
            merge = merges.get(pair)
            if merge is not None and (best_rank is None or merge.rank < best_rank):
                best_pair = pair
                best_rank = merge.rank
                best_new_id = merge.new_vid
            prev = cur

        if best_pair is None:
            return tuple(seq)

        # Merge every non-overlapping occurrence of that pair.
        seq = _merge_pair_in_sequence(seq, best_pair, best_new_id)


def _merge_pair_in_sequence(
    seq: bytes | tuple[int, ...],
    pair: tuple[int, int],
    new_id: int,
) -> tuple[int, ...]:
    lhs, rhs = pair
    out = []
    append = out.append
    i = 0
    n = len(seq)
    limit = n - 1

    while i < limit:
        if seq[i] == lhs and seq[i + 1] == rhs:
            append(new_id)
            i += 2
        else:
            append(seq[i])
            i += 1

    if i < n:
        append(seq[i])

    return tuple(out)


def _count_sequences(texts: list[str]) -> dict[tuple[int, ...], int]:
    sequences = dict[tuple[int, ...], int]()
    for text in texts:
        for piece in _pretokenize(text):
            seq = tuple(piece.encode("utf-8"))
            sequences[seq] = sequences.get(seq, 0) + 1
    return sequences


def _count_pairs(sequences: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    counts = dict[tuple[int, int], int]()
    for seq, freq in sequences.items():
        if len(seq) < 2:
            continue

        prev = seq[0]
        for cur in seq[1:]:
            pair = (prev, cur)
            counts[pair] = counts.get(pair, 0) + freq
            prev = cur

    return counts


def _merge_sequences(
    sequences: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_id: int,
) -> dict[tuple[int, ...], int]:
    merged = dict[tuple[int, ...], int]()
    for seq, freq in sequences.items():
        new_seq = _merge_pair_in_sequence(seq, pair, new_id)
        merged[new_seq] = merged.get(new_seq, 0) + freq
    return merged


def _pretokenize(text: str):
    i = 0
    n = len(text)

    while i < n:
        start = i
        if text[i].isspace():
            if text[i] == " " and i + 1 < n and not text[i + 1].isspace():
                start = i
                i += 1
            else:
                i += 1
                while i < n and text[i].isspace():
                    i += 1
                yield text[start:i]
                continue

        suffix_len = _contraction_suffix_len(text, i)
        if suffix_len:
            i += suffix_len
            yield text[start:i]
            continue

        if text[i].isalpha():
            i += 1
            while i < n and text[i].isalpha():
                i += 1
            yield text[start:i]
            continue

        if text[i].isdigit():
            i += 1
            while i < n and text[i].isdigit():
                i += 1
            yield text[start:i]
            continue

        i += 1
        while i < n and _is_symbol(text[i]):
            if _contraction_suffix_len(text, i):
                break
            i += 1
        yield text[start:i]


def _contraction_suffix_len(text: str, i: int) -> int:
    if text[i] != "'" or i + 1 >= len(text):
        return 0

    first = text[i + 1].lower()
    if i + 2 < len(text):
        second = text[i + 2].lower()
        if (first == "r" and second == "e") or (first == "v" and second == "e"):
            return 3
        if first == "l" and second == "l":
            return 3

    if first in {"d", "m", "s", "t"}:
        return 2

    return 0


def _is_symbol(ch: str) -> bool:
    return not ch.isspace() and not ch.isalpha() and not ch.isdigit()


@dataclass
class _Merge:
    rank: int
    new_vid: int

from typing import Iterable, Iterator
from array import array
import json
import pickle
from pathlib import Path

import regex as re

class EncodedTokenIds:
    def __init__(self, data: array):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[int]:
        for x in self.data:
            yield int(x)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [int(x) for x in self.data[idx]]
        return int(self.data[idx])

    def __eq__(self, other):
        try:
            if len(self) != len(other):
                return False
        except TypeError:
            return NotImplemented
        return all(a == b for a, b in zip(self, other))

    def __repr__(self) -> str:
        return f"EncodedTokenIds(len={len(self.data)})"
    
    
def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

compiled_pat = re.compile(GPT2_PAT)

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        self.token_to_id = {
            token_bytes: token_id for token_id, token_bytes in self.vocab.items()
        }

        self.merge_ranks = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        self.special_token_bytes = [
            token.encode("utf-8") for token in self.special_tokens
        ]

        
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id

        self.special_token_to_id = {
            token_bytes: self.token_to_id[token_bytes]
            for token_bytes in self.special_token_bytes
        }

        self.special_pattern = self._build_special_pattern()
        
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str],
    ):
        
        # Load vocab file
        with open(vocab_filepath,"rb") as f:
            vocab_content = pickle.load(f)
        
        vocab = {}
        for key, value in vocab.items():
            id = int(key)
            if isinstance(value, str):
                value = value.encode("utf-8")
            vocab[id] = value
        
        # Load merges file
        with open(merges_filepath,"rb") as f:
            merges_content = pickle.load(f)
            
        merges = []
        for x,y in merges_content:
            if isinstance(x,str):
                x = x.encode("utf-8")
            if isinstance(y,str):
                y = y.encode("utf-8")
            merges.append((x,y))
        
        return cls(vocab, merges, special_tokens)
    
    def _build_special_pattern(self) -> re.Pattern | None:
        if not self.special_tokens:
            return None
        
        escaped_tokens = [re.escape(tok) for tok in self.special_tokens]
        escaped_tokens = sorted(escaped_tokens, key=len, reverse=True)
        pattern = re.compile("(" + "|".join(escaped_tokens) + ")")
        return pattern
    
    def _is_special_token(self, part: str):
        return part in self.special_tokens
    
    def _split_special_tokens(self, text: str):
        """把 special token 从整段文本里切出来,保留 special token 本身"""
        if self.special_pattern is None:
            return [text]
        
        parts = self.special_pattern.split(text)
        parts = [part for part in parts if part]
        return parts
    
    def _pretokenize(self, text: str):
        for match in compiled_pat.finditer(text):
            yield match.group(0)
            
    def _pretoken_to_tokens(self, pretokens: str) -> list[bytes]:
            """pretokens 转换为 二进制字节码"""
            return [bytes([b]) for b in pretokens.encode("utf-8")]
            
    def _iter_special_parts(self, text: str):
        if self.special_pattern is None:
            if text:
                yield text
            return

        last = 0
        for match in self.special_pattern.finditer(text):
            start, end = match.span()
            if start > last:
                yield text[last:start]
            yield match.group(0)
            last = end

        if last < len(text):
            yield text[last:]
    
    def _append_encoded_normal_text(self, text: str, out: array) -> None:
        for pretoken in self._pretokenize(text):
            seq_tokens = self._encode_pretoken_to_tokens(pretoken)
            for tok in seq_tokens:
                out.append(self.token_to_id[tok])

    def _get_merge(self,tokens: list[bytes]):
            """得到 候选 合并对：(rank, position, pair)"""
            candidates = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i],tokens[i + 1])
                if pair in self.merge_ranks:
                    candidates.append((self.merge_ranks[pair], i, pair)) # 易错点 append((...))
            return candidates

    def _merge_once(self,tokens: list[bytes]):
        """合并pair"""
        candidates = self._get_merge(tokens)
        if not candidates:
            return tokens, False
        _,i,pair = min(candidates)
        merge_token = pair[0] + pair[1]
        new_token = tokens[:i] + [merge_token] + tokens[i+2:] # 易错点：[merge_token]
        return new_token, True
        
    def _encode_pretoken_to_tokens(self, pretoken: str):
        """编码分词"""
        tokens = self._pretoken_to_tokens(pretoken)
        while True:
            tokens, changed = self._merge_once(tokens)
            if not changed:
                break
        return tokens

    def _token_bytes_to_ids(self, tokens: list[bytes]):
        
        return [self.token_to_id[tok] for tok in tokens]

    def _encode_normal_text(self, text: str):
        out = array("H")
        self._append_encoded_normal_text(text, out)
        return EncodedTokenIds(out)

    def encode(self, text: str):
        if text == "":
            return EncodedTokenIds(array("H"))

        out = array("H")

        for part in self._iter_special_parts(text):
            if self._is_special_token(part):
                token_bytes = part.encode("utf-8")
                out.append(self.special_token_to_id[token_bytes])
            else:
                self._append_encoded_normal_text(part, out)

        return EncodedTokenIds(out)
    
    def encode_iterable(self, iterable: Iterable[str]):
        """
            整个文件按chunk读取
            special token / pretokenization / BPE merge 不能因为 chunk 边界而变错
        """    
        for chunk in iterable:
            yield from self.encode(chunk)
            
            
            
    def decode(self, ids: list[int]):
        seq_bytes = b"".join(self.vocab[i] for i in ids)
        return seq_bytes.decode("utf-8", errors="replace")
        


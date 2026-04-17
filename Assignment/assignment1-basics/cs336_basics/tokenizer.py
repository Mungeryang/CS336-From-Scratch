from typing import Iterable
import json
import pickle
from pathlib import Path

import regex as re


GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

compiled_pat = re.compile(GPT2_PAT)

class Tiny_BPETokenizer:
    def __init__(
        self,
        vocab: dict[int,bytes], 
        merges: list[tuple[bytes,bytes]], 
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.merge_ranks = {pair: rank for rank,pair in enumerate(self.merges)}
        
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        
        self.special_token_bytes = [
            token.encode("utf-8") 
            for token in self.special_tokens
        ] # special token 转成 bytes
        
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
        """转义 special token 并进行切段"""
        if not self.special_tokens:
            return None
        
        escaped_tokens = [re.escape(tok) for tok in self.special_tokens]
        escaped_tokens = sorted(escaped_tokens, key=len, reverse=True)
        pattern = re.compile("(" + "|".join(escaped_tokens) + ")")
        return pattern
    
    def _is_special_token(self, part: str):
        """判断是否有 special token"""
        return part in self.special_tokens
    
    def _split_special_tokens(self, text: str):
        """把 special token 从整段文本里切出来,保留 special token 本身"""
        if self.special_pattern is None:
            return [text]
        
        parts = self.special_pattern.split(text)
        parts = [part for part in parts if part]
        return parts
    
    def _pretokenize(self, text: str)  -> list[str]:
        """只处理普通文本块,把普通文本块切成 pretokens"""
        pretokens = []
        for match in compiled_pat.finditer(text):
            pretokens.append(match.group(0))
        return pretokens

        # return [match.group(0) for match in compiled_pat.finditer(text)]
    
    def _pretoken_to_tokens(self, pretokens: str) -> list[bytes]:
        """pretokens 转换为 二进制字节码"""
        return [bytes([b]) for b in pretokens.encode("utf-8")]
    
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
        """二进制字节码 到 整数ID 映射"""
        return [self.token_to_id[tok] for tok in tokens]
        
    
    def _encode_normal_text(self, text: str):
        """编码除special tokens外的文本"""
        pretokens = self._pretokenize(text)
        all_ids = []
        
        for pretoken in pretokens:
            seq_tokens = self._encode_pretoken_to_tokens(pretoken)
            token_ids = self._token_bytes_to_ids(seq_tokens)
            all_ids.extend(token_ids)
        return all_ids
    
    def encode(self, text: str):
        """编码全部输入文本"""
        if text == "":
            return []
        
        parts = self._split_special_tokens(text)
        all_ids = []
        
        for part in parts:
            if self._is_special_token(part):
                token_bytes = part.encode("utf-8")
                all_ids.append(self.special_token_to_id[token_bytes])
            else:
                all_ids.extend(self._encode_normal_text(part))
        return all_ids
        
    
    def encode_iterable(self, iterable: Iterable[str]):
        """
            整个文件按chunk读取
            special token / pretokenization / BPE merge 不能因为 chunk 边界而变错
        """    
        for chunk in iterable:
            yield from self.encode(chunk)
    
    
    def decode(self, ids: list[int]):
        """编码序列整数ID"""
        seq_bytes = b"".join(self.vocab[i] for i in ids)
        return seq_bytes.decode("utf-8", errors="replace")
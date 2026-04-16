from ast import pattern
from heapq import merge
import json
import multiprocessing as mp
from multiprocessing import process
import os 
import pickle 
from readline import write_history_file
import time
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import regex as re

from cs336_basics.pretokenization_example import boundaries, find_chunk_boundaries

from cs336_basics.train_bpe import (
    word_to_byte_tuple,
    apply_merge,
    build_pair_index
)


GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

compiled_pat = re.compile(GPT2_PAT)

BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

def build_word_counts_for_chunks(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
):
    word_counts = Counter()
    with open(input_path,"r",encoding="utf-8") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    
    chunk_text = chunk_bytes.encode("utf-8")
    
    if special_tokens:
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        pattern = "|".join(sorted(escaped_tokens, key=len, reverse=True))
        segments = [seg for seg in re.split(pattern, chunk_text) if seg]
    else:
        segments = [chunk_text]
    
    for segment in segments:
        for match in compiled_pat.finditer(segment):
            pretoken = match.group(0)
            word_counts[word_to_byte_tuple(pretoken)] += 1
    return word_counts

def _build_word_counts_for_chunks_star(args):
    return build_word_counts_for_chunks(*args)

def build_word_counts(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int,
):
    if not special_tokens:
        raise ValueError("Parallel chunking expects at least one special token boundary.")
    
    split_special_token = special_tokens[0].encode("utf-8")
    
    with open(input_path,"rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_processes,
            split_special_token=split_special_token,
        )
    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    total_word_counts = Counter()
    
    with mp.Pool(processes=num_processes) as pool:
        for partial_counts in tqdm(
            pool.imap_unordered(_build_word_counts_for_chunks_star, tasks),
            total=len(tasks),
            desc="Pretokenizing chunks",
        ):
            total_word_counts.update(partial_counts)
        
    return total_word_counts

def train_bpe_tinystories(
    word_counts: Counter[tuple[bytes, ...]],
    vocab_size: int,
    special_tokens: list[str],
):
    merges: list[tuple[bytes, bytes]] = []
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    
    pair_counts, pair_to_words = build_pair_index(word_counts)
    
    num_merges = vocab_size - len(vocab)
    
    for _ in tqdm(range(num_merges), desc="BPE merges"):
        if not pair_counts:
            break
        
        best_pair = max(pair_counts.items(), key=lambda x:(x[1],x[0]))[0]
        merge_token = best_pair[0] + best_pair[1]
        
        merges.append(best_pair)
        vocab[len(vocab)] = merge_token
        
        word_counts, pair_counts, pair_to_words = apply_merge(
            word_counts=word_counts,
            pair_counts=pair_counts,
            pair_to_words=pair_to_words,
            best_pair=best_pair,
        )
    return vocab,merges

def save_merges(merges, output_path: str | os.PathLike):
    with open(output_path, "wb") as f:
        pickle.dump(merges, f)

def save_vocab(vocab, output_path: str | os.PathLike):
    with open(output_path, "wb") as f:
        pickle.dump(vocab, f)

def find_longest_token(vocab: dict[int, bytes]):
    best_id = max(vocab.keys(),key=lambda idx: len(vocab[idx]))
    return best_id, vocab[best_id]

def main():
    input_path = "/home/ygm/llm-project/CS336-From-Scratch/Assignment/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]
    num_processes = 8   
    
    
    tqdm.write("Building word counts with parallel pretokenization")
    t0 = time.time()
    
    word_counts = build_word_counts(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=num_processes,
    )
    
    t1 = time.time()
    
    tqdm.write("Running merge")
    vocab,merges = train_bpe_tinystories(
        word_counts=word_counts,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    
    t2 = time.time()
    
    tqdm.write("Saving outputs")
    save_vocab(vocab, "tinystories_vocab.pkl")
    save_merges(merges, "tinystories_merges.pkl")
    
    token_id, token_bytes = find_longest_token(vocab)
    
    print(f"Pretokenization + word_counts time: {t1 - t0:.2f}s")
    print(f"Merge loop time: {t2 - t1:.2f}s")
    print(f"Total time: {t2 - t0:.2f}s")
    print(f"Final vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")
    print(f"Longest token id: {token_id}")
    print(f"Longest token length: {len(token_bytes)}")
    print(f"Longest token bytes: {token_bytes!r}")
    print(f"Longest token decoded: {token_bytes.decode('utf-8', errors='replace')!r}")


if __name__ == "__main__":
    main()
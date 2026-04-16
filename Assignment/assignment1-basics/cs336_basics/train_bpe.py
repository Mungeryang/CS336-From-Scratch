import regex as re
from collections import Counter,defaultdict

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

compiled_pat = re.compile(GPT2_PAT)

BYTE_TOKENS = tuple(bytes([i]) for i in range(256)) 

def load_file(file_name:str):
    with open(file_name,"r",encoding="utf-8") as f:
        text = f.read()
    return text

def word_to_byte_tuple(token: str):
    encoded = token.encode("utf-8")
    return tuple(BYTE_TOKENS[b] for b in encoded)
    # return tuple(bytes([b]) for b in token.encode("utf-8"))

# 第一版 预分词 主要使用的计算 “字节对” 的方法
def count_pair(word_counts: Counter[tuple[bytes, ...]]):
    '''
    得到的输出结果：
        Counter({(b'e', b's'): 8,
                 (b's', b't'): 8,
                 (b'l', b'o'): 7,
                 (b'o', b'w'): 7,
    '''
    pair_counts = Counter()
    for word,freq in word_counts.items():
        if len(word) < 2:
            continue
        for pair in zip(word,word[1:]):
            pair_counts[pair] += freq
    return pair_counts

# TODO: 第二版 预分词 新增方法
def pairs_in_word(word: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    """
    返回一个词内部所有相邻 pair 的频次
        (b'a', b'b', b'a', b'b') -> {(b'a', b'b'): 2, (b'b', b'a'): 1}
    """
    if len(word) < 2:
        return Counter()
    return Counter(zip(word,word[1:]))

# TODO: 第二版 预分词 新增方法
def build_pair_index(word_counts: Counter[tuple[bytes,bytes]]):
    """
    构建：
    1. pair_counts: 每个 pair 在整个语料中出现的总频次
    2. pair_to_words: 每个 pair 出现在哪些词里
       这里 value 用 Counter 而不是 set，是为了处理：
       不同旧词 merge 后变成同一个 new_word 的碰撞情况
    """
    pair_counts = Counter()
    pair_to_words = defaultdict(Counter)

    for word, freq in word_counts.items():
        word_pair_counts = pairs_in_word(word)
        for pair, multiplicity in word_pair_counts.items():
            pair_counts[pair] += multiplicity * freq
            pair_to_words[pair][word] += 1

    return pair_counts, pair_to_words

# TODO: 第二版 预分词 新增方法
def merge_word(word: tuple[bytes, ...],best_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged_token = best_pair[0] + best_pair[1]
    new_word = []
    i = 0
    n = len(word)

    while i < n:
        if i < n - 1 and (word[i], word[i + 1]) == best_pair:
            new_word.append(merged_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)

# TODO: 第二版 预分词 新增方法
def apply_merge(
    word_counts: Counter[tuple[bytes, ...]],
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_words: dict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]],
    best_pair: tuple[bytes, bytes],
):
    affected_words = list(pair_to_words.get(best_pair, {}).keys()) # e.g [(b' ', b't', b'h', b'e'),(b' ', b't', b'o'),(b' ', b't', b'h', b'a', b't')]
    if not affected_words:
        return word_counts, pair_counts, pair_to_words

    affected_freqs = {word: word_counts[word] for word in affected_words}

    for old_word in affected_words:
        # e.g old_word = (b' ', b't', b'h', b'e')
        freq = affected_freqs[old_word]

        # 1. 删除 old_word 的 pair
        old_pair_counts = pairs_in_word(old_word)
        for pair, counts in old_pair_counts.items():
            pair_counts[pair] -= counts * freq
            if pair_counts[pair] <= 0:
                del pair_counts[pair]

            pair_to_words[pair][old_word] -= 1
            if pair_to_words[pair][old_word] <= 0:
                del pair_to_words[pair][old_word]
            if not pair_to_words[pair]:
                del pair_to_words[pair]

        # 2. merge 成 new_word
        new_word = merge_word(old_word, best_pair)

        # 3. 更新 word_counts
        word_counts[old_word] -= freq
        if word_counts[old_word] <= 0:
            del word_counts[old_word]

        word_counts[new_word] += freq

        # 4. 把 new_word 的 pair 贡献加回去
        new_pair_counts = pairs_in_word(new_word)
        for pair, counts in new_pair_counts.items():
            pair_counts[pair] += counts * freq
            pair_to_words[pair][new_word] += 1

    return word_counts, pair_counts, pair_to_words
    


# 第一版 合并并更新 “字节对” 的方法
def apply_merge_token(word_counts,best_pair: tuple):
    merge_token = best_pair[0] + best_pair[1]
    new_word_counts = Counter()
    
    for word,freq in word_counts.items():
        new_word = []
        i = 0
        
        while i < len(word):
            if i < len(word) - 1 and (word[i],word[i + 1]) == best_pair:
                new_word.append(merge_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_counts[tuple(new_word)] += freq
    return new_word_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
        
    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} 太小，至少要容纳 256 个字节和 {len(special_tokens)} 个 special tokens。"
        )
    
    text = load_file(input_path)
    
    if special_tokens:
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        pattern = "|".join(sorted(escaped_tokens, key=len, reverse=True))
        segments = [seg for seg in re.split(pattern,text) if seg]
    else:
        segments = [text]
    
    word_counts = Counter()
    
    for segment in segments:
        for match in compiled_pat.finditer(segment):
            pretoken = match.group(0)
            byte_tuple = word_to_byte_tuple(pretoken)
            word_counts[byte_tuple] += 1
            
    # TODO: 第二版新增
    pair_counts, pair_to_words = build_pair_index(word_counts)
    
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)
    
    # 第一版：无法通过 test_train_bpe_speed
    for _ in range(num_merges):
        # pair_counts = count_pair(word_counts) # 全量扫描所有词,第二版无比注释掉
        if not pair_counts:
            break
    
        best_pair = max(pair_counts.items(), key=lambda x: (x[1],x[0]))[0]
        merge_token = best_pair[0] + best_pair[1]
        
        merges.append(best_pair)
        vocab[len(vocab)] = merge_token
        
        word_counts,pair_counts,pair_to_words = apply_merge(
            word_counts,
            pair_counts,
            pair_to_words,
            best_pair
        )
        
    #     word_counts = apply_merge_token(word_counts,best_pair) # 全量扫描所有词 
    return vocab,merges




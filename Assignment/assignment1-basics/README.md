# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## 代码库架构

1. cs336_basics/*: 存放我们自己手撸的全部代码，可以在这个文件夹下开始任何的创造

2. adapters.py: 测试时的统一的接口层，将我们自己实现的代码 `cs336_basics/*` 与 test/ 测试代码做桥接

3. test_*.py: 必须通过的全部测试文件

## 数据集

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
 
## BPE Tokenizer Training

### Vocabulary initialization

分词器词汇表是从 字节串token 到整数ID的一对一映射。由于我们正在训练一个字节级BPE分词器，初始词汇表就是所有字节的集合。由于存在256种可能的字节值，我们的初始词汇表大小为256。

### Pre-Tokenization


### 预分词

## ✨ What You Build In Assignment 1

This assignment asks you to build a small language-modeling stack from scratch:

1. A byte-level BPE tokenizer.
2. A decoder-only Transformer language model.
3. Cross-entropy loss and AdamW.
4. A training loop with batching, checkpointing, and learning-rate scheduling.
5. Experiments on TinyStories and OpenWebText.

At a high level, the workflow is:

```text
raw text
  -> BPE training
  -> tokenizer vocab + merges
  -> tokenized dataset (integer IDs)
  -> batching
  -> Transformer LM forward pass
  -> cross-entropy loss
  -> backward pass + AdamW + LR schedule + grad clipping
  -> checkpoints / evaluation / text generation
```

## 🧭 Repo Map

The repo is intentionally minimal. The important files are:

- `cs336_basics/*`
  This is where you write your code. The package is mostly empty by design, so you are free to organize your implementation.
- `tests/adapters.py`
  This is the interface layer between the tests and your code. Keep it as glue code only. The actual logic should live in `cs336_basics/*`.
- `tests/test_*.py`
  Public unit tests. These are the most concrete behavioral specification you have.
- `tests/fixtures/*`
  Reference vocabs, merges, sample corpora, and model fixtures.
- `cs336_spring2025_assignment1_basics.pdf`
  The assignment handout. This is the authoritative description of the tasks and constraints.
- `cs336_basics/pretokenization_example.py`
  Example helper for splitting a large file into chunks aligned to a special token for parallel pretokenization.

## 🏗️ Suggested Internal Architecture

You do not need to match this exactly, but it is a practical layout that scales well as the assignment grows:

```text
cs336_basics/
  bpe.py          # train_bpe and supporting helpers
  tokenizer.py    # Tokenizer class, encode/decode/encode_iterable
  nn.py           # Linear, Embedding, RMSNorm, SiLU, SwiGLU
  attention.py    # softmax, SDPA, RoPE, MHA
  model.py        # TransformerBlock, TransformerLM
  optim.py        # AdamW, cosine LR schedule, grad clipping
  data.py         # get_batch and dataset helpers
  checkpoint.py   # save_checkpoint, load_checkpoint
  train.py        # end-to-end training script
```

A good rule is:

- `adapters.py` only imports and calls your implementation.
- Unit-testable math components stay in small focused modules.
- Training script code should not contain reusable core logic inline.

## 🔗 Dependency Order Between Components

A reliable implementation order is:

```text
BPE training -> Tokenizer
Linear / Embedding / SiLU / RMSNorm / Softmax / CrossEntropy
-> SwiGLU
-> RoPE
-> Scaled Dot-Product Attention
-> Multi-Head Self-Attention
-> TransformerBlock
-> TransformerLM
-> get_batch / AdamW / LR schedule / gradient clipping / checkpointing
-> full training loop and experiments
```

Why this order works:

- The tokenizer stack is relatively self-contained and has its own tests.
- Low-level tensor ops are easier to debug in isolation than inside the full model.
- Attention depends on stable lower-level ops.
- The training loop should come last, after the model and utilities are trustworthy.

## 🚀 Quick Start

### Environment

We manage environments with `uv` for reproducibility and portability.
Install `uv` from the official docs, or via `pip install uv` / `brew install uv`.

Run files inside the project with:

```sh
uv run <python_file_path>
```

### Run all unit tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the functions in
[`./tests/adapters.py`](./tests/adapters.py).

### Download data

Download TinyStories and a subsample of OpenWebText:

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## 🪜 Implementation Step by Step

### Phase 0: Read before coding

Before writing code, read these files in this order:

1. `cs336_spring2025_assignment1_basics.pdf`
2. `tests/adapters.py`
3. `tests/test_train_bpe.py`
4. `tests/test_tokenizer.py`
5. `tests/test_model.py`
6. `tests/test_nn_utils.py`
7. `tests/test_data.py`
8. `tests/test_optimizer.py`
9. `tests/test_serialization.py`

This gives you three layers of specification:

- The handout tells you the intended design.
- `adapters.py` tells you the interfaces.
- The tests tell you the exact behavior expected by the public grader.

### Phase 1: Build the tokenizer pipeline

Target files and concepts:

- `bpe.py`
- `tokenizer.py`
- `tests/test_train_bpe.py`
- `tests/test_tokenizer.py`

Sub-steps:

1. Implement BPE vocabulary initialization from raw bytes.
2. Implement regex-based pretokenization using the GPT-2 pattern from the handout.
3. Count pretoken frequencies as UTF-8 byte sequences.
4. Implement iterative pair counting and merging.
5. Add support for special tokens.
6. Implement `Tokenizer.encode`.
7. Implement `Tokenizer.decode`.
8. Implement `Tokenizer.encode_iterable` for streaming tokenization.

What to validate mentally:

- Special tokens must be preserved as indivisible units during encoding.
- BPE merges happen only within a pretoken, never across pretoken boundaries.
- During BPE training, frequency ties are broken by choosing the lexicographically greater pair.
- `decode` must tolerate malformed byte sequences via replacement behavior.
- `encode_iterable` should be memory-efficient and not require loading the full file into RAM.

A practical strategy is to make `encode` correct first, then derive `encode_iterable` from the same logic with chunk-safe boundary handling.

### Phase 2: Implement low-level neural network primitives

Target files and concepts:

- `nn.py`
- `attention.py`
- `tests/test_model.py`
- `tests/test_nn_utils.py`

Sub-steps:

1. `Linear`
2. `Embedding`
3. `SiLU`
4. `RMSNorm`
5. `softmax`
6. `cross_entropy`
7. `gradient_clipping`
8. `SwiGLU`

What to validate mentally:

- Shapes are correct for arbitrary leading batch dimensions.
- `Linear` stores weights in `(out_features, in_features)` layout.
- `Embedding` indexes directly into a learnable weight matrix.
- `RMSNorm` upcasts to `float32` before squaring and normalizing, then downcasts back.
- `softmax` and `cross_entropy` use numerically stable formulations.
- Gradient clipping computes the global norm across parameters with gradients.

### Phase 3: Implement attention and the Transformer

Target files and concepts:

- `attention.py`
- `model.py`
- `tests/test_model.py`

Sub-steps:

1. Rotary positional embeddings (RoPE)
2. Scaled dot-product attention
3. Causal multi-head self-attention
4. Pre-norm `TransformerBlock`
5. Full `TransformerLM`

What to validate mentally:

- RoPE applies to `q` and `k`, not `v`.
- The head dimension should behave like a batch-like dimension.
- Causal masking prevents attention to future tokens.
- Mask semantics in this assignment are: `True` means attention is allowed.
- `TransformerBlock` is pre-norm, not post-norm.
- The language model outputs raw logits over the vocabulary, not probabilities.
- The model must handle sequence lengths shorter than `context_length`.

### Phase 4: Implement training utilities

Target files and concepts:

- `data.py`
- `optim.py`
- `checkpoint.py`
- `tests/test_data.py`
- `tests/test_optimizer.py`
- `tests/test_serialization.py`

Sub-steps:

1. Implement `get_batch` for random contiguous sampling from a token array.
2. Implement AdamW as an `Optimizer` subclass.
3. Implement cosine LR schedule with warmup.
4. Implement save/load checkpoint helpers.

What to validate mentally:

- `get_batch` samples uniformly over valid starting offsets.
- Labels are just the input shifted by one token.
- Returned tensors are placed on the requested device.
- AdamW uses decoupled weight decay, not plain L2 regularization folded into gradients.
- Optimizer state is per-parameter.
- Checkpoints must restore model state, optimizer state, and iteration count.

### Phase 5: Put everything together

Once the public tests are passing, move to end-to-end training:

1. Train a tokenizer on TinyStories.
2. Serialize vocab and merges.
3. Tokenize the train/valid corpora into integer arrays.
4. Train the Transformer LM.
5. Save checkpoints and training curves.
6. Generate samples and compute perplexity.
7. Repeat on OpenWebText.
8. Run the ablations and writeup experiments.

## 🧪 Recommended Test Order

Don't start with `uv run pytest` every time. Use a progression that matches dependencies.

### Tokenizer line

```sh
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_tokenizer.py
```

### Core neural utilities

```sh
uv run pytest -k 'test_linear or test_embedding or test_silu or test_rmsnorm or test_swiglu'
uv run pytest -k 'test_softmax_matches_pytorch or test_cross_entropy or test_gradient_clipping'
```

### Attention and model

```sh
uv run pytest -k 'test_rope or test_scaled_dot_product_attention or test_4d_scaled_dot_product_attention'
uv run pytest -k 'test_multihead_self_attention or test_multihead_self_attention_with_rope'
uv run pytest -k 'test_transformer_block or test_transformer_lm'
```

### Training utilities

```sh
uv run pytest -k 'test_get_batch or test_adamw or test_get_lr_cosine_schedule or test_checkpointing'
```

### Final full run

```sh
uv run pytest
```

## 🧠 Handout Insights That Matter In Practice

These are some of the most useful implementation hints from the handout.

### 1. Start small

Do not begin by training on the full TinyStories train split. First validate correctness on the provided fixtures and then on a much smaller debug corpus.

### 2. BPE bottleneck is usually pretokenization

The handout explicitly points out that pretokenization is often the main bottleneck. If your BPE training is too slow, profile first, then optimize the pretokenization path.

### 3. Parallelize where it actually helps

Pretokenization can be parallelized. The merge loop generally cannot be parallelized effectively in pure Python. Use `pretokenization_example.py` as a reference for chunking large files at special-token boundaries.

### 4. Remove special tokens before regex pretokenization

If your corpus contains special tokens such as `<|endoftext|>`, split on them first so merges never cross those boundaries.

### 5. Numerical stability matters

The assignment repeatedly calls this out for `RMSNorm`, `softmax`, and `cross_entropy`. If these are unstable, downstream tests and training will become misleading.

### 6. Public tests are only part of the assignment

Passing tests means your core components are likely correct. It does not complete the full assignment. You still need training runs, experiments, and a writeup.

## ⚠️ Common Pitfalls

- Implementing real logic inside `tests/adapters.py` instead of inside `cs336_basics/*`.
- Forgetting that BPE works on UTF-8 bytes, not raw Python Unicode code points.
- Letting BPE merges cross special-token boundaries.
- Breaking BPE tie frequency incorrectly. The assignment wants the lexicographically greater pair.
- Writing `encode_iterable` in a way that reads the full file into memory.
- Forgetting `errors="replace"` in tokenizer decoding.
- Using unstable softmax or cross-entropy formulas.
- Treating `False` as attendable in attention masks. Here, `True` means a position may be attended to.
- Applying RoPE to value vectors.
- Building a post-norm block when the assignment expects pre-norm.
- Assuming every input sequence equals `context_length`.
- Implementing AdamW as Adam plus coupled weight decay.

## 📌 Practical Notes For Open-Source Readers

If you are reading this repository outside the course, a productive way to use it is:

1. Re-implement each component yourself from the handout.
2. Use the public tests as behavioral checks, not as the primary source of design.
3. Keep notes on shapes, invariants, and edge cases while reading the tests.
4. Add your own tiny debug scripts before trying full training.
5. Only optimize after the implementation is demonstrably correct.

## 📝 After The Public Tests Pass

Once your unit tests pass, the remaining assignment work is mostly experimental and empirical:

- tokenizer training on TinyStories and OpenWebText
- corpus tokenization and serialization
- LM training on TinyStories
- sample generation and perplexity evaluation
- ablations such as removing RMSNorm, switching pre-norm/post-norm, removing positional information, or comparing SwiGLU vs SiLU
- OpenWebText run and optional leaderboard submission

That stage is where engineering discipline matters most: configuration management, logging, checkpointing, profiling, and carefully chosen debug-scale experiments.

## 🙌 Final Advice

A good mental model for Assignment 1 is:

- first build correct local components,
- then compose them into the model,
- then make training stable,
- then make experiments reproducible.

If you keep each stage small and testable, the assignment remains very manageable. If you skip straight to end-to-end training, debugging becomes significantly harder.


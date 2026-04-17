# <p align="center"> 🚀🚀 CS336-From-Scratch Spring 2026🚀🚀 </p>

> The NoteBook and Assignments implemention via Learning CS336 Spring 2026！

> “And in the end, the love you take is equal to the love you make.”  —— The Beatles

`I'm mungeryang, a master's student from the University of Chinese Academy of Sciences(UCAS-iie). In this Repo, I have open-sourced all of my study notes, implementation details for assignments, and results.`

<div align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTD0LpkXqKO5WIQfWEvAb3nGwrxS_AWAyU20g&s" width="70%">
</div>

Class HomePage：https://cs336.stanford.edu/

![cs336](https://miro.medium.com/v2/resize:fit:1400/1*N9A3rarbj3waA7LWn3TFwg.png)


## 📔 NoteBook

|  Chapter  |            Content             | Status |
| :-------: | :----------------------------: | :----: |
| Lecture01 |     Overview, tokenization     |   ✅    |
| Lecture02 |  PyTorch, Resource accounting  |   ✅    |
| Lecture03 | Architectures, hyperparameters |   ✅    |
| Lecture04 |       Mixture of experts       |   ❌    |
| Lecture05 |              GPUs              |   ❌    |
| Lector06  |        Kernels, Triton         |   ❌    |
| Lector07  |          Parallelism           |   ❌    |
| Lector08  |          Scaling laws          |   ❌    |
| Lector09  |           Evaluation           |   ❌    |
| Lector10  |              Data              |   ❌    |
| Lector11  |      Alignment - SFT/RLHF      |   ❌    |
| Lector12  |         Alignment - RL         |   ❌    |




## 💻 Assignments

### Assignment 1: Basics

- Implement all of the components (tokenizer, model architecture, optimizer) necessary to train a standard Transformer language model
- Train a minimal language model

|Assignment1|   Status   |  Link  |
| :-------: | :--------: | :----: |
| train_bpe |     ✅     |   [BPE Implementation](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/train_bpe.py)   |
| BPETokenizer |  ✅     |   [Tiny_BPETokenizer Class Implementation](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/train_bpe.py)    |
|  Linear   |    ing     |         |
| Emebdding |     ❌     |         |
| RMSNorm   |     ❌     |         |
| Swiglu    |     ❌     |         |
| RoPE      |     ❌     |         |
| softmax   |     ❌     |         |
| attention |     ❌     |         |
| mul-attn  |     ❌     |         |
| LM block  |     ❌     |         |
| cross-entropy |  ❌    |         |
| AdamW     |  ❌        |         |

#### Test train_bpe

[BPE Implementation](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/train_bpe.py)

<img src="results_img/train_bpe.png">


#### Test tokenizer

[Tiny_BPETokenizer Class Implementation](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/train_bpe.py)

<img src="results_img/tokenizer2.png">


### Assignment 2: Systems

- Profile and benchmark the model and layers from Assignment 1 using advanced tools, optimize Attention with your own Triton implementation of FlashAttention2

- Build a memory-efficient, distributed version of the Assignment 1 model training code





### Assignment 3: Scaling

- Understand the function of each component of the Transformer
- Query a training API to fit a scaling law to project model scaling



### Assignment 4: Data

- Convert raw Common Crawl dumps into usable pretraining data
- Perform filtering and deduplication to improve model performance



### Assignment 5: Alignment

- Apply supervised finetuning and reinforcement learning to train LMs to reason when solving math problems
- **[Optional Part 2](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)**: implement and apply safety alignment methods such as DPO






# <p align="center"> 🚀🚀 CS336-From-Scratch Spring 2026🚀🚀 </p>

> The NoteBook and Assignments implemention via Learning CS336 Spring 2026！

> “And in the end, the love you take is equal to the love you make.”  —— The Beatles

`I'm mungeryang, a master's student from the University of Chinese Academy of Sciences(UCAS-iie). In this Repo, I have open-sourced all of my study notes, implementation details for assignments, and results.`

<div align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTD0LpkXqKO5WIQfWEvAb3nGwrxS_AWAyU20g&s" width="70%">
</div>

Class HomePage：https://cs336.stanford.edu/

**图解大模型架构**参见：[Hand-Drawn-LLM](https://github.com/Mungeryang/Hand-Drawn-LLM)

![cs336](https://miro.medium.com/v2/resize:fit:1400/1*N9A3rarbj3waA7LWn3TFwg.png)

## 🧐100 $QA_{s}$ About LLM - 大模型面试100问

> ### 通过整理 `Standford CS336 Spring26` 课堂笔记，总结大模型算法经典面试**50**问

> 参考资料: CS336课堂笔记、[李博杰老师 - 大模型面试题 200 问](https://www.ituring.com.cn/book/3285)、[百面大模型](https://www.ptpress.com.cn/publishing/book/ef639cdb-d2a9-4987-8c79-14760baa4fc3)、[大模型技术30讲](https://github.com/ningg/Machine-Learning-Q-and-AI)


⭐️⭐️⭐️ [50问整理](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/tree/main/NoteBook)

P.S. 由于精力和能力有限，仅整理出我本人认为较为经典的50问。问题与作答全部开源，欢迎任何感兴趣的人fork更新～ 👏👏👏



## 💻 Assignments

### Assignment 1: Basics

- Implement all of the components (tokenizer, model architecture, optimizer) necessary to train a standard Transformer language model
- Train a minimal language model

|Assignment1|   Status   |  Link  |
| :-------: | :--------: | :----: |
| train_bpe |     ✅     |   [BPE Implementation](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/train_bpe.py)   |
| BPETokenizer |  ✅     |   [Tiny_BPETokenizer Class Implementation](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/train_bpe.py)    |
|  Linear   |    ✅      |[Linear Class](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/linear.py)         |
| Emebdding |     ✅     | [EMbedding Class](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/embedding.py)        |
| RMSNorm   |     ✅     | [RMSNorm](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/rmsnorm.py)        |
| Swiglu    |     ✅     | [SwiGLU FFN](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/swiglu.py)        |
| RoPE      |     ✅     | [RoPE Class](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/rope.py)        |
| softmax   |     ✅     | [softmax funcion](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/rope.py)        |
| attention |     ✅     | [Scaled_Dot_Attn](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/scaled_dot_product_attention.py)        |
| mul-attn  |     ✅     | [MultiHeadAttn Class](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/multihead_self_attention.py)        |
| LM block  |     ✅     | [Transformer Block](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/transformer_block.py)       |
| cross-entropy |  ✅    | [train function](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/tests/adapters.py)        |
| AdamW     |  ✅        | [Adamw optimizer](https://github.com/Mungeryang/CS336-From-Scratch-Spring2026/blob/main/Assignment/assignment1-basics/cs336_basics/adamw.py)        |

### ⌨️ Assignment 1: Results

#### Test train_bpe

<img src="results_img/train_bpe.png">

#### Test Tokenizer

<img src="results_img/tokenizer2.png">

#### Test Transformer_LM

<img src="results_img/test_model.png">

#### Test Utils and Optimizer

<img src="results_img/utils_optimizer.png">

#### Test DataLoader and Checkpoint

<img src="results_img/data_loader.png">


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






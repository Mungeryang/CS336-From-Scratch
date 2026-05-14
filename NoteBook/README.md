# <p align="center"> 🧐50 $QA_{s}$ ON LLM - 大模型面试50问🧐 </p>

#### <p align="center"> 杨桂淼整理 2026年4月 </p>

<div align="center">
  <img src="https://www.ydylcn.com/skwx_ydyl//zpimage.zhtml?ID=10600582&SiteID=1&draft=0&type=norm" width="70%">
</div>

> ### 通过整理 `Standford CS336 Spring26` 课堂笔记，总结大模型算法经典面试**50**问

> 参考资料: CS336课堂笔记、[李博杰老师 - 大模型面试题 200 问](https://www.ituring.com.cn/book/3285)、[百面大模型](https://www.ptpress.com.cn/publishing/book/ef639cdb-d2a9-4987-8c79-14760baa4fc3)、[大模型技术30讲](https://github.com/ningg/Machine-Learning-Q-and-AI)

> 以点带面，忌贪多贪全，从单个问题出发逐步深入。 不是对八股的死记硬背，而是在实操中去总结问题。

> ### ⚠️ 持续更新中...

## Topic 1: Tokenizer and PreToken

### Byte 级的BPE分词器相比于传统的BPE分词器有哪些优势?

byte-level BPE 的核心优势可以概括为**兼顾了开放词表能力和较好的压缩能力**。

<img src="https://github.com/stanford-cs336/lectures/blob/main/images/tokenized-example.png">

- Byte-level BPE 的初始词表是全部 256 个字节值, 任意 Unicode 文本都可以先编码成 UTF-8 字节序列，因此理论上任何输入都能被表示，不会出现传统词表覆盖不到的字符或词。

- 即使是很少见的字符，也总能拆成若干字节来表示。因此它对跨语言文本、emoji、罕见符号等输入更稳定。

- 如果只按字节切分，序列会很长，训练和推理成本高。BPE 会把高频字节序列继续合并成子词，从而压缩序列长度，减少计算开销。

### BPE训练过层中,预分词(Pre-tokenization)的作用是什么？

- 防止跨界合并，避免将本应该属于不用语义单元的片段合并成一个独立的ID，提高语义一致性。

- 通过正则表达式先将文本切分成单词或者短语块，可以在统计字节对频率时减少全量扫描语料的次数。


## Topic 2: Model Architecture


### Qwen架构中的使用的层归一化函数是什么？其和 *LayerNorm* 的区别体现在哪些方面?

Qwen架构中使用的层归一化函数为`RMSNorm`, 相比原始 Transformer 常用的 `LayerNorm`，现代大模型更常采用 `RMSNorm`。

- LayerNorm 会先减去均值，再除以标准差：

$$ \mathrm{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta $$

- RMSNorm 不做减均值，只按均方根进行缩放：

$$ \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon} $$

> LayerNorm 是“减均值再缩放”，而 RMSNorm 是“只按均方根缩放、不减均值”；后者更简单、更省计算(计算开销更小)。


### 使用Transformer生成文本时,在解码过程中使用的Trick有哪些?

#### Trick 1 

模型最后输出的是 `logits`，需要先经过 **softmax** 转成词表上的概率分布，才能进行采样。在 softmax 前把 logits 除以温度参数 $ \tau $ 进行**温度缩放** :

$$ \mathrm{softmax}(v, \tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j=1}^{|\mathrm{vocab}|} \exp(v_j / \tau)} $$


#### Trick 2

不直接在整个词表上采样，而是先把所有 token 按概率从大到小排序，从概率最大的 token 开始往后累加，一直加到累计概率第一次达到或超过阈值 p。

然后只在这组 token 里重新归一化并采样，截断低概率噪声 token，提升生成文本质量。


### Pre-norm 与 Post-norm 的区别，为什么现代 LLM 偏好Pre-norm 架构？

<img src="https://i-blog.csdnimg.cn/blog_migrate/d4d8a8327721f8368e1bce5f0a1b2096.png">

Transformer Block 包括两个主要的sub-layers: multi-head self-attn mechanism(MHA) 和 position-wise feed-forward netword(FFN)。

结构区别：Post-norm 在残差连接相加后进行 LayerNorm；Pre-norm 则是在进入自注意力或前馈网络之前进行 LayerNorm 。(区别如上图所示，重点抓住两个核心层即可)

Pre-norm 创造了一条从输入嵌入到最终输出的“清晰残差流”，实验证明它能显著提高大模型训练的稳定性，改善梯度流动 。

### 多模态大模型架构中，视觉编码器与语言解码器之间通过什么桥接起来？

> 视觉编码器与语言解码器之间通过线性投影层连接起来。

以LLaVA为例，ViT 输出 1024-dim 视觉特征，已经能识别物体了。Vicuna 语言模型输入 5120-dim 词嵌入，已经懂语言了。

线性投影 Linear(1024, 5120) 开始是随机初始化的，需要学会把视觉语义"翻译"成 LLM 能理解的 token。


### LLaVA训练的范式？其开启的重要研究路线是什么？

LLaVA主要采用两阶段训练策略：*Stage1: 对齐预训练 +  Stage2: 指令微调*

- Stage1 - 对齐预训练阶段，目的是让投影层学会把 CLIP 特征映射到 LLM 的词嵌入空间

视觉编码器 ❌冻结   投影层 ✅训练    Vicuna ❌冻结

- Stage2 - 指令微调阶段，目的是让投影层与语言模型联合微调，适配指令跟随任务

视觉编码器 ❌冻结   投影层 ✅训练    Vicuna ✅训练


当下很多开源多模态大模型的训练范式都是沿用了LLaVA的基础：

```shell
视觉编码器（CLIP/SigLIP,通常冻结）
         ↓
投影层（Projector,可训练）
         ↓
大语言模型（LLM,SFT 阶段解开）
         ↓
训练数据：LLM 生成的指令数据 + 少量人工标注

```

> Qwen3-VL、InternVL、DeepSeek-VL 等模型虽然架构上复杂，但是**数据生成范式**和**两阶段训练策略**(对齐预训练 + 指令微调)全部源自LLaVA。












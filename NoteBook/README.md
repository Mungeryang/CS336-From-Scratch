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



## Topic 2: Model Architecture


### Qwen架构中的使用的层归一化函数是什么？其和 *LayerNorm* 的区别体现在哪些方面?

Qwen架构中使用的层归一化函数为`RMSNorm`, 相比原始 Transformer 常用的 `LayerNorm`，现代大模型更常采用 `RMSNorm`。

- LayerNorm 会先减去均值，再除以标准差：

$$ \mathrm{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta $$

- RMSNorm 不做减均值，只按均方根进行缩放：

$$ \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon} $$

> LayerNorm 是“减均值再缩放”，而 RMSNorm 是“只按均方根缩放、不减均值”；后者更简单、更省计算。


### 使用Transformer生成文本时,在解码过程中使用的Trick有哪些?

#### Trick 1 

模型最后输出的是 `logits`，需要先经过 **softmax** 转成词表上的概率分布，才能进行采样。在 softmax 前把 logits 除以温度参数 $ \tau $ 进行**温度缩放** :

$$ \mathrm{softmax}(v, \tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j=1}^{|\mathrm{vocab}|} \exp(v_j / \tau)} $$


#### Trick 2

不直接在整个词表上采样，而是先把所有 token 按概率从大到小排序，从概率最大的 token 开始往后累加，一直加到累计概率第一次达到或超过阈值 p。

然后只在这组 token 里重新归一化并采样，截断低概率噪声 token，提升生成文本质量。





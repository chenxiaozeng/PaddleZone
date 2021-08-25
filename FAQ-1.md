## FAQ


## 写在前面

+ 我们收集整理了开源以来在Issues和用户群中的常见问题并且给出了简要解答，旨在为NLP的开发者提供一些参考，也希望帮助大家少走一些弯路。
+ NLP领域大佬众多，本文档回答主要依赖有限的项目实践，难免挂一漏万，如有遗漏和不足，也**希望有识之士帮忙补充和修正**，万分感谢。
+ 使用过程中可参考PaddleNLP[官方文档](https://paddlenlp.readthedocs.io/zh/latest/index.html) 、[AI Studio官方项目](https://aistudio.baidu.com/aistudio/projectdetail/1535371)。

## PaddleNLP常见问题汇总（持续更新）

* [【理论篇】NLP通用问题1个](#【理论篇】NLP通用问题)

  [Q1.语义索引和匹配的区别？](#语义索引和匹配有什么区别？)

* [【实战篇】PaddleNLP实战问题27个 ](#【实战篇】PaddleNLP实战问题 )
  
* [安装配置2题](#安装问题) 
  
    [Q1.如何在CUDA11安装和使用PaddlNLP?](#1-1)
  
    [Q2.在加载PaddleNLP内置的数据集或模型需要变更下载路径时，如何修改环境变量？](#1-2)
  
* [数据集和数据处理2题](#数据问题)
  
     [Q1.如何使用自己的数据集？](#2-1)
  
     [Q2.在数据类别分布不均衡的情况下， 应该如何去做处理？](#2-2)
  
* [模型训练调优16题](#训练调优问题)
  
  ​	[Q1.训练过程中，训练程序意外退出/挂起，应该如何解决？](#3-1)
  
  ​	[Q2. 如何加载自己的`bert-base-uncased`预训练模型，进而使用PaddleNLP的功能？](#3-2)
  
  ​	[Q3. 如何保存、加载训练好的模型？](#3-3)
  
  ​	[Q4. 在训练中断需要继续热启动训练的场景下，如何保证学习率和优化器能从中断地方继续迭代？](#3-4)
  
  ​	[Q5. 如何冻结模型梯度？](#3-5)
  
  ​	[Q6. 如何可视化acc,loss曲线图,模型网络结构图等？](#3-6)
  
  ​	[Q7. 如何在eval阶段打印评价指标，在各epoch保存模型参数？](#3-7)
  
  ​	[Q8. 在模型验证和测试过程中，如何保证每一次的结果是相同的？](#3-8)
  
  ​	[Q9. ERNIE模型如何返回中间层的输出？](#3-9)
  
  ​	[Q10. 在训练自定义模型的时候，需要在词表中加入额外的字典信息该如何操作？](#3-10)
  
  ​	[Q11. 在Q-A匹配场景下，当训练数据量较少时，有什么推荐的方法能提升模型效果吗？](#3-11)
  
  ​	[Q12. 如何设置parameter？](#3-12)
  
  ​	[Q13. 如何在训练时对CPU和GPU进行选择？](#3-13)
  
  ​	[Q14. 如何理解可解释性？](#3-14)
  
  ​	[Q15. 静态图如何转动态图？](#3-15)
  
  ​	[Q16. 以Ernie为例，如何使用Ernietokenizer保存模型的词表和配置文件？](#3-16)
  
* [预测部署4题](#部署问题)
  
  ​	[Q1. 请问如何进行模型压缩？](#4-1)
  
  ​	[Q2.PaddleNLP训练好的模型如何部署到服务器 ？](#4-2)
  
  ​	[Q3. NLP模型如何接入PaddleInference?](#4-3)
  
  ​	[Q4. NLP模型如何接入PaddleServing?](#4-4)
  
* [特定模型和NLP应用场景3题](#NLP应用场景)
  
  ​	[Q1.【解语】wordtag模型如何自定义添加命名实体及对应词类?](#5-1)
  
  ​    [Q2.【词法分析】LAC模型，如何自定义标签LABEL？](#5-2)
  
    ​    [Q3.【阅读理解】MapDatasets map方法中对应的batched=True是如何理解的，在阅读理解任务中为什么必须把参数batched设置为True？](#5-3)

<a name="NLP通用问题"></a>

## ⭐️【理论篇】NLP通用问题

<a name="语义索引和匹配有什么区别？"></a>

##### Q1. 语义索引和匹配有什么区别？https://github.com/PaddlePaddle/PaddleNLP/issues/699

语义索引要解决的核心问题是如何从海量 doc 中通过 ANN 索引的方式快速、准确地找出跟 query 相关的文档，语义匹配要解决的核心问题是对 query和文档更精细的语义匹配信息建模。同时，换个角度理解， [语义索引](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/semantic_indexing)是要解决搜索、推荐场景下的召回问题，而语义匹配是要解决排序问题，两者要解决的问题不同，所采用的方案也会有很大不同，但两者间存在一些共通的技术点，可以互相借鉴。

<a name="PaddleNLP实战问题"></a>

## 【实战篇】PaddleNLP实战问题 

<a name="安装问题"></a>

### 安装配置

<a name="1-1"></a>

##### Q1. 如何在CUDA11安装和使用PaddlNLP?

在CUDA11安装，可参考[issue](https://github.com/PaddlePaddle/PaddleNLP/issues/348)，其他CUDA版本安装可参考 [官方文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html)

<a name="1-2"></a>

##### Q2. 在加载PaddleNLP内置的数据集或模型需要变更下载路径时，如何修改环境变量？ https://github.com/PaddlePaddle/PaddleNLP/issues/560

1、Linux下， export PPNLP_HOME=“其他非中文的目录"即可；
		2、Windows下，配置环境变量 PPNLP_HOME 到其他非中文目录，重启即可。

<a name="数据问题"></a>

### 数据集和数据处理

<a name="2-1"></a>

##### Q1. 如何使用自己的数据集？

通过使用PaddleNLP提供的 `load_dataset`，  `MapDataset` 和 `IterDataset` ，可以方便的定义属于自己的数据集，然后可以完成相应的任务。参照官网文档：[自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。

<a name="2-2"></a>

##### Q2.在数据类别分布不均衡的情况下， 应该如何去做处理？https://github.com/PaddlePaddle/PaddleNLP/issues/791

可以参考以下几种方法：

1、欠采样，即去除一些多数类中的样本使得正例、反例数目接近，然后再进行学习。

2、过采样，即增加一些少数类样本使得正、 反例数目接近，然后再进行学习。

3、修改分类阈值，因为数据集存在类别不均衡的情况，所以直接训练分类器会使得模型在预测时更偏向于多数类，所以不再以0.5为分类阈值，而是针对少数类在模型仅有较小把握时就将样本归为少数类。

4、使用Focal loss计算损失函数值。Focal loss是一个能较好适应类别不均衡的损失函数。

<a name="训练调优问题"></a>

### 模型训练调优

<a name="3-1"></a>

##### Q1. 训练过程中，训练程序意外退出/挂起，应该如何解决？

一般先考虑内存、显存（使用GPU训练的话）是否不足，可在配置文件中，将训练和评估的batch size调小一些。需要注意，训练batch size调小时，学习率learning rate也要调小，一般可按等比例调整。<a name="3-2"></a>

##### Q2. 如何加载自己的`bert-base-uncased`预训练模型，进而使用PaddleNLP的功能？https://github.com/PaddlePaddle/PaddleNLP/issues/763

1、加载`bert-base-uncased`的tokenizer和model

```
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

2、通过`save_pretrained()`保存到`./trained_model`，`./trained_model`包含`model_config.json`，`model_state.pdparams`，`tokenizer_config.json`，`vocab.txt`:

```
tokenizer.save_pretrained('trained_model')
model.save_pretrained('trained_model')
```

3、从包含config file、模型、vocab.txt的路径加载tokenizer和model:

```
tokenizer = BertTokenizer.from_pretrained('trained_model')
model = BertModel.from_pretrained('trained_model')
```

<a name="3-3"></a>

##### Q3. 如何保存、加载训练好的模型？

1、预训练模型

保存：

```python
model.save_pretrained('./checkpoint')
tokenizer.save_pretrained('./checkpoint')
```

加载：

```python
model.from_pretrained('./checkpoint')
tokenizer.from_pretrained('./checkpoint')
```

2、其他模型

```python
paddle.save() #保存模型参数
paddle.load() #加载模型参数
```

<a name="3-4"></a>

##### Q4. 在训练中断需要继续热启动训练的场景下，如何保证学习率和优化器能从中断地方继续迭代？

1、先将lr, optimizer等参数用代码保存下来：

```python
paddle.save(lr_scheduler.state_dict(), "xx")
```

2、在恢复训练的时候再进行加载：

```python
lr_scheduler.set_state_dict(paddle.load("xx"))
```

<a name="3-5"></a>

##### Q5. 如何冻结模型梯度？https://github.com/PaddlePaddle/PaddleNLP/issues/297

1、一种方法是在optimizer上进行处理，model.parameters是一个list，可以通过parameters的name进行一定的过滤, 不让parameters进行更新；

```python
 [ p for p in model.parameters() if 'linear' not in p.name]  # 这里就可以过滤一下linear层，具体过滤策略可以根据需要来设定
```

2、另一种方法以ernie为例，将ernie输出的tensor设置stop_gradient为True。可以使用`register_forward_post_hook`按照如下的方式尝试：

```python
def forward_post_hook(layer, input, output):
    output.stop_gradient=True

self.ernie.register_forward_post_hook(forward_post_hook)
```

<a name="3-6"></a>

##### Q6. 如何可视化acc,loss曲线图,模型网络结构图等？

将配置文件里的`use_visualdl`参数设置为True即可，更多的可视化使用方法可以参考：[VisualDL使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/03_VisualDL/visualdl.html)。

<a name="3-7"></a>

##### Q7. 如何在eval阶段打印评价指标，在各epoch保存模型参数？https://github.com/PaddlePaddle/PaddleNLP/issues/170

paddle.Model.fit()在验证阶段会打印eval data的评价指标，并且打印的指标是一个累积的数值；另外可使用paddle.Model.fit指定save_freq参数，间隔一定的epoch数保存模型参数。

<a name="3-8"></a>

#####Q8. 在模型验证和测试过程中，如何保证每一次的结果是相同的？

在验证和测试过程中常常出现的结果不一致情况一般有以下几种解决方法：

1、如果是在预训练模型的微调阶段首先查看是否导入fine-tune模型，导入参数后，线性层在预测时就不会随机初始化，预测结果就是唯一的。

2、确保验证模式下排除一些随机性参数条件，例如dropout等随机因素。

3、在模型中固定随机数种子seed。

<a name="3-9"></a>

##### Q9. ERNIE模型如何返回中间层的输出？https://github.com/PaddlePaddle/PaddleNLP/issues/728

目前的API设计是不保留中间层输出的。 除了修改源码外，也可以通过加一个register_forward_post_hook来进行中间层的输出。 参考[register_forward_post_hook](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#register_forward_post_hook),在hook里把每一层encoder layer的输入存到一个全局的list里。

<a name="3-10"></a>  

##### Q10. 在训练自定义模型的时候，需要在词表中加入额外的字典信息该如何操作？https://github.com/PaddlePaddle/PaddleNLP/issues/702

一般的微调预训练模型通常使用和预训练阶段一样的字典就可以了，另外直接使用ERNIE的tokenizer会按照字粒度来切分无法产生词。另一种方式可以使用这些字典信息，可以将数据中在词典信息中的词进行整体mask进行一个mask language model的二次预训练，这样经过二次训练的模型就包含了对额外字典的表征。

<a name="3-11"></a>

##### Q11. 在Q-A匹配场景下，当训练数据量较少时，有什么推荐的方法能提升模型效果吗？

1. 针对小样本场景下训练1个QA匹配模型；
2. 基于我们开源的语义匹配模型热启，在此基础上用少量数据训练QA匹配模型。

<a name="3-12"></a>

##### Q12. 如何设置parameter？https://github.com/PaddlePaddle/PaddleNLP/issues/665

可以通过set_value来设置parameter，set_value的参数可以是numpy或者tensor

```python
   layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.ernie.config["initializer_range"],
                        shape=layer.weight.shape))
```

<a name="3-13"></a>

##### Q13. 如何在训练时对CPU和GPU进行选择？https://github.com/PaddlePaddle/PaddleNLP/issues/125

```python
#选择GPU训练
run_type_g.add_arg("use_cuda",bool,True,  "If set, use GPU for training.")
#选择CPU训练
run_type_g.add_arg("use_cuda",bool,False,  "If set, use CPU for training.")
```

<a name="3-14"></a>

##### Q14. 如何理解可解释性？https://github.com/PaddlePaddle/PaddleNLP/issues/849

请参考[可解释性示例](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/lime_tutorial_nlp_ERNIE.ipynb)。

<a name="3-15"></a>

##### Q15. 静态图如何转动态图？https://github.com/PaddlePaddle/PaddleNLP/issues/777

首先，需要将静态图参数保存成ndarray数据，然后将静态图参数名和对应动态图参数名对应，最后保存成动态图参数即可。请参考[参数转换脚本](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie/static_to_dygraph_params)。

<a name="3-16"></a>

##### Q16. 以Ernie为例，如何使用Ernietokenizer保存模型的词表和配置文件？https://github.com/PaddlePaddle/PaddleNLP/issues/277

具体流程如下：

```python
from paddlenlp.transformers import ErnieTokenizer
tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
···
tokenizer.save_pretrained(dir)
```

目录下tokenizer保存下来信息如下：

1. vocab.txt 词表

2. tokenizer具体的配置：tokenizer_config.json 

   <a name="部署问题"></a>

### 预测部署

<a name="4-1"></a>

##### Q1. 请问如何进行模型压缩？https://github.com/PaddlePaddle/PaddleNLP/issues/696

请参考[模型压缩示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/ofa)。

<a name="4-2"></a>

##### Q2. PaddleNLP训练好的模型如何部署到服务器 ？

参考[模型部署示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/electra)，其包含了动态转静态，以及部署的过程 。

<a name="4-3"></a>

##### Q3. NLP模型如何接入PaddleInference?

可参考[Paddle Inference预测示例](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_classification/pretrained_models/deploy/python/) 。

<a name="4-4"></a>

##### Q4. NLP模型如何接入PaddleServing?

可参考[Paddle Serving预测示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/pretrained_models/deploy/serving)。

<a name="NLP应用场景"></a>

### 特殊模型和NLP应用场景

<a name="5-1"></a>

##### Q1. 【解语】wordtag模型如何自定义添加命名实体及对应词类?

其主要依赖于二次构造数据来进行finetune，同时要更新termtree信息。wordtag分为两个步骤：
		1、通过BIOES体系进行分词；
		2、将分词后的信息和TermTree进行匹配。
		因此我们需要：
		1、分词正确，这里可能依赖于wordtag的finetune数据，来让分词正确；
		2、wordtag里面也需要把分词正确后term打上相应的知识信息。

可参考[issue](https://github.com/PaddlePaddle/PaddleNLP/issues/822)。

<a name="5-2"></a>

##### Q2. 【词法分析】LAC模型，如何自定义标签LABEL？

请参考[自定义标签示例](https://github.com/PaddlePaddle/PaddleNLP/issues/662)，[增量训练自定义LABLE示例](https://github.com/PaddlePaddle/PaddleNLP/issues/657)。

<a name="5-3"></a>

##### Q3. 【阅读理解】MapDatasets map方法中对应的batched=True是如何理解的，在阅读理解任务中为什么必须把参数batched设置为True？

batched=True就是对整个batch的数据进行map，而非逐条进行map。在阅读理解任务中，一个example可能变成多个feature ，对数据逐条map是行不通的，所以需要设置batched=True。




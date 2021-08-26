## 1FAQ


## 写在前面

+ 我们收集整理了开源以来在Issues和用户群中的常见问题并且给出了简要解答，旨在为NLP的开发者提供一些参考，也希望帮助大家少走一些弯路。
+ NLP领域大佬众多，本文档回答主要依赖有限的项目实践，难免挂一漏万，如有遗漏和不足，也**希望有识之士帮忙补充和修正**，万分感谢。
+ 使用过程中可参考PaddleNLP[官方文档](https://paddlenlp.readthedocs.io/zh/latest/index.html) 、[AI Studio官方项目](https://aistudio.baidu.com/aistudio/projectdetail/1535371)。

## PaddleNLP常见问题汇总（持续更新）

* ⭐[【精选】NLP精选10问](#[理论篇]精选)

* ⭐[【理论篇】NLP理论10问](#[理论篇]NLP通用问题)

    * [Q1.语义索引和匹配的区别？](#语义索引和匹配有什么区别？)

* ⭐ [【实战篇】PaddleNLP实战问题27问 ](#[实战篇]PaddleNLP实战问题 )
  
  * [安装配置2题](#安装问题) 
    
      ​	[Q1.如何在CUDA11安装和使用PaddlNLP?](#1-1)
    
      ​	[Q2.在加载PaddleNLP内置的数据集或模型需要变更下载路径时，如何修改环境变量？](#1-2)
    
  * [数据集和数据处理2题](#数据问题)
    
       ​	[Q1.如何使用自己的数据集？](#2-1)
    
       ​	[Q2.在数据类别分布不均衡的情况下， 应该如何去做处理？](#2-2)
    
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
    
       ​ [Q3.【阅读理解】MapDatasets map方法中对应的batched=True是如何理解的，在阅读理解任务中为什么必须把参数batched设置为True？](#5-3)

<a name="NLP通用问题"></a>

## 【理论篇】NLP通用问题

<a name="2-2"></a>

##### Q. 数据类别分布不均衡， 有哪些应对方法？https://github.com/PaddlePaddle/PaddleNLP/issues/791

**A：**可以采用以下几种方法优化类别分布不均衡问题：

（1）欠采样：对样本量较多的类别进行欠采样，随机去除一些样本，使得各类别数目接近。

（2）过采样：对样本量较少的类别进行过采样，随机选择样本进行复制，使得各类别数目接近。

（3）修改分类阈值：直接使用类别分布不均衡的数据训练分类器，会使得模型在预测时更偏向于多数类，所以不再以0.5为分类阈值，而是针对少数类在模型仅有较小把握时就将样本归为少数类。

（4）损失函数选用Focal loss，Focal loss能较好适应类别不均衡情况。

<a name="3-11"></a>

##### Q2. 当训练样本较少时，有什么推荐的方法能提升模型效果吗？

**A： ** 增加训练样本带来的效果是最直接的。此外，可以基于我们开源的[预训练模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers)进行热启，再用少量数据集fine-tune模型。此外，针对分类、匹配等场景，[小样本学习](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/few_shot)也能够带来不错的效果。

<a name="NLP精选"></a>

## 【精选】NLP精选5问

<a name="1-2"></a>

##### Q1. 如何加载自己的本地数据集，以便使用PaddleNLP的功能？

**A： **通过使用PaddleNLP提供的 `load_dataset`，  `MapDataset` 和 `IterDataset` ，可以方便的自定义属于自己的数据集哦，也欢迎您贡献数据集到PaddleNLP repo。

从本地文件创建数据集时，我们 **推荐** 根据本地数据集的格式给出读取function并传入 `load_dataset()` 中创建数据集。

```python
from paddlenlp.datasets import load_dataset

def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过列名
        next(f)
        for line in f:
            words, labels = line.strip("\n').split("\t')
            words = words.split("\002')
            labels = labels.split("\002')
            yield {'tokens': words, 'labels': labels}

# data_path为read()方法的参数
map_ds = load_dataset(read, data_path='train.txt',lazy=False)
iter_ds = load_dataset(read, data_path='train.txt',lazy=True)
```

如果您习惯使用`paddle.io.Dataset/IterableDataset`来创建数据集也是支持的，您也可以从其他python对象如`List`对象创建数据集，详细内容可参照[官方文档-自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。

<a name="1-2"></a>

##### Q2. PaddleNLP会将内置的数据集、模型下载到默认路径，如何修改路径？ https://github.com/PaddlePaddle/PaddleNLP/issues/560

**A：** 内置的数据集、模型默认会下载到`$HOME/.paddlenlp/`下，通过配置环境变量可下载到指定路径：

（1）Linux下，设置 `export PPNLP_HOME="xxxx"`，注意不要设置中文目录。

（2）Windows下，同样配置环境变量 PPNLP_HOME 到其他非中文目录，重启即可。

<a name="1-2"></a>

##### Q3. PaddleNLP中如何保存、加载训练好的模型？

**A： **（1）预训练模型

​	保存：

```python
model.save_pretrained("./checkpoint')
tokenizer.save_pretrained("./checkpoint')
```

​	加载：

```python
model.from_pretrained("./checkpoint')
tokenizer.from_pretrained("./checkpoint')
```

（2）其他模型

```python
paddle.save() #保存模型参数
paddle.load() #加载模型参数
```

<a name="2-2"></a>

##### Q4. 如何提升模型的性能，提升模型QPS？https://github.com/PaddlePaddle/PaddleNLP/issues/791 @郭晟

**A：** 有几种思路：

（1）模型压缩：如将复杂模型信息蒸馏到简单模型；使用较小的模型如TinyBERT等，参考[模型压缩示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression)。

（2）GPU环境，可采用TensorRT加速，参考...

（3）混合精度，参考...

...

<a name="2-2"></a>

##### Q5. 如果使用预训练模型，一般需要多少条样本？@泽阳

**A： ** 



<a name="PaddleNLP实战问题"></a>

## 【实战篇】PaddleNLP实战问题 

<a name="数据问题"></a>

### 数据集和数据处理

<a name="3-16"></a>

##### Q. 使用自己的数据集训练预训练模型时，如何引入额外的词表？https://github.com/PaddlePaddle/PaddleNLP/issues/702  @郭晟

**A： **一般的微调预训练模型通常使用和预训练阶段一样的字典就可以了，另外直接使用ERNIE的tokenizer会按照字粒度来切分无法产生词。另一种方式可以使用这些字典信息，可以将数据中在词典信息中的词进行整体mask进行一个mask language model的二次预训练，这样经过二次训练的模型就包含了对额外字典的表征。

<a name="3-16"></a>

##### Q2. 以Ernietokenizer为例， 如何使用预训练模型Tokenizer加载自定义词？https://github.com/PaddlePaddle/PaddleNLP/issues/277

**A： **具体流程如下：

```python
from paddlenlp.transformers import ErnieTokenizer
tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
···
tokenizer.save_pretrained("./checkpoint")
```

在`./checkpoint`目录下保存下来词表文件`vocab.txt `和配置文件`tokenizer_config.json`，修改相应配置即可。

<a name="训练调优问题"></a>

### 模型训练调优

<a name="3-2"></a>

##### Q. 如何加载自己的预训练模型，进而使用PaddleNLP的功能？https://github.com/PaddlePaddle/PaddleNLP/issues/763

**A： ** 以bert为例，如果是使用PaddleNLP训练，通过`save_pretrained()`接口保存的模型，可通过`from_pretrained()`来加载：

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

如果不是上述情况，可以使用如下方式加载模型，也欢迎您贡献模型到PaddleNLP repo中。

（1）加载`BertTokenizer`和`BertModel`

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

（2）调用`save_pretrained()`生成 `model_config.json`、 ``tokenizer_config.json``、`model_state.pdparams`、  `vocab.txt `文件，保存到`./checkpoint`：

```python
tokenizer.save_pretrained("./checkpoint")
model.save_pretrained("./checkpoint")
```

（3）修改`model_config.json`、 `tokenizer_config.json`这两个配置文件，指定为自己的模型，之后通过`from_pretrained()`加载模型。

```python
tokenizer = BertTokenizer.from_pretrained("./checkpoint")
model = BertModel.from_pretrained("./checkpoint")
```

<a name="3-4"></a>

##### Q. 如果训练中断，需要继续热启动训练，如何保证学习率和优化器能从中断地方继续迭代？

**A：**

 （1）先将`lr`、` optimizer`等参数保存下来：

```python
paddle.save(lr_scheduler.state_dict(), "xxxx")
```

（2）加载参数恢复训练：

```python
lr_scheduler.set_state_dict(paddle.load("xxxx"))
```

<a name="3-5"></a>

##### Q3. 如何冻结模型梯度？https://github.com/PaddlePaddle/PaddleNLP/issues/297  @ 郭晟@泽阳

**A： **

可以直接修改PaddleNLP内部代码实现：在forward里用paddle.no_grad()包裹一下...

或者，也可以采取以下方法：

（1）一种方法以Ernie为例，将模型输出的tensor设置`stop_gradient`为True。可以使用`register_forward_post_hook`按照如下的方式尝试：

```python
def forward_post_hook(layer, input, output):
    output.stop_gradient=True

self.ernie.register_forward_post_hook(forward_post_hook)
```

（2）另一种方法是在`optimizer`上进行处理，`model.parameters`是一个`List`，可以通过`name`进行相应的过滤, 不更新某些参数，这种方法需要对网络结构的名字有整体了解，因为网络结构的实体名字决定了参数的名字，这个使用方法有一定的门槛：

```python
 [ p for p in model.parameters() if 'linear' not in p.name]  # 这里就可以过滤一下linear层，具体过滤策略可以根据需要来设定
```

<a name="3-7"></a>

##### Q4. 如何在eval阶段打印评价指标，在各epoch保存模型参数？https://github.com/PaddlePaddle/PaddleNLP/issues/170  @燚标

**A：**  `paddle.Model.fit() `在验证阶段会打印eval_data的评价指标，并且打印的指标是一个累积的数值。另外可使用`paddle.Model.fit()`指定`save_freq`参数，间隔一定的epoch数保存模型参数。

<a name="3-1"></a>

##### Q5. 训练过程中，训练程序意外退出或Hang住，应该如何排查？

**A： ** 一般先考虑内存、显存（使用GPU训练的话）是否不足，可将训练和评估的batch size调小一些。

需要注意，batch size调小时，学习率learning rate也要调小，一般可按等比例调整。

<a name="3-8"></a>

##### Q6. 在模型验证和测试过程中，如何保证每一次的结果是相同的？

**A： **在验证和测试过程中常常出现的结果不一致情况一般有以下几种解决方法：

（1）如果是在预训练模型的微调阶段首先查看是否导入fine-tune模型，导入参数后，线性层在预测时就不会随机初始化，预测结果就是唯一的。

（2）确保验证模式下排除一些随机性参数条件，例如dropout等随机因素。

（3）在模型中固定随机数种子seed，如何固定？

<a name="3-9"></a>

##### Q7. ERNIE模型如何返回中间层的输出？https://github.com/PaddlePaddle/PaddleNLP/issues/728

**A： **目前的API设计是不保留中间层输出的，当然在PaddleNLP里可以很方便得修改源码。此外，也可以通过`register_forward_post_hook()`为Layer注册一个 `forward post-hook` 函数，该 `hook` 函数将会在 `forward` 函数调用之后被调用，输出中间层信息。详情参考[register_forward_post_hook说明文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#register_forward_post_hook)，在hook里把每一层encoder layer的输入存到一个全局的`List`里。



<a name="部署问题"></a>

### 预测部署

<a name="4-1"></a>

##### Q1. PaddleNLP训练好的模型如何部署到服务器 ？@政锡

**A：** 我们推荐在动态图模式下开发，静态图模式部署。

（1）动转静

（2）借助Paddle Inference部署

   参考[/PaddleNLP/examples](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/)下的deploy目录，如[基于ERNIE的命名实体识别模型部署](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/waybill_ie/deploy/python)。

<a name="4-4"></a>

##### Q2. PaddleNLP模型如何接入PaddleServing? @政锡

**A： ** 首先需要将动态图模型转成静态图模型，再借助Paddle Inferenc和Paddle Serving。

可参考[Paddle Serving预测示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/pretrained_models/deploy/serving)。

<a name="3-15"></a>

##### Q3. 静态图模型如何转换成动态图模型？https://github.com/PaddlePaddle/PaddleNLP/issues/777 @泽阳

**A： **首先，需要将静态图参数保存成`ndarray`数据，然后将静态图参数名和对应动态图参数名对应，最后保存成动态图参数即可。详情可参考[参数转换脚本](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie/static_to_dygraph_params)。

<a name="NLP应用场景"></a>

### 特定模型和应用场景咨询

<a name="5-2"></a>

##### Q1. 【词法分析】LAC模型，如何自定义标签label，并继续训练？ @舜杰

**A： **更新label文件`tag.dict`，修改下CRF的标签数即可....

请参考[自定义标签示例](https://github.com/PaddlePaddle/PaddleNLP/issues/662)，[增量训练自定义LABLE示例](https://github.com/PaddlePaddle/PaddleNLP/issues/657)。

<a name="3-13"></a>

##### Q2. 信息抽取任务中，是否推荐使用预训练模型+CRF，怎么实现呢？@

**A：** 

<a name="5-1"></a>

##### Q3. 【阅读理解】`MapDatasets`的`map()`方法中对应的`batched=True`怎么理解，在阅读理解任务中为什么必须把参数`batched`设置为`True`？

**A： **`batched=True`就是对整个batch的数据进行map，而非逐条进行map。在阅读理解任务中，一条样本需要拆分成多条处理 ，对数据逐条map是行不通的，所以需要设置`batched=True`。

<a name="语义索引和匹配有什么区别？"></a>

##### Q4. 【语义匹配】语义索引和语义匹配有什么区别？https://github.com/PaddlePaddle/PaddleNLP/issues/699

**A：**语义索引要解决的核心问题是如何从海量 Doc 中通过 ANN 索引的方式快速、准确地找出与 query 相关的文档，语义匹配要解决的核心问题是对 query和文档更精细的语义匹配信息建模。换个角度理解， [语义索引](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/semantic_indexing)是要解决搜索、推荐场景下的召回问题，而[语义匹配](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_matching)是要解决排序问题，两者要解决的问题不同，所采用的方案也会有很大不同，但两者间存在一些共通的技术点，可以互相借鉴。

<a name="5-3"></a>

##### Q5. 【解语】wordtag模型如何自定义添加命名实体及对应词类?  @泽阳

**A： **其主要依赖于二次构造数据来进行finetune，同时要更新termtree信息。wordtag分为两个步骤：
（1）通过BIOES体系进行分词；
（2）将分词后的信息和TermTree进行匹配。
	因此我们需要：
（1）分词正确，这里可能依赖于wordtag的finetune数据，来让分词正确；
（2）wordtag里面也需要把分词正确后term打上相应的知识信息。

可参考[issue](https://github.com/PaddlePaddle/PaddleNLP/issues/822)。

<a name="使用咨询问题"></a>

### 其他使用咨询

<a name="1-1"></a>

##### Q1. 如何在CUDA11安装和使用PaddlNLP?

**A： **在CUDA11安装，可参考[issue](https://github.com/PaddlePaddle/PaddleNLP/issues/348)，其他CUDA版本安装可参考 [官方文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html)

<a name="1-2"></a>

##### Q2. 如何设置parameter？https://github.com/PaddlePaddle/PaddleNLP/issues/665

**A： **可以通过`set_value()`来设置parameter，`set_value()`的参数可以是`numpy`或者`tensor`。

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

##### Q3. GPU版的Paddle虽然能在CPU上运行，但是必须要有GPU设备吗？

**A：**  `export CUDA_VISIBLE_DEVICES='"`，CPU是可以正常跑的。

##### Q4:  如何指定用CPU还是GPU训练模型？https://github.com/PaddlePaddle/PaddleNLP/issues/125

**A： **

<a name="3-14"></a>

##### Q5. 模型可解释性问题，如何解释模型预测出的结果？https://github.com/PaddlePaddle/PaddleNLP/issues/849

**A：** [InterpretDL](https://github.com/PaddlePaddle/InterpretDL)提供了一系列模型可解释性算法，可参考[可解释性示例](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/lime_tutorial_nlp_ERNIE.ipynb)。

<a name="3-14"></a>

##### Q6. 动态图模型和静态图模型的预测结果一致吗？@

**A：** 

<a name="3-13"></a>

##### Q7. 如何可视化acc,loss曲线图,模型网络结构图等？

**A： **将配置文件里的`use_visualdl`参数设置为True即可，更多的可视化使用方法可以参考：[VisualDL使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/03_VisualDL/visualdl.html)。




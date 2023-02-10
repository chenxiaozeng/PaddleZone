简体中文 | [English](./README_en.md)

<p align="center">
  <img src="./docs/imgs/paddlenlp.png" width="718" height ="100" />
</p>


------------------------------------------------------------------------------------------

[![PyPI - PaddleNLP Version](https://img.shields.io/pypi/v/paddlenlp.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/paddlenlp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paddlenlp)](https://pypi.org/project/paddlenlp/)
[![PyPI Status](https://pepy.tech/badge/paddlenlp/month)](https://pepy.tech/project/paddlenlp)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
![GitHub](https://img.shields.io/github/license/paddlepaddle/paddlenlp)

这里考虑加下各种门户链接，to update...

## News  <img src="./docs/imgs/news_icon.png" width="40"/>

<details><summary>News</summary><div>
* [2021-10-12] PaddleNLP 2.1版本已发布！新增开箱即用的NLP任务能力、Prompt Tuning应用示例与生成任务的高性能推理！:tada:更多详细升级信息请查看[Release Note](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.1.0)。
* [2021-09-16][《千言-问题匹配鲁棒性评测》](https://www.datafountain.cn/competitions/516)正式开赛啦🔥🔥🔥，欢迎大家踊跃报名!! [官方基线地址](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_matching/question_matching)。
* [2021-08-22][《千言：面向事实一致性的生成评测比赛》](https://aistudio.baidu.com/aistudio/competition/detail/105)正式开赛啦🔥🔥🔥，欢迎大家踊跃报名!! [官方基线地址](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_generation/unimo-text)。
</div></details>


## 简介

PaddleNLP是飞桨自然语言处理开发库，具备**易用的文本领域API**，**多场景的应用示例**、和**高性能分布式训练**三大特点，旨在提升开发者在文本领域的开发效率，并提供丰富的NLP应用示例。

- **易用的文本领域API**
  - 提供丰富的产业级预置任务能力[Taskflow](./docs/model_zoo/taskflow.md)和全流程的文本领域API：支持丰富中文数据集加载的[Dataset API](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html)；灵活高效地完成数据预处理的[Data API](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.data.html)；提供100+预训练模型的[Transformer API](./docs/model_zoo/transformers.rst)等，可大幅提升NLP任务建模的效率。

- **多场景的应用示例**
  - 覆盖从学术到产业级的NLP[应用示例](#多场景的应用示例)，涵盖NLP基础技术、NLP系统应用以及相关拓展应用。全面基于飞桨核心框架2.0全新API体系开发，为开发者提供飞桨文本领域的最佳实践。

- **高性能分布式训练**
  - 基于飞桨核心框架领先的自动混合精度优化策略，结合分布式Fleet API，支持4D混合并行策略，可高效地完成大规模预训练模型训练。

## 安装

### 环境依赖

- python >= 3.6
- paddlepaddle >= 2.1

### pip安装

```shell
pip install --upgrade paddlenlp
```

更多关于PaddlePaddle和PaddleNLP安装的详细教程请查看[Installation](./docs/get_started/installation.rst)。

## 易用的文本领域API

### Taskflow：开箱即用的产业级NLP能力

Taskflow旨在提供**开箱即用**的NLP预置任务能力，覆盖自然语言理解与生成两大场景，提供**产业级的效果**与**极致的预测性能**。

```python
from paddlenlp import Taskflow

# 中文分词
seg = Taskflow("word_segmentation")
seg("第十四届全运会在西安举办")
>>> ['第十四届', '全运会', '在', '西安', '举办']

# 词性标注
tag = Taskflow("pos_tagging")
tag("第十四届全运会在西安举办")
>>> [('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')]

# 命名实体识别
ner = Taskflow("ner")
ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
>>> [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]

# 句法分析
ddp = Taskflow("dependency_parsing")
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]

# 情感分析
senta = Taskflow("sentiment_analysis")
senta("这个产品用起来真的很流畅，我非常喜欢")
>>> [{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9938690066337585}]
```

更多使用方法请参考[Taskflow文档](./docs/model_zoo/taskflow.md)。

### Transformer API: 强大的预训练模型生态底座

覆盖**30**个网络结构和**100**余个预训练模型参数，既包括百度自研的预训练模型如ERNIE系列, PLATO, SKEP等，也涵盖业界主流的中文预训练模型如BERT，GPT，XLNet，BART等。使用AutoModel可以下载不同网络结构的预训练模型。欢迎开发者加入贡献更多预训练模型！🤗

```python
from paddlenlp.transformers import *

ernie = AutoModel.from_pretrained('ernie-1.0')
ernie_gram = AutoModel.from_pretrained('ernie-gram-zh')
bert = AutoModel.from_pretrained('bert-wwm-chinese')
albert = AutoModel.from_pretrained('albert-chinese-tiny')
roberta = AutoModel.from_pretrained('roberta-wwm-ext')
electra = AutoModel.from_pretrained('chinese-electra-small')
gpt = AutoModelForPretraining.from_pretrained('gpt-cpm-large-cn')
```

对预训练模型应用范式如语义表示、文本分类、句对匹配、序列标注、问答等，提供统一的API体验。

```python
import paddle
from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')
text = tokenizer('自然语言处理')

# 语义表示
model = AutoModel.from_pretrained('ernie-1.0')
sequence_output, pooled_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
# 文本分类 & 句对匹配
model = AutoModelForSequenceClassification.from_pretrained('ernie-1.0')
# 序列标注
model = AutoModelForTokenClassification.from_pretrained('ernie-1.0')
# 问答
model = AutoModelForQuestionAnswering.from_pretrained('ernie-1.0')
```

请参考[Transformer API文档](./docs/model_zoo/transformers.rst)查看目前支持的预训练模型结构、参数和详细用法。

### Dataset API: 丰富的中文数据集

Dataset API提供便捷、高效的数据集加载功能；内置[千言数据集](https://www.luge.ai/)，提供丰富的面向自然语言理解与生成场景的中文数据集，为NLP研究人员提供一站式的科研体验。

```python
from paddlenlp.datasets import load_dataset

train_ds, dev_ds, test_ds = load_dataset("chnsenticorp", splits=["train", "dev", "test"])

train_ds, dev_ds = load_dataset("lcqmc", splits=["train", "dev"])
```

可参考[Dataset文档](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html) 查看更多数据集。

### Embedding API: 一键加载预训练词向量

```python
from paddlenlp.embeddings import TokenEmbedding

wordemb = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")
print(wordemb.cosine_sim("国王", "王后"))
>>> 0.63395125
wordemb.cosine_sim("艺术", "火车")
>>> 0.14792643
```

内置50+中文词向量，覆盖多种领域语料、如百科、新闻、微博等。更多使用方法请参考[Embedding文档](./docs/model_zoo/embeddings.md)。

### 更多API使用文档

- [Data API](./docs/data.md): 提供便捷高效的文本数据处理功能
- [Metrics API](./docs/metrics.md): 提供NLP任务的评估指标，与飞桨高层API兼容。

更多的API示例与使用说明请查阅[PaddleNLP官方文档](https://paddlenlp.readthedocs.io/)

## 多场景的应用示例

PaddleNLP提供了多粒度、多场景的NLP应用示例，面向动态图模式和全新的API体系开发，更加简单易懂。
涵盖了[NLP基础技术](#nlp-基础技术)、[NLP系统应用](#nlp-系统应用)以及文本相关的[NLP拓展应用](#拓展应用)、与知识库结合的[文本知识关联](./examples/text_to_knowledge)、与图结合的[文本图学习](./examples/text_graph/)等。

这里来一张场景（模型）全景图（类似detection这种），高亮体现notebook交互式教程。细节另起一个文档。详情请参考[Task List文档](./docs/task_list/README.md)。
![image](https://user-images.githubusercontent.com/11793384/145320976-2ef97187-0b2c-4ee6-ab69-1ba092e470a9.png)

## 文档教程(Tutorial)

- [运行环境准备](./doc/doc_ch/environment.md)
- [快速开始（10分钟完成高精度中文情感分析）](./doc/doc_ch/quickstart.md)
- 数据准备
  - [内置数据集列表](./doc/doc_ch/models.md)
  - [加载数据集](./doc/doc_ch/models_list.md)
  - [自定义数据集](./doc/doc_ch/inference_ppocr.md)    
- [模型库](./ppstructure/README_ch.md)
  - [预训练词向量](./ppstructure/layout/README_ch.md)
  - [预训练模型](./ppstructure/table/README_ch.md)
- 推理部署（以文本分类为例）
  - [基于C++预测引擎推理](./deploy/cpp_infer/readme.md)
  - [服务化部署](./deploy/pdserving/README_CN.md)
  - [端侧部署](./deploy/lite/readme.md)
  - [Benchmark](./doc/doc_ch/benchmark.md)
- [面向场景实践](./ppstructure/README_ch.md)
  - [Taskflow](./ppstructure/layout/README_ch.md)
  - [场景list](./ppstructure/table/README_ch.md)
    -  [Notebook](./ppstructure/table/README_ch.md)
- FAQ
  - [【精选】NLP精选10个问题](./doc/doc_ch/FAQ.md)
  - [【理论篇】NLP通用50个问题](./doc/doc_ch/FAQ.md)
  - [【实战篇】PaddleNLP实战183个问题](./doc/doc_ch/FAQ.md)
- 理论学习素材
  - [预训练模型前世今生](./doc/doc_ch/FAQ.md)  
- [代码组织结构](./doc/doc_ch/tree.md)

## 社区贡献与技术交流

### 特殊兴趣小组

- 欢迎您加入PaddleNLP的SIG社区，贡献优秀的模型实现、公开数据集、教程与案例等。

### QQ

- 现在就加入PaddleNLP的QQ技术交流群，一起交流NLP技术吧！⬇️

<div align="center">
  <img src="./docs/imgs/qq.png" width="200" height="200" />
</div>  


## 版本更新

更多版本更新说明请查看[ChangeLog](./docs/changelog.md)

## Acknowledge

我们借鉴了Hugging Face的[Transformers](https://github.com/huggingface/transformers)🤗关于预训练模型使用的优秀设计，在此对Hugging Face作者及其开源社区表示感谢。

## License

PaddleNLP遵循[Apache-2.0开源协议](./LICENSE)。

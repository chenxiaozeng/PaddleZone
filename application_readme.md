# 产业级端到端系统范例  

## 1、简介

PaddleNLP 从预训练模型库出发，提供了经典预训练模型在主流 NLP 任务上丰富的[应用示例](../examples)，满足了大量开发者的学习科研与基础应用需求。

针对更广泛的产业落地需求、更复杂的 NLP 场景任务，PaddleNLP 推出**产业级端到端系统范例库**（下文简称产业范例），提供单个模型之上的产业解决方案。

- 最强模型与实践———产业范例针对具体业务场景，提供最佳模型（组合），兼顾模型精度与性能，降低开发者模型选型成本；
- 全流程端到端———打通数据标注-模型训练-模型调优-模型压缩—预测部署全流程，帮助开发者更低成本得完成产业落地。

## 2、基于 Pipelines 构建产业范例，加速落地

在面向不同场景任务建设一系列产业方案的过程中，不难发现，从技术基础设施角度看：     

（1）NLP系统都可以抽象为由多个基础组件串接而成的流水线系统；       
（2）多个NLP流水线系统可共享使用相同的基础组件。     

因此，PaddleNLP 逐渐孵化出了一套 NLP 流水线系统 [Pipelines](../pipelines)，将各个 NLP 复杂系统的通用模块抽象封装为标准组件，支持开发者通过配置文件对标准组件进行组合，仅需几分钟即可定制化构建智能系统，让解决NLP任务像搭积木一样便捷、灵活、高效。同时，Pipelines 中预置了前沿的预训练模型和算法，在研发效率、模型效果和性能方面提供多重保障。因此，Pipelines 能够大幅加快开发者使用飞桨落地的效率。


<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/212836991-d9132e46-b5bf-4389-80e1-4f9dee32f1fe.png" width="90%" length="90%">
</div>

**PaddleNLP 提供了多个版本的产业范例:**

- 如果你希望快速体验、直接应用、从零搭建一套完整系统，推荐使用 **Pipelines 版本**。这里集成了训练好的模型，无需关心模型训练细节；提供 Docker 环境，可快速一键部署端到端系统；打通前端 Demo 界面，便于直观展示、分析、调试效果。

- 如果你希望使用自己的业务数据进行二次开发，推荐使用`./applications`目录下的**可定制版本**，训练好的模型可以直接集成进 Pipelines 中进行使用。

| 场景任务   | Pipelines版本地址 | 可定制版本地址 | 
| :--------------- | ------- | ------- | 
| 检索系统 | [字面+语义检索系统](../pipelines/examples/semantic-search) | [语义检索系统](./neural_search) |  
| 智能问答系统 | [FAQ问答](../pipelines/examples/FAQ/)、[无监督问答](../pipelines/examples/unsupervised-question-answering)、[检索式问答](../pipelines/examples/question-answering) | [智能问答系统](./question_answering) |  
| 文本分类系统 | 暂无 | [文本分类系统](./text_classification)  |  
| 零样本分类系统 | 暂无 | [零样本分类系统](./zero_shot_text_classification) |  
| 通用信息抽取系统 | 暂无 | [通用信息抽取系统](./information_extraction) |  
| 观点抽取与情感分析系统  | [情感分析](../pipelines/examples/sentiment_analysis)  | [观点抽取与情感分析系统](./sentiment_analysis) |  
| 语音指令解析系统  | 暂无 | [语音指令解析系统](./speech_cmd_analysis) |  
| 文档智能系统  | [文档抽取问答](../pipelines/examples/document-intelligence) |  [跨模态文档问答](./document_intelligence/doc_vqa)|  
| 文本摘要系统  | 暂无 | [文本摘要系统](./text_summarization) | 
| 文生图系统  | [文生图系统](../pipelines/examples/text_to_image)  | 可参考[PPDiffusers]() |   

## 3、典型范例介绍

#### 🔍 语义检索系统

针对无监督数据、有监督数据等多种数据情况，结合SimCSE、In-batch Negatives、ERNIE-Gram单塔模型等，推出前沿的语义检索方案，包含召回、排序环节，打通训练、调优、高效向量检索引擎建库和查询全流程。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/213134465-30cae5fd-4cd1-4e5b-a1cb-fa55c72980a7.gif" width="60%" length="60%">
</div>


更多使用说明请参考[语义检索系统](./neural_search)。

#### ❓ 智能问答系统

基于[🚀RocketQA](https://github.com/PaddlePaddle/RocketQA)技术的检索式问答系统，支持FAQ问答、说明书问答等多种业务场景。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168514868-1babe981-c675-4f89-9168-dd0a3eede315.gif" width="400">
</div>


更多使用说明请参考[智能问答系统](./question_answering)与[文档智能问答](./document_intelligence/doc_vqa)

#### 💌 评论观点抽取与情感分析

基于情感知识增强预训练模型SKEP，针对产品评论进行评价维度和观点抽取，以及细粒度的情感分析。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407260-b7f92800-861c-4207-98f3-2291e0102bbe.png" width="400">
</div>

更多使用说明请参考[情感分析](./sentiment_analysis)。

#### 🎙️ 智能语音指令解析

集成了[PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)和[百度开放平台](https://ai.baidu.com/)的的语音识别和[UIE](./model_zoo/uie)通用信息抽取等技术，打造智能一体化的语音指令解析系统范例，该方案可应用于智能语音填单、智能语音交互、智能语音检索等场景，提高人机交互效率。

<div align="center">
    <img src="https://user-images.githubusercontent.com/16698950/168589100-a6c6f346-97bb-47b2-ac26-8d50e71fddc5.png" width="400">
</div>

更多使用说明请参考[智能语音指令解析](./applications/speech_cmd_analysis)。

# 面向深度学习模型的可靠性测试综述
这个 Github 存储库总结了深度学习模型的可靠性测试资源的精选列表。 有关更多详细信息和分类标准，请参阅我们的综述论文。

为什么研究可靠性测试？深度学习模型由于其出色的性能表现而在各个领域被广泛应用，但它们在面对不确定输入时，往往会出现意料之外的错误行为，在诸如自动驾驶系统等安全关键应用，可能会造成灾难性的后果。深度模型的可靠性问题引起了学术界和工业界的广泛关注。因此，在深度模型部署前迫切需要对模型进行系统性测试，通过生成测试样本，并由模型的输出得到测试报告，以评估模型的可靠性，提前发现潜在缺陷。然而，深度测试虽然已在多个领域得到应用，但尚缺少对其任务性能、安全性、公平性与隐私性四个方面展开全面测试的方法综述。

 1. 任务性能测试
 - 模型准确率测试
 - 训练程度测试
 2. 安全性测试
 - 推理阶段安全性测试
 - 训练阶段安全性测试
 - 测试样本选取方法
 3. 公平性和隐私性测试
 - 公平性测试
 - 隐私性测试
 4. 可靠性测试的应用
 - 自动驾驶
 - 语音识别
 - 自然语言处理

# 安全性测试方法

## 模型准确率测试

Classifier variability: Accounting for training and testing [\[pdf\]](https://reader.elsevier.com/reader/sd/pii/S0031320312000180?token=7C5D9FD0D6EE41064EA1B9A3A80DE4E28C930BEEEC195BB9C4D45E50DEC6069A149A53DCD1C59FAC988D0BDC82459849&originRegion=us-east-1&originCreation=20211112132635)

SynEva: Evaluating ML Programs by Mirror Program Synthesis [\[pdf\]](https://cs.nju.edu.cn/changxu/1_publications/18/QRS18.pdf)
## 训练程度测试
Perturbed Model Validation: A New Framework to Validate Model Relevance [\[pdf\]](https://hal.inria.fr/hal-02139208/document)

Detecting Overfitting via Adversarial Examples [\[pdf\]](https://arxiv.org/pdf/1903.02380.pdf)

Circuit-Based Intrinsic Methods to Detect Overfitting [\[pdf\]](http://proceedings.mlr.press/v119/chatterjee20a/chatterjee20a.pdf)

Test data reuse for evaluation of adaptive machine learning algorithms: over-fitting to a fixed 'test' dataset and a potential solution [\[pdf\]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10577/2293818/Test-data-reuse-for-evaluation-of-adaptive-machine-learning-algorithms/10.1117/12.2293818.short?SSO=1)

MODE: Automated neural network model debugging via state differential analysis and input selection [\[pdf\]](https://www.researchwithrutgers.com/en/publications/mode-automated-neural-network-model-debugging-via-state-different)


## 推理阶段安全性测试
###  基于覆盖率的测试方法
DeepXplore: Automated Whitebox Testing of Deep Learning Systems [\[pdf\]](https://arxiv.org/pdf/1705.06640.pdf)

DLFuzz: Differential Fuzzing Testing of Deep Learning Systems [\[pdf\]](https://arxiv.org/pdf/1808.09413.pdf)

TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing [\[pdf\]](http://proceedings.mlr.press/v97/odena19a/odena19a.pdf)

DeepTest: Automated Testing of Deep-Neural-Network-driven Autonomous Cars [\[pdf\]](https://dl.acm.org/doi/pdf/10.1145/3180155.3180220)

DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems [\[pdf\]](https://arxiv.org/pdf/1803.07519.pdf)

Deephunter: A coverage-guided fuzz testing framework for deep neural networks [\[pdf\]](https://experts.illinois.edu/en/publications/deephunter-a-coverage-guided-fuzz-testing-framework-for-deep-neur)

Effective White-Box Testing of Deep Neural Networks with Adaptive Neuron-Selection Strategy [\[pdf\]](http://prl.korea.ac.kr/~sooyoung/papers/ISSTA20.pdf)

DeepCT: Tomographic Combinatorial Testing for Deep Learning Systems [\[pdf\]](http://stap.ait.kyushu-u.ac.jp/~zhao/pub/pdf/saner2019.pdf)

Testing Deep Neural Networks [\[pdf\]](https://arxiv.org/pdf/1803.04792.pdf)

Concolic Testing for Deep Neural Networks [\[pdf\]](https://arxiv.org/pdf/1805.00089.pdf)

DeepCruiser: Automated Guided Testing for Stateful Deep Learning Systems [\[pdf\]](https://arxiv.org/pdf/1812.05339.pdf)

DeepStellar: Model-Based Quantitative Analysis of Stateful Deep Learning Systems [\[pdf\]](http://stap.ait.kyushu-u.ac.jp/~zhao/pub/pdf/esec-fse2019.pdf)

testRNN: Coverage-guided Testing on Recurrent Neural Networks [\[pdf\]](https://arxiv.org/pdf/1906.08557.pdf)

###  覆盖率方法的局限性

Structural Coverage Criteria for Neural Networks Could Be Misleading [\[pdf\]](https://cs.nju.edu.cn/changxu/1_publications/19/ICSE19_01.pdf)

Is Neuron Coverage a Meaningful Measure for Testing Deep Neural Networks? [\[pdf\]](https://dl.acm.org/doi/pdf/10.1145/3368089.3409754)

There is Limited Correlation between Coverage and Robustness for Deep Neural Networks [\[pdf\]](https://arxiv.org/pdf/1911.05904.pdf)

###  基于变异的测试方法
An Analysis and Survey of the Development of Mutation Testing [\[pdf\]](http://crest.cs.ucl.ac.uk/fileadmin/crest/sebasepaper/JiaH10.pdf)

DeepMutation: Mutation Testing of Deep Learning Systems [\[pdf\]](https://arxiv.org/pdf/1805.05206.pdf)

DeepMutation++: a Mutation Testing Framework for Deep Learning Systems [\[pdf\]](http://stap.ait.kyushu-u.ac.jp/~zhao/pub/pdf/ase2019db.pdf)

DeepCrime: mutation testing of deep learning systems based on real faults [\[pdf\]](https://dl.acm.org/doi/abs/10.1145/3460319.3464825)

MuNN: Mutation Analysis of Neural Networks [\[pdf\]](https://ieeexplore.ieee.org/abstract/document/8431960)

DEEPMETIS: Augmenting a Deep Learning Test Set to Increase its Mutation Score [\[pdf\]](https://arxiv.org/pdf/2109.07514.pdf)

###  基于修复的测试方法
Apricot: A Weight-Adaptation Approach to Fixing Deep Learning Models [\[pdf\]](https://www.cs.cityu.edu.hk/~wkchan/papers/ase2019-zhang+chan.pdf)

Plum: Exploration and Prioritization of Model Repair Strategies for Fixing Deep Learning Models [\[pdf\]](https://dsa21.techconf.org/download/DSA2021_FULL/pdfs/DSA2021-1sP33wTCujRJmnnXDjv3mG/439100a140/439100a140.pdf)

DeepCorrect: Correcting DNN Models against Image Distortions [\[pdf\]](https://arxiv.org/pdf/1705.02406.pdf)

DeepFault: Fault Localization for Deep Neural Networks [\[pdf\]](https://arxiv.org/pdf/1902.05974.pdf)

RobOT: Robustness-Oriented Testing for Deep Learning Systems [\[pdf\]](https://arxiv.org/pdf/2102.05913.pdf)

Fuzz Testing based Data Augmentation to Improve Robustness of Deep Neural Networks [\[pdf\]](https://www.comp.nus.edu.sg/~gaoxiang/papers/Sensei_ICSE20.pdf)

DialTest: automated testing for recurrent-neural-network-driven dialogue systems [\[pdf\]](https://dl.acm.org/doi/abs/10.1145/3460319.3464829)

DeepRepair: Style-Guided Repairing for DNNs in the Real-world Operational Environment [\[pdf\]](https://arxiv.org/pdf/2011.09884.pdf)

TauMed: test augmentation of deep learning in medical diagnosis [\[pdf\]](https://dl.acm.org/doi/abs/10.1145/3460319.3469080)

## 训练阶段安全性测试

###  离线检测

Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks [\[pdf\]](https://par.nsf.gov/servlets/purl/10120302)

TABOR: A Highly Accurate Approach to Inspecting and Restoring Trojan Backdoors in AI Systems [\[pdf\]](https://arxiv.org/pdf/1908.01763.pdf)

Scalable Backdoor Detection in Neural Networks [\[pdf\]](https://arxiv.org/pdf/2006.05646.pdf)

DeepInspect: A Black-box Trojan Detection and Mitigation Framework for Deep Neural Networks [\[pdf\]](http://www.aceslab.org/sites/default/files/DeepInspect.pdf)

Detecting AI Trojans Using Meta Neural Analysis [\[pdf\]](https://arxiv.org/pdf/1910.03137.pdf)

###  在线检测

ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation [\[pdf\]](https://dl.acm.org/doi/pdf/10.1145/3319535.3363216)

EX-RAY: Distinguishing Injected Backdoor from Natural Features in Neural Networks by Examining Differential Feature Symmetry [\[pdf\]](https://arxiv.org/pdf/2103.08820.pdf)

## 测试样本的选取方法

DeepGini: Prioritizing Massive Tests to Enhance the Robustness of Deep Neural Networks [\[pdf\]](https://arxiv.org/pdf/1903.00661.pdf)

Input Prioritization for Testing Neural Networks [\[pdf\]](https://arxiv.org/pdf/1901.03768.pdf)

A Noise-Sensitivity-Analysis-Based Test Prioritization Technique for Deep Neural Networks [\[pdf\]](https://arxiv.org/pdf/1901.00054.pdf)

Neuron Activation Frequency Based Test Case Prioritization [\[pdf\]](https://ieeexplore.ieee.org/abstract/document/9405318)

Test Selection for Deep Learning Systems [\[pdf\]](https://dl.acm.org/doi/pdf/10.1145/3417330)

Prioritizing Test Inputs for Deep Neural Networks via Mutation Analysis [\[pdf\]](https://ieeexplore.ieee.org/abstract/document/9402064)

Guiding Deep Learning System Testing using Surprise Adequacy [\[pdf\]](https://arxiv.org/pdf/1808.08444.pdf)

Multiple-boundary clustering and prioritization to promote neural network retraining [\[pdf\]](https://dl.acm.org/doi/abs/10.1145/3324884.3416621)

Measuring Discrimination to Boost Comparative Testing for Multiple Deep Learning Models [\[pdf\]](https://arxiv.org/pdf/2103.04333.pdf)

Operation is the hardest teacher: estimating DNN accuracy looking for mispredictions [\[pdf\]](https://arxiv.org/pdf/2102.04287.pdf)

## 公平性测试

### 个体公平性
Automated Directed Fairness Testing [\[pdf\]](https://arxiv.org/pdf/1807.00468.pdf)

Automated Test Generation to Detect Individual Discrimination in AI Models [\[pdf\]](https://arxiv.org/pdf/1809.03260.pdf)

White-box fairness testing through adversarial sampling [\[pdf\]](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=5635&context=sis_research)

### 群体公平性

Fairness Testing: Testing Software for Discrimination [\[pdf\]](https://dl.acm.org/doi/pdf/10.1145/3106237.3106277)

## 隐私性测试

DP-Finder: Finding Differential Privacy Violations by Sampling and Optimization [\[pdf\]](https://files.sri.inf.ethz.ch/website/papers/ccs18-dpfinder.pdf)

Testing Differential Privacy with Dual Interpreters [\[pdf\]](https://arxiv.org/pdf/2010.04126.pdf)

## 可靠性测试的应用

### 自动驾驶

DeepBillboard: Systematic Physical-World Testing of Autonomous Driving Systems [\[pdf\]](https://arxiv.org/pdf/1812.10812.pdf)

Model-based Exploration of the Frontier of Behaviours for Deep Learning System Testing [\[pdf\]](https://arxiv.org/pdf/2007.02787.pdf)

Automated Test Cases Prioritization for Self-driving Cars in Virtual Environments [\[pdf\]](https://arxiv.org/pdf/2107.09614.pdf)

### 语音识别

CrossASR: Efficient differential testing of automatic speech recognition via text-to-speech [\[pdf\]](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=6539&context=sis_research)

CrossASR++ : A Modular Differential Testing Framework for Automatic Speech Recognition [\[pdf\]](https://arxiv.org/pdf/2105.14881.pdf)

### 自然语言处理

Metamorphic testing for machine translations: MT4MT [\[pdf\]](https://ro.uow.edu.au/cgi/viewcontent.cgi?article=3331&context=eispapers1)

Automatic Testing and Improvement of Machine Translation [\[pdf\]](https://dl.acm.org/doi/pdf/10.1145/3377811.3380420)

## 在线模型库
### Caffe Model Zoo
Caffe是一个考虑了表达、运行速度和模块化的深度学习框架。在Caffe Model Zoo中，集成了由许多研究人员和工程师使用各种架构和数据为不同的任务制作的Caffe模型，这些预训练模型可以应用于多种任务和研究中，从简单回归到大规模视觉分类，再到语音和机器人应用。[\[Web\]](https://github.com/BVLC/caffe/wiki/Model-Zoo)

### ONNX Model Zoo

开放神经网络交换（Open Neural Network Exchange, ONNX）是一种用于表示机器学习模型的开放标准格式。ONNX定义了一组通用运算符、机器学习和深度学习模型的构建块，以及一种通用文件格式，使AI开发人员能够使用具有各种框架、工具、运行时和编译器的模型。ONNX Model Zoo是由社区成员贡献的ONNX格式的预训练的、最先进的集成模型库。模型任务涵盖了图像分类、目标检测、机器翻译等十种多领域任务。[\[Web\]](https://github.com/onnx/models)

### BigML model market

BigML是一个可消耗，可编程且可扩展的机器学习平台，可轻松解决分类、回归、时间序列预报、聚类分析、异常检测、关联发现和主题建模任务，并将它们自动化。BigML促进了跨行业的无限预测应用，包括航空航天、汽车、能源、娱乐、金融服务、食品、医疗保健、物联网、制药、运输、电信等等。 [\[Web\]](https://bigml.com/)

### Amazon SageMaker

Amazon SageMaker是由亚马逊提供的机器学习服务平台，通过整合专门为机器学习构建的广泛功能集，帮助数据科学家和开发人员快速准备、构建、训练和部署高质量的机器学习模型。SageMaker消除了机器学习过程中每个步骤的繁重工作，让开发高质量模型变得更加轻松。SageMaker在单个工具集中提供了用于机器学习的所有组件，因此模型将可以通过更少的工作量和更低的成本更快地投入生产。 [\[Web\]](https://aws.amazon.com/cn/sagemaker/)

## 常用工具包

### Themis

Galhotra等人提出了Themis，一个开源的、用于检测因果偏见的公平性测试工具。它可以通过生成有效的测试套件来测量歧视是否存在。在给定描述有效系统输入的模式时，Themis会自动生成判别测试。应用场景包括金融贷款、医疗诊断和治疗、促销行为、刑事司法系统等。 [\[Web\]](https://github.com/LASER-UMASS/Themis)

### mltest

测试工具mltest，是一个用于为基于Tensorflow的机器学习系统编写单元测试的测试框架。它可以通过极少的设置，实现包括变量变化、变量恒定、对数范围检查、输入依赖、NaN和Inf张量检查等多种不同的常见机器学习问题进行综合测试。遗憾的是，Tensorflow2.0的发布，破坏了该测试工具的大部分功能。 [\[Web\]](https://github.com/Thenerdstation/mltest)

### torchtest

torchtest受mltest启发，与mltest功能类似，torchtest用于为基于pytorch的机器学习系统编写单元测试的测试框架。 [\[Web\]](https://github.com/suriyadeepan/torchtest)

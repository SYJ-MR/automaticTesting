

# 2022-自动化测试-警告识别

[TOC]

## 1.简介

本项目主要是基于置信学习技术，对apache项目中的警告数据集进行去噪。

本项目的工作主要分为两个阶段：第一个阶段，利用github上现有的项目

findbugs-violation (https://github.com/lxyeah/findbugs-violations) ，收集github上的apache项目的警告数据集。第二个阶段，利用cleanlab这一python包，对数据集进行去噪，并对模型进行训练。

下面我们将对于以下两个部分分别展开介绍我们的工作过程和心路历程(2,3)，最后会附上本项目的说明书(4)。

## 2.数据集收集与标记

### 2.1收集数据集

//todo

### 2.2数据集标记

如上一部分所提到的，对于该apache项目biojava我们一共收集到了1967份有效警告和10万余份无效警告。其中，我们对于所有的有效警告进行了人工标记，对于所有的无效警告我们按50：1的比例进行了分层抽样然后标记。也就是总的标记数量在4000份左右。

我们首先了解了对于每一种警告类型所代表的含义，根据findbugs在github上的文档 [colin2wang/findbugs-description](https://github.com/colin2wang/findbugs-description) 。这样便于我们识别该警告是否属于findbugs所标记的警告类型。

了解了警告类型的含义之后我们正式启动了我们的标记之路。对于每一条警告而言，我们都需要大约一分钟的时间去更换到对应的版本，找到警告所在位置，并根据上下文和自身的java基础知识来判断该警告是否为有效警告。但是受限于我们对于java的深度，我们不能保证每一个被标记的警告都是正确的，但是能保证我们标记的每一条警告都是我们经过自身思考之后给出的自己认为最稳妥的答案。

在标记中我们发现，对于一些特定的警告类型标记的正确性比较高，我们推测是因为这一类的警告比较容易检测。例如MS_SHOULD_BE_FINAL，只需要扫描一个变量被声明后有无改动即可。这对于我们后续的置信学习部分提供了一些启发，也许可以利用警告类型这一维度的数据特征对模型进行训练。

另外，在标记的过程中阅读源码时我们发现，对于很多警告而言，源码中在跟踪不相邻commit时会出现问题，导致警告生命周期链断裂，造成误报 。具体来说就是，对于同一个位置源码的警告，可能在某一个版本被修复了，但是会在另外一个版本在次被抛出，这就给我们的标记工作增添不必要的工作量。

## 3.基于置信学习去噪

### 3.1置信学习初探

首先，我们初步了解了置信学习领域，发现这是一个较为前沿的领域，可以借鉴的内容并不多。于是我们自己研读了有关置信学习的论文：Confident Learning: Estimating Uncertainty in Dataset Labels。对于置信学习有了一个初步的了解：大致就是先计算出数据标签的概率分布，然后跟人工标记的标签进行比对（此处则是findbugs自动标注的标签），最后找出最有可能是噪声的样本。对于大体的框架其实不难理解，但是落实到具体的算法之上就有些许黔驴技穷的感觉了。

### 3.2数据集预处理

对于第一阶段收集到的数据集而言，大致形如下表

有效警告：

| 警告类型     | 报警告版本的commit id | 警告在项目中的位置     | 警告消失版本的commit id | 警告消失版本时在项目中的位置 | 有效 |
| ------------ | --------------------- | ---------------------- | ----------------------- | ---------------------------- | ---- |
| 未使用的数据 | ....                  | ..../main/..../xx.java | ......                  | ..../main/..../xx.java       | 是   |

无效警告：

| 警告类型     | 报警告版本的commit id | 在项目中的位置         | 有效 |
| ------------ | --------------------- | ---------------------- | ---- |
| 未使用的数据 | ....                  | ..../main/..../xx.java | 否   |



得到数据集之后要考虑的就是选择一个合适的训练模型来输入数据集进行训练。

#### 3.2.1训练模型的选择：

我们也有考虑过基于表格数据和基于文本的形式来训练模型，但是受限于“警告在项目中的位置”之类的数据范围太大（即可能出现各种各样的路径），而且字符串长度过长，不利于当作表格数据进行训练。而对于基于文本训练的模型来说，数据范围略大并不会对于模型训练产生影响。比如对于表格数据来说，表项中的数据无非就是数字（int，float，double等），或者是一些类型（category）（eg：cash，credit card，wechat-pay，ali-pay等），数据范围略小。而文本数据来说，数据项类似于一段话这样的长字符串，而且没有任何限制，这样的数据范围较为适合训练警告数据集。

相比较而言，二者的共同点是：它们都将数据进行拆分然后转化成向量放入模型进行训练。不同点是：感觉上表格数据的取值是“离散”的，而文本数据的取值是“连续”的。

#### 3.2.2数据特征的增加和优化

对于第一阶段产生的数据而言，要想直接训练出一个拟合效果不错的模型无疑是困难的，所以说在这里我们小组也跟其他小组就数据集处理这个一问题进行了一些讨论。

对于这样的数据集来说，主要有两个问题：第一：特征维度来说过于低。第二：有效的训练特征更是不足。下面将分别展开对于这两个问题我们提供的改进方法。

> /*
>
> 在阐述第一第二点之前，先阐述一下在模型训练过程中犯下的一个严重而又低级的错误
>
> 第零点：统一数据格式
>
> 因为正告数据集，跟误告数据集格式不同，所以如果直接把二者打乱进行训练的话模型很可能就仅仅根据格式上的不同而产生输出，这也是为什么在没有统一格式之前的拟合结果接近于100%的原因。这也是对于机器学习领域的不熟悉而犯下的可笑的错误，看来以后还要填补一下机器学习领域的知识。😥
>
> */

**第一：增加特征的维度**

在警告数据集中的每一个警告都是来自于findbugs工具自带的警告类型。幸运的是，findbugs的文档中对于每个警告类型都归结为了一个大类型（也可以称之为警告级别（rank）），分别是  Bad practice ， Correctness ， Experimental ， Internationalization ， Malicious code vulnerability ， Malicious code vulnerability ， Performance ， Security ， Dodgy code 。其中每个类型也有对应的严重程度，例如 Bad practice就是一些编码的坏习惯，对于程序的正确正常运行没有太大影响，Correctness 就是会影响程序的正确运行的类型。所以说我们根据每个警告类型对应的警告级别，作为一个全新的维度也加入到了数据集当中。这将会更有利于模型学习到警告类型的特征。



**第二：增加特征的有效性**

对于commit id来说，本质上就是随机产生的一串字符，没有任何可以提取出来的特征。不难预料对于这样的数据类型放入模型训练之后学习的效果只会少之又少。通过小组间的讨论之后我们一致认为，commit id需要转换为一个可能会有产出的数据——commit 的日期，如果有了一个commit先后顺序的信息加入其中，可能会对模型的训练有所帮助。例如，2015年之前的警告大多都是无效警告。所以我们利用python自带的git包来把数据中所有的commit id更换成了commit的日期。



综合以上两点，将初始的数据集处理之后的数据集形如下表

| 警告类型rank | 警告类型     | 提交时间       | 警告位置             | 有效  |
| ------------ | ------------ | -------------- | -------------------- | ----- |
| correctness  | 未使用的数据 | 20151012140506 | ../../src/../xx.java | 是/否 |

可以看到，修改之后的数据集的特征维度有所增加，而且特征的有效性得到了提高。虽然本次实验过程中没有用原始的数据集进行对照实验，但是不难预想处理后的数据集对于模型的训练效果会有明显的提升。

### 3.3置信学习去噪

本次实验使用了基于python的置信学习工具cleanlab来对噪声数据进行置信学习。置信学习的大致流程分为了三个部分：

1. **count**： 估计噪声标签和真实标签的联合分布 
2. **clean**：  找出并过滤掉错误样本  
3. **retrain**：过滤错误样本后，重新调整样本类别权重，重新训练 

下面简要介绍三个步骤具体的实现。

#### 3.3.1count阶段

首先，count阶段，我们需要估计噪声标签和真实标签的联合分布。常用的估计联合分布的方法就是使用k-折交叉验证，我们在本次实验中也是使用的该方法进行估计。

虽然说cleanlab已经将其封装成一个简单的函数接口，但是我们作为使用者，为了提高分布估计的准确性，也了解了其中的原理。所谓k-折交叉验证，即将样本分为k份，其中1份当作测试集，另外k-1份当作训练集，用进行训练后的模型对于测试集的样本进行预测，如此即可得到那1份样本是有效警告和无效警告的概率。不难看出，在噪声数量不变、分布均匀的情况下，k取值越大，预测出的概率将越接近真实值。但是为了防止概率过于接近真实概率，而出现模型过拟合的情况，k值也不能过大。

在本实验中，我们选择k=10作为交叉验证的参数，使用的是sklearn包下的交叉验证的方法，训练模型选择的是keras包下的kerasClassifier()，由于对于深度学习领域了解有限，所以其中神经网络的构建借鉴于cleanlab官方文档。其中k-折交叉验证的过程被cross_val_predict方法封装。代码如下所示：

```python
import tensorflow as tf
from sklearn.model_selection import cross_val_predict
from scikeras.wrappers import KerasClassifier
def get_net():#构建神经网络
    net = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None,), dtype="int64"),
            layers.Embedding(max_features + 1, 16),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(2),
            layers.Softmax()
        ]
    )  
    net.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.CategoricalAccuracy(),
    )
    return net
model = KerasClassifier(get_net(), epochs=10)#构建训练模型
k_folds = 10  # 设定k值
pred_probs = cross_val_predict(
    model,
    full_texts,
    full_labels,
    cv=k_folds,
    method="predict_proba",
)#k-折交叉验证
```

其中我们尝试过从k=3一直到k=12来进行交叉验证，其中loss值和噪声数量随k值变化如下图所示：

![屏幕截图 2022-12-02 100538](https://user-images.githubusercontent.com/81347141/205199512-47abb2b3-4486-4492-9ca3-ead2079c8fb6.png)

![](https://user-images.githubusercontent.com/81347141/205203413-a5352320-cc51-4127-b799-7c809e925013.png)

从以上两张图可以看出，在k的不断增加的情况下，loss的值无疑是一个持续下降的趋势。耐人寻味的是，按理说拟合效果在k增大的情况下一直增加，那么噪声的数量就会呈一个逐步减少的趋势。但是根据统计的数据来看噪声的数量只是在某些离散区间呈下降趋势，我们推测为是因为数据集分布不均，导致在k取到每个特定值的时候拟合效果会异常高，导致噪声的数量急剧下降。

#### 3.3.2clean阶段

其次，clean阶段，我们需要找出并过滤掉错误样本。通常来说有几种常用的方法来预测噪音：

1. 直接选取预测概率中概率较大的一类与人工标签进行比对，不一致的样本就是噪音
2. 构造计数矩阵，然后把非对角单元的样本作为噪音。
3. Prune by class，即对于人工标记的有效和无效， 筛选每个类属于给定类标签的概率最小的样本 ，按照最低概率进行排序。
4. Prune by Noise Rate，对于计数矩阵的非对角单元，，选取一定数量样本进行过滤，并按照概率相差最大（最大间隔）进行排序。

其中如果使用第一种方法的话那么噪音的数量就会过于多而导致噪音甄别效果不佳。

这里cleanlab也为我们封装好了一个接口，其中我们可以通过filter_by字段进行clean方法的选择，上述除了第1种方法的第2，3，4种方法都可以选择。默认使用PBC方法。直接调用就可以得到结果。我们选择的也是默认的PBC方法。代码如下图所示

```python
from cleanlab.filter import find_label_issues

ranked_label_issues = find_label_issues(
    labels=full_labels, 
    pred_probs=pred_probs, 								           return_indices_ranked_by="self_confidence"
)#得到一个最有可能是噪声的样本的下标的list，按照可能性从高到低进行排列。
```

根据结果，我们找到了140个噪音，其中有效警告中的噪音有121个，无效警告中的噪音有19个。当然，找到的噪音数量也与我们之前设定的参数有关。比如说，在k-折交叉验证阶段设定k值的时候，如果k值设定得越大，那么标签概率的分布的预测越接近真实情况，那么对于去噪来说，噪声就会因为概率的接近而变少，但是，如果k值设定得越小，那么标签概率分布的预测就会变得模糊，偏离真实情况，那么噪声就会因为概率的偏差而变多。

#### 3.3.3retrain阶段

最后，retrain阶段，过滤错误样本后，重新调整样本类别权重，重新训练。在cleanlab中提供的是将后两个阶段甚至三个阶段封装到一起的接口CleanLearning.fit() 。其中可以提供联合分布概率，也可以只提供训练数据和训练标签。如果不提供分布概率的话fit()能够自动计算其概率。我们的选择是让其自动计算其分布概率。代码如下所示

```python
from cleanlab.classification import CleanLearning
model = KerasClassifier(get_net(), epochs=10)  # 初始化模型
cl = CleanLearning(clf=model, seed=SEED)  # 将模型放入cleanlab封装好的模型中
_ = cl.fit(train_texts, train_labels) #在cleanlab下进行训练
```

然后我们需要将用cleanlab训练出来的模型与不使用cleanlab训练出的模型进行对照实验，所以我们还使用了常规的训练方法训练了一个模型。代码如下所示：

```python
model = KerasClassifier(get_net(), epochs=10)# 初始化模型
model.fit(train_texts, train_labels) #在常规模型下进行训练
```

经过多次训练测试实验之后我们发现，对于在cleanlab上的训练的模型在相当数量的情况下拟合度会略优于常规模型，但是在一部分情况下cleanlab模型的拟合度甚至会低于常规模型。我们推测为，虽然说训练集经过了去噪环节，但是测试集还没有经过去噪，在这样的情况下用测试集去测试模型的拟合度完全有可能下降。具体地，我们对两种模型做了10次拟合训练，最终得出模型在测试集上的准确率数据如下图所示：

![](https://user-images.githubusercontent.com/81347141/205212022-6766e8ff-16e8-4a7d-b5fb-0949dba9c288.png)

对于模型偶尔有准确度大幅下降情况（比如第五次训练、第九次训练），我们认为是数据特征不够突出，加上每一次训练都重新打乱了训练集和测试集，所以导致了在某些个别的训练中产生测试集准确率下降。

### 3.4模型标记与人工标记比对

即使之前我们已经使用了cleanlab对数据集进行了去噪和再训练也对比了cleanlab和常规训练模型的拟合度。但是这些始终都是在机器自己识别的警告中进行互相的训练、计算、比对、去噪等等。要想验证模型的有效性，还需要和人工标记真实警告标记进行比对。

正如上文提到的，我们在去噪过程中发现了140个噪音，其中121个为有效警告的噪音，19个为无效警告的噪音。对于约4000条警告数据而言，这个数据量无疑是较少的，相应的模型的召回率可能就比较低。我们人工标记出的噪音有668条，但是鉴于我们不能保证668条中每一条都是正确的，我们这个比对也只能仅供参考。但是鉴于本实验主要关注的是精度，也就是在我们找到的140个噪音中，有多少噪音是真正的噪音（TP）。经过我们的统计，140个噪音之中，106个是真噪音（TP），即精度大约为75.7%。然后具体的混淆矩阵如下图所示：

<img src="https://user-images.githubusercontent.com/81347141/205199553-a2b8e80b-126c-482f-b6c7-5d0837303fe9.png" alt="混淆矩阵" style="zoom:200%;" />

> 注：其中TP表示预测为噪声而实际上也是噪声，FP表示预测为噪声但实际上不是噪声，FN表示预测为不是噪声但实际上是噪声，TN表示预测为不是噪声而实际上也不是噪声

除了对基础的混淆矩阵进行了统计，我们同样计算了混淆矩阵所包含的高阶数据：精度(precision)、准确率(accuracy)、召回率(recall)。
$$
precision = \frac{TP}{TP+FP} \\
accuracy = \frac{TP+TN}{TP+TN+FP+FN}\\
recall = \frac{TP}{TP+FN}
$$

![](https://user-images.githubusercontent.com/81347141/205199471-8f913a71-943c-4afb-8f27-4178ac527122.png)

从以上两张图标当中不难发现，我们的对于我们的结果来说，精度和准确度较高，但是召回率较低。对于去噪类型的工具来说，我们认为可以接受这样的结果。因为我们必须首先确保精度，才能保证去噪的有效性，不能在去噪的过程中引入新的噪声，这样才能让整个数据集朝向更加clean的方向移动。比方说自动化找到了1500个噪声，那么就算召回率接近100%，把668个真实噪声全都找到了，但是我们又向数据集中自动化添加了800余个噪声。

另外，我们之前还使用过其他的参数组合。经过多次实验我们发现，当模型的拟合效果越好的时候，噪声就越小，但是其精度就会大大提升。例如，我们曾经只发现了21个噪声，但是其中18个都是真噪声（TP)，精度达到了85.7%。可见，我们需要在精度和召回率之间做一个权衡，我们既要保证我们找到的噪声精度达到一定水平，也要保证我们的噪声数量有一定保证。

但是我们认为对于训练参数的调整而交换来的收益只是治标不治本的，需要从根本上提高模型的拟合度和噪声的精度。至少先从训练数据集上入手，有很多有效的信息在收集数据集项目中都没有收集到，我们完全可以收集对应位置的源码相关的信息。例如对于DEAD_LOCAL_STORE类型的警告，我们就应该继续扫描源码后面的部分，将其后面是否有使用过这一变量体现在警告数据集中，这样模型可以直观地学习到该警告是否是一个有效警告。再比如CATCH_EXCEPTION类型的警告，我们就应该扫描前面所使用的变量类型需要catch哪些类型的exception，这样直观的信息将会比警告在源码中的位置这种模糊信息更容易训练。

## 4.项目说明

### 4.1运行环境

数据集收集部分：

//todo

置信学习部分:

Ubuntu 22.04

conda 22.9.0

python 3.9.13

tensorflow 2.7.0

cleanlab 2.1.0

### 4.2项目构成

本项目大概分成以下几大板块

1. 收集警告数据集的项目findbugs-violation
2. 收集警告数据集的目标项目biojava
3. 初始警告数据集txt文件(有效警告.txt、无效警告.txt、全部警告统计.txt、有效警告统计.txt)
4. 标记后的警告数据集txt文件(警告标记整合.txt)
5. 进行数据集处理的python文件(process-dataset.py)
6. 预处理后的数据集txt文件（有效警告(modified).txt、无效警告(modified).txt）
7. 进行置信学习的python文件（confident-learning.py）
8. 进行结果比对以及可视化结果的python文件（data-comparision-and-drawing.py）

下面我分别介绍各文件的内容和作用

#### 4.2.1 findbugs-violation

//todo

#### 4.2.2 biojava

收集警告数据集的目标项目，对于这个项目的使用只存在于对警告进行人工标记的阶段。先将项目clone到本地，然后根据每一条警告的commit id ，在项目根目录下命令行键入

```
git reset --hard [commit id]
```

将对应commit id版本的项目代码拉取到本地，然后根据警告数据集中的路径找到对应报警告的位置，进行人工识别警告是否为有效警告。

#### 4.2.3初始警告数据集txt文件

其中包括四个文件：有效警告.txt、无效警告.txt、全部警告统计.txt、有效警告统计.txt，下面分别介绍其中内容。

**有效警告.txt**:记录了所有biojava中通过findbugs标记出的有效警告，形如下表（表项数据为示例）



| 警告类型     | 报警告版本的commit id | 警告在项目中的位置     | 警告消失版本的commit id | 警告消失版本时在项目中的位置 | 有效 |
| ------------ | --------------------- | ---------------------- | ----------------------- | ---------------------------- | ---- |
| 未使用的数据 | ....                  | ..../main/..../xx.java | ......                  | ..../main/..../xx.java       | 是   |

**无效警告.txt**:记录了所有biojava中通过findbugs标记处的无效警告，形如下表（表项数据为示例）



| 警告类型     | 报警告版本的commit id | 在项目中的位置         | 有效 |
| ------------ | --------------------- | ---------------------- | ---- |
| 未使用的数据 | ....                  | ..../main/..../xx.java | 否   |

**全部警告统计.txt：**记录了所有警告中警告类型的出现次数，以及警告类型对应的大类，形如下表（表项数据为示例）

| Violation Type        | Occurrence | Category     |
| --------------------- | ---------- | ------------ |
| SE_NO_SERIALVERSIONID | 5195       | Bad practice |

**有效警告统计.txt**:记录了所有有效警告中警告类型的出现次数，以及警告类型对应的大类，形如下表（表项数据为示例）

| Violation Type                  | Occurrence | Category    |
| ------------------------------- | ---------- | ----------- |
| SIC_INNER_SHOULD_BE_STATIC_ANON | 10         | Performance |

#### 4.2.4标记后的警告数据集txt文件

包含一个文件：**警告标记整合.txt**：其中记录了2000条正告和2000条误告的人工标记，其中标记中的“1”表示close，“2”表示open，“3”表示unknown，形如下表（表项数据为示例）

有效警告标记样本：

| 警告类型     | 报警告版本的commit id | 警告在项目中的位置     | 警告消失版本的commit id | 警告消失版本时在项目中的位置 | 有效 | 标记 |
| ------------ | --------------------- | ---------------------- | ----------------------- | ---------------------------- | ---- | ---- |
| 未使用的数据 | ....                  | ..../main/..../xx.java | ......                  | ..../main/..../xx.java       | 是   | 1    |

无效警告标记样本

| 警告类型     | 报警告版本的commit id | 在项目中的位置         | 有效 | 标记 |
| ------------ | --------------------- | ---------------------- | ---- | ---- |
| 未使用的数据 | ....                  | ..../main/..../xx.java | 否   | 2    |

#### 4.2.5进行数据集处理的python文件

包含一个文件：**process-dataset.py**：

ps：对于代码中的路径需要修改为对应的路径，当前展示的是代码在Ubuntu虚拟机运行时的路径

其中对于有效警告.txt和无效警告.txt做了如下处理：

1. 首先根据python自带的git包，获取每条数据对应commit id的时间先后顺序，然后将commit id用commit的日期时间替换之。

   ```python
   from git.repo import Repo #python 自带的git包，便于我们撰写脚本
   import datetime
   
   git_repo_dir='/home/shenyujie/automaticTesting/biojava'
   
   def get_commit_time(commit_id):#根据commit id获取commit时间的方法
       repo = Repo(git_repo_dir)
       commit = repo.commit(commit_id)
       return commit.committed_date
   
   #下面分别将有效警告.txt和无效警告.txt中的commit id 全部替换为 commit时间并分别存放在有效警告(modified).txt和无效警告(modified).txt中
   f = open("/home/shenyujie/dataset/有效警告.txt")
   s = f.read() 
   s = s.replace(":"," ")
   s = s.replace("=>"," ")
   f.close()
   res = ""
   array = s.split("\n")
   for i in range(0,len(array)):
       subArray = array.split(" ")
       for j in range(0,len(subArray)):
           if j != 3:
               res = res + subArray + " "
           else:
               strDate = datetime.datetime.strftime(get_commit_time(subArray[j]),"%Y%m%d%H%M%S")
               res = res + strDate + " "
       res = res + "\n"    
   f = open("/home/shenyujie/dataset/有效警告(modified).txt","w")
   f.write(res)
   f.close()
   
   f = open("/home/shenyujie/dataset/无效警告.txt")
   s = f.read() 
   s = s.replace(":"," ")
   s = s.replace("，"," ")
   f.close()
   res = ""
   array = s.split("\n")
   for i in range(0,len(array)):
       subArray = array.split(" ")
       for j in range(0,len(subArray)):
           if j != 3:
               res = res + subArray + " "
           else:
               strDate = datetime.datetime.strftime(get_commit_time(subArray[j]),"%Y%m%d%H%M%S")
               res = res + strDate + " "
       res = res + "\n"    
   f = open("/home/shenyujie/dataset/无效警告(modified).txt","w")
   f.write(res)
   f.close()
   
   ```

   

2. 然后根据警告类型属于某一个警告大类，将其警告大类的信息添加到对应的样本中

   ps：对于代码中的路径需要修改为对应的路径，当前展示的是代码在Ubuntu虚拟机运行时的路径

   ```python
   #建立一个警告类型到大类的映射表
   f = open("/home/shenyujie/dataset/全部警告统计.txt")
   s = f.read()
   s = s.replace(";"," ")
   f.close()
   typeMap = split("\n") 
   for i in range(0,len(typeMap)):
       typeMap[i] = typeMap[i].split(" ")
   
   #给有效警告(modified).txt的数据集添加上警告类型大类
   f = open("/home/shenyujie/dataset/有效警告(modified).txt")
   s = f.read()
   f.close()
   res = ""
   array = s.split("\n")
   for i in range(0,len(array)):
       subArray = array[i].split(" ")
       bigType = ""
       for j in range(0,len(typeMap)):
           if subArray[0] == typeMap[j][1]:
               res = res + typeMap[j][3] + " "+ array[i] + "\n"
               break
   f = open("/home/shenyujie/dataset/有效警告(modified).txt","w")
   f.write(res)
   f.close()
   
   #给无效警告(modified).txt的数据集添加上警告类型大类
   f = open("/home/shenyujie/dataset/无效警告(modified).txt")
   s = f.read()
   f.close()
   res = ""
   array = s.split("\n")
   for i in range(0,len(array)):
       subArray = array[i].split(" ")
       bigType = ""
       for j in range(0,len(typeMap)):
           if subArray[0] == typeMap[j][1]:
               res = res + typeMap[j][3] + " "+ array[i] + "\n"
               break
   f = open("/home/shenyujie/dataset/无效警告(modified).txt","w")
   f.write(res)
   f.close()
   ```

   

#### 4.2.6 预处理后的数据集txt文件

包含两个文件：有效警告(modified).txt、无效警告(modified).txt

**有效警告(modified).txt**：

经过process-dataset.py文件处理之后，其中的commit id一列更换为了commit time，新增了一列表示警告类型大类。形如下表（数据项为示例）：



| 警告类型     | 报警告版本的commit时间 | 警告在项目中的位置     | 警告消失版本的commit id | 警告消失版本时在项目中的位置 | 警告大类      |
| ------------ | ---------------------- | ---------------------- | ----------------------- | ---------------------------- | ------------- |
| 未使用的数据 | 20151022180536         | ..../main/..../xx.java | 201510232304            | ..../main/..../xx.java       | Bad  Practice |



**无效警告(modified).txt：**

经过process-dataset.py文件处理之后，其中的commit id一列更换为了commit time，新增了一列表示警告类型大类。形如下表（数据项为示例）：

| 警告类型     | 报警告版本的commit id | 在项目中的位置         | 警告大类      |
| ------------ | --------------------- | ---------------------- | ------------- |
| 未使用的数据 | ....                  | ..../main/..../xx.java | Bad  Practice |

#### 4.2.7 进行置信学习的python文件

包含一个文件：confident-learning.py

本python文件为置信学习的主要文件。主要完成了以下工作。

1. 数据集准备

2. 使用cleanlab进行去噪。

3. 将cleanlab模型训练效果与常规训练模型进行比对。

具体细节见代码注释。

ps：对于代码中的路径需要修改为对应的路径，当前展示的是代码在Ubuntu虚拟机运行时的路径

   

```python
import re
import string
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_predict
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from scikeras.wrappers import KerasClassifier

SEED = 123456  

#加载数据集
f = open("/home/shenyujie/dataset/有效警告修改(modified).txt")
s = f.read()
f.close()
raw_full_texts = s.split("\n")
for i in range(0,len(raw_full_texts)-1):
    temp = raw_full_texts[i].split(" ")
    raw_full_texts[i] = ""    
    for j in range(0,7):
        raw_full_texts[i] = raw_full_texts[i] + temp[j] +" "


f = open("/home/shenyujie/dataset/无效警告修改(modified).txt")
s = f.read()
f.close()
temp = s.split("\n")
for i in range(0,len(temp)):
    raw_full_texts.append(temp[i])

full_labels = []
for i in range(0,1967):
    full_labels.append(1)
for i in range(0,2186):
    full_labels.append(0)
    

# 打乱数据集
import numpy as np
 
np.random.seed(123456)
np.random.shuffle(raw_full_texts)
np.random.seed(123456)
np.random.shuffle(full_labels)
np.random.seed(123456)

# 设定将文本参数化的参数，并将输入文本参数化
max_features = 10000
sequence_length = 100

vectorize_layer = layers.TextVectorization(
    
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

vectorize_layer.adapt(raw_full_texts)
full_texts = vectorize_layer(raw_full_texts)
full_texts = full_texts.numpy()

# 构建神经网络
def get_net():
    net = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None,), dtype="int64"),
            layers.Embedding(max_features + 1, 16),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes),
            layers.Softmax()
        ]
    )  

    net.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.CategoricalAccuracy(),
    )
    return net

# 初始化模型
num_classes = len(set(full_labels))
model = KerasClassifier(get_net(), epochs=10)

# 设定k值并进行k折交叉验证
k_folds = 10 
pred_probs = cross_val_predict(
    model,
    full_texts,
    full_labels,
    cv=k_folds,
    method="predict_proba",
)


loss = log_loss(full_labels, pred_probs)  #对于k-折交叉验证基于loss值进行评估
print(f"Cross-validated estimate of log loss: {loss:.3f}")

# 使用cleanlab寻找噪声
from cleanlab.filter import find_label_issues

ranked_label_issues = find_label_issues(
    labels=full_labels, pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
)
print(ranked_label_issues)

#--------------------------------------------------------------
# 下面将cleanlab与常规模型的拟合程度进行比对
#--------------------------------------------------------------

# 加载数据集
f = open("/home/shenyujie/dataset/有效警告修改(modified).txt")
s = f.read()
f.close()
raw_full_texts = s.split("\n")

f = open("/home/shenyujie/dataset/无效警告修改(modified).txt")
s = f.read()
f.close()
temp = s.split("\n")
print(len(temp))
for i in range(0,len(temp)):
    raw_full_texts.append(temp[i])
print(len(raw_full_texts))

full_labels = []
for i in range(0,1967):
    full_labels.append(1)
for i in range(0,2186):
    full_labels.append(0)
    
# 打乱数据集
import numpy as np
 
np.random.seed(123456)
np.random.shuffle(raw_full_texts)
np.random.seed(123456)
np.random.shuffle(full_labels)
np.random.seed(123456)

# 划分训练集和测试集
raw_train_texts = raw_full_texts[0:3200]
train_labels = full_labels[0:3200]
raw_test_texts = raw_full_texts[3200:len(raw_full_texts)]
test_labels = full_labels[3200:len(full_labels)]

# 向量化数据
vectorize_layer.reset_state()
vectorize_layer.adapt(raw_train_texts)

train_texts = vectorize_layer(raw_train_texts)
test_texts = vectorize_layer(raw_test_texts)

train_texts = train_texts.numpy()
test_texts = test_texts.numpy()

# 常规模型进行训练并测试拟合度
model = KerasClassifier(get_net(), epochs=10)
model.fit(train_texts, train_labels)

preds = model.predict(test_texts)
acc_og = accuracy_score(test_labels, preds)
print(f"\n Test accuracy of original neural net: {acc_og}")


# 使用cleanlab进行训练并测试拟合度
from cleanlab.classification import CleanLearning

model = KerasClassifier(get_net(), epochs=10)  # Note we first re-instantiate the model
cl = CleanLearning(clf=model, seed=SEED)  # cl has same methods/attributes as model

_ = cl.fit(train_texts, train_labels)

pred_labels = cl.predict(test_texts)
acc_cl = accuracy_score(test_labels, pred_labels)
print(f"Test accuracy of cleanlab's neural net: {acc_cl}")
```



#### 4.2.8 进行结果比对以及可视化结果的python文件

包含一个文件：data-comparision-and-drawing.py

其中将cleanlab找出的噪声与人工标注的噪声进行比对，形成混淆矩阵，并可视化混淆矩阵以及精度、准确度召回率。

ps：对于代码中的路径需要修改为对应的路径，当前展示的是代码在Ubuntu虚拟机运行时的路径

```python
# 加载数据
f = open("/home/shenyujie/dataset/result-0.txt")
s1 = f.read()
f.close()
f=open("/home/shenyujie/dataset/标记整合.txt")
s2 = f.read()
s2 = s2.replace(" ","")
s2 = s2.replace(":"," ")
s2 = s2.replace("=>"," ")
f.close()

from git.repo import Repo #python 自带的git包，便于我们撰写脚本
import datetime

git_repo_dir='/home/shenyujie/dataset/biojava'

def get_commit_time(commit_id): #根据commit id获取commit时间的方法
    repo = Repo(git_repo_dir)
    commit = repo.commit(commit_id)
    
    return commit.committed_datetime


array1 = s1.split("\n")
array2 = s2.split("\n")
for i in range(1,len(array2)):
    subArray = array2[i].split(" ")
    if len(subArray)>=3:
        strDate = datetime.datetime.strftime(get_commit_time(subArray[2]),"%Y%m%d%H%M%S")
        array2[i] = array2[i].replace(subArray[2],strDate)

#计算TP
TP = 0 
for i in range(0,len(array1)):
    subArray1 = array1[i].split(" ")
    for j in range(1,len(array2)):
        subArray2 = array2[j].split(" ")
        if len(subArray2)>=3 and len(subArray1)>=3:
            if subArray1[3]==subArray2[2]:
                if(subArray1[-1]==subArray2[-1][-1]):
                    TP = TP + 1
                    break
        
        
print(TP)

import matplotlib.pyplot as plt


# 计算TP + FN
T = 0
for i in range(0,len(array2)):
    subArray = array2[i].split(" ")
    
    if len(subArray)>=3 and subArray[-1][-1] == '2':
        T = T+1
print(T)

# ----------------------------------------------------------------------------------------
# 下面根据比对的数据进行可视化，主要体现出混淆矩阵、精度、准确度、召回率
#------------------------------------------------------------------------------------------
#已知总的数据量为4074
total = 4074
TP=TP
FP=140-TP
FN=T -TP
TN= total - TP - FP - FN
X = ["TP","FN","FP","TN"]
Y = [TP,FN,FP,TN]
plt.bar(X,Y,color='b')
plt.show()

precision = TP/(TP+FP)
accuracy = (TP+TN)/(TP+TN+FP+FN)
recall = (TP)/(TP+FN)


X = ["precision","accuracy","recall"]
Y = [precision,accuracy,recall]
plt.bar(X,Y,color='b')
plt.show()

# ----------------------------------------------------------------------------------------
# 下面绘制的是不同k取值下交叉验证的结果：主要体现为loss，和噪声数量
#------------------------------------------------------------------------------------------

x_axis_data = [3, 4, 5,6,7,8,9,10,11,12]
y_axis_data = [201, 57, 278, 269, 48,186,111,140,215,160]
plt.plot(x_axis_data, y_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='number of noise')
plt.legend(loc="upper right")
plt.xlabel('K')
plt.ylabel('number of noise')
for i in range(0,len(y_axis_data)):
    plt.text(x_axis_data[i],y_axis_data[i]+4,str(y_axis_data[i]))
plt.show()



x_axis_data = [3, 4, 5,6,7,8,9,10,11,12]
y_axis_data = [0.672, 0.665, 0.662, 0.659, 0.655,0.654,0.653,0.651,0.650,0.647]
plt.plot(x_axis_data, y_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='loss')
plt.legend(loc="upper right")
plt.xlabel('K')
plt.ylabel('loss')
for i in range(0,len(y_axis_data)):
    if i==9:
        plt.text(x_axis_data[i]-0.3,y_axis_data[i]+0.0008,str(y_axis_data[i]))
    elif i!=0:
        plt.text(x_axis_data[i],y_axis_data[i]+0.001,str(y_axis_data[i]))
    else :
        
        plt.text(x_axis_data[i]+0.3,y_axis_data[i]+0.0004,str(y_axis_data[i]))
plt.show()

# ----------------------------------------------------------------------------------------
# 下面绘制的是10次训练模型中常规模型和cleanlab模型测试集准确率的结果
#------------------------------------------------------------------------------------------


x_axis_data = [1, 2, 3,4,5,6,7,8,9,10]
y_axis_data = [0.9133, 0.9271, 0.9259, 0.8927, 0.5858,0.7379,0.9133,0.9202,0.8596,0.8318]
plt.plot(x_axis_data, y_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='normal model')
x_axis_data = [1, 2, 3,4,5,6,7,8,9,10]
y_axis_data = [0.9248, 0.9076, 0.9259, 0.9110, 0.9271,0.9156,0.9271,0.9271,0.7345,0.8561]
plt.plot(x_axis_data, y_axis_data, 'ro-', color='#ff0000', alpha=0.8, linewidth=1, label='cleanlab model')
plt.legend(loc="upper right")
plt.xlabel('')
plt.ylabel('accuracy')
plt.show()
```










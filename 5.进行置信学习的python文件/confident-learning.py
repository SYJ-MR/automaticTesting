import re
import string
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_predict
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from scikeras.wrappers import KerasClassifier

SEED = 123456  # for reproducibility

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
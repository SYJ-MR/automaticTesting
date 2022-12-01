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

f = open("/home/shenyujie/dataset/有效警告修改(modified).txt")
s = f.read()
f.close()
raw_full_texts = s.split("\n")
for i in range(0,len(raw_full_texts)-1):
    temp = raw_full_texts[i].split(" ")
    raw_full_texts[i] = ""
#     print(len(temp))
#     print(i)
    
    for j in range(0,7):
        raw_full_texts[i] = raw_full_texts[i] + temp[j] +" "
# print(raw_full_texts)

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
    
import numpy as np
 
np.random.seed(123456)
np.random.shuffle(raw_full_texts)
np.random.seed(123456)
np.random.shuffle(full_labels)
np.random.seed(123456)


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
print((full_texts[1]))


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
    )  # outputs probability that text belongs to class 1

    net.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.CategoricalAccuracy(),
    )
    return net


num_classes = len(set(full_labels))
model = KerasClassifier(get_net(), epochs=10)


num_crossval_folds = 10  # for efficiency; values like 5 or 10 will generally work better
pred_probs = cross_val_predict(
    model,
    full_texts,
    full_labels,
    cv=num_crossval_folds,
    method="predict_proba",
)

loss = log_loss(full_labels, pred_probs)  # score to evaluate probabilistic predictions, lower values are better
print(f"Cross-validated estimate of log loss: {loss:.3f}")

from cleanlab.filter import find_label_issues

ranked_label_issues = find_label_issues(
    labels=full_labels, pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
)
print(len(ranked_label_issues))


print(
    f"cleanlab found {len(ranked_label_issues)} potential label errors.\n"
    f"Here are indices of the top 10 most likely errors: \n {ranked_label_issues[:10]}"
)


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
    
import numpy as np
 
np.random.seed(123456)
np.random.shuffle(raw_full_texts)
np.random.seed(123456)
np.random.shuffle(full_labels)
np.random.seed(123456)

raw_train_texts = raw_full_texts[0:3200]
train_labels = full_labels[0:3200]
raw_test_texts = raw_full_texts[3200:len(raw_full_texts)]
test_labels = full_labels[3200:len(full_labels)]

vectorize_layer.reset_state()
vectorize_layer.adapt(raw_train_texts)

train_texts = vectorize_layer(raw_train_texts)
test_texts = vectorize_layer(raw_test_texts)

train_texts = train_texts.numpy()
test_texts = test_texts.numpy()

model = KerasClassifier(get_net(), epochs=10)
model.fit(train_texts, train_labels)

preds = model.predict(test_texts)
acc_og = accuracy_score(test_labels, preds)
print(f"\n Test accuracy of original neural net: {acc_og}")

from cleanlab.classification import CleanLearning

model = KerasClassifier(get_net(), epochs=10)  # Note we first re-instantiate the model
cl = CleanLearning(clf=model, seed=SEED)  # cl has same methods/attributes as model

_ = cl.fit(train_texts, train_labels)

pred_labels = cl.predict(test_texts)
acc_cl = accuracy_score(test_labels, pred_labels)
print(f"Test accuracy of cleanlab's neural net: {acc_cl}")
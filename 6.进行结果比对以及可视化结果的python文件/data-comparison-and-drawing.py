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
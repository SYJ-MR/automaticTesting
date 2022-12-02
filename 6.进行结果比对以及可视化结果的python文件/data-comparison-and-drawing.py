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
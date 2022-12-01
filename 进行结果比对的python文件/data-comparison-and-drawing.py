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

# strDate = datetime.datetime.strftime(get_commit_time(subArray[j]),"%Y%m%d%H%M%S")

array1 = s1.split("\n")
array2 = s2.split("\n")
for i in range(1,len(array2)):
    subArray = array2[i].split(" ")
    if len(subArray)>=3:
        strDate = datetime.datetime.strftime(get_commit_time(subArray[2]),"%Y%m%d%H%M%S")
        array2[i] = array2[i].replace(subArray[2],strDate)
        print(array2[i])
    
#     temp = ""
#     for j in range(0,len(subArray)):
#         if j!=2:
#             temp = temp + subArray[j] + " "
#         else:
#             temp = temp + strDate
            
            
# TP = 0
# for i in range(0,len(array1)):
#     subArray1 = array1[i].split(" ")
#     for j in range(1,len(array2)):
#         subArray2 = array2[j].split(" ")
        
# #         print(subArray2)
#         if len(subArray2)>=3:
# #             print(subArray2[2])
# #             print(get_commit_time(subArray2[2]))
#             strDate = datetime.datetime.strftime(get_commit_time(subArray2[2]),"%Y%m%d%H%M%S")
#             if subArray1[3]==strDate:
#                 if(subArray1[-1]==subArray2[-1]):
#                     TP = TP + 1
#                     print(array1[i])
#                     break
        
        
# print(TP)

TP = 0
for i in range(0,len(array1)):
    subArray1 = array1[i].split(" ")
    for j in range(1,len(array2)):
        subArray2 = array2[j].split(" ")
        
#         print(subArray2)
        if len(subArray2)>=3 and len(subArray1)>=3:
#             print(subArray2[2])
#             print(get_commit_time(subArray2[2]))
            
            if subArray1[3]==subArray2[2]:
#                 print(subArray1[-1])
#                 print(subArray2[-1][-1])
                if(subArray1[-1]==subArray2[-1][-1]):
                    
                    TP = TP + 1
                    print(array1[i])
                    break
        
        
print(TP)

import matplotlib.pyplot as plt

T = 0
for i in range(0,len(array2)):
    subArray = array2[i].split(" ")
    
    if len(subArray)>=3 and subArray[-1][-1] == '2':
        T = T+1
print(T)


TP=106
FP=34
FN=562
TN=3372
X = ["TP","FN","FP","TN"]
Y = [106,562,34,3372]
plt.bar(X,Y,color='b')
plt.show()

precision = TP/(TP+FP)
accuracy = (TP+TN)/(TP+TN+FP+FN)
recall = (TP)/(TP+FN)


X = ["precision","accuracy","recall"]
Y = [precision,accuracy,recall]
plt.bar(X,Y,color='b')
plt.show()
from git.repo import Repo #python 自带的git包，便于我们撰写脚本
import datetime

git_repo_dir='/home/shenyujie/automaticTesting/biojava'

def get_commit_time(commit_id):#根据commit id获取commit时间的方法
    repo = Repo(git_repo_dir)
    commit = repo.commit(commit_id)
    return commit.committed_datetime

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
array = array[0:len(array):50]
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
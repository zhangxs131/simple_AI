
with open('../data/thucnews/train.txt','r',encoding='utf-8') as f:
    content=f.read().splitlines()

data=[]
labels=[]

nums=20
for i in content:
    i=i.split()
    if i[-1]=='4':
        print(i[0])
        nums=nums-1
        if nums==0:
            break
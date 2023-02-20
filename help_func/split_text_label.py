txt_file='../data/thucnews/dev.txt'
save_file='../data/thucnews/test.txt'
nums=100

with open(txt_file,'r',encoding='utf-8') as f:
    content=f.read().splitlines()

text=[]
for i in content:
    text.append(i)

text=text[:nums]

with open(save_file,'w',encoding='utf-8') as f:
    f.writelines([i+'\n' for i in text])

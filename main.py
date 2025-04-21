import csv,os
import random as rd
import torch
from prework import Random_embedding,Glove_embedding
from comparison import plot


file_path=os.path.join('..','task1','sentiment-analysis-on-movie-reviews', 'train.tsv', 'train.tsv')
with open(file_path,'r',encoding='utf-8') as f:
    train_tsv=csv.reader(f,delimiter='\t')
    tmp=list(train_tsv)
file_path=os.path.join('glove.6B','glove.6B.50d.txt')

with open(file_path,'rb') as f:
    lines=f.readlines()

#用glove创建词典
trained_dict=dict()
n=len(lines)
for i in range(n):
    line=lines[i].split()
    trained_dict[line[0].decode('utf-8').upper()]=[float(line[j]) for j in range(1,51)]

#初始化
epoch=20
alpha=0.001

data=tmp[1:]
batch_size=500

#随机初始化

rd.seed(2025)
random_embedding=Random_embedding(data=data)
random_embedding.get_id()


rd.seed(2025)
glove_embedding=Glove_embedding(data=data,trained_dict=trained_dict)
glove_embedding.get_id()

plot(random_embedding,glove_embedding,alpha,batch_size,epoch)



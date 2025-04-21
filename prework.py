import random as rd
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

def data_split(data):
    train=[]
    test=[]
    rate=0.3
    for it in data:
        if rd.random()>rate:
            train.append(it)
        else:
            test.append(it)
    return train,test

#随机初始化
class Random_embedding():
    def __init__(self,data,test_rate=0.3):
        self.data=data
        self.dict_words=dict()
        data.sort(key=lambda x:len(x[2].split())) #按照句子的长短排序，短的在前，避免后面一个batch内句子的长短不一，导致padding过度:就是使得同一个batch里面的句子长度相似，要padding的数量少
        self.len=0   #单词数量(包含padding的ID:0)
        self.train,self.test=data_split(data)
        self.train_y=[int(it[3]) for it in self.train]
        self.test_y=[int(it[3]) for it in self.test]
        self.train_matrix=list()
        self.test_matrix=list()
        self.longest=0    #记录最长的句子

    def get_id(self):
        for it in self.data:
            s=it[2]
            s=s.upper()
            s_split=s.split()
            for word in s_split:
                if word not in self.dict_words:
                    self.dict_words[word]=len(self.dict_words)+1        #padding 为0
        self.len=len(self.dict_words)        #暂时不包括padding

        for it in self.train:
            s=it[2]
            s=s.upper()
            s_split=s.split()
            item=[self.dict_words[word] for word in s_split]   #找到ID列表，未进行padding
            self.longest=max(self.longest,len(item))
            self.train_matrix.append(item)
        for it in self.test:
            s=it[2]
            s=s.upper()
            s_split=s.split()
            item=[self.dict_words[word] for word in s_split]
            self.longest=max(self.longest,len(item))
            self.test_matrix.append(item)
        self.len+=1           #单词数目，包含padding的id0

class Glove_embedding():
    def __init__(self,data,trained_dict,test_rate=0.3):
        self.dict_words=dict()
        self.trained_dict=trained_dict
        data.sort(key=lambda x:len(x[2].split()))
        self.len=0
        self.data=data
        self.train,self.test=data_split(data)
        self.train_y=[int(it[3]) for it in self.train]
        self.test_y=[int(it[3]) for it in self.test]
        self.train_matrix=list()
        self.test_matrix=list()
        self.longest=0
        self.embedding=list() #抽取出用到的，即预训练模型的单词

    def get_id(self):
        self.embedding.append([0] * 50)
        for it in self.data:
            s = it[2]
            s = s.upper()
            s_split = s.split()
            for word in s_split:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1
                    if word in self.trained_dict:
                        self.embedding.append(self.trained_dict[word])
                    else:
                        self.embedding.append([0] * 50)
        self.len = len(self.dict_words)

        for it in self.train:
            s=it[2]
            s=s.upper()
            s_split=s.split()
            item=[self.dict_words[word] for word in s_split]
            self.longest=max(self.longest,len(item))
            self.train_matrix.append(item)

        for it in self.test:
            s=it[2]
            s=s.upper()
            s_split=s.split()
            item=[self.dict_words[word] for word in s_split]
            self.longest=max(self.longest,len(item))
            self.test_matrix.append(item)
        self.len+=1

#自定义数据集的结构
class ClsDataset(Dataset):
    def __init__(self,sentence,emotion):
        self.sentence=sentence
        self.emotion=emotion

    def __getitem__(self,item):
        return self.sentence[item],self.emotion[item]

    def __len__(self):
        return len(self.sentence)

#自定义数据集的内数据返回类型，并且进行padding
def collate_fn(batch_data):
    sentence,emotion=zip(*batch_data)
    sentences=[torch.LongTensor(x) for x in sentence] #将句子转化为LongTensor类型
    padded_sents=pad_sequence(sentences,batch_first=True,padding_value=0) #自动padding操作
    return torch.LongTensor(padded_sents),torch.LongTensor(emotion)

#利用dataloader划分batch
def get_batch(x,y,batch_size):
    dataset=ClsDataset(x,y)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn)
    return dataloader

# shuffle是指每个epoch都随机打乱数据再分batch，设置成False，否则之前的顺序会直接打乱
# drop_last是指不利用最后一个不完整的batch（数据大小不能被batch_size整除）


















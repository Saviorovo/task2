import torch
import torch.nn as nn
import torch.nn.functional as F

# 选择设备：优先 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words,
                 typenum=5, weight=None, layer=1,
                 nonlinearity='tanh', batch_first=True,
                 drop_out=0.35):
        super(RNN, self).__init__()
        self.len_feature = len_feature
        self.len_hidden = len_hidden
        self.layer = layer
        self.dropout = nn.Dropout(drop_out)
        # Embedding 初始化
        if weight is None:
            emb_weight = torch.empty(len_words, len_feature, device=device)
            nn.init.xavier_normal_(emb_weight)
            self.embedding = nn.Embedding(len_words, len_feature)
            self.embedding.weight = nn.Parameter(emb_weight)
        else:
            emb_weight = torch.tensor(weight, dtype=torch.float, device=device)
            self.embedding = nn.Embedding(len_words, len_feature)
            self.embedding.weight = nn.Parameter(emb_weight)

        # RNN 隐层
        self.rnn = nn.RNN(
            input_size=len_feature,
            hidden_size=len_hidden,
            num_layers=layer,
            nonlinearity=nonlinearity,
            batch_first=batch_first,
            dropout=drop_out
        )
        # 全连接层
        self.fc = nn.Linear(len_hidden, typenum)

    def forward(self, x):
        # x: list or ndarray of shape [B, L]
        x = torch.tensor(x, dtype=torch.long, device=device)
        emb = self.embedding(x)                     # [B, L, D]
        out = self.dropout(emb)
        # 初始化隐藏状态，直接在 device 上
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer, batch_size,
                         self.len_hidden, device=device)
        # 前向 RNN
        _, hh = self.rnn(out, h0)                   # hh: [num_layers, B, H]
        last_h = hh[-1]                             # [B, H]
        return self.fc(last_h)                      # [B, typenum]

class CNN(nn.Module):
    def __init__(self, len_feature, len_words, longest,
                 typenum=5, weight=None, drop_out=0.3):
        super(CNN, self).__init__()
        self.len_feature = len_feature
        # Embedding 初始化
        if weight is None:
            emb_weight = torch.empty(len_words, len_feature, device=device)
            nn.init.xavier_normal_(emb_weight)
            self.embedding = nn.Embedding(len_words, len_feature)
            self.embedding.weight = nn.Parameter(emb_weight)
        else:
            emb_weight = torch.tensor(weight, dtype=torch.float, device=device)
            self.embedding = nn.Embedding(len_words, len_feature)
            self.embedding.weight = nn.Parameter(emb_weight)
        # 卷积层
        self.conv1 = nn.Conv2d(1, longest, (2, len_feature), padding=(1,0))
        self.conv2 = nn.Conv2d(1, longest, (3, len_feature), padding=(1,0))
        self.conv3 = nn.Conv2d(1, longest, (4, len_feature), padding=(2,0))
        self.conv4 = nn.Conv2d(1, longest, (5, len_feature), padding=(2,0))
        self.act = nn.ReLU()
        # 全连接层
        self.fc = nn.Linear(4 * longest, typenum)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        # x: list or ndarray [B, L]
        x = torch.tensor(x, dtype=torch.long, device=device)
        emb = self.embedding(x)                     # [B, L, D]
        out = emb.view(x.size(0), 1, x.size(1), self.len_feature)
        out = self.dropout(out)
        # 四路卷积 + ReLU
        c1 = self.act(self.conv1(out)).squeeze(3)   # [B, C_out, L+1]
        c2 = self.act(self.conv2(out)).squeeze(3)
        c3 = self.act(self.conv3(out)).squeeze(3)
        c4 = self.act(self.conv4(out)).squeeze(3)
        # 全局最大池化
        p1 = F.max_pool1d(c1, c1.size(2)).squeeze(2)  # [B, C_out]
        p2 = F.max_pool1d(c2, c2.size(2)).squeeze(2)
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)
        p4 = F.max_pool1d(c4, c4.size(2)).squeeze(2)
        # 拼接
        feat = torch.cat([p1, p2, p3, p4], dim=1)   # [B, 4*C_out]
        return self.fc(feat)                         # [B, typenum]


import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score


# ------------------超参数设置------------------------------------------------------------------------------
MAX_SEQ_LEN = 30
BATCH_SIZE = 256
EPOCH = 1
N_FOLD = 10
LEARNING_RATE = 0.0001
Dim_Embedding = 512
N_Head = 8
D_K = D_V = 64
N_layers = 6
Dim_FF = 1024

# -----------------固定随机种子-----------------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else 'cpu'

# -----------------读取数据--------------------------------------------------------------------------------
train = pd.read_csv("oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv",
                    sep='\t', names=['q1', 'q2', 'label'])
test = pd.read_csv("oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv",
                   sep='\t', names=['q1', 'q2', 'label'])
test["label"] = 0

# -----------------构建字典--------------------------------------------------------------------------------
vocab_pd = pd.read_csv('Bert-vocab.txt', names=['word'])
vocab_dict = {}
for key, value in vocab_pd.word.to_dict().items():
    vocab_dict[value] = key


# -------------------定义数据集类--------------------------------------------------------------------------
class OPPODataSet(Dataset):
    def __init__(self, data_, dict_, seq_length=MAX_SEQ_LEN):
        self.data = data_
        self.vocab = dict_
        self.seq_len = seq_length

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        text_a, text_b, label_ = self.data.iloc[item].values
        text_a = self.get_sentence(text_a)
        text_b = self.get_sentence(text_b)
        q1_ = torch.tensor(text_a, dtype=torch.long)
        q2_ = torch.tensor(text_b, dtype=torch.long)
        label_ = torch.tensor(label_, dtype=torch.long)

        return q1_, q2_, label_

    def get_sentence(self, sentence, max_seq_len=30):
        tokens = sentence.split()
        for i in range(len(tokens)):
            tokens[i] = self.vocab.get(tokens[i], self.vocab["<unk>"])
        if max_seq_len is None:
            return tokens
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        else:
            tokens += [self.vocab["<pad>"]] * (max_seq_len - len(tokens))
        return tokens


# -------------------------位置编码-------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_emb, dropout=0.1, max_len=MAX_SEQ_LEN):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_emb, 2).float() * (-math.log(10000.0) / d_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(vocab_dict["<pad>"]).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q_, K_, V_, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q_, K_.transpose(-1, -2)) / np.sqrt(D_K)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V_)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(Dim_Embedding, D_K * N_Head, bias=False)
        self.W_K = nn.Linear(Dim_Embedding, D_K * N_Head, bias=False)
        self.W_V = nn.Linear(Dim_Embedding, D_K * N_Head, bias=False)
        self.fc = nn.Linear(N_Head * D_V, Dim_Embedding, bias=False)

    def forward(self, emb, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = emb, emb.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(emb).view(batch_size, -1, N_Head, D_K).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(emb).view(batch_size, -1, N_Head, D_K).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(emb).view(batch_size, -1, N_Head, D_V).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, N_Head, 1, 1)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, N_Head * D_V)
        # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(Dim_Embedding)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(Dim_Embedding, Dim_FF, bias=False),
            nn.ReLU(),
            nn.Linear(Dim_FF, Dim_Embedding, bias=False)
        )

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(Dim_Embedding)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs,  enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(len(vocab_dict), 512, padding_idx=vocab_dict["<pad>"])
        self.pos_emb = PositionalEncoding(Dim_Embedding)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(N_layers)])

    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs


class MatchNet(nn.Module):
    # 基于RNN的文本匹配网络
    def __init__(self, hidden_size, dropout_prob):
        super(MatchNet, self).__init__()
        # RNN的编码器
        self.encoder = Encoder()
        # 2层全连接神经网络， 内部使用ReLu激活函数， 采用dropout避免过拟合
        self.mlp = nn.Sequential(nn.Dropout(dropout_prob),
                                 nn.Linear(hidden_size * 2, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout_prob),
                                 nn.Linear(hidden_size, 1))
        # 初始化全连接参数
        nn.init.xavier_normal_(self.mlp[1].weight)
        nn.init.xavier_normal_(self.mlp[4].weight)
        nn.init.constant_(self.mlp[1].bias, 0)
        nn.init.constant_(self.mlp[4].bias, 0)

    def forward(self, seq1, seq2):
        # (batch_size, max_seq_len, hidden_size * 2)  使用孪生网络分别对每个句子进行上下文编码
        seq1_states = self.encoder(seq1)
        seq2_states = self.encoder(seq2)

        # (batch_size, hidden_size * 2)  平均池化得到每个句子的表示
        seq1_state = seq1_states.mean(dim=1)
        seq2_state = seq2_states.mean(dim=1)

        # (batch_size, hidden_size * 4)  拼接得到句子对的表示
        pair_state = torch.cat([seq1_state, seq2_state], dim=-1)

        # (batch_size,)  经过全连接层得到句子对的label
        y = self.mlp(pair_state).squeeze()

        return y


# ------------------------评估函数-------------------------------------------------------------------
def evaluate(model_, eval_loader_, criterion_):
    predict_ = None
    all_labels = None
    with torch.no_grad():
        for batch in eval_loader_:
            model_.eval()  # 停用Dropout 和Batch Normalization等
            question1, question2, label_ = tuple(x for x in batch)
            predict_batch = model(question1, question2)
            predict_ = predict_batch.sigmoid() if predict_ is None else torch.cat([predict_, predict_batch.sigmoid()], dim=0)
            all_labels = label_ if all_labels is None else torch.cat([all_labels, label_], dim=0)
        eval_loss_ = criterion_(predict, label.float()).item()
        label_list_ = all_labels.tolist()
        predict_labels = (predict_ > 0.5).long().tolist()
        eval_acc_ = accuracy_score(label_list_, predict_labels)
    return eval_loss_, eval_acc_


# ------------------------测试函数-------------------------------------------------------------------
def commit_result(model_, test_loader_):
    predict_set_ = None
    with torch.no_grad():
        for batch in test_loader_:
            model_.eval()  # 停用Dropout和Batch Normalization等

            question1, question2, label_ = tuple(x for x in batch)
            predict_ = model_(question1, question2)

            predict_set_ = predict_.sigmoid() if predict_set_ is None else torch.cat([predict_set_, predict_.sigmoid()], dim=0)
    result_ = predict_set_.numpy()
    # pd.DataFrame(result[:]).to_csv("result.csv", index=False, header=False)
    return result_


# RNN的隐状态维度设为200，dropout概率设为0.1，固定预训练的词向量
model = MatchNet(hidden_size=512, dropout_prob=0.1)
best_model = model
best_acc = 0
print(model)

# TIPS: 使用二分类交叉熵作为训练的损失函数。
#       为了提高loss计算的精度，模型输出前没有做sigmoid激活，故采用BCEWithLogitsLoss
#       若模型输出前进行了sigmoid激活，可采用BCEWLoss
criterion = nn.BCEWithLogitsLoss()
# 利用Adam作为优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

k_fold = KFold(n_splits=N_FOLD, random_state=427,  shuffle=True)
test_results = 0
agv_acc = 0
result = None

for fold, (train_index, valid_index) in enumerate(k_fold.split(range(train.shape[0]))):
    print('=-'*50)
    print(" "*40, f'This is {fold} fold')
    print("=-"*50)
    test_data_set = OPPODataSet(test, vocab_dict, MAX_SEQ_LEN)
    test_loader = DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=False)
    train_data_set = OPPODataSet(train.iloc[train_index], vocab_dict, MAX_SEQ_LEN)
    eval_data_set = OPPODataSet(train.iloc[valid_index], vocab_dict, MAX_SEQ_LEN)
    train_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_data_set, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCH):
        model.train()
        losses = []
        true_labels = None
        predict_set = None
        p_bar = tqdm(train_loader)
        for data in p_bar:
            q1, q2, label = tuple(x for x in data)
            optimizer.zero_grad()
            predict = model(q1, q2)
            loss = criterion(predict, label.float())
            predict_set = predict.sigmoid() if predict_set is None else torch.cat([predict_set, predict.sigmoid()], dim=0)
            true_labels = label if true_labels is None else torch.cat([true_labels, label], dim=0)
            losses.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()  # 更新参数
            p_bar.set_description(f'epoch:{epoch} loss:{np.mean(losses)}')
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion)
        labels_list = true_labels.tolist()
        predict_list = (predict_set > 0.5).long().tolist()
        train_acc = accuracy_score(labels_list, predict_list)
        print('=-' * 50)
        print(f'epoch:{epoch} valid_acc{eval_acc}  train_acc:{train_acc}')
        print("=-" * 50)
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model = model
    result = (result+commit_result(model, test_loader)) if result is not None else commit_result(model, test_loader)

result = result / N_FOLD
pd.DataFrame(result[:]).to_csv("result1.csv", index=False, header=False)
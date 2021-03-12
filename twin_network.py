import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import accuracy_score


# ------------------超参数设置------------------------------------------------------------------------------
MAX_SEQ_LEN = 50
BATCH_SIZE = 128
EPOCH = 20
N_FOLD = 5
LEARNING_RATE = 0.0001
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


# -----------------------定义编码器------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True,
                          batch_first=True)  # batch_size=True,就将batch放在输入的第一维（批大小，序列长度，输入维数

    def forward(self, input_states, input_len):
        # 按序列长度对序列进行排序
        sorted_input_states, sorted_input_len, _, unsorted_idxes = self.sort_by_lens(input_states, input_len)
        # 对不同实际长度的输入序列进行打包，去除其中用来补齐的<pad>
        packed_input_states = nn.utils.rnn.pack_padded_sequence(sorted_input_states, sorted_input_len, batch_first=True)
        output, _ = self.rnn(packed_input_states)
        # 重新对实际长度不同的序列进行补齐
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # 还原原始序列的顺序
        return output.index_select(0, unsorted_idxes)

    def sort_by_lens(self, seqs, seq_lens, descending=True):
        # 对序列长度排序，sorted_seq_lens记录排完序的张量，sorted_idxes记录排完序的下标
        # [3, 0, 1, 2, 4]
        sorted_seq_lens, sorted_idxes = seq_lens.sort(0, descending=descending)
        # 根据sorted_idxes对序列进行排序
        sorted_seqs = seqs.index_select(0, sorted_idxes)
        # type_as 张量转化为指定张量
        # [0, 1, 2, 3, 4]
        orig_idxes = torch.arange(0, len(seq_lens)).type_as(seq_lens)
        # [1, 2, 3, 0, 4]
        _, revers_mapping = sorted_idxes.sort(0, descending=False)
        unsorted_idxes = orig_idxes.index_select(0, revers_mapping)
        return sorted_seqs, sorted_seq_lens, sorted_idxes, unsorted_idxes


# ------------------------定义模型-------------------------------------------------------------------
class MatchNet(nn.Module):
    # 基于RNN的文本匹配网络
    def __init__(self, hidden_size, dropout_prob, padding_idx=vocab_dict["<pad>"], freeze_embedding=True):
        super(MatchNet, self).__init__()
        self.padding_idx = padding_idx
        # 使用预训练的词嵌入
        self.embedding = nn.Embedding(len(vocab_dict), 300, padding_idx=padding_idx)
        # RNN的编码器
        self.encoder = Encoder(300, hidden_size)
        # 2层全连接神经网络， 内部使用ReLu激活函数， 采用dropout避免过拟合
        self.mlp = nn.Sequential(nn.Dropout(dropout_prob),
                                 nn.Linear(hidden_size * 4, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout_prob),
                                 nn.Linear(hidden_size, 1))
        # 初始化全连接参数
        nn.init.xavier_normal_(self.mlp[1].weight)
        nn.init.xavier_normal_(self.mlp[4].weight)
        nn.init.constant_(self.mlp[1].bias, 0)
        nn.init.constant_(self.mlp[4].bias, 0)

    def forward(self, seq1, seq2):
        # (batch_size, max_seq_len, embedding_size)  获取每个词的初始分布式表示
        seq1_embeddings = self.embedding(seq1)
        seq2_embeddings = self.embedding(seq2)

        # (batch_size,)  获得每个序列的实际长度
        seq1_len = torch.sum((seq1 != self.padding_idx).long(), dim=-1)
        seq2_len = torch.sum((seq2 != self.padding_idx).long(), dim=-1)

        # (batch_size, max_seq_len, hidden_size * 2)  使用孪生网络分别对每个句子进行上下文编码
        seq1_states = self.encoder(seq1_embeddings, seq1_len)
        seq2_states = self.encoder(seq2_embeddings, seq2_len)

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
    result = predict_set_.numpy()
    pd.DataFrame(result[:]).to_csv("result.csv", index=False, header=False)


# RNN的隐状态维度设为200，dropout概率设为0.1，固定预训练的词向量
model = MatchNet(hidden_size=200, dropout_prob=0.1, padding_idx=vocab_dict["<pad>"], freeze_embedding=True)
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

for fold, (train_index, valid_index) in enumerate(k_fold.split(range(train.shape[0]))):
    print('=-'*50)
    print(" "*40, f'This is {fold} fold')
    print("=-"*50)
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
torch.save(best_model.state_dict(), 'match_model.pt')

# ---------------------------预测-----------------------------------------------------------------
test_data_set = OPPODataSet(test, vocab_dict, MAX_SEQ_LEN)
test_loader = DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=False)
commit_result(best_model, test_loader)

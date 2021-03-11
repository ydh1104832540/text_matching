import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import log_loss, accuracy_score


# ------------------固定随机种子，便于复现实验-----------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
# -------------------读取数据-------------------
train = pd.read_csv("oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv", sep='\t', names=['q1', 'q2', 'label'])
test = pd.read_csv("oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv", sep='\t', names=['q1', 'q2', 'label'])

vocab_pd = pd.read_csv('Bert-vocab.txt', names=['word'])
vocab_dict = {}
for key, value in vocab_pd.word.to_dict().items():
    vocab_dict[value] = key
# ----------------构建训练集与测试集--------------
# 最大的序列长度
MAX_SEQ_LEN = 50


def get_sentence(sentence, vocab, max_seq_len):
    tokens = sentence.split()
    for i in range(len(tokens)):
        tokens[i] = vocab.get(tokens[i], vocab["<unk>"])
    if max_seq_len is None:
        return tokens
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    else:
        tokens += [vocab["<pad>"]] * (max_seq_len - len(tokens))
    return tokens


question1 = [get_sentence(s, vocab=vocab_dict, max_seq_len=MAX_SEQ_LEN) for s in train["q1"]]
question2 = [get_sentence(s, vocab=vocab_dict, max_seq_len=MAX_SEQ_LEN) for s in train["q2"]]
labels = [int(x) for x in train["label"]]

# 构建数据集
question1 = torch.tensor(question1, dtype=torch.long)
question2 = torch.tensor(question2, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.float)
dataSet = TensorDataset(question1, question2, labels)
train_set, test_set = random_split(dataSet, lengths=[90000, 10000])
print(len(train_set), len(test_set))


def sort_by_lens(seqs, seq_lens, descending=True):
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


# -----------------------构建模型--------------
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True,
                          batch_first=True)  # batch_size=True,就将batch放在输入的第一维（批大小，序列长度，输入维数

    def forward(self, input_states, input_len):
        # 按序列长度对序列进行排序
        sorted_input_states, sorted_input_len, _, unsorted_idxes = sort_by_lens(input_states, input_len)
        # 对不同实际长度的输入序列进行打包，去除其中用来补齐的<pad>
        packed_input_states = nn.utils.rnn.pack_padded_sequence(sorted_input_states, sorted_input_len, batch_first=True)
        output, _ = self.rnn(packed_input_states)
        # 重新对实际长度不同的序列进行补齐
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # 还原原始序列的顺序
        return output.index_select(0, unsorted_idxes)


class MatchNet(nn.Module):
    # 基于RNN的文本匹配网络
    def __init__(self, hidden_size, dropout_prob, padding_idx=vocab_dict["<pad>"], freeze_embedding=True):
        super(MatchNet, self).__init__()
        self.padding_idx = padding_idx
        # 使用预训练的词嵌入
        self.embedding = nn.Embedding(len(vocab_dict), 300)
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

        # (batch_size,)  经过全连接层得到句子对的logit
        pair_logit = self.mlp(pair_state).squeeze()

        return pair_logit


# RNN的隐状态维度设为200，dropout概率设为0.1，固定预训练的词向量
model = MatchNet(hidden_size=200, dropout_prob=0.1, padding_idx=vocab_dict["<pad>"], freeze_embedding=True)
print(model)


def eval(model, data_loder, citerion):
    all_logits = None
    all_labels = None
    with torch.no_grad():
        for batch in data_loder:
            model.eval()  # 停用Dropout 和Batch Normalization等
            question1, question2, label = tuple(x for x in batch)
            logit = model(question1, question2)
            all_logits = logit if all_logits is None else torch.cat([all_logits, logit], dim=0)
            all_labels = label if all_labels is None else torch.cat([all_labels, label], dim=0)
        eval_loss = citerion(logit, label).item()
    return eval_loss


# TIPS: `BATCH_SIZE`是指在使用梯度下降训练模型的过程中，一次训练所使用的样本数。
#       当`BATCH_SIZE`=1时，每次训练只取一个样本，受样本中随机性的影响，梯度会比较不稳定，模型难以训练；
#       随着`BATCH_SIZE`的增大，梯度也变得更准确，但同时也需要更大的内存或GPU显存。
#       因此，`BATCH_SIZE`的设置既不能过小导致训练不稳定，也要注意不能超过内存（显存）限制。
BATCH_SIZE = 128
# TIPS: 在训练神经网络时，使用学习率(`LEARNING_RATE`)控制参数的更新速度。
#       当学习率较小时，参数更新速度慢，适用于对模型进行微调；
#       当学习率较大时，会使得模型在训练过程中不断震荡，导致训练不稳定
LEARNING_RATE = 0.0001
# TIPS: 一个epoch是指用训练集所有样本训练一轮的周期。
#       如果训练的轮数太少，网络有可能发生欠拟合，而epoch数太多则又可能导致过拟合。
#       因此可以在训练过程中定期在验证集上评估模型性能，并把性能最好的模型保存下来
EPOCHS = 50  # 5
# 使用显卡1
# device = torch.device('cuda:0')
# device = torch.device('cpu')
# model.to(device)

train_sampler = RandomSampler(train_set)
train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=BATCH_SIZE)
dev_sampler = SequentialSampler(test_set)
dev_loader = DataLoader(test_set, sampler=dev_sampler, batch_size=BATCH_SIZE)

# TIPS: 使用二分类交叉熵作为训练的损失函数。
#       为了提高loss计算的精度，模型输出前没有做sigmoid激活，故采用BCEWithLogitsLoss
#       若模型输出前进行了sigmoid激活，可采用BCEWLoss
criterion = nn.BCEWithLogitsLoss()

# 利用Adam作为优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

global_steps = 0
min_eval_loss = float('inf')
steps = []
train_losses = []
dev_losses = []
for epoch in range(EPOCHS):
    for batch in train_loader:
        model.train()  # 训练模式，启用Dropout和Batch Normalization等
        question1, question2, label = tuple(x for x in batch)
        # 将参数的梯度清零
        optimizer.zero_grad()
        # 前向传播
        logit = model(question1, question2)
        loss = criterion(logit, label)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        global_steps += 1
        if global_steps % 10 == 0:
            train_loss = eval(model, train_loader, criterion)
            dev_loss = eval(model, dev_loader, criterion)
            steps.append(global_steps)
            train_losses.append(train_loss)
            dev_losses.append(dev_loss)
            print('Epoch: %d, Global steps: %3d, '
                  'Train loss: %.3f, Dev loss: %.3f' % (epoch + 1, global_steps, train_loss, dev_loss))
            if dev_loss < min_eval_loss:
                # 将评价损失最低的模型保存到磁盘
                min_eval_loss = dev_loss
                torch.save(model.state_dict(), 'match_net.pt')
            break
print('训练结束')

# 从文件加载训练好的模型
model = MatchNet(hidden_size=200, dropout_prob=0.1, padding_idx=vocab_dict["<pad>"], freeze_embedding=True)
model.load_state_dict(torch.load('match_net.pt'))

batches_probs = []
true_labels = []
pred_probs = None
true_labels = None
with torch.no_grad():
    for batch in dev_loader:
        model.eval()  # 停用Dropout和Batch Normalization等

        question1, question2, label = tuple(x for x in batch)
        logit = model(question1, question2)

        pred_probs = logit.sigmoid() if pred_probs is None else torch.cat([pred_probs, logit.sigmoid()], dim=0)
        true_labels = label if true_labels is None else torch.cat([true_labels, label], dim=0)

pred_labels = (pred_probs > 0.5).long().tolist()
pred_probs = pred_probs.tolist()
true_labels = true_labels.tolist()
logloss = log_loss(true_labels, pred_probs)
print('log loss:', logloss)
accuracy = accuracy_score(true_labels, pred_labels)
print('accuracy:', accuracy)












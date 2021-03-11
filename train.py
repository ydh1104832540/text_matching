import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# -------读取数据--------
train = pd.read_csv("oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv",
                    sep='\t', names=['q1', 'q2', 'label'])
test = pd.read_csv("oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv",
                   sep='\t', names=['q1', 'q2', 'label'])
test['label'] = 0


def get_dict(data):
    words_dict = defaultdict(int)
    for i in tqdm(range(data.shape[0])):
        text = data.q1.iloc[i].split() + data.q2.iloc[i].split()
        for c in text:
            words_dict[c] += 1
    return words_dict


# -------------------------构建词典--------------------
test_dict = get_dict(test)
train_dict = get_dict(train)
word_dict = list(test_dict.keys()) + list(train_dict.keys())
word_dict = set(word_dict)
word_dict = set(map(int, word_dict))
word_dict = list(word_dict)
special_tokens = ['<pad>', '<unk>', '<cls>', '<sep>', '<mask>']
WORDS = special_tokens + word_dict
pd.Series(WORDS).to_csv('Bert-vocab.txt', header=False, index=0)

vocab = pd.read_csv('Bert-vocab.txt', names=['word'])
vocab_dict = {}
for key, value in vocab.word.to_dict().items():
    vocab_dict[value] = key


class OPPODataSet(Dataset):
    def __init__(self, data, dict, seq_length=50):
        self.data = data
        self.vocab = dict
        self.seq_len = seq_length

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        text_a, text_b, label = self.data.iloc[item].values
        text_a = self.get_sentence(text_a)
        text_b = self.get_sentence(text_b)
        text_a = [self.vocab['<cls>']] + text_a + [self.vocab['<sep>']]
        text_b = text_b + [self.vocab["<sep>"]]

        token_type_ids = ([0 for _ in range(len(text_a))] + [1 for _ in range(len(text_b))])[: self.seq_len]
        text = (text_a + text_b)[: self.seq_len]

        padding = [self.vocab['<pad>'] for _ in range(self.seq_len - len(text))]
        attention_mask = len(text) * [1]

        text.extend(padding), token_type_ids.extend(padding), attention_mask.extend(padding)
        attention_mask = np.array(attention_mask)
        token_type_ids = np.array(token_type_ids)
        text = torch.tensor(text, dtype=torch.long)
        return {
            'input_ids': text,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        }, torch.tensor(self.data.label.iloc[item])

    def get_sentence(self, sentence):
        tokens = sentence.split()
        for i in range(len(tokens)):
            tokens[i] = self.vocab.get(tokens[i], self.vocab["<unk>"])
        return tokens


class FastText(nn.Module):
    def __init__(self, emd_dim=128, vocab_size=23000, seq_length=128):
        super(FastText, self).__init__()
        self.emd = nn.Embedding(vocab_size, emd_dim)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, bidirectional=True, dropout=0.2)
        self.avg_pool = nn.AvgPool1d(kernel_size=90)
        self.max_pool = nn.MaxPool1d(kernel_size=90)
        self.lr = nn.Linear(in_features=1024, out_features=256)
        self.lr1 = nn.Linear(in_features=256, out_features=2)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(num_features=1024)
        self.batch_norm1 = nn.BatchNorm1d(num_features=256)

    def forward(self, x):
        x = self.emd(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x = x.transpose(2, 1)
        x1 = self.avg_pool(x).squeeze(-1)
        x2 = self.max_pool(x).squeeze(-1)
        x = torch.cat([x1, x2], dim=-1)
        x = self.batch_norm(x)
        x = self.lr(x)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.batch_norm1(x)
        x = self.lr1(x)
        return x


def evaluate(model_, data_loader):
    model_.eval()
    labels_list_ = []
    predict_list_ = []
    for data_labels in tqdm(data_loader):
        data = data_labels[0]
        labels = data_labels[1]
        inputs_ids = data['input_ids']
        predict_ = model_(inputs_ids)
        predict_ = torch.softmax(predict_, dim=-1)
        predict_list_ += predict_.argmax(-1).tolist()
        labels_list_ += labels.detach().tolist()
    acc = accuracy_score(labels_list_, predict_list_)
    return acc


def get_result(model_, test_loader_):
    model_.eval()
    results = []
    for data_labels in tqdm(test_loader_):
        data = data_labels[0]
        inputs_ids = data["input_ids"].long()
        predict_ = model_(inputs_ids)
        predict_ = torch.softmax(predict_, dim=-1)
        results += predict_.argmax(-1).tolist()
    return results


k_fold = KFold(n_splits=5, random_state=427,  shuffle=True)
test_results = 0
agv_acc = 0
model = FastText(128, vocab_size=len(vocab_dict), seq_length=90)
best_model = model
# model = model.to(device)
criterion = nn.CrossEntropyLoss()
# criterion = criterion.to(device)
optim = torch.optim.Adam(model.parameters(), lr=2e-3)
best_acc = 0
for fold, (train_index, valid_index) in enumerate(k_fold.split(range(train.shape[0]))):
    print('=-'*50)
    print(" "*40, f'This is {fold} fold')
    print("=-"*50)
    nums_epoch = 10
    # device = "cuda" if torch.cuda.is_available() else 'cpu'
    train_dataset = OPPODataSet(train.iloc[train_index], vocab_dict, 90)
    valid_dataset = OPPODataSet(train.iloc[valid_index], vocab_dict, 90)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)
    for epoch in range(nums_epoch):
        model.train()
        losses = []
        labels_list = []
        predict_list = []
        pbar = tqdm(train_loader)
        for data_labels in pbar:
            data = data_labels[0]
            labels = data_labels[1]
            inputs_ids = data["input_ids"]
            predict = model(inputs_ids)
            loss = criterion(predict, labels)
            predict_list += predict.argmax(-1).tolist()
            losses.append(loss.detach().numpy())
            labels_list += labels.numpy().tolist()
            loss.backward()
            optim.step()
            optim.zero_grad()
            pbar.set_description(f'epoch:{epoch} loss:{np.mean(losses)}')
        valid_acc = evaluate(model, valid_loader)
        train_acc = accuracy_score(labels_list, predict_list)
        print('=-' * 50)
        print(f'epoch:{epoch} valid_acc{valid_acc}')
        print(f'epoch:{epoch} train_acc:{train_acc}')
        print("=-" * 50)
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model

torch.save(best_model.state_dict(), 'match_model.pt')
test_dataset = OPPODataSet(test, vocab_dict, 90)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
result = get_result(best_model, test_loader)
pd.DataFrame(result[:, 1]).to_csv("prediction_result/result1.csv", index=False, header=False)

# test_results = get_result(best_model, test_loader)/5
# agv_acc += best_acc/5









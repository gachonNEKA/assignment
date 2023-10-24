from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from kobert_tokenizer import KoBERTTokenizer
from tqdm import tqdm, tqdm_notebook
from transformers import BertModel
from transformers import AdamW
from torch import nn

import torch.nn.functional as F
import torch.optim as optim
import gluonnlp as nlp
import pandas as pd
import numpy as np
import datetime
import argparse
import pickle
import torch
#import wandb
import os

# GPU 사용 시
device = torch.device("cuda:0")

parser = argparse.ArgumentParser(description='Porarity Recognition Model')

parser.add_argument('--train_data',
                    type=str,
                    default=True,
                    help='train data')

parser.add_argument('--test_data',
                    type=str,
                    default=True,
                    help='test data')

parser.add_argument('--num_epoch',
                    type=str,
                    default=True,
                    help='the number of epoch')

args = parser.parse_args()

training_file_path = args.train_data
test_file_path = args.test_data
num_epoch = args.num_epoch

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

train = pd.read_csv(training_file_path)
test = pd.read_csv(test_file_path)

train_sentences = ["[CLS] " + sen + " [SEP]" for sen in train.Sentence]
test_sentences = ["[CLS] " + sen + " [SEP]" for sen in test.Sentence]

train_labels = list(train['Emotion'].values)
test_labels = list(train['Emotion'].values)

idx = {}
for l_train in train_labels :
  if l_train not in idx :
    idx[l_train] = len(idx)


train_labels = train['Emotion'].map(idx)
test_labels = test['Emotion'].map(idx)

train_data_list = []
for sentence, label in zip(train_sentences, train_labels)  :
    data = []
    data.append(sentence)
    data.append(str(label))

    train_data_list.append(data)

test_data_list = []
for sentence, label in zip(test_sentences, test_labels)  :
    data = []
    data.append(sentence)
    data.append(str(label))

    test_data_list.append(data)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = num_epoch
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

tok = tokenizer.tokenize

data_train = BERTDataset(train_data_list, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTDataset(test_data_list, 0, 1, tok, vocab, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=3,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

'''
wandb.init(
    # set the wandb project where this run will be logged
    project="Polarity Recognition",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "KoBERT",
    "epochs": num_epoch,
    }
)
'''

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            train_step_accuracy = loss.data.cpu().numpy()
            train_step_loss = train_acc / (batch_id+1)
    print("epoch {} train acc {} loss {}".format(e+1, train_acc / (batch_id+1), train_step_loss))
    train_accuracy = train_acc / (batch_id+1)
    train_loss = train_step_loss
  
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
        loss = loss_fn(out, label)
    print("epoch {} test acc {} loss {}".format(e+1, test_acc / (batch_id+1), loss))
    test_accuracy = test_acc / (batch_id+1)
    test_loss = loss

# wandb.log({"Training Accuracy per Step": train_step_accuracy, "Training Loss per Step": train_step_loss, "Training Accuracy": train_accuracy, "Training Loss": train_loss, "Validation Accuracy": test_accuracy, "Validation Loss": test_loss})

folder_path = os.path.join(os.path.dirname(__file__), '.', 'pt')

current_date = datetime.date.today()
pt_name = "model_" + current_date.strftime("%Y%m%d") + ".pt"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

pt_path = os.path.join(folder_path, pt_name)

torch.save(model.state_dict(), pt_path)

import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from seqeval import metrics

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoConfig

from models.ner.bert_ner import BertForTokenClassification
from models.ner.bert_crf import BertCRFforNER
from utils import set_seed, read_label_list


def get_args():
    parser = argparse.ArgumentParser(description='Chinese BIO NER')
    parser.add_argument('--do_predict', type=bool, default=False, help='predict the pictures')
    parser.add_argument('--class_num', type=int, default=10, help='labels for classification')
    parser.add_argument('--lr', type=float, default=3e-5, help='lr rate for training')
    parser.add_argument('--num_epoch', type=int, default=8, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--warm_up_ratio', type=float, default=0.03, help='warmup rate')
    parser.add_argument('--log_steps', type=int, default=2, help='steps for logging')
    parser.add_argument('--pretrained_path', type=str, default='pretrained_model/bert-chinese-wwm-ext',
                        help='choose a model_path')
    parser.add_argument('--model_save_path', type=str, default='save_model/thucnew_bert.bin',
                        help='save path for model')
    parser.add_argument('--data_path', type=str, default='./data/ner/weibo', help='save path for model')
    parser.add_argument('--label_txt', type=str, default='./data/ner/weibo/label.txt', help='label txt file')
    parser.add_argument('--predict_dir', type=str, default='./data/ner/weibo/test_dir', help='save path for model')

    args = parser.parse_args()
    return args


# =========================================== data process =================================================================

class NERDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data = self._load_file(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _load_file(self, file_name):
        data = []

        if file_name.split('.')[-1] == 'json':
            with open(file_name, encoding='utf-8') as f:
                content = f.readlines()
            for i in content:
                data.append(json.loads(i))
        elif file_name.split('.')[-1] == 'txt':
            with open(file_name, encoding='utf-8') as f:
                content = f.readlines()
            text = []
            label = []
            for i in content:
                if i == '\n':
                    data.append({'text': text, 'label': label})
                    text = []
                    label = []
                else:
                    t = i.split(' ')
                    text.append(t[0])
                    label.append(t[1].strip('\n'))

        return data  # text,label


# collate_fn=lambda x: collate_fn(x, info)
def collate_fn(batch, tokenizer, label_list):
    label2id = {k: v for v, k in enumerate(label_list)}
    inputs = tokenizer([i['text'] for i in batch], padding="max_length", truncation=True, max_length=512,
                       return_tensors='pt', is_split_into_words=True)

    labels = []
    for id, i in enumerate(batch):
        label = i['label']
        word_ids = inputs.word_ids(batch_index=id)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    inputs['labels'] = torch.LongTensor(labels)

    return inputs


def collate_fn_test(batch, tokenizer):
    inputs = tokenizer([i['text'] for i in batch], padding="max_length", truncation=True, max_length=512,
                       return_tensors='pt', is_split_into_words=True)
    return inputs


def gen_dataloader(data_path, tokenizer, batch_size=32, label_txt=None):
    train_dataset = NERDataset(os.path.join(data_path, 'train.txt'))
    dev_dataset = NERDataset(os.path.join(data_path, 'dev.txt'))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda x: collate_fn(x, tokenizer, label_txt))
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda x: collate_fn(x, tokenizer, label_txt))

    return train_dataloader, dev_dataloader


def gen_test_dataloader(file_dir, tokenizer, batch_size=1):
    if file_dir.split(',')[-1] == 'txt':
        text = read_label_list(file_dir)
        test_loader = DataLoader(text, batch_size=batch_size, shuffle=False,
                                 collate_fn=lambda x: collate_fn(x, tokenizer=tokenizer))
    else:
        test_loader = None
        print('predict file should be txt line')

    return test_loader


# ================================================= train & evaluate =================================================================
def train(model, train_dataloader, dev_dataloader, num_epochs=5, lr=1e-3, device='cpu', log_steps=100,
          model_save_path='./save_model/mnist_lr.pth', warm_up_ratio=0.05, label_list=None):
    # optim

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=lr)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)

    total_batch = 1  # 记录进行到多少batch
    dev_best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        model.train()
        for i, input in enumerate(tqdm(train_dataloader)):

            input['input_ids'] = input['input_ids'].to(device)
            input['token_type_ids'] = input['token_type_ids'].to(device)
            input['attention_mask'] = input['attention_mask'].to(device)
            input['labels'] = input['labels'].to(device)

            outputs = model(**input)
            loss = outputs['loss']

            loss.backward()
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            if total_batch % log_steps == 0:
                # 每多少轮输出在训练集和验证集上的效果
                labels = input['labels'].data.cpu().numpy()
                predic = outputs["logits"].argmax(dim=-1).cpu().numpy()

                train_result = compute_metrics(label_list, predic, labels)
                dev_f1, dev_loss = evaluate(model, dev_dataloader, device, label_list=label_list)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), model_save_path)

                msg = 'Epoch: {0:>6} Iter: {1:>6},  Train Loss: {2:>6.2},  Train f1: {3:>6.2%},  Val Loss: {4:>6.2},  Val f1: {5:>6.2%}'
                print(msg.format(epoch + 1, total_batch, loss.item(), train_result["f1"], dev_loss, dev_f1))
                model.train()
            total_batch += 1


def evaluate(model, data_loader, device='cpu', label_list=None, seq_len=512):
    model.eval()

    loss_total = 0

    predict_all = []
    labels_all = []

    with torch.no_grad():
        for input in tqdm(data_loader):
            input['input_ids'] = input['input_ids'].to(device)
            input['token_type_ids'] = input['token_type_ids'].to(device)
            input['attention_mask'] = input['attention_mask'].to(device)
            input['labels'] = input['labels'].to(device)

            outputs = model(**input,decode=True)
            loss = outputs["loss"]
            loss_total += loss

            labels = input['labels'].data.cpu().numpy().tolist()
            predic = outputs["logits"].argmax(dim=-1).cpu().numpy().tolist()
            predict_all = predict_all + predic
            labels_all = labels_all + labels

    predict_all = np.array(predict_all)
    labels_all = np.array(labels_all)

    print("p.shape", predict_all.shape)
    print("l.shape", labels_all.shape)

    # predict_all = np.empty([0, seq_len])
    # labels_all = np.empty([0, seq_len])
    #
    # with torch.no_grad():
    #     for input in tqdm(data_loader):
    #         input['input_ids'] = input['input_ids'].to(device)
    #         input['token_type_ids'] = input['token_type_ids'].to(device)
    #         input['attention_mask'] = input['attention_mask'].to(device)
    #         input['labels'] = input['labels'].to(device)
    #
    #         outputs = model(**input)
    #         loss = outputs["loss"]
    #         loss_total += loss
    #
    #         labels = input['labels'].data.cpu().numpy()
    #         predic = outputs["logits"].argmax(dim=-1).cpu().numpy()
    #
    #         labels_all = np.append(labels_all, labels, axis=0)
    #         predict_all = np.append(predict_all, predic, axis=0)
    # print(labels_all.shape)
    # print(predict_all.shape)

    result = compute_metrics(label_list, predict_all, labels_all)

    return result['f1'], loss_total / len(data_loader)


def predict(model, data_loader, label_list=None, device='cpu'):
    model.eval()

    predict_all = np.array([], dtype=int)

    with torch.no_grad():
        for input in tqdm(data_loader):
            input['input_ids'] = input['input_ids'].to(device)
            input['token_type_ids'] = input['token_type_ids'].to(device)
            input['attention_mask'] = input['attention_mask'].to(device)

            outputs = model(**input,decode=True)

            predic = outputs['logits'].argmax(dim=-1).cpu().numpy()

            predict_all = np.append(predict_all, predic)

        true_predictions = [
            [label_list[p] for p in prediction]
            for prediction in predict_all
        ]

    return true_predictions


def compute_metrics(label_list, predictions, labels):
    # predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    f1 = metrics.f1_score(true_predictions, true_labels)

    return {
        # "precision": results["overall_precision"],
        # "recall": results["overall_recall"],
        "f1": f1,
        # "accuracy": results["overall_accuracy"],
    }


# =============================================== main ===========================================================
def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    label_list = read_label_list(args.label_txt)

    if not args.do_predict:
        train_dataloader, dev_dataloader = gen_dataloader(args.data_path, tokenizer, batch_size=args.batch_size,
                                                          label_txt=label_list)

        # model
        config = AutoConfig.from_pretrained(args.pretrained_path, num_labels=len(label_list))
        model = BertCRFforNER.from_pretrained(args.pretrained_path, config=config)
        model = model.to(device)

        # train and eval
        train(model, train_dataloader, dev_dataloader, num_epochs=args.num_epoch, lr=args.lr, device=device,
              log_steps=args.log_steps, model_save_path=args.model_save_path, label_list=label_list)
    else:
        test_dataloader = gen_test_dataloader(args.predict_dir, tokenizer=tokenizer)

        config = AutoConfig.from_pretrained(args.pretrained_path, num_labels=args.class_num)
        model = BertCRFforNER.from_config(config)
        model.load_state_dict(torch.load(args.model_save_path))
        model.to(device)
        result = predict(model, test_dataloader, label_list, device)

        print([label_list[i] for i in result])


if __name__ == '__main__':
    main()

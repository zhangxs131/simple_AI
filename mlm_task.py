import torch
import numpy as np
import argparse
import os
from tqdm import  tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW,get_linear_schedule_with_warmup
from transformers import AutoTokenizer,AutoConfig
from transformers import BertForMaskedLM

from utils import set_seed,read_label_list

def get_args():

    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--do_predict', type=bool, default=False, help='predict the pictures')
    parser.add_argument('--lr', type=float, default=0.001, help='lr rate for training')
    parser.add_argument('--num_epoch', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--seed', type=int, default=2022,help='random seed')
    parser.add_argument('--warm_up_ratio', type=float, default=0.03,help='warmup rate')
    parser.add_argument('--log_steps', type=int, default=200, help='steps for logging')
    parser.add_argument('--pretrained_path', type=str, default='pretrained_model/bert-chinese-wwm-ext', help='choose a model_path')
    parser.add_argument('--model_save_path', type=str, default='save_model/fill_mask_query_bert.bin', help='save path for model')
    parser.add_argument('--data_path', type=str, default='./data/fill_mask_query', help='save path for model')
    parser.add_argument('--predict_dir', type=str, default='./data/thucnews/test_dir', help='save path for model')

    args = parser.parse_args()
    return args

class FillMaskDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        data_list = read_label_list(data_path)
        self.data=[{'text':i} for i in data_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# collate_fn=lambda x: collate_fn(x, info)
def collate_fn(batch, tokenizer,mlm_probability=0.15):
    inputs = tokenizer([i['text'] for i in batch], padding=True, truncation=True, max_length=128, return_tensors='pt')


    labels = inputs['input_ids'].clone()
    # We sample a few tokens in each sequence for MLM training
    probability_matrix = torch.full(labels.shape, mlm_probability)

    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs['input_ids'][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs['input_ids'][indices_random] = random_words[indices_random]

    inputs['labels'] = labels

    return inputs

def collate_test_fn(batch, tokenizer):
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
    return inputs


#加载数据
def gen_dataloader(data_path,tokenizer,batch_size=32):

    train_dataset = FillMaskDataset(os.path.join(data_path,'train.txt'))
    test_dataset = FillMaskDataset(os.path.join(data_path,'dev.txt'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=lambda x:collate_fn(x,tokenizer=tokenizer))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=lambda x:collate_fn(x,tokenizer=tokenizer))

    return train_loader,test_loader

def gen_test_dataloader(file_dir,tokenizer,batch_size=1):
    if file_dir.split(',')[-1]=='txt':
        with open(file_dir,'r',encoding='utf-8') as f:
            text=f.read().splitlines()
        test_loader = DataLoader(text, batch_size=batch_size, shuffle=False,
                                 collate_fn=lambda x: collate_fn(x, tokenizer=tokenizer))
    else:
        test_loader=None
        print('predict file should be txt line')

    return test_loader

def train(model, train_dataloader, dev_dataloader,num_epochs=5,lr=1e-3, device='cpu',log_steps=100,model_save_path='./save_model/mnist_lr.pth',warm_up_ratio=0.05):
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
                true = input['labels'].data.cpu()
                predic = torch.max(outputs['logits'], 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, dev_dataloader, device)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), model_save_path)

                msg = 'Epoch: {0:>6} Iter: {1:>6},  Train Loss: {2:>5.2},  Train Acc: {3:>6.2%},  Val Loss: {4:>5.2},  Val Acc: {5:>6.2%}'
                print(msg.format(epoch + 1, total_batch, loss.item(), train_acc, dev_loss, dev_acc))
                model.train()
            total_batch += 1


def evaluate(model, data_loader, device='cpu'):
    model.eval()

    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for input in tqdm(data_loader):
            input['input_ids'] = input['input_ids'].to(device)
            input['token_type_ids'] = input['token_type_ids'].to(device)
            input['attention_mask'] = input['attention_mask'].to(device)
            input['labels'] = input['labels'].to(device)

            outputs = model(**input)
            loss = outputs['loss']
            loss_total += loss

            labels = input['labels'].data.cpu().numpy()
            predic = outputs['logits'].argmax(dim=-1).cpu().numpy()

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    labels = labels_all.reshape(-1)
    preds = predict_all.reshape(-1)
    mask = labels != -100  #只计算mask的词准确率
    labels = labels[mask]
    preds = preds[mask]

    acc = metrics.accuracy_score(labels, preds)


    return acc, loss_total / len(data_loader)

def predict(model, data_loader, device='cpu'):
    model.eval()

    predict_all = np.array([], dtype=int)

    with torch.no_grad():
        for input in tqdm(data_loader):
            input['input_ids'] = input['input_ids'].to(device)
            input['token_type_ids'] = input['token_type_ids'].to(device)
            input['attention_mask'] = input['attention_mask'].to(device)

            outputs = model(**input)

            predic = outputs['logits'].argmax(dim=-1).cpu().numpy()

            predict_all = np.append(predict_all, predic)

    return predict_all

def main():
    args=get_args()
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(args.pretrained_path)

    if not args.do_predict:
        train_dataloader,test_dataloader=gen_dataloader(args.data_path,tokenizer,batch_size=args.batch_size)
        config=AutoConfig.from_pretrained(args.pretrained_path)
        model=BertForMaskedLM.from_pretrained(args.pretrained_path,config=config)
        model.to(device)

        train(model,train_dataloader,test_dataloader,num_epochs=args.num_epoch,lr=args.lr,device=device,log_steps=args.log_steps,model_save_path=args.model_save_path)

    else:

        test_dataloader=gen_test_dataloader(args.predict_dir,tokenizer=tokenizer)

        config = AutoConfig.from_pretrained(args.pretrained_path)
        model=BertForMaskedLM.from_config(config)
        model.load_state_dict(torch.load(args.model_save_path))
        model.to(device)
        result=predict(model,test_dataloader,device)
        label_list=read_label_list(args.label_txt)

        print([label_list[i] for i in result])


if __name__=='__main__':
    main()
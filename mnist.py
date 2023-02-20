import torch
import numpy as np
import argparse
import os
from PIL import Image
from tqdm import  tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from sklearn import metrics
from torch.optim import  AdamW


from models.img_classification.lr import  LR
from models.img_classification.lenet5 import Lenet5


def get_args():
    parser=argparse.ArgumentParser(description='mnist')
    parser.add_argument('--do_predict', type=bool, default=False, help='predict the pictures')
    parser.add_argument('--class_num', type=int, default=10, help='labels for classification')
    parser.add_argument('--lr',type=float,default=0.001,help='lr rate for training')
    parser.add_argument('--num_epoch',type=int,default=10,help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--log_steps', type=int, default=200, help='steps for logging')
    parser.add_argument('--model_save_path', type=str, default='save_model/mnist_lr.pth', help='save path for model')
    parser.add_argument('--data_path', type=str, default='./data/mnist', help='save path for model')
    parser.add_argument('--predict_dir', type=str, default='./data/mnist/test_dir', help='save path for model')

    args=parser.parse_args()

    return args


#加载数据
def gen_dataloader(data_path,batch_size=32,size=28):

    transform = transforms.Compose([transforms.Resize([size,size]),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)  # train=True训练集，=False测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,test_loader

def gen_test_dataloader(file_dir,batch_size=1):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    img_name=os.listdir(file_dir)

    imgs=[]
    for i in img_name:
        img=Image.open(os.path.join(file_dir,i))
        img=transform(img)
        imgs.append(img)
    # print(imgs[0].shape)
    # imgs=torch.stack(imgs, dim=0)
    imgs=DataLoader(imgs,batch_size=batch_size,shuffle=False)# torch.Tensor


    return imgs


def train(model, train_dataloader, dev_dataloader,num_epochs=5,lr=1e-3, device='cpu',log_steps=100,model_save_path='./save_model/mnist_lr.pth'):
    # optim

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=lr)

    total_batch = 1  # 记录进行到多少batch
    dev_best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        model.train()
        for i, input in enumerate(tqdm(train_dataloader)):
            data,label=input
            data=data.to(device)
            label=label.to(device)

            outputs = model(data,label=label)
            loss = outputs['loss']

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            if total_batch % log_steps == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = label.data.cpu()
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
            input=[i.to(device) for i in input]
            data,label=input

            outputs = model(data,label=label)
            loss = outputs['loss']
            loss_total += loss

            labels = label.data.cpu().numpy()
            predic = outputs['logits'].argmax(dim=-1).cpu().numpy()

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)

    if False:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_loader), report, confusion

    return acc, loss_total / len(data_loader)

def predict(model, data_loader, device='cpu'):
    model.eval()


    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for input in tqdm(data_loader):
            data=input.to(device)

            outputs = model(data)

            predic = outputs['logits'].argmax(dim=-1).cpu().numpy()

            predict_all = np.append(predict_all, predic)



    return predict_all


def main():

    args=get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not args.do_predict:
        train_dataloader,test_dataloader=gen_dataloader(args.data_path,batch_size=args.batch_size)
        model=LR(class_num=args.class_num)
        model.to(device)

        train(model,train_dataloader,test_dataloader,num_epochs=args.num_epoch,lr=args.lr,device=device,log_steps=args.log_steps,model_save_path=args.model_save_path)

    else:

        test_dataloader=gen_test_dataloader(args.predict_dir,args.batch_size)

        model=LR(class_num=args.class_num)
        model.load_state_dict(torch.load(args.model_save_path))
        model.to(device)
        result=predict(model,test_dataloader,device)
        print(result)


if __name__=='__main__':
    main()
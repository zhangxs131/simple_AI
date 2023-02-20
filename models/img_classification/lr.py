import torch
import torch.nn as nn
import torch.nn.functional as F


class LR(nn.Module):
    def __init__(self,class_num=10,chanels=1,size=28,loss_fun=None):
        super().__init__()
        self.class_num=class_num

        self.fc1=nn.Linear(chanels*size*size,100)
        self.fc2=nn.Linear(100,class_num)

        self.loss_fun=loss_fun
        if self.loss_fun==None:
            self.loss_fun=nn.CrossEntropyLoss()

    def forward(self,x,label=None):
        """

        :param x:[bs,channel,28,28]
        :return:
        """
        batch_size=x.shape[0]
        channel=x.shape[1]

        x=x.view(batch_size,-1)
        x=F.relu(self.fc1(x))
        x=F.softmax(self.fc2(x),dim=-1)

        if label==None:
            return {'logits':x}
        else:
            loss=self.loss_fun(x,label)
            return {'logits':x,'loss':loss}
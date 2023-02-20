import torch
import torch.nn as nn
import torch.nn.functional as F


class Lenet5(nn.Module):
    def __init__(self,class_num=10,chanels=1,size=28,loss_fun=None):
        super().__init__()
        self.class_num=class_num

        if size==28:
            self.conv1=nn.Conv2d(1,6,(5,5),padding=2)
        elif size==32:
            self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.conv2=nn.Conv2d(6,16,(5,5))


        self.fc1=nn.Linear(400,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,class_num)

        self.loss_fun=loss_fun
        if self.loss_fun==None:
            self.loss_fun=nn.CrossEntropyLoss()

    def forward(self,x,label=None):
        """

        :param x:[bs,channel,28,28]
        :return:
        """
        batch_size=x.shape[0]

        x=F.avg_pool2d(self.conv1(x),kernel_size=2)
        x=F.avg_pool2d(self.conv2(x),kernel_size=2)

        x=x.view(batch_size,-1)

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x),dim=-1)

        if label==None:
            return {'logits':x}
        else:
            loss=self.loss_fun(x,label)
            return {'logits':x,'loss':loss}
def main():
    test_tensor=torch.randn((8,1,28,28))
    print(test_tensor.shape)

    model=Lenet5()
    outputs=model(test_tensor)
    print(outputs['logits'].shape)


if __name__=='__main__':
    main()
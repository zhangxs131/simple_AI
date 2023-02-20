import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self,class_num=1000,chanels=3,size=227,loss_fun=None):
        super().__init__()
        self.class_num=class_num

        if size==224:
            self.conv1 = nn.Conv2d(chanels, 96, (11, 11), stride=4,padding=2)
        else:
            self.conv1 = nn.Conv2d(chanels, 96, (11, 11),stride=4)
        self.conv2=nn.Conv2d(96,256,(5,5),padding=2)
        self.conv3=nn.Conv2d(256,384,(3,3),padding=1)
        self.conv4=nn.Conv2d(384,384,(3,3),padding=1)
        self.conv5= nn.Conv2d(384, 256, (3, 3), padding=1)

        self.fc1=nn.Linear(9216,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,class_num)

        self.loss_fun=loss_fun
        if self.loss_fun==None:
            self.loss_fun=nn.CrossEntropyLoss()

    def forward(self,x,label=None):
        """

        :param x:[bs,channel,28,28]
        :return:
        """
        batch_size=x.shape[0]

        x=F.max_pool2d(self.conv1(x),kernel_size=3,stride=2)
        x=F.max_pool2d(self.conv2(x),kernel_size=3,stride=2)
        x=self.conv3(x)
        x=self.conv4(x)
        x=F.max_pool2d(self.conv5(x),kernel_size=3,stride=2)

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
    test_tensor=torch.randn((8,3,224,224))
    print(test_tensor.shape)

    model=AlexNet(size=224)
    outputs=model(test_tensor)
    print(outputs['logits'].shape)


if __name__=='__main__':
    main()
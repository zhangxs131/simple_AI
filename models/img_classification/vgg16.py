import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self,class_num=1000,chanels=3,size=224,loss_fun=None):
        super().__init__()
        self.class_num=class_num


        self.maxpool1 = nn.Sequential(
            nn.Conv2d(chanels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dense = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, class_num)
        )


        self.loss_fun=loss_fun
        if self.loss_fun==None:
            self.loss_fun=nn.CrossEntropyLoss()

    def forward(self,x,label=None):
        """

        :param x:[bs,channel,28,28]
        :return:
        """
        batch_size=x.shape[0]

        x = self.maxpool1(x)
        x = self.maxpool2(x)
        x = self.maxpool3(x)
        x = self.maxpool4(x)
        x = self.maxpool5(x)
        x=x.view(batch_size,-1)

        x = self.dense(x)

        x=F.softmax(x,dim=-1)

        if label==None:
            return {'logits':x}
        else:
            loss=self.loss_fun(x,label)
            return {'logits':x,'loss':loss}
def main():
    test_tensor=torch.randn((8,3,224,224))
    print(test_tensor.shape)

    model=VGG16(size=224)
    outputs=model(test_tensor)
    print(outputs['logits'].shape)


if __name__=='__main__':
    main()
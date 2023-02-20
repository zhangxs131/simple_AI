from PIL import Image
from torchvision import datasets
import os

save_dir='../data/mnist/test_dir'
save_name='test_7.jpg'
test_data=datasets.MNIST(root='../data/mnist', train=False, download=True)

example_num=5
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for i in range(example_num):
    d=test_data[i][0]
    l=test_data[i][1]

    test_data[i][0].save(os.path.join(save_dir, 'test_{}.jpg'.format(l)))

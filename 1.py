import gzip
import struct
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torchvision


def load_mnist_train(path, kind='train'):
    """
    path:数据集的路径
    kind:值为train，代表读取训练集
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    # 使用gzip打开文件
    with gzip.open(labels_path, 'rb') as lbpath:
        # 使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
        # 这样读到的前两个数据分别是magic number和样本个数
        magic, n = struct.unpack('>II', lbpath.read(8))
        # 使用np.fromstring读取剩下的数据，lbpath.read()表示读取所有的数据
        labels = np.fromstring(lbpath.read(), dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromstring(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


path = 'E:\\xuexi\\二年级\\最优化\\相关示例程序\\Neural-Network-master\\Neural-Network-master'
X_train, y_train = load_mnist_train(path, kind='train')
X_test, y_test = load_mnist_train(path, kind='t10k')
# print(X_train[0])
# print(y_train[0])
# print(X_test[0])
# print(y_test[0])
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )

# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

input = torch.tensor(X_train).float()
label = torch.tensor(y_train).float()

labels=[]
for i in range(len(y_train)):
    labels.append([0,0,0,0,0,0,0,0,0,0])
    labels[i][y_train[i]]=100
temp=np.array(labels)
label=torch.tensor(temp).float()


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out = self.predict(out)
        return out


net = Net(784,16,10)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
compute_loss = nn.MSELoss()
for epoch in range(100):
    # print(epoch)
    predict = net(input)

    loss = compute_loss(predict, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.ioff()
plt.show()

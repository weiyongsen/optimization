import gzip
import struct
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys


def load_mnist_train(path, kind='train'):
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


# 获得训练集的数据和标签
x_train, y_train = load_mnist_train(r'E:\xuexi\二年级\最优化\相关示例程序\Neural-Network-master\Neural-Network-master', kind='train')
x_test, y_test = load_mnist_train(r'E:\xuexi\二年级\最优化\相关示例程序\Neural-Network-master\Neural-Network-master', kind='t10k')


# 构造神经网络
class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):  # 输入层，隐藏层(两个)和输出层神经元个数
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = torch.nn.functional.relu(out)
        out = self.hidden2(out)
        out = torch.nn.functional.relu(out)
        out = self.predict(out)
        return out


model = Net(784, 16, 10)  # 输入层784，隐藏层一16，隐藏层二16，输出层10
learning_rate = 1e-5    # 学习率
batch_size = 1000       # 每组划分样本个数
loss = []               # 损失值
train_epoch = 1000        # 训练次数
print("开始训练网络...")

y_train_label = np.zeros((60000, 10))  # 构造训练集的标签
for i in range(60000):
    y_train_label[i][y_train[i]] = 100
# 将训练集和标签集转换成张量
x_train = torch.Tensor(x_train).float()
y_train_label = torch.Tensor(y_train_label).float()

for epoch in range(train_epoch):
    for start in range(0, len(x_train), batch_size):
        end = start + batch_size
        batch_x_train = x_train[start:end]
        batch_y_train = y_train_label[start:end]
        y_pre = model(batch_x_train)
        loss_single = torch.nn.functional.mse_loss(y_pre, batch_y_train)  # 以均方误差作为损失值
        loss_single.backward()  # 前向传播
        with torch.no_grad():  # 梯度下降更新参数
            for param in model.parameters():
                param -= learning_rate * param.grad
        model.zero_grad()  # 将所有参数的梯度归零（因为网络中梯度有累加的性值，不归零的话会影响下一步梯度的计算结果）
    # 计算一个epoch后的损失值
    loss_single = torch.nn.functional.mse_loss(y_pre, batch_y_train)
    loss.append(loss_single.detach().numpy().tolist())
    sys.stderr.write('\rEpoch: %d/%d,Loss: %f' % (epoch + 1, train_epoch, loss[epoch]))  # 控制台动态输出err为红字，out为白字。
    sys.stderr.flush()

# 绘制损失值变化曲线
plt.plot(range(len(loss)), loss)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Learning Rate:1e-5  Batch Size:1000')
plt.show()

# 预测训练集
print("\n根据模型预测训练集...")
count_train = 0
wrong_train = 0
for i in range(60000):
    y = model(x_train[i])
    y = y.detach().numpy().tolist()
    y_pre = y.index(max(y))
    if y_train[i] == y_pre:
        count_train += 1
    else:
        wrong_train += 1
    sys.stdout.write('\rEpoch: %d/%d,准确值: %d,预测值: %d,错误次数: %d' % (i + 1, 60000, y_train[i], y_pre, wrong_train))
    sys.stdout.flush()

# 将测试集转换成张量
x_test = torch.Tensor(x_test).float()
y_wrong_pre = []
x_wrong_pre = []
y_wrong_pre_rt = []
y_rt_pre = []
x_rt_pre = []
# 预测测试集
print("\n根据模型预测测试集...")
count_test = 0  # 正确样本数
wrong_test = 0
for j in range(10000):
    y = model(x_test[j])
    y = y.detach().numpy().tolist()
    y_pre = y.index(max(y))
    if y_test[j] == y_pre:
        count_test += 1
        x_rt_pre.append(x_test[j])
        y_rt_pre.append(y_pre)
    else:
        wrong_test += 1
        x_wrong_pre.append(x_test[j])
        y_wrong_pre.append(y_pre)
        y_wrong_pre_rt.append(y_test[j])
    sys.stdout.write('\rEpoch: %d/%d,准确值: %d,预测值: %d，错误次数: %d' % (j + 1, 10000, y_test[j], y_pre, wrong_test))
    sys.stdout.flush()

# 绘制部分正确预测的图
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, )
ax = ax.flatten()
for i in range(25):
    img = x_rt_pre[i].reshape(28, 28)
    ax[i].set_title("Predict:" + str(y_rt_pre[i]) + ",Correct:" + str(y_rt_pre[i]))
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# 绘制部分错误预测的图
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, )
ax = ax.flatten()
for i in range(25):
    img = x_wrong_pre[i].reshape(28, 28)
    ax[i].set_title("Predict:" + str(y_wrong_pre[i]) + ",Correct:" + str(y_wrong_pre_rt[i]))
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# 输出准确率
print()     # stdout后的换行
print("训练准确率：" + str(count_train / 60000))
print("测试准确率：" + str(count_test / 10000))

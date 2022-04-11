import random
import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
i = 0
while i != 20:
    a = random.randint(0, 50)
    if a in x:
        i = i
    else:
        x = x + [a]
        i = i + 1
x.sort()
i = 0
while i != 20:
    a = random.randint(0, 50)
    if a in y:
        i = i
    else:
        y = y + [a]
        i = i + 1
y.sort()

# print(x)
# print(y)

lv = []       # 损失
epoch = []    # 迭代次数
m = 20  # 数据个数
alpha = 0.001  # 学习率

pd_1 = 0.0
pd_2 = 0.0
x1 = 0.0
x2 = 0.0
i = 1  # 迭代次数

while True:
    sum_1 = 0
    sum_2 = 0
    # 求梯度
    for j in range(m):
        sum_1 += x2 + x1 * x[j] - y[j]
        sum_2 += (x2 + x1 * x[j] - y[j]) * x[j]

    pd_2 = x2 - alpha * (1 / m) * sum_1
    pd_1 = x1 - alpha * (1 / m) * sum_2

    epoch.append(i)
    pd_lv = 0.0
    # 计算损失值
    for j in range(m):
        pd_lv = pd_lv + (pd_1 * x[j] + pd_2 - y[j]) * (pd_1 * x[j] + pd_2 - y[j])
    pd_lv = pd_lv / (2 * m)
    lv.append(pd_lv)
    # 如果迭代结果和上一次相同说明已收敛，则终止迭代
    if (pd_2 == x2) & (pd_1 == x1):
        break
    x2 = pd_2
    x1 = pd_1
    i = i + 1

print("拟合的结果为：y=", x1, "*x+", x2)
print("共进行了", i, "次迭代")

# 可视化
plt.figure(1)
plt.scatter(x, y, color='b')  # 绘制散点图
plt.xlabel('x label')
plt.ylabel('y label')
plt.title('Linear')
X = np.linspace(-2, 51, 10000)
# 建立线性方程
Y = x1 * X + x2
plt.plot(X, Y)  # 绘制直线

plt.figure(2)
plt.plot(epoch, lv)     # 绘制损失图
plt.xlim(-200, 10000)
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.title('Loss-Epoch')
plt.show()

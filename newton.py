# f(x)=60-10*x1-4*x2+x1^2+x2^2-x1*x2
import numpy as np


def f(x, y):   # 目标函数
    return 60 - 10 * x - 4 * y + x ** 2 + y ** 2 - x * y


def fx(x, y):   # 对x求偏导
    return -10 + 2 * x - y


def fy(x, y):   # 对y求偏导
    return -4 + 2 * y - x


# def fxx(x, y):
#     return 2
# def fyy(x, y):
#     return 2
# def fxy(x, y):
#     return -1

def hessian(x, y):  # 海森矩阵
    return [[2, -1], [-1, 2]]


def hessian_inv(x, y):  # 海森矩阵的逆矩阵
    return np.linalg.inv(hessian(x, y))


x, y = input("请输入初始点(空格分开):").split()   # 输入初始点
x1 = [int(x), int(y)]
h = hessian_inv(x, y)
x2 = [1, 1]
print("初始点是(", x1[0], ",", x1[1], ")")
epsilon = float(input("请输入精度："))
i=0
while fx(x2[0], x2[1]) ** 2 + fy(x2[0], x2[1]) ** 2 > epsilon:  # 如果一阶导数不接近0，继续迭代
    x2[0] = x1[0] - h[0][0] * fx(x1[0], x1[1]) - h[0][1] * fy(x1[0], x1[1])     # 模拟矩阵运算
    x2[1] = x1[1] - h[1][0] * fx(x1[0], x1[1]) - h[1][1] * fy(x1[0], x1[1])
    x1[0] = x2[0]   # 记录上次迭代的值
    x1[1] = x2[1]
    i += 1  # 记录迭代次数
    print("第", i, "次迭代后的点是(", x2[0], ",", x2[1], ")","它的一阶导数是(",fx(x2[0],x2[1]),",",fy(x2[0],x2[1]),")", sep='')
print("共迭代",i,"次")
print("最终得到极小值点在(", x2[0], ",", x2[1], ")")
print("极小值是", f(x2[0], x2[1]))

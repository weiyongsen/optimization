import math


# 目标函数

def f(x):
    if x > 0:
        return 8 * math.e ** (1 - x) + 7 * math.log(x)
    else:
        return 0


# 进退法
def inout(a0, h):
    a1 = a0  # 第二步，得出f1，f2
    a2 = a1 + h
    f1 = f(a1)
    f2 = f(a2)

    if f2 < f1:  # 判断前进还是后退
        a3 = a2 + h
        f3 = f(a3)
        while True:  # 第三步，重复比较
            if f2 < f3:
                a = a1
                b = a3
                break
            else:
                h = 2 * h  # 步长扩大
                a1 = a2
                a2 = a3
                f1 = f2
                f2 = f3
                a3 = a2 + h
                f3 = f(a3)
    else:

        while True:
            a3 = a1 - h
            f3 = f(a3)
            if f1 < f3:
                a = a3
                b = a2
                break
            else:
                h = h  # 步长不扩大
                a2 = a1
                a1 = a3
                f3 = f1
                f1 = f3
                a3 = a1 - h
                f3 = f(a3)
    return [a, b]


# 黄金分割法
def gold(a, b, ep):
    b1 = a + 0.382 * (b - a)  # 初始区间
    b2 = a + 0.618 * (b - a)
    global i
    i=0
    while b - a > ep:  # 条件
        fb1 = f(b1)
        fb2 = f(b2)
        if fb1 > fb2:  # 区间向右收缩
            a = b1
            b1 = b2
            b2 = a + 0.618 * (b - a)
            i += 1  # 记录迭代次数
            print("第", i, "次迭代后的区间是[", a, ",", b, "]", sep='')
        if fb1 < fb2:  # 区间向左收缩
            b = b2
            b2 = b1
            b1 = a + 0.382 * (b - a)
            i += 1
            print("第", i, "次迭代后的区间是[", a, ",", b, "]", sep='')
    minnum = (a + b) / 2
    return minnum


chushi = float(input("请输入初始值:"))  # 给定初始值
buchang = float(input("请输入步长:"))
e = inout(chushi, buchang)
print("由进退法得到的初始区间为:", e)  # 打印结果
epsilon = float(input("请输入黄金分割精度:"))  # 设置精度
end = gold(e[0], e[1], epsilon)
print("共迭代",i,"次")
print("精度", epsilon, "时黄金分割法确定的极小点在:", end, "处")
print("精度", epsilon, "时黄金分割法确定的极小值为:", f(end))
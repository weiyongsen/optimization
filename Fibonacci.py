# 爬楼梯

def ladder(n):
    if n <= 0:
        return 0
    else:
        l = [1, 2]
        for num in range(2, n):
            l.append(l[num - 1] + l[num - 2])
        return l[n - 1]


n = int(input("请输入楼梯阶数:"))
print("一共有", ladder(n), "种不同的方法爬完", n, "层楼梯")

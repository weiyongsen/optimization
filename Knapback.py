# 0-1背包问题
def knap_DP(n, m):
    x = [False for raw in range(n + 1)]     # False 代表不被选择， True 代表被选择
    a = [[0 for col in range(m + 1)] for raw in range(n + 1)]   # 生成动态规划表并赋值0

    for i in range(1, n + 1):   # 更新动态规划表
        for j in range(1, m + 1):
            a[i][j] = a[i - 1][j]
            if (j >= w[i]) and (a[i - 1][j - w[i]] + v[i] > a[i - 1][j]):   # 状态转移方程
                a[i][j] = a[i - 1][j - w[i]] + v[i]

    j = m       # 回溯寻找放入背包的物品
    for i in range(n, 0, -1):
        if a[i][j] > a[i - 1][j]:
            x[i] = True
            j = j - w[i]
    Mv = a[n][m]

    for i in range(n + 1):  # 将x种的True值位置记录
        if x[i]:
            chosen.append(i + 1)
    return Mv


w = [3, 4, 3, 5, 5]     # 重量
v = [200, 300, 350, 400, 500]   # 价值
n = len(w) - 1  # 减一，与m统一，方便循环操作
m = 10          # 背包容量
chosen = []     # 被选择的物品编号
Max = knap_DP(n, m)     # 最大利润
print("应选择", chosen, "号矿", "可获利", Max, "金")
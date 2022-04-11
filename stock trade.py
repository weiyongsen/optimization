# 买卖股票的最佳时机2

def maxprofit(prices):
    n = len(prices)
    dp = [[0 for _ in range(2)] for _ in range(n)]
    dp[0][0] = 0
    dp[0][1] = -prices[0]
    for i in range(1, n):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
    return dp[n - 1][0]


# prices = [7, 1, 3, 5, 4, 6]
prices = [7,1,5,3,6,4]
print("最大获利:", maxprofit(prices))
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

a=(("hello world", 1), ("goodnight moon", -1))
feature={word:value for words, value in a for word in words.split()}
print(feature)
# Sales1 = [100, 118, 107, 83, 76, 31, 65, 97, 91, 84, 91, 60, 61, 80, 49]
# Sales2 = [80, 115, 68, 111, 80, 81, 82, 78, 46, 90, 126, 59, 127, 91, 130]
# data1 = np.array(Sales1)
# data2 = np.array(Sales2)

# np.random.seed(0)



# # 데이터를 정렬하여 분위수를 구합니다.
# q1 = np.sort(data1)
# q2 = np.sort(data2)

# median1 = np.median(data1)
# median2 = np.median(data2)
# print(median1,median2)
# # Create the QQ plot comparing the two samples
# plt.figure(figsize=(6, 6))
# plt.scatter(q1, q2, color='blue', label='Quantile points')
# plt.xlabel("Data1 Quantiles")
# plt.ylabel("Data2 Quantiles")
# plt.title("QQ Plot Comparing Two Distributions")

# # Plot the 45-degree reference line
# min_val = min(q1[0], q2[0])
# max_val = max(q1[-1], q2[-1])
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='45° line')

# # Mark the median point on the plot
# plt.scatter(median1, median2, color='green', s=100, marker='o', label='Median')

# plt.legend()
# plt.show()
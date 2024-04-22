import os
import pandas as pd
file_path = "com-lj.ungraph.txt"

# 1. 读取txt文件
with open(file_path, "r") as file:
    lines = file.readlines()

# 2. 数据预处理
# 假设数据格式为：编号\t数据1\t数据2\t...
data = []
for line in lines:
    # 去除空格并按制表符分割数据
    items = line.strip().split("\t")
    # 如果是标题行，则跳过
    if items[0] == "ToNodeId":  # 这里的条件根据实际情况调整
        continue
    # 提取编号和数据列
    number = items[0]
    values = items[1:]
    # 将数据列转换成浮点数，如果不能转换则跳过该行
    try:
        values = [float(value) for value in values]
    except ValueError:
        continue
    # 将编号和数据列组合成一个元组，并添加到data列表中
    data.append((number, values))

# 3. 将数据转换成DataFrame
# 假设第一列是编号，后面的列是数据
# 可以根据需要调整列名
columns = ["Number"] + [f"Data_{i}" for i in range(len(data[0][1]))]
df = pd.DataFrame(data, columns=columns)

# 打印前几行数据，检查是否正确读取和预处理
print(df.head())



# 在读取文件时，请确保文件路径正确，并且根据实际情况调整文件读取和数据处理的代码。

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx

# 1. 数据获取与预处理
# 假设您已经从数据集中获取了社交网络数据，并将其存储在一个DataFrame中
# 这里假设您的数据集已经预处理过，可以直接用于挖掘

# 2. 频繁模式挖掘
# 使用Apriori算法挖掘频繁项集
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

# 使用关联规则挖掘
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# 3. 可视化展示
# 可视化频繁项集
plt.bar(range(len(frequent_itemsets)), frequent_itemsets.support)
plt.xticks(range(len(frequent_itemsets)), frequent_itemsets.itemsets, rotation='vertical')
plt.xlabel('Frequent Itemsets')
plt.ylabel('Support')
plt.title('Frequent Itemsets and Their Support')
plt.show()

# 可视化关联规则
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents')
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', edge_color='gray', linewidths=1, font_size=15)
plt.title('Association Rules Graph')
plt.show()
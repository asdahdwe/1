import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整列宽

base = pd.read_csv('Alzheimer Disease and Healthy Aging Data In US.csv',low_memory=False)
base.head()

# 标称属性摘要
nominal_attributes = ['LocationAbbr', 'LocationDesc', 'Datasource','Class','Topic','Question','Data_Value_Unit','DataValueTypeID','Data_Value_Type','StratificationCategory1','Stratification1','StratificationCategory2','Stratification2','Geolocation','ClassID','TopicID','QuestionID','StratificationCategoryID1','StratificationID1','StratificationCategoryID2']
for attribute in nominal_attributes:
    frequency_counts = base[attribute].value_counts()
    print("标称属性 {} 的频数统计：\n{}".format(attribute, frequency_counts))

# 数值属性摘要
numeric_attributes = ['YearStart','YearEnd','Data_Value','Data_Value_Alt','Low_Confidence_Limit','High_Confidence_Limit','LocationID']
numeric_summary = base[numeric_attributes].describe()
missing_values_count = base[numeric_attributes].isnull().sum()
print("数值属性的五数概括：\n", numeric_summary)
print("\n数值属性的缺失值个数：\n", missing_values_count)
#
# 直方图
base[numeric_attributes].hist(bins=20, figsize=(10, 6))
plt.suptitle('Histogram of Numeric Attributes')
plt.show()

# 盒图
base[numeric_attributes].plot(kind='box', vert=False, figsize=(10, 6))
plt.title('Boxplot of Numeric Attributes')
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer

def remove_missing_values(base):
    """将缺失部分剔除"""
    return base.dropna()

def fill_with_mode(base):
    """用最高频率值来填补缺失值"""
    filled_base = base.copy()
    for column in filled_base.columns:
        if filled_base[column].notnull().any():
           most_frequent_value = filled_base[column].mode()[0]
           filled_base[column].fillna(most_frequent_value, inplace=True)
    return filled_base

def fill_with_regression(base):
    """通过属性的相关关系来填补缺失值（线性回归）"""
    filled_base = base.copy()
    for column in filled_base.columns:
        if filled_base[column].isnull().sum() > 0:  # 如果存在缺失值
            features = filled_base.dropna().drop(column, axis=1)  # 删除含有缺失值的行，并且排除目标列
            if not features.empty:  # 检查特征矩阵是否为空
                target = filled_base[column].dropna()  # 目标列
                linear_model = LinearRegression().fit(features, target)
                missing_data = filled_base.loc[filled_base[column].isnull(), features.columns]
                filled_base.loc[filled_base[column].isnull(), column] = linear_model.predict(missing_data)
    return filled_base

def fill_with_knn(base):
    """通过数据对象之间的相似性来填补缺失值（K近邻算法）"""
    # 删除字符串类型的列
    base_numeric = base.select_dtypes(include=['number'])
    knn_imputer = KNNImputer()
    return pd.DataFrame(knn_imputer.fit_transform(base_numeric), columns=base_numeric.columns, index=base_numeric.index)

# 1. 将缺失部分剔除
base_without_missing = remove_missing_values(base)
# 2. 用最高频率值来填补缺失值
base_filled_with_mode = fill_with_mode(base)
# 3. 通过属性的相关关系来填补缺失值（线性回归）
base_filled_with_regression = fill_with_regression(base)
# 4. 通过数据对象之间的相似性来填补缺失值（K近邻算法）
base_filled_with_knn = fill_with_knn(base)

base_without_missing.to_csv('base_without_missing.csv', index=False)
base_filled_with_mode.to_csv('base_filled_with_mode.csv', index=False)
base_filled_with_regression.to_csv('base_filled_with_regression.csv', index=False)
base_filled_with_knn.to_csv('base_filled_with_knn.csv', index=False)


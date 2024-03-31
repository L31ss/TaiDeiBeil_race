import os
import time
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# 读取附件1中的数据
data = pd.read_csv('C:\\Users\\MI\\Desktop\\泰迪杯A题\\数据\\附件1\\M101.csv')

# 时间戳
dates = data['日期']
times = data['时间']

# 创建一个新的列来保存转换后的结果
data['秒_小时'] = pd.to_datetime(times, unit='s').dt.strftime('%H:%M:%S')

# 获取当前年份
current_year = datetime.datetime.now().year

data['日期格式'] = pd.to_datetime(dates, format='%j').dt.strftime(f'{current_year}-%m-%d')

# 将日期和转换后的时间列合并成一个字符串列
data['时间戳'] = pd.to_datetime(data['日期格式'] + ' ' + data['秒_小时'])

# 将时间戳转换为 datetime 类型
data['时间戳'] = pd.to_datetime(data['时间戳'])

# 数据预处理
# 处理缺失值

data.ffill(axis=0, inplace=True)

# 探索性数据分析（EDA）
# 统计各个装置故障字段的情况
fault_columns = ['物料推送装置故障1001', '物料检测装置故障2001', '填装装置检测故障4001',
                 '填装装置定位故障4002', '填装装置填装故障4003', '加盖装置定位故障5001',
                 '加盖装置加盖故障5002', '拧盖装置定位故障6001', '拧盖装置拧盖故障6002']

device_columns = ['物料推送气缸推送状态', '物料推送气缸收回状态', '物料推送数',
                  '物料待抓取数', '放置容器数', '容器上传检测数',
                  '填装检测数', '填装定位器固定状态', '填装定位器放开状态',
                  '物料抓取数', '填装旋转数', '填装下降数',
                  '填装数', '加盖检测数', '加盖定位数',
                  '推盖数', '加盖下降数', '加盖数',
                  '拧盖检测数', '拧盖定位数', '拧盖下降数',
                  '拧盖旋转数', '拧盖数', '合格数', '不合格数']
# # 统计每个故障的总数
# for column in fault_columns:
#     print(f"故障{column}统计情况：")
#     print(data[column].value_counts())

# 将故障内容不为0的值全部置为1
for column in fault_columns:
    data[column] = data[column].apply(lambda x: 1 if x != 0 else x)

data.to_csv("unprocessed.csv", index=False)
# 选择第6列到第26列的数据进行处理，并将其转换为数值类型
data_subset = data.iloc[:, 5:26].apply(pd.to_numeric, errors='coerce')

# 对数据进行从下往上的差分处理，后一行比前一行有变化设为1，无变化设为0
data_diff = data_subset.diff().fillna(0).apply(lambda x: x.map(lambda x: 1 if x > 0 else 0))

# 将处理后的结果与原始数据的其他列合并
processed_data = pd.concat([data.iloc[:, :5], data_diff, data.iloc[:, 26:]], axis=1)

data.drop(columns=['日期'], inplace=True)
data.drop(columns=['秒_小时'], inplace=True)
data.drop(columns=['日期格式'], inplace=True)

# 将合并后的数据保存到 CSV 文件中
data.to_csv('processed_data_with_all_columns.csv', index=False)

pre_test = ['时间', '物料推送气缸推送状态', '物料推送气缸收回状态', '物料推送数', '物料待抓取数',
                   '填装数', '加盖数', '拧盖数', '合格数', '不合格数']

target_columns = fault_columns

data_len = len(data)
print(data_len)
X = data[pre_test][0:data_len]
y = data[target_columns][0:data_len]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('混淆矩阵 : ')
print(confusion_matrix(y_test.values.flatten(), y_pred.flatten()))
print('\n分类报告 : ')
print(classification_report(y_test.values.flatten(), y_pred.flatten()))

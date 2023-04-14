import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
sns.set(font='SimHei', font_scale=1.2, style='darkgrid')  # 解决Seaborn中文显示问题
palette = 'deep'

df = pd.read_excel('../原始数据/附件1：物流网络历史货量数据.xlsx')
dateDf = df.groupby(by=['日期']).count().sort_index()

plt.figure(figsize=(14, 8))
sns.lineplot(dateDf, x='日期', y='货量')
plt.xticks(rotation=90)
plt.ylabel('数量')
plt.tight_layout()
plt.show()

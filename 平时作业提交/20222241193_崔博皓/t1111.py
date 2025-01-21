import matplotlib
from matplotlib.font_manager import FontProperties
# 设置 matplotlib 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
load_data_path = "附件1：各园区典型日负荷数据.xlsx"
generation_data_path = "附件2：各园区典型日风光发电数据.xlsx"

load_data = pd.read_excel(load_data_path)
generation_data = pd.read_excel(generation_data_path)

# 设置正确的标题
generation_data.columns = ['时间', '太阳能_A', '风力_B', '太阳能_C', '风力_C']

# 将数据转换为数值类型
load_data.iloc[:, 1:] = load_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
generation_data.iloc[:, 1:] = generation_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# 根据装机容量计算实际发电量
generation_data['太阳能_A'] = generation_data['太阳能_A'] * 750
generation_data['风力_B'] = generation_data['风力_B'] * 1000
generation_data['太阳能_C'] = generation_data['太阳能_C'] * 600
generation_data['风力_C'] = generation_data['风力_C'] * 500
# 每小时总发电量求和
generation_data['总发电量'] = generation_data[['太阳能_A', '风力_B', '太阳能_C', '风力_C']].sum(axis=1)

# 每小时总负荷求和
load_data['总负荷'] = load_data.sum(axis=1)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(generation_data['总发电量'], label='总发电量 (kW)', color='green')
plt.plot(load_data['总负荷'], label='总负荷 (kW)', color='red')
plt.title('各园区总发电量与总负荷')
plt.xlabel('每天小时数')
plt.ylabel('功率 (kW)')
plt.legend()
plt.grid(True)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 载入数据
load_data_path = "附件1：各园区典型日负荷数据.xlsx"
generation_data_path = "附件2：各园区典型日风光发电数据.xlsx"

load_data = pd.read_excel(load_data_path)
generation_data = pd.read_excel(generation_data_path)

# 设置正确的表头
generation_data.columns = ['时间', '太阳能_A', '风力_B', '太阳能_C', '风力_C']

# 将数据转换为数值
load_data.iloc[:, 1:] = load_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
generation_data.iloc[:, 1:] = generation_data.iloc[:, 1:]. apply(pd.to_numeric, errors='coerce')

# 根据装机容量计算实际发电量
generation_data['太阳能_A'] = generation_data['太阳能_A'] * 750
generation_data['风力_B'] = generation_data['风力_B'] * 1000
generation_data['太阳能_C'] = generation_data['太阳能_C'] * 600
generation_data['风力_C'] = generation_data['风力_C'] * 500
generation_data['总发电量'] = generation_data[['太阳能_A', '风力_B', '太阳能_C', '风力_C']].sum(axis=1)

# 每小时负荷总和
load_data['总负荷'] = load_data.sum(axis=1)

# 初始化存储浪费和购买能量的变量
total_wasted = 0
total_purchased = 0

# 每小时计算浪费和购买的能量
for i in range(len(load_data)):
    hourly_generation = generation_data.loc[i, '总发电量']
    hourly_load = load_data.loc[i, '总负荷']

    if hourly_generation > hourly_load:
        total_wasted += hourly_generation - hourly_load
    elif hourly_load > hourly_generation:
        total_purchased += hourly_load - hourly_generation

# 假设太阳能和风力等比购买时的总成本
C_solar = 0.4
C_wind = 0.5
total_cost = (total_purchased / 2) * C_solar + (total_purchased / 2) * C_wind
average_cost_per_unit = total_cost / total_purchased if total_purchased > 0 else 0

# 显示结果
print(f"总购买电量（千瓦时）：{total_purchased}")
print(f"总浪费能量（千瓦时）：{total_wasted}")
print(f"总供应成本（元）：{total_cost}")
print(f"单位平均成本（元/千瓦时）：{average_cost_per_unit}")
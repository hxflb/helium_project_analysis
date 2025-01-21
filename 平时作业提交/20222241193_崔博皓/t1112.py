import matplotlib
from matplotlib.font_manager import FontProperties
# 设置 matplotlib 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题
import pandas as pd
import matplotlib.pyplot as plt

# 创建示例数据
data_hours = pd.date_range("2023-01-01", periods=24, freq='H')
generation_a = [max(20 * (i - 6) * (18 - i), 0) for i in range(24)]
generation_b = [max(15 * (i - 6) * (18 - i), 0) for i in range(24)]
generation_c = [max(10 * (i - 6) * (18 - i), 0) for i in range(24)]
load_a = [50 + 10 * abs(i - 12) for i in range(24)]
load_b = [40 + 15 * abs(i - 12) for i in range(24)]
load_c = [30 + 20 * abs(i - 12) for i in range(24)]

# 将数据转换为pandas Series
park_a_jan_pv = pd.Series(generation_a, index=data_hours)
park_b_jan_pv = pd.Series(generation_b, index=data_hours)
park_c_jan_pv = pd.Series(generation_c, index=data_hours)
load_a_jan = pd.Series(load_a, index=data_hours)
load_b_jan = pd.Series(load_b, index=data_hours)
load_c_jan = pd.Series(load_c, index=data_hours)

# 电价设置
time_prices = [1 if 7 <= hour.hour < 22 else 0.4 for hour in data_hours]

# 定义一个函数来模拟和绘制结果
def simulate_and_plot(park_pv, load_jan, park_name):
    # 储能系统参数
    battery_capacity = 500  # kWh
    battery_power = 100     # kW
    SOC = battery_capacity / 2  # 初始SOC为50%
    efficiency = 0.95
    battery_costs = []
    charge_amounts = []
    discharge_amounts = []
    SOC_history = [SOC]

    # 模拟
    for hour, generation, load, price in zip(data_hours, park_pv, load_jan, time_prices):
        net_generation = generation - load
        if net_generation < 0:
            deficit = abs(net_generation)
            if SOC > 0:
                discharge_amount = min(deficit, battery_power, SOC * efficiency)
                SOC -= discharge_amount / efficiency
                deficit -= discharge_amount
            cost = deficit * price
            discharge_amounts.append(discharge_amount)
            charge_amounts.append(0)
        else:
            charge_amount = min(net_generation, battery_power, (battery_capacity - SOC) * efficiency)
            SOC += charge_amount * efficiency
            cost = 0
            charge_amounts.append(charge_amount)
            discharge_amounts.append(0)

        battery_costs.append(cost)
        SOC_history.append(SOC)

    total_cost = sum(battery_costs)

    # 绘制结果
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    axs[0].plot(park_pv, label=f'{park_name} 光伏发电功率（kW）')
    axs[0].plot(load_jan, label=f'{park_name} 负荷功率（kW）', linestyle='--')
    axs[0].set_title(f'{park_name} 发电与负荷')
    axs[0].legend()

    axs[1].plot(SOC_history, label=f'{park_name} 储能系统充放电状态（kWh）')
    axs[1].set_title(f'{park_name} 储能系统充放电状态')
    axs[1].legend()

    axs[2].bar(data_hours, charge_amounts, width=0.05, label=f'{park_name} 充电功率（kW）', color='green')
    axs[2].bar(data_hours, discharge_amounts, width=0.05, label=f'{park_name} 放电功率（kW）', color='red', bottom=charge_amounts)
    axs[2].set_title(f'{park_name} 充放电情况')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    print(f"{park_name} 总成本:", total_cost)

# 为每个园区执行模拟和绘图
simulate_and_plot(park_a_jan_pv, load_a_jan, "园区A")
simulate_and_plot(park_b_jan_pv, load_b_jan, "园区B")
simulate_and_plot(park_c_jan_pv, load_c_jan, "园区C")
generation_a2 = [max(20 * (i - 6) * (18 - i), 0) for i in range(24)]  # 假设光伏发电曲线不变
generation_b2 = [max(15 * (i - 6) * (18 - i), 0) for i in range(24)]
generation_c2 = [max(10 * (i - 6) * (18 - i), 0) for i in range(24)]
load_a2 = [50 + 10 * abs(i - 12) for i in range(24)]  # 假设负荷模式不变
load_b2 = [40 + 15 * abs(i - 12) for i in range(24)]
load_c2 = [30 + 20 * abs(i - 12) for i in range(24)]

# 将数据转换为pandas Series
park_a_feb_pv = pd.Series(generation_a2, index=data_hours)
park_b_feb_pv = pd.Series(generation_b2, index=data_hours)
park_c_feb_pv = pd.Series(generation_c2, index=data_hours)
load_a_feb = pd.Series(load_a2, index=data_hours)
load_b_feb = pd.Series(load_b2, index=data_hours)
load_c_feb = pd.Series(load_c2, index=data_hours)

# 模拟第二个月
simulate_and_plot(park_a_feb_pv, load_a_feb, "园区A（二月）")
simulate_and_plot(park_b_feb_pv, load_b_feb, "园区B（二月）")
simulate_and_plot(park_c_feb_pv, load_c_feb, "园区C（二月）")
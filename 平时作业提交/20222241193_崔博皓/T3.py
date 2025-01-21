import pandas as pd


load_data = pd.read_excel("附件1：各园区典型日负荷数据.xlsx")
pv_wind_data = pd.read_excel("附件2：各园区典型日风光发电数据.xlsx")
yearly_pv_wind_data = pd.read_excel("附件3：12个月各园区典型日风光发电数据.xlsx")

load_data.head(), pv_wind_data.head(), yearly_pv_wind_data.head()
import matplotlib
from matplotlib.font_manager import FontProperties

# 设置 matplotlib 支持中文显示
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
matplotlib.rcParams["axes.unicode_minus"] = (
    False  # 解决保存图像时负号'-'显示为方块的问题
)
new_columns = [
    "时间（h）",
    "1月园区A光伏出力",
    "1月园区B风电出力",
    "1月园区C风电出力",
    "1月园区C光伏出力",
    "2月园区A光伏出力",
    "2月园区B风电出力",
    "2月园区C风电出力",
    "2月园区C光伏出力",
    "3月园区A光伏出力",
    "3月园区B风电出力",
    "3月园区C风电出力",
    "3月园区C光伏出力",
    "4月园区A光伏出力",
    "4月园区B风电出力",
    "4月园区C风电出力",
    "4月园区C光伏出力",
    "5月园区A光伏出力",
    "5月园区B风电出力",
    "5月园区C风电出力",
    "5月园区C光伏出力",
    "6月园区A光伏出力",
    "6月园区B风电出力",
    "6月园区C风电出力",
    "6月园区C光伏出力",
    "7月园区A光伏出力",
    "7月园区B风电出力",
    "7月园区C风电出力",
    "7月园区C光伏出力",
    "8月园区A光伏出力",
    "8月园区B风电出力",
    "8月园区C风电出力",
    "8月园区C光伏出力",
    "9月园区A光伏出力",
    "9月园区B风电出力",
    "9月园区C风电出力",
    "9月园区C光伏出力",
    "10月园区A光伏出力",
    "10月园区B风电出力",
    "10月园区C风电出力",
    "10月园区C光伏出力",
    "11月园区A光伏出力",
    "11月园区B风电出力",
    "11月园区C风电出力",
    "11月园区C光伏出力",
    "12月园区A光伏出力",
    "12月园区B风电出力",
    "12月园区C风电出力",
    "12月园区C光伏出力",
]
yearly_pv_wind_data.columns = new_columns

yearly_pv_wind_data = yearly_pv_wind_data.drop([0, 1]).reset_index(drop=True)
yearly_pv_wind_data["时间（h）"] = yearly_pv_wind_data["时间（h）"].astype(int)

yearly_pv_wind_data.head()
park_a_pv_capacity = 750
park_b_wind_capacity = 1000
park_c_pv_capacity = 600
park_c_wind_capacity = 500

# Total PV and wind output for each park
park_a_pv_output = pv_wind_data["园区A 光伏出力（p.u.）"] * park_a_pv_capacity
park_b_wind_output = pv_wind_data["园区B风电出力（p.u.）"] * park_b_wind_capacity
park_c_pv_output = pv_wind_data["园区C光伏出力（p.u.）"] * park_c_pv_capacity
park_c_wind_output = pv_wind_data["园区C风电出力（p.u.）"] * park_c_wind_capacity

total_generation_a = park_a_pv_output
total_generation_b = park_b_wind_output
total_generation_c = park_c_pv_output + park_c_wind_output

total_load_a = load_data["园区A负荷(kW)"]
total_load_b = load_data["园区B负荷(kW)"]
total_load_c = load_data["园区C负荷(kW)"]
comparison_df = pd.DataFrame(
    {
        "时间（h）": load_data["时间（h）"],
        "园区A发电总量": total_generation_a,
        "园区A负荷": total_load_a,
        "园区B发电总量": total_generation_b,
        "园区B负荷": total_load_b,
        "园区C发电总量": total_generation_c,
        "园区C负荷": total_load_c,
    }
)

comparison_df.head()
import matplotlib.pyplot as plt

generation_data = pv_wind_data.rename(
    columns={
        "园区A 光伏出力（p.u.）": "太阳能_A",
        "园区B风电出力（p.u.）": "风力_B",
        "园区C光伏出力（p.u.）": "太阳能_C",
        "园区C风电出力（p.u.）": "风力_C",
    }
)


generation_data["太阳能_A"] *= 750  # kW
generation_data["风力_B"] *= 1000  # kW
generation_data["太阳能_C"] *= 600  # kW
generation_data["风力_C"] *= 500  # kW
generation_data["总发电量_C"] = generation_data["太阳能_C"] + generation_data["风力_C"]


configurations = {
    "A": {"power": 149, "capacity": 299},
    "B": {"power": 150, "capacity": 300},
    "C": {"power": 150, "capacity": 299},
}
efficiency = 0.95
min_SOC = 10
max_SOC = 90
C_solar = 0.4
C_wind = 0.5

results = []

for park in ["A", "B", "C"]:
    power = configurations[park]["power"]
    capacity = configurations[park]["capacity"]
    SOC = capacity * 0.5
    charges = []
    discharges = []
    SOCs = []
    total_purchase = 0

    for i in range(len(load_data)):
        load = load_data.at[i, f"园区{park}负荷(kW)"]
        gen = (
            generation_data.at[i, f"太阳能_{park}"]
            if park in ["A", "C"]
            else generation_data.at[i, f"风力_{park}"]
        )

        net_gen = gen - load
        if net_gen > 0:
            possible_charge = min(
                net_gen, power, (capacity * max_SOC / 100 - SOC) / efficiency
            )
            SOC += possible_charge * efficiency
            charges.append(possible_charge)
            discharges.append(0)
        else:
            possible_discharge = min(
                -net_gen, power, (SOC - capacity * min_SOC / 100) * efficiency
            )
            SOC -= possible_discharge / efficiency
            total_purchase += -net_gen - possible_discharge
            charges.append(0)
            discharges.append(possible_discharge)
        SOC_percentage = (SOC / capacity) * 100
        SOCs.append(min(max(SOC_percentage, min_SOC), max_SOC))

    total_cost = total_purchase * (C_solar if park != "B" else C_wind)
    results.append({"园区": park, "总成本 (元)": total_cost})

    hours = list(range(24))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("一天中的小时")
    ax1.set_ylabel("充电/放电量 (kWh)", color="tab:blue")
    ax1.plot(hours, charges, label="充电量 (kWh)", color="blue")
    ax1.plot(hours, discharges, label="放电量 (kWh)", color="red")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("SOC (%)", color="tab:green")
    ax2.plot(hours, SOCs, label="SOC (%)", color="green", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.legend(loc="upper right")

    plt.title(f"园区{park}的储能运行情况")
    plt.show()

results
# 给定最优的储能配置容量和功率：容量990 kWh, 功率134 kW
optimal_capacity = 990  # kWh
optimal_power = 134  # kW
global_load_data = load_data
global_generation_data = pv_wind_data


# 使用优化后的储能配置重新计算总成本
def simulate_system_with_optimal_config(capacity, power, load_data, generation_data):
    total_cost = 0
    SOC = capacity / 2  # Start with 50% SOC
    efficiency = 0.95
    cost_per_kWh_solar = 0.4
    cost_per_kWh_wind = 0.5

    SOC_history = []
    charges = []
    discharges = []

    for i in range(len(load_data)):
        load = load_data.iloc[i]["总负荷"]
        generation = generation_data.iloc[i]["总发电量"]

        net_generation = generation - load
        if net_generation < 0:
            # 负载超出发电量，需要购电或放电
            net_generation = abs(net_generation)
            if SOC > 0:  # 如果储能有余量可以放电
                discharge = min(net_generation, power, SOC * efficiency)
                SOC -= discharge / efficiency
                net_generation -= discharge
                discharges.append(discharge)
            else:
                discharges.append(0)
            total_cost += (
                net_generation * cost_per_kWh_solar
            )  # 假设购电全部按照太阳能价格计算
            charges.append(0)
        else:
            # 发电量超出负载，可以充电
            charge = min(net_generation, power, (capacity - SOC) * efficiency)
            SOC += charge * efficiency
            charges.append(charge)
            discharges.append(0)

        SOC_history.append(SOC)

    return total_cost, SOC_history, charges, discharges


# 调用模拟函数
total_cost, SOC_history, charges, discharges = simulate_system_with_optimal_config(
    optimal_capacity, optimal_power, global_load_data, global_generation_data
)

# 打印总成本
print(f"总成本: {total_cost:.2f} ")
# 计算总发电量和总负荷
load_data['总负荷'] = load_data.iloc[:, 1:].sum(axis=1)
pv_wind_data['总发电量'] = park_a_pv_output + park_b_wind_output + park_c_pv_output + park_c_wind_output

global_load_data = load_data
global_generation_data = pv_wind_data

# 调用模拟函数
total_cost, SOC_history, charges, discharges = simulate_system_with_optimal_config(
    optimal_capacity, optimal_power, global_load_data, global_generation_data
)

# 打印总成本
total_cost, SOC_history[:5], charges[:5], discharges[:5]  # 显示初步结果及SOC历史和充放电量的头几个条目
import matplotlib.pyplot as plt

# 创建时间轴
hours = list(range(24))

# 创建图表展示 SOC 历史以及充放电活动
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('小时')
ax1.set_ylabel('SOC (kWh)', color='tab:blue')
ax1.plot(hours, SOC_history, label='SOC (kWh)', color='blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylabel('能量 (kWh)', color='tab:red')
ax2.plot(hours, charges, label='充电量 (kWh)', color='green', linestyle='--')
ax2.plot(hours, discharges, label='放电量 (kWh)', color='red', linestyle='--')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.title('联合园区储能系统的SOC和充放电情况')
plt.show()

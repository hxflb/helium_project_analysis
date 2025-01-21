import matplotlib
from matplotlib.font_manager import FontProperties

# 设置 matplotlib 支持中文显示
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
matplotlib.rcParams["axes.unicode_minus"] = (
    False  # 解决保存图像时负号'-'显示为方块的问题
)
import pandas as pd
import matplotlib.pyplot as plt

# 载入数据
load_data_path = "附件1：各园区典型日负荷数据.xlsx"
generation_data_path = "附件2：各园区典型日风光发电数据.xlsx"

load_data = pd.read_excel(load_data_path)
generation_data = pd.read_excel(generation_data_path)

# 设置正确的表头
generation_data.columns = ["时间", "太阳能_A", "风力_B", "太阳能_C", "风力_C"]

# 数据转换为数值类型
load_data.iloc[:, 1:] = load_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
generation_data.iloc[:, 1:] = generation_data.iloc[:, 1:].apply(
    pd.to_numeric, errors="coerce"
)

# 根据装机容量计算实际发电量
generation_data["太阳能_A"] *= 750  # A园区太阳能装机容量
generation_data["风力_B"] *= 1000  # B园区风力装机容量
generation_data["太阳能_C"] *= 600  # C园区太阳能装机容量
generation_data["风力_C"] *= 500  # C园区风力装机容量

# 为每个园区绘制数据图
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
fig.subplots_adjust(hspace=0.5)

# 为每个园区绘制图表
parks = ["A", "B", "C"]
for idx, park in enumerate(parks):
    axes[idx].plot(
        generation_data.index,
        load_data[f"园区{park}负荷(kW)"],
        label="负载",
        color="red",
    )

    if park == "C":
        # C园区同时具有太阳能和风力发电
        axes[idx].plot(
            generation_data.index,
            generation_data["太阳能_C"],
            label="太阳能发电 (C)",
            color="green",
        )
        axes[idx].plot(
            generation_data.index,
            generation_data["风力_C"],
            label="风力发电 (C)",
            color="blue",
        )
    elif park == "A":
        axes[idx].plot(
            generation_data.index,
            generation_data["太阳能_A"],
            label="太阳能发电 (A)",
            color="green",
        )
    elif park == "B":
        axes[idx].plot(
            generation_data.index,
            generation_data["风力_B"],
            label="Wind Generation (B)",
            color="blue",
        )

    axes[idx].set_title(f"园区{park} - 负荷 vs. 发电情况")
    axes[idx].set_xlabel("小时")
    axes[idx].set_ylabel("功率 (kW)")
    axes[idx].legend()

plt.show()
import pandas as pd

# 加载数据
load_data_path = "附件1：各园区典型日负荷数据.xlsx"
generation_data_path = "附件2：各园区典型日风光发电数据.xlsx"

load_data = pd.read_excel(load_data_path)
generation_data = pd.read_excel(generation_data_path)

# 设置正确的表头
generation_data.columns = ["时间", "太阳能_A", "风力_B", "太阳能_C", "风力_C"]

# 将数据转换为数字，处理潜在的非数字类型
load_data.iloc[:, 1:] = load_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
generation_data.iloc[:, 1:] = generation_data.iloc[:, 1:].apply(
    pd.to_numeric, errors="coerce"
)

# 根据装机容量计算实际发电量
generation_data["太阳能_A"] *= 750  # A园区太阳能装机容量
generation_data["风力_B"] *= 1000  # B园区风力装机容量
generation_data["太阳能_C"] *= 600  # C园区太阳能装机容量
generation_data["风力_C"] *= 500  # C园区风力装机容量
generation_data["总发电量_C"] = generation_data["太阳能_C"] + generation_data["风力_C"]

# 最大负荷值
max_loads = {"A": 447, "B": 419, "C": 506}

# 初始化结果变量
results = []

# 设置风力和太阳能电费
C_solar = 0.4
C_wind = 0.5

# 计算每个园区
for park in ["A", "B", "C"]:
    load_data[f"购买_{park}"] = 0
    load_data[f"浪费_{park}"] = 0

    # 遍历每个小时计算所需变量
    for i, row in load_data.iterrows():
        load = row[f"园区{park}负荷(kW)"]
        max_load = max_loads[park]
        load = min(load, max_load)  # 确保负荷不超过最大负荷

        if park == "C":
            gen = generation_data.loc[i, "总发电量_C"]  # C园区的综合发电量
        elif park == "A":
            gen = generation_data.loc[i, "太阳能_A"]
        elif park == "B":
            gen = generation_data.loc[i, "风力_B"]

        if gen >= load:
            load_data.loc[i, f"浪费_{park}"] = gen - load
            load_data.loc[i, f"购买_{park}"] = 0
        else:
            load_data.loc[i, f"购买_{park}"] = load - gen

    # 计算经济指标
    total_purchase = load_data[f"购买_{park}"].sum()
    total_wasted = load_data[f"浪费_{park}"].sum()
    total_cost = (
        load_data.loc[load_data[f"购买_{park}"] > 0, f"购买_{park}"]
        * (C_solar if park != "B" else C_wind)
    ).sum()
    average_cost = total_cost / total_purchase if total_purchase > 0 else float("inf")

    results.append(
        {
            "园区": park,
            "总购电量 (kWh)": total_purchase,
            "总浪费能量 (kWh)": total_wasted,
            "总供电成本 (元)": total_cost,
            "单位平均成本 (元/kWh)": average_cost,
        }
    )

# 将结果转换为数据框
result_df = pd.DataFrame(results)
print(result_df)
import pandas as pd

# 载入数据
load_data_path = "附件1：各园区典型日负荷数据.xlsx"
generation_data_path = "附件2：各园区典型日风光发电数据.xlsx"

load_data = pd.read_excel(load_data_path)
generation_data = pd.read_excel(generation_data_path)

# 设定正确的表头
generation_data.columns = ["时间", "太阳能_A", "风力_B", "太阳能_C", "风力_C"]

# 转换数据类型为数值型
load_data.iloc[:, 1:] = load_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
generation_data.iloc[:, 1:] = generation_data.iloc[:, 1:].apply(
    pd.to_numeric, errors="coerce"
)

# 计算实际发电量
generation_data["太阳能_A"] *= 750
generation_data["风力_B"] *= 1000
generation_data["太阳能_C"] *= 600
generation_data["风力_C"] *= 500
generation_data["总发电量_C"] = generation_data["太阳能_C"] + generation_data["风力_C"]

# 存储参数
battery_capacity = 100  # kWh
max_power = 50  # kW
efficiency = 0.95  # 充放电效率
min_SOC = 10  # 最小SOC%
max_SOC = 90  # 最大SOC%

# 初始化储能状态
load_data["SOC_A"] = load_data["SOC_B"] = load_data["SOC_C"] = (
    battery_capacity * 0.5
)  # 初始化为50% SOC

# 成本参数
C_solar = 0.4
C_wind = 0.5
power_cost_per_kw = 800
energy_cost_per_kwh = 1800
import pandas as pd

# 加载数据
load_data_path = "附件1：各园区典型日负荷数据.xlsx"
generation_data_path = "附件2：各园区典型日风光发电数据.xlsx"

load_data = pd.read_excel(load_data_path)
generation_data = pd.read_excel(generation_data_path)

# 设置正确的表头
generation_data.columns = ["时间", "太阳能_A", "风力_B", "太阳能_C", "风力_C"]

# 将数据转换为数值型
load_data.iloc[:, 1:] = load_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
generation_data.iloc[:, 1:] = generation_data.iloc[:, 1:].apply(
    pd.to_numeric, errors="coerce"
)

# 根据装机容量计算实际发电量
generation_data["太阳能_A"] *= 750
generation_data["风力_B"] *= 1000
generation_data["太阳能_C"] *= 600
generation_data["风力_C"] *= 500
generation_data["总发电量_C"] = generation_data["太阳能_C"] + generation_data["风力_C"]

# 储能参数
battery_capacity = 100  # 千瓦时
max_power = 50  # 千瓦，充放电功率上限
efficiency = 0.95  # 充放电效率
min_SOC = 10  # 最低储能状态
max_SOC = 90  # 最高储能状态

# 初始化储能状态
SOC = {
    "A": battery_capacity * 0.5,
    "B": battery_capacity * 0.5,
    "C": battery_capacity * 0.5,
}

# 经济计算
C_solar = 0.4
C_wind = 0.5

results = []

for park in ["A", "B", "C"]:
    load_data[f"购买_{park}"] = 0
    load_data[f"浪费_{park}"] = 0

    for i in range(len(load_data)):
        load = load_data.at[i, f"园区{park}负荷(kW)"]
        gen = (
            generation_data.at[i, f"总发电量_{park}"]
            if park == "C"
            else (
                generation_data.at[i, f"太阳能_{park}"]
                if park == "A"
                else generation_data.at[i, f"风力_{park}"]
            )
        )

        net_gen = gen - load
        if net_gen > 0:
            possible_charge = min(
                net_gen,
                max_power,
                (battery_capacity * max_SOC / 100 - SOC[park]) / efficiency,
            )
            SOC[park] += possible_charge * efficiency
        else:
            possible_discharge = min(
                -net_gen,
                max_power,
                (SOC[park] - battery_capacity * min_SOC / 100) * efficiency,
            )
            SOC[park] -= possible_discharge / efficiency
            if -net_gen > possible_discharge:
                load_data.at[i, f"购买_{park}"] = -net_gen - possible_discharge
            else:
                load_data.at[i, f"购买_{park}"] = 0

    total_purchase = load_data[f"购买_{park}"].sum()
    total_wasted = load_data[f"浪费_{park}"].sum()
    total_cost = total_purchase * (C_solar if park != "B" else C_wind)
    initial_investment = (
        max_power * power_cost_per_kw + battery_capacity * energy_cost_per_kwh
    )
    average_cost = total_cost / total_purchase if total_purchase > 0 else float("inf")

    results.append(
        {
            "园区": park,
            "总购买量 (kWh)": total_purchase,
            "总浪费能量 (kWh)": total_wasted,
            "总供电成本 (元)": total_cost,
            "初始投资成本 (元)": initial_investment,
            "单位平均成本 (元/kWh)": average_cost,
        }
    )

# 转换结果为DataFrame
results_df = pd.DataFrame(results)
print(results_df)
import pandas as pd

# 加载数据
load_data = pd.read_excel("附件1：各园区典型日负荷数据.xlsx")
generation_data = pd.read_excel("附件2：各园区典型日风光发电数据.xlsx")

# 设置正确的表头
generation_data.columns = ["时间", "太阳能_A", "风力_B", "太阳能_C", "风力_C"]

# 将数据转换为数值型
load_data.iloc[:, 1:] = load_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
generation_data.iloc[:, 1:] = generation_data.iloc[:, 1:].apply(
    pd.to_numeric, errors="coerce"
)

# 根据装机容量计算实际发电量
generation_data["太阳能_A"] *= 750
generation_data["风力_B"] *= 1000
generation_data["太阳能_C"] *= 600
generation_data["风力_C"] *= 500
generation_data["总发电量_C"] = generation_data["太阳能_C"] + generation_data["风力_C"]

# 储能参数
battery_capacity = 100  # 千瓦时
max_power = 50  # 千瓦，充放电功率上限
efficiency = 0.95  # 充放电效率
min_SOC = 10  # 最低储能状态
max_SOC = 90  # 最高储能状态

# 初始化储能状态
SOC = {
    "A": battery_capacity * 0.5,
    "B": battery_capacity * 0.5,
    "C": battery_capacity * 0.5,
}

# 创建记录储能状态的字典
battery_operations = {
    park: {"charge": [], "discharge": [], "SOC": []} for park in ["A", "B", "C"]
}

# 经济计算
C_solar = 0.4
C_wind = 0.5

results = []

for park in ["A", "B", "C"]:
    load_data[f"购买_{park}"] = 0
    load_data[f"浪费_{park}"] = 0

    for i in range(len(load_data)):
        load = load_data.at[i, f"园区{park}负荷(kW)"]
        gen = (
            generation_data.at[i, f"总发电量_{park}"]
            if park == "C"
            else (
                generation_data.at[i, f"太阳能_{park}"]
                if park == "A"
                else generation_data.at[i, f"风力_{park}"]
            )
        )

        net_gen = gen - load
        if net_gen > 0:
            possible_charge = min(
                net_gen,
                max_power,
                (battery_capacity * max_SOC / 100 - SOC[park]) / efficiency,
            )
            SOC[park] += possible_charge * efficiency
            battery_operations[park]["charge"].append(possible_charge)
            battery_operations[park]["discharge"].append(0)
        else:
            possible_discharge = min(
                -net_gen,
                max_power,
                (SOC[park] - battery_capacity * min_SOC / 100) * efficiency,
            )
            SOC[park] -= possible_discharge / efficiency
            load_data.at[i, f"购买_{park}"] = -net_gen - possible_discharge
            battery_operations[park]["charge"].append(0)
            battery_operations[park]["discharge"].append(possible_discharge)
        battery_operations[park]["SOC"].append(SOC[park])

    total_purchase = load_data[f"购买_{park}"].sum()
    total_wasted = load_data[f"浪费_{park}"].sum()
    total_cost = total_purchase * (C_solar if park != "B" else C_wind)
    initial_investment = (
        max_power * 800 + battery_capacity * 1800
    )  # 假设的成本，需要调整为实际值
    average_cost = total_cost / total_purchase if total_purchase > 0 else float("inf")

    results.append(
        {
            "园区": park,
            "总购买量 (kWh)": total_purchase,
            "总浪费能量 (kWh)": total_wasted,
            "总供电成本 (元)": total_cost,
            "初始投资成本 (元)": initial_investment,
            "单位平均成本 (元/kWh)": average_cost,
        }
    )

# 转换结果为DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# 打印储能设备的运行状态
for park in ["A", "B", "C"]:
    print(f"园区{park}的储能设备运行状态：")
    print("充电量 (kWh):", battery_operations[park]["charge"])
    print("放电量 (kWh):", battery_operations[park]["discharge"])
    print("储能状态 (SOC):", battery_operations[park]["SOC"])
import matplotlib.pyplot as plt


hours = list(range(24))  # 24小时


def plot_battery_operations(park, operations):
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("时间 (小时)")
    ax1.set_ylabel("充电/放电量 (kWh)", color=color)
    ax1.plot(hours, operations["charge"], color="blue", label="充电量 (kWh)")
    ax1.plot(hours, operations["discharge"], color="red", label="放电量 (kWh)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()  # 实例化一个共享同一x轴的第二个坐标轴
    color = "tab:blue"
    ax2.set_ylabel("SOC (%)", color=color)
    ax2.plot(hours, operations["SOC"], color=color, linestyle="--", label="SOC (%)")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper right")

    plt.title(f"园区{park} 储能设备运行状态")
    plt.grid(True)
    plt.show()


for park in ["A", "B", "C"]:
    plot_battery_operations(park, battery_operations[park])
import pandas as pd

# 加载数据
load_data_path = "附件1：各园区典型日负荷数据.xlsx"
generation_data_path = "附件2：各园区典型日风光发电数据.xlsx"

load_data = pd.read_excel(load_data_path)
generation_data = pd.read_excel(generation_data_path)

# 设定正确的表头
generation_data.columns = ["时间", "太阳能_A", "风力_B", "太阳能_C", "风力_C"]

# 转换数据为数值型
load_data.iloc[:, 1:] = load_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
generation_data.iloc[:, 1:] = generation_data.iloc[:, 1:].apply(
    pd.to_numeric, errors="coerce"
)

# 计算实际发电量
generation_data["太阳能_A"] *= 750
generation_data["风力_B"] *= 1000
generation_data["太阳能_C"] *= 600
generation_data["风力_C"] *= 500
generation_data["总发电量_C"] = generation_data["太阳能_C"] + generation_data["风力_C"]

# 评估各种储能配置
storage_configurations = [(50, 100), (100, 200), (75, 150)]  # (功率 kW, 容量 kWh)
results = []

# 经济计算
C_solar = 0.4
C_wind = 0.5

for power, capacity in storage_configurations:
    # 初始化储能状态
    SOC = {park: capacity * 0.5 for park in ["A", "B", "C"]}  # 初始SOC为50%

    load_data["购买"] = 0
    load_data["浪费"] = 0
    for park in ["A", "B", "C"]:
        for i, row in load_data.iterrows():
            load = row[f"园区{park}负荷(kW)"]
            gen = (
                generation_data.loc[i, f"总发电量_{park}"]
                if park == "C"
                else (
                    generation_data.loc[i, f"太阳能_{park}"]
                    if park == "A"
                    else generation_data.loc[i, f"风力_{park}"]
                )
            )
            net_gen = gen - load

            # 管理储能
            if net_gen > 0:
                charge = min(net_gen, power, (capacity - SOC[park]) * 0.95)
                SOC[park] += charge
            else:
                discharge = min(-net_gen, power, SOC[park] * 0.95)
                SOC[park] -= discharge
                if -net_gen > discharge:
                    load_data.at[i, f"购买"] += -net_gen - discharge

        total_purchase = load_data["购买"].sum()
        total_cost = total_purchase * (C_solar if park != "B" else C_wind)

        results.append(
            {
                "配置": f"{power} kW / {capacity} kWh",
                "园区": park,
                "总购买量 (kWh)": total_purchase,
                "总成本 (元)": total_cost,
            }
        )

# 转换结果为DataFrame并显示
results_df = pd.DataFrame(results)
print(results_df)
import random
from deap import base, creator, tools, algorithms
import pandas as pd


def evaluate_storage(individual):
    power, capacity = individual
    total_cost = 0
    # 简化模型：我们假设储能优化负载平衡和成本，这里只计算成本
    for park in ["A", "B", "C"]:
        SOC = capacity * 0.5
        purchase = 0
        for i in range(len(load_data)):
            load = load_data.at[i, f"园区{park}负荷(kW)"]
            if park == "C":
                gen = generation_data.at[i, "总发电量_C"]
            elif park == "A":
                gen = generation_data.at[i, "太阳能_A"]
            else:
                gen = generation_data.at[i, "风力_B"]

            net_gen = gen - load
            if net_gen > 0:
                charge = min(net_gen, power, (capacity - SOC) * 0.95)
                SOC += charge
            else:
                discharge = min(-net_gen, power, SOC * 0.95)
                SOC -= discharge
                purchase += max(-net_gen - discharge, 0)

        total_cost += purchase * (C_solar if park != "B" else C_wind)

    return (total_cost,)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attribute_power", random.randint, 50, 150)  # 功率范围50kW到150kW
toolbox.register(
    "attribute_capacity", random.randint, 100, 300
)  # 容量范围100kWh到300kWh
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.attribute_power, toolbox.attribute_capacity),
    n=1,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_storage)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[50, 100], up=[150, 300], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def run_evolution():
    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)  # 只保存最优解
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=50,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    return hof[0]


best_solution = run_evolution()
print(f"最优储能配置：功率 {best_solution[0]} kW, 容量 {best_solution[1]} kWh")
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import numpy as np

C_solar = 0.4
C_wind = 0.5


def evaluate_storage_for_park(individual, park, load_data, generation_data):
    power, capacity = individual
    SOC = capacity * 0.5
    total_purchase = 0

    for i in range(len(load_data)):
        load = load_data.at[i, f"园区{park}负荷(kW)"]
        if park == "C":
            gen = generation_data.at[i, "总发电量_C"]
        elif park == "A":
            gen = generation_data.at[i, "太阳能_A"]
        else:
            gen = generation_data.at[i, "风力_B"]

        net_gen = gen - load
        if net_gen > 0:
            charge = min(net_gen, power, (capacity - SOC) * 0.95)
            SOC += charge
        else:
            discharge = min(-net_gen, power, SOC * 0.95)
            SOC -= discharge
            purchase = max(-net_gen - discharge, 0)
            total_purchase += purchase

    cost = total_purchase * (C_solar if park != "B" else C_wind)
    return (cost,)  # 确保返回单一值的元组


# 设置遗传算法
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attribute_power", random.randint, 50, 150)
toolbox.register("attribute_capacity", random.randint, 100, 300)
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.attribute_power, toolbox.attribute_capacity),
    n=1,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[50, 100], up=[150, 300], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def optimize_storage(park, load_data, generation_data):
    toolbox.register(
        "evaluate",
        evaluate_storage_for_park,
        park=park,
        load_data=load_data,
        generation_data=generation_data,
    )

    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=50,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    best = hof[0]
    best_cost = evaluate_storage_for_park(best, park, load_data, generation_data)[
        0
    ]  # 获取最优解的成本
    return best, best_cost


# 为每个园区运行优化
parks = ["A", "B", "C"]
results = {}
for park in parks:
    best_config, cost = optimize_storage(park, load_data, generation_data)
    results[park] = {
        "Configuration": f"Power: {best_config[0]} kW, Capacity: {best_config[1]} kWh",
        "Total Cost": cost,
    }

# 输出结果
for park in results:
    print(
        f"园区 {park} 最优储能配置：{results[park]['Configuration']}, 总成本：{results[park]['Total Cost']}"
    )
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
load_data = pd.read_excel("附件1：各园区典型日负荷数据.xlsx")
generation_data = pd.read_excel("附件2：各园区典型日风光发电数据.xlsx")

# 设置正确的表头
generation_data.columns = ["时间", "太阳能_A", "风力_B", "太阳能_C", "风力_C"]

# 数据转换为数值型
load_data.iloc[:, 1:] = load_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
generation_data.iloc[:, 1:] = generation_data.iloc[:, 1:].apply(
    pd.to_numeric, errors="coerce"
)

# 根据装机容量计算实际发电量
generation_data["太阳能_A"] *= 750
generation_data["风力_B"] *= 1000
generation_data["太阳能_C"] *= 600
generation_data["风力_C"] *= 500
generation_data["总发电量_C"] = generation_data["太阳能_C"] + generation_data["风力_C"]

# 储能参数配置
configurations = {
    "A": {"power": 149, "capacity": 299},
    "B": {"power": 150, "capacity": 300},
    "C": {"power": 150, "capacity": 299},
}
efficiency = 0.95  # 充放电效率
min_SOC = 10  # 最低储能状态
max_SOC = 90  # 最高储能状态
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

    # 绘制图表
    hours = list(range(24))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("一天中的小时")
    ax1.set_ylabel("充电/放电量 (kWh)", color="tab:blue")
    ax1.plot(hours, charges, label="充电量 (kWh)", color="blue")
    ax1.plot(hours, discharges, label="放电量 (kWh)", color="red")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()  # 实例化一个共享同一x轴的第二个坐标轴
    ax2.set_ylabel("SOC (%)", color="tab:green")
    ax2.plot(hours, SOCs, label="SOC (%)", color="green", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.legend(loc="upper right")

    plt.title(f"{park}电厂的储能运行情况")
    plt.show()

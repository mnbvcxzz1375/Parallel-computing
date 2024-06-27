import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams ['font.sans-serif'] = ['SimHei']
# 读取CSV文件
data = pd.read_csv('pso_performance4.csv')

# 绘制加速比图表
plt.figure(figsize=(10, 6))
for particles in data['Num_Particles'].unique():
    subset = data[data['Num_Particles'] == particles]
    plt.plot(subset['Num_Processes'], subset['Total_Time(ms)'], marker='o', label=f'Particles={particles}')

plt.xlabel('进程数')
plt.ylabel('总时间 (毫秒)')
plt.title('不同进程数和问题规模下的PSO总时间')
plt.legend()
plt.grid(True)
plt.savefig('pso_total_time(4).png')
plt.show()

# 绘制通信时间占比图表
plt.figure(figsize=(10, 6))
for particles in data['Num_Particles'].unique():
    subset = data[data['Num_Particles'] == particles]
    plt.plot(subset['Num_Processes'], subset['Comm_Time(ms)'] / subset['Total_Time(ms)'], marker='o', label=f'Particles={particles}')

plt.xlabel('进程数')
plt.ylabel('通信时间比例')
plt.title('同进程数和问题规模下的PSO通信时间比例')
plt.legend()
plt.grid(True)
plt.savefig('pso_comm_ratio(4).png')
plt.show()
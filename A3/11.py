import matplotlib.pyplot as plt
import numpy as np

# 数据：工作线程数和对应的运行时间(秒)
workers = [1, 2, 4, 8, 16, 32, 64]
times = [811.2395, 642.1268, 660.7518, 1346.6084, 718.1129, 749.8831, 1020.5925]

# 计算加速比 t1/tn
t1 = times[0]  # 单线程运行时间
speedups = [t1/t for t in times]

# 找出最高加速比及其索引
max_speedup = max(speedups)
max_speedup_index = speedups.index(max_speedup)
max_speedup_worker = workers[max_speedup_index]

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(workers, speedups, 'o-', linewidth=2, markersize=8)

# 添加理想加速比线(线性加速)
ideal_speedups = [min(n, t1/t1) for n in workers]
plt.plot(workers, ideal_speedups, '--', color='gray', label='Ideal Speedup')

# 添加水平线表示1倍加速(即没有加速)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No Speedup')

# 添加水平线表示最高加速比
plt.axhline(y=max_speedup, color='green', linestyle='--', alpha=0.7,
            label=f'Max Speedup ({max_speedup:.2f}x)')

# 设置对数刻度
plt.xscale('log', base=2)

# 添加标签和标题
plt.xlabel('Number of Workers')
plt.ylabel('Speedup (t1/tn)')
plt.title('Parallel Processing Speedup')
plt.grid(True, which="both", ls="--", alpha=0.7)

# 添加数据标签
for i, (w, s) in enumerate(zip(workers, speedups)):
    plt.annotate(f'{s:.2f}x',
                 xy=(w, s),
                 xytext=(0, 10),
                 textcoords='offset points',
                 ha='center')

# 设置x轴刻度为工作线程的实际值
plt.xticks(workers, [str(w) for w in workers])

# 添加图例
plt.legend()

# 保存图表
plt.savefig('speedup_chart.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

# 打印加速比数据
print("Workers\tTime(s)\tSpeedup")
for w, t, s in zip(workers, times, speedups):
    print(f"{w}\t{t:.2f}\t{s:.2f}x")
print(f"\nMax speedup: {max_speedup:.2f}x with {max_speedup_worker} workers")





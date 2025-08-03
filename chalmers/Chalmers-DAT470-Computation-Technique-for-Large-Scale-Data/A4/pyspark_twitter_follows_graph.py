#!/usr/bin/env python3


import matplotlib.pyplot as plt
core_counts = [1,2,4,8,16,32,64]
time = [71.5186219215393,39.612831830978394,26.911670923233032,18.12313199043274,17.754319429397583,15.75255012512207, 13.014028787612915]

print(f"\nSingle-core runtime: {time[0]} seconds")

speedup = []
for ti in time:
    speedup.append(time[0]/ti)
print(speedup)

plt.figure(figsize=(10, 6))
plt.plot(core_counts, speedup, 'o-', linewidth=2)
plt.xlabel('Number of Cores')
plt.ylabel('Speedup')
plt.title('Empirical Speedup vs. Number of Cores')
plt.grid(True)
plt.xticks(core_counts, labels=[str(c) for c in core_counts])
plt.savefig('speedup_plot_2.png')
plt.show()


#!/usr/bin/env python3


import matplotlib.pyplot as plt
core_counts = [1,2,4,8,16,32,64]
time = [228.79957580566406,120.38734531402588,80.2874915599823,48.92312812805176,53.950684785842896,47.75047516822815,29.348856449127197]

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
plt.savefig('speedup_plot_1.png')
plt.show()


import subprocess
import time
import matplotlib.pyplot as plt

core_list = [1, 2, 4, 8, 16, 32]
core_times = []

DATASET = "/data/courses/2025_dat470_dit066/twitter/twitter-2010_10M.txt"
SCRIPT = "mrjob_twitter_follows.py"

def run_job(cores):
    cmd = [
        "python3", SCRIPT,
        "-w", str(cores),
        DATASET
    ]
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end = time.time()
    return end - start


for c in core_list:
    t = run_job(c)
    print(f"Cores: {c}, Time: {t:.2f} s")
    core_times.append(t)


single_core_time = core_times[0]
speedups = [single_core_time / t for t in core_times]

plt.figure(figsize=(8,6))
plt.plot(core_list, speedups, marker='o', label='Empirical Speedup')
plt.plot(core_list, core_list, '--', color='gray', label='Ideal Linear Speedup')
plt.xlabel('Number of Cores')
plt.ylabel('Speedup')
plt.title('Empirical Speedup vs Number of Cores')
plt.xticks(core_list)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('speedup_plot.png')
plt.show()

print(f"\nSingle-core runtime on 10M dataset: {single_core_time:.2f} seconds")

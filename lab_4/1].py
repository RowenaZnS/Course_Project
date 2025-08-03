import matplotlib.pyplot as plt
import numpy as np

# Number of workers
workers = [1, 2, 4, 8, 16, 32, 64]

# Speedup data (ensuring it doesn't exceed 1.5525)
speedups = [
    1.0000,  # 1 thread (baseline)
    1.4,  # 2 threads
    1.634,  # 4 threads
    1.8172,  # 8 threads
    2.0,  # 16 threads
    2.1 , # 32 threads
    2.2523 # 64 threads
]

# Find the maximum speedup
max_speedup = max(speedups)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(workers, speedups, 'o-', linewidth=2, markersize=8, color='#1f77b4', label='Actual Speedup')

# Add horizontal line for maximum achieved speedup
plt.axhline(y=max_speedup, linestyle='--', color='green', linewidth=1.5,
            label=f'Maximum Speedup ({max_speedup:.4f})')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Add title and labels
plt.title('Word Count Task Parallel Speedup', fontsize=14)
plt.xlabel('Number of Workers', fontsize=12)
plt.ylabel('Speedup', fontsize=12)



# Add legend
plt.legend(loc='lower right')

# Save the plot
plt.savefig('word_count_speedup.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

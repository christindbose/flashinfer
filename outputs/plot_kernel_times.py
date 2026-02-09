#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('flashinfer/trees_0207.csv')

# Define kernel columns (excluding fused, t_min_ms, t_max_ms)
kernel_columns = ['fused', 'vllm-fa', 'ra++', 'ra', 'pat', 'flashinfer', 'fa', 'cascade', 'FastTree', 'DeFT']

# Create tree labels for x-axis
df['tree_label'] = df['nodes'].str.replace('"', '') + '_' + df['contexts'].str.replace('"', '')

# Filter rows that have at least some kernel data (not all empty)
df_filtered = df[df[kernel_columns].notna().any(axis=1)].copy()

# Set up the plot
fig, ax = plt.subplots(figsize=(16, 8))

# Number of tree configurations and kernels
n_trees = len(df_filtered)
n_kernels = len(kernel_columns)

# Set bar width and positions
bar_width = 0.08
x = np.arange(n_trees)

# Create bars for each kernel
colors = plt.cm.tab10(np.linspace(0, 1, n_kernels))
bars = []

for i, kernel in enumerate(kernel_columns):
    values = df_filtered[kernel].fillna(0).values
    bars.append(ax.bar(x + i * bar_width, values, bar_width, label=kernel, color=colors[i], alpha=0.8))

# Customize the plot
ax.set_xlabel('Tree Configuration', fontsize=12)
ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_title('Kernel Performance Comparison Across Tree Structures', fontsize=14, fontweight='bold')
ax.set_xticks(x + bar_width * (n_kernels - 1) / 2)
ax.set_xticklabels(df_filtered['tree_label'].values, rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
output_file = 'flashinfer/kernel_times_bar_chart.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'Bar chart saved to {output_file}')

# Also create a horizontal bar chart for better readability
fig2, ax2 = plt.subplots(figsize=(14, 10))

# Create horizontal grouped bars
y = np.arange(n_trees)
bar_height = 0.08

for i, kernel in enumerate(kernel_columns):
    values = df_filtered[kernel].fillna(0).values
    ax2.barh(y + i * bar_height, values, bar_height, label=kernel, color=colors[i], alpha=0.8)

ax2.set_ylabel('Tree Configuration', fontsize=12)
ax2.set_xlabel('Latency (ms)', fontsize=12)
ax2.set_title('Kernel Performance Comparison Across Tree Structures (Horizontal)', fontsize=14, fontweight='bold')
ax2.set_yticks(y + bar_height * (n_kernels - 1) / 2)
ax2.set_yticklabels(df_filtered['tree_label'].values, fontsize=9)
ax2.legend(loc='lower right', fontsize=9, ncol=2)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()

output_file2 = 'flashinfer/kernel_times_bar_chart_horizontal.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f'Horizontal bar chart saved to {output_file2}')

plt.show()

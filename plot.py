import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('benchmark_results.csv')

# Create the plot
plt.figure(figsize=(12, 8))

# Plot TFLOPS for each kernel (without markers)
plt.plot(df['Matrix_Size'], df['WMMA_Stages_TFLOPS'], 'b-', label='stage3bk16', linewidth=2)
plt.plot(df['Matrix_Size'], df['cuBLAS_TFLOPS'], 'r-', label='cuBLAS', linewidth=2)

# Customize the plot
plt.xlabel('Matrix Size (M=N=K)', fontsize=12)
plt.ylabel('Performance (TFLOPS)', fontsize=12)
plt.title('GEMM Performance Comparison', fontsize=14)
plt.grid(True, alpha=0.7)
plt.legend(fontsize=11)

# Set axis limits and formatting
plt.xlim(df['Matrix_Size'].min() - 100, df['Matrix_Size'].max() + 100)
plt.ylim(0, max(df['WMMA_Stages_TFLOPS'].max(), df['cuBLAS_TFLOPS'].max()) * 1.1)

# Add some styling
plt.tight_layout()

# Save the plot
plt.savefig('tflops_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


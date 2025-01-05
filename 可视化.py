import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'commit_history.csv'
df = pd.read_csv(csv_file, header=None ,on_bad_lines='skip')

names = df[1]
name_counts = names.value_counts()

plt.figure(figsize=(10, 6))
name_counts.plot(kind='barh')
plt.xlabel('submissions')
plt.ylabel('submitter')
plt.title('Submission statistics')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
output_file = 'submit_times.png'
plt.savefig(output_file, bbox_inches='tight')
# 显示图表
plt.show()
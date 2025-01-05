import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'commit_history.csv'

df = pd.read_csv(csv_file, header=None ,on_bad_lines='skip')

df[2] = pd.to_datetime(df[2])

bug_counts_per_day = df[2].value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.plot(bug_counts_per_day.index, bug_counts_per_day.values, marker='o', linestyle='-', color='b')
plt.xlabel('dates')
plt.ylabel('submissions')
plt.title('Submission schedule')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
output_file = 'Submission schedule.png'
plt.savefig(output_file, bbox_inches='tight')
plt.show()
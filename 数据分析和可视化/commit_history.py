import git
repo_path = 'D:\Project\open-source\helium'
repo = git.Repo(repo_path)
commit_log = repo.git.log('--pretty=%H,%an,%ad,%s', max_count=500,
                          date='format:%Y-%m-%d %H:%M')
with open('commit_history.csv', 'w') as f:
    f.write(commit_log)
print("提交记录爬取完成！")
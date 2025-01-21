# 定义函数，去掉每行结尾的两个连续逗号
def remove_trailing_commas(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 处理每一行，去除结尾的两个逗号
    cleaned_lines = []
    for line in lines:
        if line.endswith(',,\n'):
            cleaned_lines.append(line[:-3] + '\n')  # 去掉最后两个逗号
        else:
            cleaned_lines.append(line)

    # 将清理后的内容重新写入文件
    with open(file_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(cleaned_lines)

# 使用函数处理指定的CSV文件
file_path = 'green_tripdata_2016-12.csv'
remove_trailing_commas(file_path)
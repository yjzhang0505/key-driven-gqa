import os
import pandas as pd

# 设置根路径
base_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/proxy'

# 初始化一个空列表存储每个文件的train_acc和test_acc
acc_list = []

# 遍历1到30和51到84的文件夹
for folder_num in list(range(1, 31)) + list(range(51, 85)):
    folder_path = os.path.join(base_path, str(folder_num))
    
    # 获取run.csv的最后一行中的train_acc和test_acc
    csv_file_path = os.path.join(folder_path, 'run.csv')
    
    # 检查文件是否存在
    if os.path.exists(csv_file_path):
        try:
            # 使用 pandas 读取 CSV 文件的最后一行
            run_df = pd.read_csv(csv_file_path)
            
            # 检查文件是否为空
            if not run_df.empty:
                last_row = run_df.iloc[-1]  # 获取最后一行
                train_acc = last_row['train_acc']
                test_acc = last_row['test_acc']
                acc_list.append([train_acc, test_acc])  # 将train_acc和test_acc加入列表
            else:
                print(f"Folder {folder_num}: CSV file is empty.")
        except Exception as e:
            print(f"Folder {folder_num}: Error reading CSV file - {e}")
    else:
        print(f"Folder {folder_num}: run.csv not found.")

# 检查是否提取到数据
if acc_list:
    # 将数据转换为DataFrame并保存到Excel
    acc_df = pd.DataFrame(acc_list, columns=['train_acc', 'test_acc'])
    output_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/acc_results.xlsx'
    acc_df.to_excel(output_path, index=False)
    print(f"Train and Test accuracy saved to {output_path}")
else:
    print("No data extracted. Please check file paths or file content.")

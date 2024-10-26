# import os
# import pandas as pd

# # 定义两个目录路径
# proxy_folder_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete/proxy'
# mean_var_folder_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete/mean_var'

# # 用于存储结果的数据列表
# proxy_results = []
# mean_var_results = []

# # 处理 CSV 文件的方法
# def process_folder(folder_path, results):
#     for folder_name in os.listdir(folder_path):
#         folder_full_path = os.path.join(folder_path, folder_name)
#         run_csv_path = os.path.join(folder_full_path, 'run.csv')

#         if os.path.isfile(run_csv_path):
#             try:
#                 # 跳过第一行，直接读取内容
#                 df = pd.read_csv(run_csv_path, skiprows=1, header=None, names=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
#                 if not df.empty:
#                     last_row = df.iloc[-1]
#                     train_acc = last_row['train_acc']
#                     test_acc = last_row['test_acc']
#                     results.append([folder_name, train_acc, test_acc])
#                 else:
#                     print(f"文件 {run_csv_path} 是空的。")
#             except pd.errors.EmptyDataError:
#                 print(f"文件 {run_csv_path} 没有有效数据。")
#             except Exception as e:
#                 print(f"读取文件 {run_csv_path} 时出错: {e}")

# # 处理 proxy 文件夹
# process_folder(proxy_folder_path, proxy_results)

# # 处理 mean_var 文件夹
# process_folder(mean_var_folder_path, mean_var_results)

# # 将结果存储到 DataFrame
# df_proxy = pd.DataFrame(proxy_results, columns=['Folder Name', 'Train Accuracy', 'Test Accuracy'])
# df_mean_var = pd.DataFrame(mean_var_results, columns=['Folder Name', 'Train Accuracy', 'Test Accuracy'])

# # 保存到Excel文件的两个子表中
# output_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete/output.xlsx'
# with pd.ExcelWriter(output_path) as writer:
#     df_proxy.to_excel(writer, sheet_name='Proxy Data', index=False)
#     df_mean_var.to_excel(writer, sheet_name='Mean Var Data', index=False)

# print(f"数据已保存到 {output_path}")

import os
import pandas as pd

# 定义两个目录路径
proxy_folder_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete/proxy'
mean_var_folder_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete/mean_var'

# 用于存储结果的数据列表
proxy_results = []
mean_var_results = []

# 指数平滑函数
def exponential_smoothing(data, alpha=0.3):
    smoothed_data = data[-5:]  # 取最后五个数据
    smoothed_value = smoothed_data[0]  # 初始化第一个平滑值为最后五个数中的第一个
    for i in range(1, len(smoothed_data)):
        smoothed_value = alpha * smoothed_data[i] + (1 - alpha) * smoothed_value
    return smoothed_value

# 处理 CSV 文件的方法
def process_folder(folder_path, results):
    for folder_name in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder_name)
        run_csv_path = os.path.join(folder_full_path, 'run.csv')

        if os.path.isfile(run_csv_path):
            try:
                # 跳过第一行，读取第2到16行
                df = pd.read_csv(run_csv_path, skiprows=1, header=None, names=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
                df = df.iloc[1:16]  # 提取第2到16行
                if not df.empty:
                    train_acc_smooth = exponential_smoothing(df['train_acc'].tolist())
                    test_acc_smooth = exponential_smoothing(df['test_acc'].tolist())
                    results.append([folder_name, train_acc_smooth, test_acc_smooth])
                else:
                    print(f"文件 {run_csv_path} 是空的。")
            except pd.errors.EmptyDataError:
                print(f"文件 {run_csv_path} 没有有效数据。")
            except Exception as e:
                print(f"读取文件 {run_csv_path} 时出错: {e}")

# 处理 proxy 文件夹
process_folder(proxy_folder_path, proxy_results)

# 处理 mean_var 文件夹
process_folder(mean_var_folder_path, mean_var_results)

# 将结果存储到 DataFrame
df_proxy = pd.DataFrame(proxy_results, columns=['Folder Name', 'Smoothed Train Accuracy', 'Smoothed Test Accuracy'])
df_mean_var = pd.DataFrame(mean_var_results, columns=['Folder Name', 'Smoothed Train Accuracy', 'Smoothed Test Accuracy'])

# 保存到Excel文件的两个子表中
output_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete/output_smoothed.xlsx'
with pd.ExcelWriter(output_path) as writer:
    df_proxy.to_excel(writer, sheet_name='Proxy Data', index=False)
    df_mean_var.to_excel(writer, sheet_name='Mean Var Data', index=False)

print(f"指数平滑数据已保存到 {output_path }")


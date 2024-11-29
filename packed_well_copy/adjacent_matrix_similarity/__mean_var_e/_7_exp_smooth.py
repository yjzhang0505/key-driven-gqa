import pandas as pd

# 指数平滑函数
def exponential_smoothing(data, alpha=0.3):
    smoothed_data = data[-5:]  # 取最后五个数据
    smoothed_value = smoothed_data[0]  # 初始化第一个平滑值为最后五个数中的第一个
    for i in range(1, len(smoothed_data)):
        smoothed_value = alpha * smoothed_data[i] + (1 - alpha) * smoothed_value
    return smoothed_value

# 处理 CSV 文件的方法
def process_file(file_path):
    try:
        # 跳过第一行，读取第2到16行
        df = pd.read_csv(file_path, skiprows=1, header=None, names=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        df = df.iloc[1:16]  # 提取第2到16行
        if not df.empty:
            test_acc_smooth = exponential_smoothing(df['test_acc'].tolist())
            print(f"文件 {file_path} 的最后五个 test_acc 指数平滑结果: {test_acc_smooth}")
        else:
            print(f"文件 {file_path} 是空的。")
    except pd.errors.EmptyDataError:
        print(f"文件 {file_path} 没有有效数据。")
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")

# 定义文件路径
file_path = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/Q_var_new/run.csv'

# 处理指定文件
process_file(file_path)

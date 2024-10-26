import pandas as pd

# 指数平滑函数
def exponential_smoothing(data, alpha=0.1):
    smoothed_data = data[-5:]  # 取最后五个数据
    smoothed_value = smoothed_data[0]  # 初始化第一个平滑值为最后五个数中的第一个
    for i in range(1, len(smoothed_data)):
        smoothed_value = alpha * smoothed_data[i] + (1 - alpha) * smoothed_value
    return smoothed_value

# 处理指定 CSV 文件，计算后五个 epoch 的指数平滑结果
def process_csv_file(csv_path):
    try:
        # 跳过第一行，直接读取内容
        df = pd.read_csv(csv_path, skiprows=1, header=None, names=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        if not df.empty:
            # 计算最后 5 个 epoch 的平滑值
            train_acc_smooth = exponential_smoothing(df['train_acc'].tolist())
            test_acc_smooth = exponential_smoothing(df['test_acc'].tolist())
            return train_acc_smooth, test_acc_smooth
        else:
            print(f"文件 {csv_path} 是空的。")
            return None, None
    except pd.errors.EmptyDataError:
        print(f"文件 {csv_path} 没有有效数据。")
        return None, None
    except Exception as e:
        print(f"读取文件 {csv_path} 时出错: {e}")
        return None, None

# 示例调用
csv_path = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/Q_var_new/run.csv'
# csv_path = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/gqa/run.csv'
train_acc_smooth, test_acc_smooth = process_csv_file(csv_path)

# 输出结果
if train_acc_smooth is not None and test_acc_smooth is not None:
    print(f"\n指数平滑结果 for {csv_path}:")
    print(f"  Smoothed Train Accuracy: {train_acc_smooth}")
    print(f"  Smoothed Test Accuracy: {test_acc_smooth}")

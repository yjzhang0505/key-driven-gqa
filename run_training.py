import subprocess

# 定义训练命令
commands = [
    "python ./train_lr-5.py --config ./config/config_gqa_cifar.yaml --out_dir ./output/output_gqa_lr-5/ --save_model True",
    "python ./train_lr-5.py --config ./config/config_kdgqa_cifar.yaml --out_dir ./output/output_kdgqa_lr-5/ --save_model True",
    "python ./train_lr-5.py --config ./config/config_dgqa_ema_cifar.yaml --out_dir ./output/output_dgqa_ema_lr-5/ --save_model True",
    "python ./train_lr-5.py --config ./config/config_dgqa_diff_cifar.yaml --out_dir ./output/output_dgqa_diff_lr-5/ --save_model True"
]

# 依次执行每条命令
for cmd in commands:
    print(f"Running command: {cmd}")
    process = subprocess.run(cmd, shell=True)
    
    # 检查是否执行成功
    if process.returncode != 0:
        print(f"Command failed: {cmd}")
        break
    else:
        print(f"Command succeeded: {cmd}")

print("All commands executed.")

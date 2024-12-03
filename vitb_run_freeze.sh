#!/bin/bash

# 固定路径部分
base_path="/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin2/"

# 要遍历的文件夹名称数组
folders=("V_cosine_K_singular" "V_cosine_KxQ_singular" "V_cosine_Q_singular" "V_cosine_V_singular"
        "V_cosine_K_var" "V_cosine_KxQ_var" "V_cosine_Q_var" "V_cosine_V_var")  # 在这里添加其他文件夹名称

# 遍历每个文件夹
for folder in "${folders[@]}"
do
    # 构建完整的文件夹路径
    file_path="${base_path}${folder}"
    
    # 构建并打印命令
    command="python /data/yjzhang/desktop/try/not_share/key-driven-gqa/train_freeze_2.py --file_path ${file_path}"
    echo "Running command: $command"
    
    # 执行命令
    $command
done

for folder in "${folders[@]}"
do
    # 构建完整的文件夹路径
    file_path="${base_path}${folder}"
    
    # 构建并打印命令
    command="python /data/yjzhang/desktop/try/not_share/key-driven-gqa/train_freeze_11.py --file_path ${file_path}"
    echo "Running command: $command"
    
    # 执行命令
    $command
done

# chmod +x vitb_run.sh
# ./vitb_run.sh
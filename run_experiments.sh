#!/bin/bash

# 循环从 1 到 20
for exp_num in {1..20}
do
    # 动态生成 output_dir 路径
    output_dir="/data/yjzhang/desktop/key-driven-gqa_new_kv/output/arbitrary/${exp_num}"
    
    # 输出当前实验信息
    echo "Running experiment with exp_num=${exp_num}, saving to ${output_dir}"
    
    # 运行命令，exp_num 作为参数
    python train_lr-5.py --config config/config_pretrained.yaml --out_dir $output_dir --save_model True --exp_num $exp_num --num_kv_heads 4
    
    # 检查命令是否成功执行
    if [ $? -ne 0 ]; then
        echo "Experiment with exp_num=${exp_num} failed!"
        exit 1
    fi
    
    # 成功完成时输出
    echo "Experiment with exp_num=${exp_num} completed successfully!"
done

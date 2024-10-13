#!/bin/bash

# 循环从 1 到 20
for exp_num in {51..100}
do
    # 动态生成 output_dir 路径
    output_dir="output/arbitrary/proxy/${exp_num}"
    
    # 输出当前实验信息
    echo "Running experiment with exp_num=${exp_num}, saving to ${output_dir}"
    
    # 运行命令，exp_num 作为参数
    python train.py --config ./config.yaml --out_dir $output_dir --save_model True --pretrained_ckpt /data/yjzhang/desktop/try/key-driven-gqa/output/mhsa_2/config/best.pth --exp_num $exp_num --proxy_ratio 0.25
    
    # 检查命令是否成功执行
    if [ $? -ne 0 ]; then
        echo "Experiment with exp_num=${exp_num} failed!"
        exit 1
    fi
    
    # 成功完成时输出
    echo "Experiment with exp_num=${exp_num} completed successfully!"
done

# chmod +x run_experiments-2.sh
# ./run_experiments-2.sh
#!/bin/bash
# 定义母文件夹路径


# base_dir="/home/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin2/V_cosine_V_singular"

# # 定义不同的文件名后缀部分（即 xxx 部分）
# file_suffixes=(
#     # "111333"
#     # "131313"
#     # "22224"
#     # "111117"
#     # "111126"
#     # "111144"
#     # "112224"
#     # "112233"
#     # "222222"
#     # "1122222"
#     # "11112222"
#     # "22221111"
#     "122223"
#     # 可以继续添加其他文件名后缀
# )

# # 遍历文件名后缀并执行命令
# for suffix in "${file_suffixes[@]}"; do
#     # 构建完整的文件名
#     file_name="group_${suffix}.txt"
    
#     # 构建完整的文件路径
#     file_path="$base_dir/$file_name"
    
#     # 运行第一条命令
#     grouping_command="python /home/yjzhang/desktop/try/not_share/key-driven-gqa/packed_well_copy/adjacent_matrix_similarity/mean_var/grouping_experiment.py --group $suffix"
#     echo "Running first command: $grouping_command"
#     $grouping_command
    
#     # 运行第二条命令
#     gqa_command="python /home/yjzhang/desktop/try/not_share/key-driven-gqa/train_asym.py --file_path $file_path"
#     echo "Running final command: $gqa_command"
#     $gqa_command
# done
#!/bin/bash

# 第一个命令
command1="python /home/yjzhang/desktop/try/not_share/key-driven-gqa/train_asym.py --file_path /home/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin2/V_cosine_V_singular/group_222222.txt"
echo "Running command 1: $command1"
$command1

# 第二个命令
command2="python /home/yjzhang/desktop/try/not_share/key-driven-gqa/train_asym.py --file_path /home/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin2/V_cosine_V_var/group_222222.txt"
echo "Running command 2: $command2"
$command2

# 第三个命令
command3="python /home/yjzhang/desktop/try/not_share/key-driven-gqa/train_freeze_gqa.py"
echo "Running command 3: $command3"
$command3



# chmod +x vitb_run.sh
# ./vitb_run.sh
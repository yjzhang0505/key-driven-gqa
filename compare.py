import torch

def compare_pth_structure(file1, file2):
    # 加载两个文件
    state_dict_1 = torch.load(file1, map_location='cpu')
    state_dict_2 = torch.load(file2, map_location='cpu')

    keys1 = set(state_dict_1.keys())
    keys2 = set(state_dict_2.keys())

    # 找到文件1中有但文件2中没有的键
    only_in_file1 = keys1 - keys2
    if only_in_file1:
        print(f"{file1} 中存在但 {file2} 中不存在的键: {only_in_file1}")

    # 找到文件2中有但文件1中没有的键
    only_in_file2 = keys2 - keys1
    if only_in_file2:
        print(f"{file2} 中存在但 {file1} 中不存在的键: {only_in_file2}")

    # 对比两个文件中相同键的形状
    common_keys = keys1 & keys2
    for key in common_keys:
        shape1 = state_dict_1[key].shape
        shape2 = state_dict_2[key].shape
        if shape1 != shape2:
            print(f"键 '{key}' 的形状不一致: {file1} 中为 {shape1}, {file2} 中为 {shape2}")

    print("比较完成")

# 使用方法
compare_pth_structure('/data/yjzhang/desktop/try/local_checkpoint/finetuned_vit_base_patch16_224.pth', '/data/yjzhang/desktop/key-driven-gqa_new_kv/final.pth')

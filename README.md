

## git 指令

- 查看当前分支 git branch
- 创建并进入新的分支 git checkout -b vitb-1.
- 查看当前暂存区状态 git status
- 将文件添加到暂存区 git add <file_name>
- 提交更改并添加修改说明 git commit -m "vitb-1."
- 将特定分支推送到网页 git push origin vitb-1.


## 更新版本

#### vitb-1.0

- vit_base_patch16_224.py  timm库的vit_base_patch16_224转化为不用调用的形式
- 只需运行 /data/yjzhang/desktop/try/not_share/key-driven-gqa/train_freeze.py
- 检查点路径 /data/yjzhang/desktop/try/ckpt/cifar100/4/model.pth
- 检查点来源 https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/blob/main/hyperparameters.json
- 只test，无train


#### vitb-1.1
- vit_base_patch16_224.py 与原始代码对齐，便于后面改为gqa
- vitb_gqa.py 只是vit_base_patch16_224.py的复制版本
- 将检查点的qkv分开，split后的检查点路径 /data/yjzhang/desktop/try/ckpt/split_qkv.pth
- 只test，无train


#### vitb-1.2
- train_freeze.py 调用mhsa，检查点为分立/data/yjzhang/desktop/try/ckpt/split_qkv.pth
- vitb_mhsa.py 加载检查点分散到各个子函数中（类原始github）
- vitb_gqa.py 暂未实现（只是mhsa的复制）


#### vitb-1.3

终于，原来是init里的加载检查点部分有问题。现在把所有外部接口维护好了。可以直接把以前的mhsa整个部分（包括加载检查点）粘过来直接用了。。。

- vitb_mhsa.py 实现原始mhsa
- vitb_gqa.py 实现原始gqa
- train_freeze.py 主代码，在import部分切换mhsa和gqa
- ckpt用原始的合并qkv
- cifar100_gqa训练检测结果保存在/data/yjzhang/desktop/try/not_share/key-driven-gqa/cifar100_results/gqa_1e-4_5_0.1_10.txt


#### vitb-1.4
自己的gqa

- packed_well_copy/adjacent_matrix_similarity/mean_var/grouping_experiment.py 重要性相似性分组，保存在/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin2
- vitb_my_gqa.py 自己的gqa，来自分支not-shared-2.1（对接口进行了改动，参数的传入传出）

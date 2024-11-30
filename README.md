
## 更新版本

#### vitb-1.0

- vit_base_patch16_224.py  timm库的vit_base_patch16_224转化为不用调用的形式
- 只需运行 /data/yjzhang/desktop/try/not_share/key-driven-gqa/train_freeze.py
- 检查点路径 /data/yjzhang/desktop/try/ckpt/cifar100/4/model.pth
- 检查点来源 https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/blob/main/hyperparameters.json


#### vitb-1.1
- vit_base_patch16_224.py 与原始代码对齐，便于后面改为gqa
- vitb_gqa.py 只是vit_base_patch16_224.py的复制版本
- 将检查点的qkv分开，split后的检查点路径 /data/yjzhang/desktop/try/ckpt/split_qkv.pth


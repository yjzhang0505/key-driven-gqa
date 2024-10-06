# Beyond Uniform Query Distribution: Key-Driven Grouped Query Attention

Official implementation of Key-Driven GQA as presented in our paper:
**Beyond Uniform Query Distribution: Key-Driven Grouped Query Attention** </br>
*Zohaib Khan\*, Muhammad Khaquan\*, Omer Tafveez, Burhanuddin Samiwala, Agha Ali Raza (* indicates equal contribution) <br>
Lahore University of Management Sciences  <br>


## Setup

Run `pip install -r requirements.txt`

* Download tiny-imagenet-200 from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
* Download CINIC-10 from [here](https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz)
* To use ImageNet-1k, you would have to log in to HuggingFace and provide a token - the simplest way is through `huggingface-cli login` on your terminal.

## Defining an experiment

We use `yaml` files to create our configurations. We expect the following structure:
```yaml
dataset: cifar10   # 'cifar10' # one of {cifar10, cifar100, food101, tiny-imagenet-200}
in_chans: 3
size: 'b'   # one of {s,b,l}
att_scheme: 'dgqa_diff'   # one of {gqa, kdgqa, dgqa_diff, dgqa_ema, pgqa}
num_classes: 10
pretrained: False
window_size: 2
num_kv_heads: 6   # dependent on the model size, and its number of heads
out_dir: "cifar10-base-gqa-pretrained"   # directory to which the outputs are saved
```

## Running

Use `train.py` and provide arguments:
- `--config`: path to the configuration file (must be `yaml`)
- `--out_dir`: path to the directory where to save outputs
- `--save_model`: whether to save the model checkpoints in the output directory
- `--pretrained_ckpt`: path to the checkpoint, if any, to use for the training (could be for uptraining or fine-tuning)

Example usage: `python train.py --config path/to/config.yaml --out_dir output_dir/ --save_model True --pretrained-ckpt path/to/ckpt.pth`

Actually usage: `python ./train_lr-5.py --config ./config/config_dgqa_diff_cifar.yaml --out_dir output_dgqa_diff_lr-5/ --save_model True`

## 更新

```
# pretrained=True时使用本地检查点
# 训练精度进度条
# train的main中设置为ViT-b标准参数
# 保存检查点前先把q、k、v合并为qkv，标准vit-b检查点格式
  可以在mhsa，pretrained=False下预训练，在gqa，pretrained=True下继续不报错了，精度有待进一步实验
# 加载检查点时先拆分。mhsa可以继续精度了，但gqa又mismatch了，把num_kv_head改为12就不报错了
--> 提交 restart
```

```
# 之前载检查点的时候用的是load_state_dict而非自己设置的函数，改掉后正常
# mhsa可以接着从0.5开始训练，gqa从0.1开始（比从0开始快一点）
--> 提交 restart1
```

```
# 手打exp_num版，可随机生成分组序列
# 输出路径去掉多余拼接，best.pth和final.pth放在正确文件夹，验证通过
# 可从config.yaml换num_kv_heads
# python train_lr-5.py --config config/config_pretrained.yaml --out_dir output/arbitrary/1 --save_model True
# 可从终端输入控制exp_num了
# 加上了run_experiments.sh（运行方法在该文件最后注释里），正跑20个实验，num_kv_heads=4, pretrained=True,att=gqa
--> 提交restart2
```

```
# 暂停实验，改为所有层共享相同参数，重新跑检查点
# is_same.py检验所有层是否共享相同参数，通过，开始跑随机实验了
--> 提交 rst-same-param
```

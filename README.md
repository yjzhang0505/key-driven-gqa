# Beyond Uniform Query Distribution: Key-Driven Grouped Query Attention

Official implementation of Key-Driven GQA as presented in our paper:
**Beyond Uniform Query Distribution: Key-Driven Grouped Query Attention** </br>
*Zohaib Khan\*, Muhammad Khaquan\*, Omer Tafveez, Burhanuddin Samiwala, Agha Ali Raza (* indicates equal contribution) <br>
Lahore University of Management Sciences  <br>

```bibtex
@misc{khan2024uniformquerydistributionkeydriven,
      title={Beyond Uniform Query Distribution: Key-Driven Grouped Query Attention}, 
      author={Zohaib Khan and Muhammad Khaquan and Omer Tafveez and Agha Ali Raza},
      year={2024},
      eprint={2408.08454},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.08454}, 
}
```

## Setup

Run `pip install -r requirements.txt`

* Download tiny-imagenet-200 from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
* Download CINIC-10 from [here](https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz)
* To use ImageNet-1k, you would have to log in to HuggingFace and provide a token - the simplest way is through `huggingface-cli login` on your terminal.

## Defining an experiment

We use `yaml` files to create our configurations. We expect the following structure:
```yaml
dataset: 					# 'cifar10' # one of {cifar10, cifar100, food101, tiny-imagenet-200}
in_chans: 3
size: 'b'					# one of {s,b,l}
att_scheme: 'gqa'				# one of {gqa, kdgqa, dgqa_diff, dgqa_ema, pgqa}
num_classes: 10
pretrained: True
window_size: 300
num_kv_heads: 6					# dependent on the model size, and its number of heads
out_dir: "cifar10-base-gqa-pretrained"		# directory to which the outputs are saved
```

## Running

Use `train.py` and provide arguments:
- `--config`: path to the configuration file (must be `yaml`)
- `--out_dir`: path to the directory where to save outputs
- `--save_model`: whether to save the model checkpoints in the output directory
- `--pretrained_ckpt`: path to the checkpoint, if any, to use for the training (could be for uptraining or fine-tuning)

Example usage: `python train.py --config path/to/config.yaml --out_dir output_dir/ --save_model True --pretrained-ckpt path/to/ckpt.pth`

## 更新版本

#### retry-2.0

- config.yaml
- 加上了随时精度显示
- change_ckpt.py: 把官方pth转换为本代码适用的pth命名
- ckpt_to_txt.py: 打印pth文件的模型结构到txt文件中
- 运行代码 `python train.py --config ./config.yaml --out_dir output --save_model True --pretrained_ckpt /data/yjzhang/desktop/PyTorch-Pretrained-ViT/jax_to_pytorch/weights/B_16_renamed.pth`  （其中，yaml中的 pretrained = False，代表不使用huggingface的预训练库，终端要写 --pretrained_ckpt，才是从本地加载检查点的路径
- 保存检查点时加上了qkv合并（使得保存的检查点可以重新用在代码里）
- 设置了随机分组
- run_experiments.sh
- 第三个epoch的test_acc<0.786则early stop
- 发现问题：随机的只是头，但权重矩阵还是用的平均分组平均池化的，要大改

#### retry-2.1

- (updated) 权重矩阵继承了随机的头，使用load，True则为加载检查点，要更新分组，False则为forward，继承分组，读取文件
- 出现问题：12层，每层更新一次。改为检查所要输出文件目录下的group.txt是否存在，不存在则分组，存在则读取文件。好处：将来复现实验可以直接读取，坏处：更新实验时记得先把这个文件/文件夹删除了才能跑，无法自动覆盖
- 头顺序随机后，算完还要把顺序返回来？（或者把proj的顺序也变了？）
- 每个头分别计算，其中对每个K和V和相应的多个Q计算，按Q的头的顺序保存，Linear的权重要对两个维度进行顺序调整（KV要先调整头的顺序，才能把原本不相邻的头变相邻然后合并，Linear之后对应的头应该不该序号，故Linear之后的顺序也是调整后的）
- batch_size, proxy_ratio, epoch
- 记得在group.py的读文件处改路径，有两处要改

### retry-2.2
- 待整理
- L2-norm.py    /data/yjzhang/desktop/try/key-driven-gqa/figure/L2_norm.py 依据L2范数画图
- box_plot.py  /data/yjzhang/desktop/try/key-driven-gqa/box_plot.py  箱线图（效果不好）
- dot-product_similarity.py  /data/yjzhang/desktop/try/key-driven-gqa/dot-product_similarity.py  点积相似性，保存到Excel，点积相似性效果不好
- grouping.py  /data/yjzhang/desktop/try/key-driven-gqa/grouping.py 读取相似性和重要性，两种标准5*6结合分组，分组打印到同一个txt
- importance.py  /data/yjzhang/desktop/try/key-driven-gqa/importance.py  计算五个标准的权重分布，均值和方差，保存在txt
- layer_is_same.py  /data/yjzhang/desktop/try/key-driven-gqa/layer_is_same.py  判断使用了层共享参数后是否真的共享了参数
- max_min.py  /data/yjzhang/desktop/try/key-driven-gqa/max_min.py  尝试提取每个运行结果的准确度最大最小值，这个文件还不能用，目前是手动和肉眼搜索和排序
- mhsa.yaml  /data/yjzhang/desktop/try/key-driven-gqa/mhsa.yaml  用来训练检查点的，att_scheme='mhsa'
- similarity.py  /data/yjzhang/desktop/try/key-driven-gqa/similarity.py  计算各头之间的余弦相似度，保存到Excel，每个子表为一个标准
- to_group_txt.py  /data/yjzhang/desktop/try/key-driven-gqa/to_group_txt.py  从刚刚grouping.py生成的单一txt，重新保存为多个数字文件夹下的group.txt
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

#### retry2.0

- config.yaml
- change_ckpt.py: 把官方pth转换为本代码适用的pth命名
- ckpt_to_txt.py: 打印pth文件的模型结构到txt文件中
- 运行代码 `python train.py --config ./config.yaml --out_dir output --save_model True --pretrained_ckpt /data/yjzhang/desktop/PyTorch-Pretrained-ViT/jax_to_pytorch/weights/B_16_renamed.pth`  （其中，yaml中的 pretrained = False，代表不使用huggingface的预训练库，终端要写 --pretrained_ckpt，才是从本地加载检查点的路径

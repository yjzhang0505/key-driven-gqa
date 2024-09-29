# Beyond Uniform Query Distribution: Key-Driven Grouped Query Attention

Official implementation of Key-Driven GQA as presented in our paper:
**Beyond Uniform Query Distribution: Key-Driven Grouped Query Attention** </br>
*Zohaib Khan\*, Muhammad Khaquan\*, Omer Tafveez, Burhanuddin Samiwala, Agha Ali Raza (* indicates equal contribution) <br>
Lahore University of Management Sciences  <br>


## Setup

Run `pip install -r requirements.txt`

* Python == 3.9
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
## 小记
- 本地检查点位置在model.py的checkpoint_path，目前只改了b的模型
- 训练检查点时要改一下路径


## Running

Use `train.py` and provide arguments:
- `--config`: path to the configuration file (must be `yaml`)
- `--out_dir`: path to the directory where to save outputs
- `--save_model`: whether to save the model checkpoints in the output directory
- `--pretrained_ckpt`: path to the checkpoint, if any, to use for the training (could be for uptraining or fine-tuning)

Example usage: `python train.py --config path/to/config.yaml --out_dir output_dir/ --save_model True --pretrained-ckpt path/to/ckpt.pth`

Actually usage: `python ./train_lr-5.py --config ./config/config_dgqa_diff_cifar.yaml --out_dir output_dgqa_diff_lr-5/ --save_model True`

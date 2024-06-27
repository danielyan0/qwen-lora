# 基于Intel GPU&CPU的模型微调及推理

## 方案介绍

方案的目的在于基于Intel的GPU&CPU，完成模型的微调和推理。

实施过程中，使用Intel的Arc 790完成模型的微调，因为显存限制，微调过程中使用Qwen1.5-7B-Chat。

微调使用Lora方法，数据集使用繁体唐诗，区分原始模型。

模型推理基于Intel CPU，使用Int4量化模型，加载微调后的Lora增量，以增量方式进行。

方案的目标是在通用大模型基础上，完成古典唐诗的自动生成；方案落地过程中，使用了Intel GPU训练、CPU推理，GPTQ量化等多项技术。

## 环境介绍

### 硬件

CPU：Intel Core i7-12700K（微调）/Intel Platinum 6462C（推理）(aliyun)

GPU：Arc 790（微调）/-（推理）

内存：64G （微调）/32G（推理）

### 模型&数据集

基础模型：Qwen1.5-7B-Chat

微调数据集：唐诗（https://github.com/chinese-poetry/chinese-poetry/tree/master/%E5%85%A8%E5%94%90%E8%AF%97）

### Intel软件工具

1. **[intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch)**
2. **[intel-extension-for-deepspeed](https://github.com/intel/intel-extension-for-deepspeed)**
3. **[neural-compressor](https://github.com/intel/neural-compressor)**

## 环境准备

```bash
docker run -ti --gpus all -d --net=host --name python_lora -v .:/root/app ubuntu:22.04
docker run -ti --name python_lora /bin/bash 
# 安装依赖
apt update && apt install -y python3 python3-pip python-is-python3 vim git
```

## 模型微调

1. 下载模型文件

```bash
apt install git-lfs
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen1.5-7B-Chat
cd Qwen1.5-7B-Chat
git lfs pull
```
2. 拉取微调项目

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
```

3. 安装依赖

```bash
docker run -ti --gpus all -d --net=host --name python_lora -v .:/root/app ubuntu:22.04
docker run -ti -d --name python_lora 
# 安装torch & intel-extension-for-pytorch
python -m pip install torch==2.0.0 torchvision==0.14.0 intel-extension-for-pytorch==2.0.110+xpu intel_extension_for_pytorch_deepspeed==2.1.30 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

cd LLaMA-Factory
pip install -r requirements.txt
pip install -e .
# Qwen1.5 依赖
pip install auto_gptq optimum
```

4. 新建配置文件qwen_lora.yaml

```yaml
### model
model_name_or_path: models/Qwen1.5-7B-Chat
quantization_bit: 4

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: 8
lora_alpha: 16
lora_dropout: 0

### dataset
dataset: tang
template: qwen
cutoff_len: 1024
#max_samples: 311855
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/Qwen1.5-7B-Chat/lora/tang
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.0001
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_steps: 0.1
fp16: true

### eval
val_size: 0.1
per_device_eval_batch_size: 1
evaluation_strategy: steps
eval_steps: 500
```

5. 开始微调

微调过程中，如果有依赖缺失，根据情况安装。

```bash
GRADIO_SHARE=1 llamafactory-cli train qwen_lora.yaml 1>output.txt 2>&1 &
```

等待几十个小时，微调完成。

## 模型推理

推理使用经过4bit gptq量化后的模型Qwen/Qwen1.5-7B-Chat-GPTQ-Int4，模型可以在Intel CPU上进行推理，推理继续使用LLaMA-Factory框架。


```dockerfile
docker run -ti -d --net=host --name excute_lora -v .:/root/app ubuntu:22.04
docker run -ti --name python_lora /bin/bash 
# 安装依赖
apt update && apt install -y python3 python3-pip python-is-python3 vim git
```

1. 下载量化模型

```bash
apt install git-lfs
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ-Int4
cd Qwen1.5-7B-Chat-GPTQ-Int4
git lfs pull
```
2. 拉取微调项目

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
```

3. 安装依赖

```bash
docker run -ti --gpus all -d -p 8000:7860 --name python_lora -v .:/root/app ubuntu:22.04
docker run -ti -d --name python_lora 
# 安装torch & intel-extension-for-pytorch
cd LLaMA-Factory
pip install -r requirements.txt
pip install -e .
# Qwen1.5 依赖
pip install auto_gptq optimum
```

### 使用命令行推理

1. 创建配置文件qwen_lora_gptq.yaml

```yaml
model_name_or_path: models/Qwen1.5-7B-Chat-GPTQ-Int4
adapter_name_or_path: saves/Qwen1.5-7B-Chat/lora/tang
template: qwen
finetuning_type: lora
```

2. 启动推理

```bash
llamafactory-cli chat qwen_lora_gptq.yaml
```

3. 推理输出

```bash
Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.

User: 写一首唐诗
Assistant: 春風和暖日輝輝，柳岸花汀水滿溪。 却憶故園行樂處，醉歸紅粉笑相攜。

User: 写一首柳树相关的唐诗
Assistant: 春風吹綠野，春雨漲春塘。 聞道東郊路，千門萬戶香。
```

### 使用Web页面推理

启动Web页面

```bash
python src/webui.py
```

访问页面，开始推理：

http://{IP}:8000

## 微调前后对比

| 模型                             | 输入       | 输出                                                         |
| -------------------------------- | ---------- | ------------------------------------------------------------ |
| Qwen1.5-7B-Chat-GPTQ-Int4        | 写一首唐诗 | 题目：春望<br>国破山河在，城春草木深。<br/> 感时花溅泪，恨别鸟惊心。<br/> 烽火连三月，家书抵万金。<br/> 白头搔更短，浑欲不胜簪。 |
| Qwen1.5-7B-Chat-GPTQ-Int4 + Lora | 写一首唐诗 | 風葉打窗時，獨坐書一卷。 秋來亦已多，吾志亦未遠。            |

